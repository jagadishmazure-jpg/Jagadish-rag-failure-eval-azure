"""RAG vs a fine-tuned model vs RAG + fine-tuning (RAFT), on the same questions and taxonomy.

Offline, the fine-tuned arm is a behavioral mock of a QLoRA adapter served somewhere else:

* about 70% of facts (by SHA-256 of the fact id) are in the training split and are answered from memory,
* facts that have a superseded value were learned from that old number, so a later policy edit is missed,
* held-out facts and unanswerable questions are answered with 9999, a number that is not in the corpus.

RAFT retrieves with the guarded hybrid retriever, drops chunks from the wrong district, then uses the
extractive reader. That is the offline stand-in for a reader trained to ignore distractor passages.

Set ``FINETUNE_ENDPOINT`` to POST the fine-tune and RAFT arms at a served adapter. RAG stays local.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request

from . import data
from .extractive import extractive_answer
from .judges import judge_rows
from .metrics import rate
from .query_rewrite import detect_district
from .scoring import CATEGORIES, score

ARMS = ("rag", "finetune", "raft")
HELD_OUT = "9999"
K = 3


def in_train(fact_id: str) -> bool:
    """Stable 70% split. The first byte of SHA-256 is uniform, and 179/256 is about 0.70."""
    return int(hashlib.sha256(fact_id.encode()).hexdigest()[:2], 16) < 179


def finetune_payload(question: str, passages: list[dict] | None, mode: str) -> dict:
    if mode not in ("finetune", "raft"):
        raise ValueError("mode must be finetune or raft")
    shown = [] if mode == "finetune" else [
        {"chunk_id": p["chunk_id"], "text": p.get("titled") or p.get("text") or ""} for p in (passages or [])
    ]
    return {"mode": mode, "question": question, "passages": shown}


def live_finetune_enabled() -> bool:
    return bool(os.environ.get("FINETUNE_ENDPOINT"))


def endpoint_accepts_token(url: str, token: str | None) -> None:
    """A bearer token goes only over HTTPS, or to localhost for a local adapter."""
    if not token:
        return
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" or host in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError("FINETUNE_TOKEN requires an https endpoint, or localhost")


def _post_finetune(question: str, passages: list[dict] | None, mode: str) -> str:
    url = os.environ["FINETUNE_ENDPOINT"]
    token = os.environ.get("FINETUNE_TOKEN")
    endpoint_accepts_token(url, token)
    payload = finetune_payload(question, passages, mode)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - endpoint is the caller's
        body = json.loads(resp.read().decode())
    if "answer" not in body:
        raise RuntimeError("FINETUNE_ENDPOINT response has no answer")
    return str(body["answer"])


def _fmt(value: float) -> str:
    return f"{value:g}"


def mock_finetune_answer(q: dict) -> str:
    """Memory-only answer. No passage, so a right number still fails the citation check."""
    if not q["answerable"] or not q["fact_id"] or not in_train(q["fact_id"]):
        return HELD_OUT
    if q.get("stale_value") is not None:
        return _fmt(float(q["stale_value"]))
    return _fmt(float(q["gold"]))


def _raft_chunks(q: dict, retriever) -> list[dict]:
    chunks = retriever.top(q, "hybrid_rerank", K)
    district = detect_district(q["question"])
    if not district:
        return chunks
    return [c for c in chunks if c["district"] in (district, None)]


def _arm_rows(name: str, questions: list[dict], retriever) -> tuple[list[dict], int]:
    rows = []
    dropped = 0
    live = live_finetune_enabled()
    for q in questions:
        if name == "rag":
            chunks = retriever.top(q, "hybrid_rerank", K)
            answer = extractive_answer(q["question"], chunks)
        elif name == "finetune":
            chunks = []
            answer = _post_finetune(q["question"], None, "finetune") if live else mock_finetune_answer(q)
        else:
            raw = retriever.top(q, "hybrid_rerank", K)
            chunks = _raft_chunks(q, retriever)
            dropped += len(raw) - len(chunks)
            answer = _post_finetune(q["question"], chunks, "raft") if live else extractive_answer(q["question"], chunks)
        shown = [c["titled"] for c in chunks]
        scored = score(q, answer, shown, chunks)
        reference = "NOT_FOUND" if not q["answerable"] else f"{q['gold']:g}. {q['gold_sentence']}"
        rows.append({"question": q, "answer": answer, "chunks": chunks, "all_chunks": retriever.chunks,
                     "scored": scored, "contexts_text": shown, "reference": reference})
    return rows, dropped


def run_adaptation(*, ragas_eval=None, azure_eval=None) -> dict:
    questions = data.questions()
    retriever = data.retriever("w80")
    fact_ids = sorted({q["fact_id"] for q in questions if q["fact_id"]})
    arms = {}
    dropped = 0
    scale = "unit"
    for name in ARMS:
        rows, dropped_n = _arm_rows(name, questions, retriever)
        dropped += dropped_n
        answerable = [r for r in rows if r["question"]["answerable"]]
        unanswerable = [r for r in rows if not r["question"]["answerable"]]
        stale = [r for r in answerable if r["question"]["stale_value"] is not None]
        backend, judged = judge_rows(rows, ragas_eval=ragas_eval, azure_eval=azure_eval)
        scale = "azure-1-to-5" if backend != "mock" else "unit"
        arms[name] = {
            "accuracy": rate(sum(r["scored"]["correct"] for r in answerable), len(answerable)),
            "abstain_when_unanswerable": rate(sum(r["scored"]["abstained"] for r in unanswerable), len(unanswerable)),
            "stale_answers": rate(sum(r["scored"]["stale_answer"] for r in stale), len(stale)) if stale else None,
            "citation_ok": rate(sum(r["scored"]["citation_ok"] for r in answerable), len(answerable)),
            "taxonomy": {c: sum(r["scored"]["category"] == c for r in rows) for c in CATEGORIES},
            "judge": judged,
            "judge_backend": backend,
        }
    live_ft = live_finetune_enabled()
    report = {
        "mode": "live-finetune" if live_ft else "mock",
        "judge_scale": scale,
        "questions": len(questions),
        "train_facts": sum(in_train(f) for f in fact_ids),
        "held_out_facts": sum(not in_train(f) for f in fact_ids),
        "finetune_description": (
            "Fine-tune and RAFT answers came from FINETUNE_ENDPOINT. The RAG arm stayed local."
            if live_ft else
            "Mock QLoRA adapter. Trained facts are answered from memory. Facts with a superseded "
            "value were learned from the old number. Held-out and unanswerable questions return 9999."
        ),
        "distractors_dropped": dropped,
        "arms": arms,
    }
    report["decision"] = decision_guide(report)
    return report


def decision_guide(report: dict) -> str:
    rag, ft, raft = (report["arms"][name] for name in ARMS)
    lines = []
    if ft["abstain_when_unanswerable"]["rate"] + 0.05 < rag["abstain_when_unanswerable"]["rate"]:
        lines.append("Unanswerable questions: the fine-tuned arm abstains less often than RAG. "
                     "Prefer RAG or RAFT when staff can ask for something the corpus does not contain.")
    if ft["stale_answers"] and rag["stale_answers"] and ft["stale_answers"]["rate"] > rag["stale_answers"]["rate"]:
        lines.append("Policy changes: the fine-tuned arm repeats superseded numbers it was trained on. "
                     "Prefer RAG with the metadata guard when a fact can change after training.")
    raft_ahead = (raft["accuracy"]["rate"] > rag["accuracy"]["rate"] + 0.01
                  or raft["abstain_when_unanswerable"]["rate"] > rag["abstain_when_unanswerable"]["rate"] + 0.01)
    if raft_ahead:
        lines.append("RAFT (retrieve, then a reader that drops the wrong district) beats guarded RAG "
                     "on accuracy or abstention in this mock. Prefer it when you can train on the question, "
                     "the gold passage and distractor passages.")
    elif (raft["accuracy"]["rate"] + 0.01 >= rag["accuracy"]["rate"]
          and raft["abstain_when_unanswerable"]["rate"] + 0.01 >= rag["abstain_when_unanswerable"]["rate"]):
        n = report.get("distractors_dropped", 0)
        lines.append(f"RAFT matches guarded RAG here. The district filter dropped {n} distractor chunks, "
                     "and the metadata guard had already removed superseded text, so the score did not move. "
                     "Prefer plain RAG until a trained reader has to face distractors the guard does not catch.")
    elif rag["accuracy"]["rate"] >= ft["accuracy"]["rate"]:
        lines.append("RAG is ahead of the fine-tuned arm on numeric accuracy. Fine-tune when the job is "
                     "style or format and the facts stay in the prompt.")
    if ft["citation_ok"]["rate"] + 0.05 < rag["citation_ok"]["rate"]:
        lines.append("Citations: a fine-tuned answer has no passage, so a correct number is "
                     "citation_mismatch in the taxonomy. Use that arm only when a citation is not required.")
    if report.get("mode") == "mock":
        lines.append("Point FINETUNE_ENDPOINT at a served QLoRA adapter to replace the mock fine-tune and RAFT answers. "
                     "Live rows are not measured.")
    else:
        lines.append("Fine-tune and RAFT answers came from FINETUNE_ENDPOINT. RAG stayed on the local retriever.")
    return " ".join(lines)


def summary_lines(report: dict) -> list[str]:
    lines = [
        f"mode: {report['mode']}. Train facts: {report['train_facts']}. Held-out facts: {report['held_out_facts']}.",
        report["finetune_description"],
        "",
        f"{'arm':<10} {'accuracy':>8} {'abstain':>8} {'stale':>8} {'cited':>8}  top_failure",
    ]
    for name, row in report["arms"].items():
        tax = row["taxonomy"]
        top = max((c for c in tax if c not in ("correct", "correct_abstain")), key=lambda c: (tax[c], c))
        stale = "n/a" if row["stale_answers"] is None else f"{row['stale_answers']['rate']:.4f}"
        lines.append(f"{name:<10} {row['accuracy']['rate']:>8.4f} {row['abstain_when_unanswerable']['rate']:>8.4f} "
                     f"{stale:>8} {row['citation_ok']['rate']:>8.4f}  {top} ({tax[top]})")
    lines += ["", report["decision"]]
    return lines
