"""Advanced RAG bench on the Tessmoor corpus.

Strategies: hybrid + rerank (the main retriever), HyDE, multi-query, step-back, query decomposition,
a Self-RAG critique loop, Azure AI Search agentic retrieval (hybrid + semantic ranker), and
GraphRAG local search against dense vector RAG.

Each answer is the extractive reader, then the ordered failure taxonomy plus the RAGAS / Azure AI
Evaluation metrics (mock unless ``RAGFAIL_LIVE_JUDGES=1``).
"""

from __future__ import annotations

import numpy as np

from . import data
from .agentic_search import agentic_chunks, live_agentic_enabled
from .extractive import extractive_answer
from .graphrag import live_graphrag_enabled, local_chunks
from .judges import judge_rows, live_judges_enabled
from .metrics import rate
from .query_rewrite import live_llm_enabled, live_rewrites
from .retrieval import apply_guards, rank, rrf
from .scoring import CATEGORIES, score
from .self_rag import self_rag

STRATEGIES = (
    "hybrid_rerank",
    "hyde",
    "multi_query",
    "step_back",
    "decomposition",
    "self_rag",
    "agentic",
    "graphrag",
    "vector",
)
K = 3


def _dense(retriever, q: dict) -> list[int]:
    return rank(retriever.emb @ retriever.qemb[q["qid"]].astype(np.float32))


def _hyde_vector(retriever, text: str) -> np.ndarray:
    """Embed a hypothetical document as a BM25-weighted average of chunk vectors."""
    scores = retriever.bm25.scores(text)
    idx = np.argsort(-scores, kind="mergesort")[:20]
    weights = np.maximum(scores[idx], 0)
    if float(weights.sum()) == 0.0:
        weights = np.ones(len(idx))
    weights = weights / weights.sum()
    pseudo = (retriever.emb[idx] * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(pseudo))
    return pseudo / norm if norm else pseudo


def _fuse_bm25(retriever, queries: list[str], q: dict) -> list[int]:
    rankings = [rank(retriever.bm25.scores(text)) for text in queries]
    rankings.append(_dense(retriever, q))
    return rrf(*rankings)


def _guarded(retriever, order: list[int], k: int = K) -> list[dict]:
    order = apply_guards(order, retriever.chunks)
    return [retriever.chunks[i] for i in order[:k]]


def retrieve(name: str, q: dict, retriever) -> tuple[list[dict], str | None, list[str]]:
    """Return chunks, an optional graph route, and critique notes."""
    if name == "hybrid_rerank":
        return retriever.top(q, "hybrid_rerank", K), None, []
    if name == "vector":
        return retriever.top(q, "dense", K), None, []
    if name == "hyde":
        hypo = live_rewrites(q["question"], "hyde")[0]
        pseudo = _hyde_vector(retriever, hypo).astype(np.float32)
        order = rrf(rank(retriever.bm25.scores(hypo)), rank(retriever.emb @ pseudo), _dense(retriever, q))
        return _guarded(retriever, order), None, []
    if name in ("multi_query", "step_back", "decomposition"):
        return _guarded(retriever, _fuse_bm25(retriever, live_rewrites(q["question"], name), q)), None, []
    if name == "agentic":
        return agentic_chunks(q, retriever, K), None, []
    if name == "graphrag":
        chunks, route = local_chunks(q, retriever, K)
        return chunks, route, []
    if name == "self_rag":
        answer, chunks, notes = self_rag(q, retriever, K)
        return chunks, None, notes + [f"answer:{answer}"]
    raise ValueError(f"unknown strategy {name}")


def _reference(q: dict) -> str:
    if not q["answerable"]:
        return "NOT_FOUND"
    return f"{q['gold']:g}. {q['gold_sentence']}"


def run_advanced(strategies: tuple[str, ...] = STRATEGIES, *, ragas_eval=None, azure_eval=None) -> dict:
    questions = data.questions()
    retriever = data.retriever("w80")
    out_strategies = {}
    routes = {"local": 0, "vector_fallback": 0}
    retries = 0
    for name in strategies:
        rows = []
        for q in questions:
            chunks, route, notes = retrieve(name, q, retriever)
            if route:
                routes[route] = routes.get(route, 0) + 1
            if any(n.endswith("retry") for n in notes):
                retries += 1
            answer = notes[-1].split("answer:", 1)[1] if notes and notes[-1].startswith("answer:") else extractive_answer(q["question"], chunks)
            shown = [c["titled"] for c in chunks]
            scored = score(q, answer, shown, chunks)
            rows.append({
                "question": q, "answer": answer, "chunks": chunks, "all_chunks": retriever.chunks,
                "scored": scored, "contexts_text": shown, "reference": _reference(q),
            })
        answerable = [r for r in rows if r["question"]["answerable"]]
        unanswerable = [r for r in rows if not r["question"]["answerable"]]
        stale = [r for r in answerable if r["question"]["stale_value"] is not None]
        backend, judged = judge_rows(rows, ragas_eval=ragas_eval, azure_eval=azure_eval)
        out_strategies[name] = {
            "accuracy": rate(sum(r["scored"]["correct"] for r in answerable), len(answerable)),
            "abstain_when_unanswerable": rate(sum(r["scored"]["abstained"] for r in unanswerable), len(unanswerable)),
            "stale_answers": rate(sum(r["scored"]["stale_answer"] for r in stale), len(stale)) if stale else None,
            "taxonomy": {c: sum(r["scored"]["category"] == c for r in rows) for c in CATEGORIES},
            "judge": judged,
            "judge_backend": backend,
        }
    backend = next(iter(out_strategies.values()), {}).get("judge_backend", "mock")
    live = [name for name, on in (("llm", live_llm_enabled()), ("agentic", live_agentic_enabled()),
                                  ("graphrag", live_graphrag_enabled()),
                                  ("judges", bool(live_judges_enabled() or ragas_eval or azure_eval))) if on]
    return {
        "mode": "mock" if not live else "live:" + ",".join(live),
        "judge_scale": "azure-1-to-5" if backend != "mock" else "unit",
        "reader": "extractive",
        "k": K,
        "questions": len(questions),
        "strategies": out_strategies,
        "graphrag_routes": routes,
        "self_rag_retries": retries if "self_rag" in strategies else 0,
        "failure_categories": list(CATEGORIES),
    }


def summary_lines(report: dict) -> list[str]:
    if report.get("mode", "mock") == "mock":
        head = ("mode: mock. Judges: RAGAS and Azure AI Evaluation metric names, scored by code. "
                "Set RAGFAIL_LIVE_JUDGES=1 (and install the [judges] extra) to call the libraries.")
    else:
        head = f"mode: {report['mode']}. judge_scale: {report.get('judge_scale', 'unit')}."
    lines = [
        head,
        f"questions: {report['questions']}. Reader: extractive. k={report['k']}.",
        "",
        f"{'strategy':<16} {'accuracy':>8} {'abstain':>8} {'faith':>8} {'ctx_rec':>8} {'ctx_prec':>8} {'stale':>8}  top_failure",
    ]
    for name, row in report["strategies"].items():
        tax = row["taxonomy"]
        top = max((c for c in tax if c not in ("correct", "correct_abstain")), key=lambda c: (tax[c], c))
        judge = row["judge"]
        stale = "n/a" if row["stale_answers"] is None else f"{row['stale_answers']['rate']:.4f}"
        lines.append(
            f"{name:<16} {row['accuracy']['rate']:>8.4f} {row['abstain_when_unanswerable']['rate']:>8.4f} "
            f"{judge['faithfulness']:>8.4f} {judge['context_recall']:>8.4f} {judge['context_precision']:>8.4f} "
            f"{stale:>8}  {top} ({tax[top]})"
        )
    g, v = report["strategies"].get("graphrag"), report["strategies"].get("vector")
    if g and v:
        lines += ["",
                  (f"GraphRAG context_recall {g['judge']['context_recall']:.4f} "
                   f"(local {report['graphrag_routes'].get('local', 0)}, "
                   f"vector fallback {report['graphrag_routes'].get('vector_fallback', 0)}) "
                   f"vs vector RAG {v['judge']['context_recall']:.4f}.")]
    if "self_rag" in report["strategies"]:
        lines.append(f"Self-RAG questions with a critique retry: {report['self_rag_retries']}.")
    if "agentic" in report["strategies"]:
        lines.append("Agentic retrieval: offline planner + hybrid + cross-encoder stand-in for the semantic ranker. "
                     "Live: AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KNOWLEDGE_BASE.")
    lines.append("Live Azure columns for this bench are not measured yet.")
    return lines
