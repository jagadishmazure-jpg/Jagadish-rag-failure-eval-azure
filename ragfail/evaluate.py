"""Full evaluation: retrieval metrics computed live, generation metrics replayed from measured answers."""

from __future__ import annotations

from collections import Counter

from . import data
from .chunking import STRATEGIES, gold_chunks
from .configs import CONFIGS, config, context_for
from .guards import strip_instructions
from .metrics import mcnemar_exact, percentile, rate
from .retrieval import Retriever
from .scoring import CATEGORIES, FAILURES, score

KS = (1, 3, 5, 10)


def _injected_chunk(c: dict) -> bool:
    return any(d["doc_id"] == c["doc_id"] and d["injection_canary"] for d in data.corpus())


def retrieval_metrics(qs: list[dict]) -> dict:
    ans = [q for q in qs if q["answerable"]]
    out: dict = {"grid": {}}
    for s in STRATEGIES:
        for field in ("titled", "text"):
            r = data.retriever(s, field)
            for setup in Retriever.SETUPS:
                if setup == "hybrid_rerank" and field == "text":
                    continue  # the cross-encoder only scored titled chunks
                ranks = []
                for q in ans:
                    gold = gold_chunks(r.chunks, q)
                    order = r.order(q, setup)
                    ranks.append(next((i + 1 for i, idx in enumerate(order[:10]) if r.chunks[idx]["chunk_id"] in gold), None))
                row = {f"recall@{k}": rate(sum(1 for x in ranks if x and x <= k), len(ranks)) for k in KS}
                row["mrr@10"] = round(sum(1 / x for x in ranks if x) / len(ranks), 4)
                out["grid"][f"{s}.{field}.{setup}"] = row
    r = data.retriever("w80")
    out["by_style_recall@3"] = {
        setup: {st: rate(sum(bool(gold_chunks(r.chunks, q) & {c["chunk_id"] for c in r.top(q, setup, 3)})
                             for q in ans if q["style"] == st), sum(1 for q in ans if q["style"] == st))
                for st in ("lexical", "paraphrase")} for setup in Retriever.SETUPS}
    sc = [q for q in ans if q["stale_value"] is not None]
    out["stale_exposure@3"] = {
        "guards_off": rate(sum(any(c["status"] != "current" or c["doc_type"] != "policy"
                                   for c in r.top(q, "hybrid_rerank", 3, guards=False)) for q in sc), len(sc)),
        "guards_on": rate(sum(any(c["status"] != "current" or c["doc_type"] != "policy"
                                  for c in r.top(q, "hybrid_rerank", 3, guards=True)) for q in sc), len(sc))}
    out["wrong_place_top1"] = {}
    for field in ("titled", "text"):
        rr = data.retriever("para", field)
        fam = [q for q in ans if q["family"] and "-" in q["family"]]
        bad = 0
        for q in fam:
            top = rr.top(q, "hybrid", 1)[0]
            bad += top["topic"] == q["family"].split("-")[0] and top["district"] is not None and top["family"] != q["family"]
        out["wrong_place_top1"][f"para.{field}.hybrid"] = rate(bad, len(fam))
    clean = [c for st in STRATEGIES for c in data.chunks(st) if not _injected_chunk(c)]
    out["instruction_filter"] = {
        "clean_chunks": len(clean),
        "false_positives": sum(strip_instructions(c["titled"])[1] > 0 for c in clean),
        "injected_chunks": sum(1 for st in STRATEGIES for c in data.chunks(st) if _injected_chunk(c) and
                               any(x in c["text"] for x in ("example", "RED-7"))),
    }
    return out


def score_answers(rows: list[dict], qs: dict[str, dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        q = qs[row["qid"]]
        cfg = config(row["config"])
        ctx = context_for(q, cfg, data.retriever(cfg["strategy"]))
        if [c["chunk_id"] for c in ctx] != row["context"]:
            raise data.DataError(f"answers.jsonl row {row['config']}/{row['qid']}: context no longer matches "
                                 "the retriever; rerun scripts/run_reader.py")
        s = score(q, row["answer"], row["shown"], ctx)
        s.update(qid=q["qid"], latency_s=row["latency_s"], prompt_tokens=row["prompt_tokens"],
                 exposed=any(_injected_chunk(c) and any(x in t for x in ("example", "RED-7"))
                             for c, t in zip(ctx, row["shown"])))
        out.setdefault(row["config"], []).append(s)
    return out


def generation_metrics(scored: dict[str, list[dict]], qs: dict[str, dict]) -> dict:
    out = {}
    for name, rows in scored.items():
        a = [r for r in rows if qs[r["qid"]]["answerable"]]
        u = [r for r in rows if not qs[r["qid"]]["answerable"]]
        answered = [r for r in rows if not r["abstained"]]
        st = [r for r in a if qs[r["qid"]]["stale_value"] is not None]
        exp = [r for r in rows if r["exposed"]]
        m = {
            "accuracy": rate(sum(r["correct"] for r in a), len(a)),
            "correct_and_cited": rate(sum(r["correct"] and r["citation_ok"] for r in a), len(a)),
            "false_abstain": rate(sum(r["abstained"] for r in a), len(a)),
            "faithfulness": rate(sum(r["faithful"] for r in answered), len(answered)) if answered else None,
            "grounded": rate(sum(r["grounded"] for r in answered), len(answered)) if answered else None,
            "stale_answers": rate(sum(r["stale_answer"] for r in st), len(st)) if st else None,
            "latency_p50_s": round(percentile([r["latency_s"] for r in rows], 50), 3),
            "latency_p95_s": round(percentile([r["latency_s"] for r in rows], 95), 3),
            "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in rows) / len(rows), 1),
            "categories": {c: n for c, n in Counter(r["category"] for r in rows).most_common()},
        }
        if u:
            m["abstain_when_unanswerable"] = rate(sum(r["abstained"] for r in u), len(u))
        m["injection_exposed"] = len(exp)
        if exp:
            m["injection_success"] = rate(sum(r["injected"] for r in exp), len(exp))
        out[name] = m
    return out


def paired(scored: dict[str, list[dict]], qs: dict[str, dict], base: str = "main") -> dict:
    b = {r["qid"]: r["correct"] for r in scored[base] if qs[r["qid"]]["answerable"]}
    out = {}
    for name, rows in scored.items():
        if name == base:
            continue
        o = {r["qid"]: r["correct"] for r in rows if qs[r["qid"]]["answerable"]}
        common = sorted(set(b) & set(o))
        only_base = sum(b[q] and not o[q] for q in common)
        only_other = sum(o[q] and not b[q] for q in common)
        out[name] = {"main_only": only_base, "other_only": only_other, "p_value": round(mcnemar_exact(only_base, only_other), 4)}
    return out


def taxonomy(scored: dict[str, list[dict]]) -> dict:
    return {name: {c: sum(r["category"] == c for r in rows) for c in CATEGORIES} for name, rows in scored.items()}


def by_kind(scored: dict[str, list[dict]], qs: dict[str, dict], name: str) -> dict:
    out = {}
    for kind in ("plain", "stale", "conflict", "injection", "unanswerable"):
        rows = [r for r in scored[name] if kind in qs[r["qid"]]["kinds"]]
        if not rows:
            continue
        good = sum(r["category"] in ("correct", "correct_abstain") for r in rows)
        out[kind] = rate(good, len(rows))
    return out


def evaluate() -> dict:
    qlist = data.questions()
    qs = {q["qid"]: q for q in qlist}
    scored = score_answers(data.read_jsonl(data.DATA / "answers.jsonl"), qs)
    missing = [n for n in CONFIGS if n not in scored]
    if missing:
        raise data.DataError(f"answers.jsonl has no rows for: {', '.join(missing)}")
    kinds = Counter(k for q in qlist for k in q["kinds"])
    return {
        "data": {"documents": len(data.corpus()), "questions": len(qlist),
                 "answerable": sum(q["answerable"] for q in qlist), "facts": len({q["fact_id"] for q in qlist if q["fact_id"]}),
                 "kinds": dict(sorted(kinds.items())),
                 "chunks": {s: len(data.chunks(s)) for s in STRATEGIES}},
        "retrieval": retrieval_metrics(qlist),
        "generation": generation_metrics(scored, qs),
        "paired_vs_main": paired(scored, qs),
        "taxonomy": taxonomy(scored),
        "main_by_kind": by_kind(scored, qs, "main"),
        "no_guards_by_kind": by_kind(scored, qs, "no_guards"),
        "failure_categories": list(FAILURES),
        "reader_calls": {"rows": sum(len(v) for v in scored.values())},
    }
