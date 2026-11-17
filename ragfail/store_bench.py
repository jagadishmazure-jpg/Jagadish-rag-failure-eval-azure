"""Benchmark the pluggable stores on recall@k, metadata filters and estimated monthly cost.

p95 latency is measured around the in-process call and returned to the caller. It is not part of the
committed table: it changes with the machine. Live service latency stays unmeasured until the store
env vars are set and ``vectorstores.live_search`` is used.
"""

from __future__ import annotations

import copy
import time

from . import data
from .chunking import gold_chunks
from .metrics import percentile, rate
from .vectorstores import STORE_NAMES, build_stores, monthly_cost

K = 3
STATUS_FILTER = {"status": "current"}
POLICY_FILTER = {"status": "current", "doc_type": "policy"}


def _hits(store, q: dict, qemb, flt: dict | None) -> tuple[list, float]:
    start = time.perf_counter()
    found = store.search(qemb[q["qid"]], K, query_text=q["question"], flt=flt, qid=q["qid"])
    return found, (time.perf_counter() - start) * 1000


def run_store_bench() -> dict:
    questions = data.questions()
    retriever = data.retriever("w80")
    stores = build_stores(retriever.chunks, retriever.emb, retriever.rr)
    answerable = [q for q in questions if q["answerable"]]
    stale_qs = [q for q in answerable if q["stale_value"] is not None]
    rows = []
    for store in stores:
        lat, hit, hit_f, hit_p = [], 0, 0, 0
        stale_off, stale_status, stale_policy = 0, 0, 0
        district_ok = 0
        district_n = 0
        for q in answerable:
            found, ms = _hits(store, q, retriever.qemb, None)
            lat.append(ms)
            gold = gold_chunks(retriever.chunks, q)
            hit += any(h.chunk_id in gold for h in found)
            found_f, ms_f = _hits(store, q, retriever.qemb, STATUS_FILTER)
            lat.append(ms_f)
            hit_f += any(h.chunk_id in gold for h in found_f)
            found_p, ms_p = _hits(store, q, retriever.qemb, POLICY_FILTER)
            lat.append(ms_p)
            hit_p += any(h.chunk_id in gold for h in found_p)
            if q["family"] and "-" in q["family"]:
                # District values in the corpus are capitalised; the family suffix is lower case.
                wanted = next((c["district"] for c in retriever.chunks if c["family"] == q["family"] and c["district"]), None)
                if wanted:
                    scoped, _ = _hits(store, q, retriever.qemb, {"district": wanted})
                    district_n += 1
                    district_ok += bool(scoped) and all(h.chunk.get("district") == wanted for h in scoped)
        def _exposed(found) -> bool:
            return any(h.chunk["status"] != "current" or h.chunk["doc_type"] != "policy" for h in found)

        for q in stale_qs:
            off, _ = _hits(store, q, retriever.qemb, None)
            status_only, _ = _hits(store, q, retriever.qemb, STATUS_FILTER)
            policy, _ = _hits(store, q, retriever.qemb, POLICY_FILTER)
            stale_off += _exposed(off)
            stale_status += _exposed(status_only)
            stale_policy += _exposed(policy)
        cost = monthly_cost(store.kind)
        rows.append({
            "store": store.name,
            "ann": store.ann,
            "live": store.live,
            "recall@3": rate(hit, len(answerable)),
            "recall@3_filtered": rate(hit_f, len(answerable)),
            "recall@3_policy_filtered": rate(hit_p, len(answerable)),
            "stale_in_top3": rate(stale_off, len(stale_qs)),
            "stale_in_top3_status_filter": rate(stale_status, len(stale_qs)),
            "stale_in_top3_policy_filter": rate(stale_policy, len(stale_qs)),
            "district_filter_ok": rate(district_ok, district_n),
            "p95_ms": round(percentile(lat, 95), 3),
            "usd_month": cost["usd_month"],
            "sku": cost["sku"],
            "cost_breakdown": cost["breakdown"],
        })
    return {"mode": "mock", "k": K, "questions_answerable": len(answerable), "stores": rows,
            "live_placeholders": {name: "not measured" for name in STORE_NAMES if name != "memory"}}


def committed_view(report: dict) -> dict:
    """Drop machine-local latency so the committed table replays in CI."""
    out = copy.deepcopy(report)
    for row in out["stores"]:
        row.pop("p95_ms", None)
    return out


def summary_lines(report: dict) -> list[str]:
    lines = [
        "mode: mock. Exact cosine in-process, except Azure AI Search which also fuses BM25 and the committed cross-encoder.",
        "p95_ms is measured around the in-process call and is not printed here (it depends on the machine).",
        "Live service p95 and live recall: not measured. Set AZURE_SEARCH_ENDPOINT, COSMOS_ENDPOINT or PGVECTOR_DSN.",
        "",
        f"{'store':<16} {'recall@3':>8} {'status':>8} {'policy':>8} {'stale_off':>9} {'stale_pol':>9} {'usd':>8}",
    ]
    for row in report["stores"]:
        lines.append(
            f"{row['store']:<16} {row['recall@3']['rate']:>8.4f} {row['recall@3_filtered']['rate']:>8.4f} "
            f"{row['recall@3_policy_filtered']['rate']:>8.4f} {row['stale_in_top3']['rate']:>9.4f} "
            f"{row['stale_in_top3_policy_filter']['rate']:>9.4f} {row['usd_month']:>8.2f}"
        )
    lines += ["", selection_guide(report)]
    return lines


def selection_guide(report: dict) -> str:
    """Pick the cheapest store whose filtered recall is within 0.01 of the best, then say why the
    other two still win some deployments. The mock uses exact cosine, so a tie is not a live ANN result."""
    rows = [r for r in report["stores"] if r["store"] != "memory"]
    best = max(r["recall@3_policy_filtered"]["rate"] for r in rows)
    near = [r for r in rows if best - r["recall@3_policy_filtered"]["rate"] <= 0.01]
    pick = min(near, key=lambda r: (r["usd_month"], r["store"]))
    rates = [r["recall@3_policy_filtered"]["rate"] for r in rows]
    spread = round(max(rates) - min(rates), 4)
    return "\n".join([
        "Store ids: azure_ai_search, cosmos_nosql, pgvector. memory is the free baseline and is not a deployment choice.",
        f"Policy-filtered recall@3 spread across the three Azure stores on this mock: {spread:.4f}.",
        f"Among stores within 0.01 of the best policy-filtered recall ({best:.4f}), "
        f"the lowest estimated monthly cost is {pick['store']} at ${pick['usd_month']:.2f} ({pick['sku']}).",
        "The mock searches exact cosine, so Cosmos DB (diskANN) and pgvector (HNSW) are not showing "
        "approximate-index loss. Re-measure recall@k and service p95 with COSMOS_ENDPOINT, PGVECTOR_DSN "
        "and AZURE_SEARCH_ENDPOINT before choosing.",
        "Choose Azure AI Search when the workload needs hybrid keyword plus vector plus the semantic "
        "ranker, or the guarded retrieval service is already pointed at it.",
        "Choose Azure Cosmos DB for NoSQL when the documents already live as items and the filter must "
        "sit in the same VectorDistance query.",
        "Choose Azure Database for PostgreSQL with pgvector when you want SQL predicates and joins, "
        "the lowest always-on estimate in this table, and a local docker-compose database for development.",
        "In-process p95 is measured and returned as p95_ms. It is machine-local and is not a service latency. "
        "Live service p95 is not measured.",
    ])
