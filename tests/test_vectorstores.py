"""Pluggable stores, the mock benchmark, and the live query builders."""

import json

import numpy as np
import pytest

from ragfail import data
from ragfail.store_bench import committed_view, run_store_bench, selection_guide
from ragfail.vectorstores import (
    HOURS_PER_MONTH,
    PG_STORAGE_GB,
    PG_USD_PER_GB,
    PG_USD_PER_HOUR,
    SEARCH_USD_PER_HOUR,
    CosmosStore,
    MemoryStore,
    build_stores,
    cosmos_query,
    live_search,
    monthly_cost,
    odata_filter,
    pg_sql,
)

RESULTS = data.ROOT / "results" / "vector_stores.json"


def _tiny():
    chunks = [
        {"chunk_id": "a", "doc_id": "a", "title": "A", "text": "alpha", "titled": "A alpha",
         "status": "current", "doc_type": "policy", "district": "Quillby", "topic": "leak"},
        {"chunk_id": "b", "doc_id": "b", "title": "B", "text": "beta", "titled": "B beta",
         "status": "superseded", "doc_type": "policy", "district": "Quillby", "topic": "leak"},
        {"chunk_id": "c", "doc_id": "c", "title": "C", "text": "gamma", "titled": "C gamma",
         "status": "current", "doc_type": "faq", "district": None, "topic": "faq"},
    ]
    emb = np.eye(3, dtype=np.float32)
    return chunks, emb


def test_filter_drops_superseded_on_every_store():
    chunks, emb = _tiny()
    for store in (MemoryStore(chunks, emb), CosmosStore(chunks, emb)):
        hits = store.search(emb[0], k=3, flt={"status": "current"})
        assert [h.chunk_id for h in hits] == ["a", "c"]
        district = store.search(emb[0], k=3, flt={"district": "Quillby"})
        assert [h.chunk_id for h in district] == ["a", "b"]


def test_query_builders_bind_values_and_reject_unknown_fields():
    sql = cosmos_query(3, {"status": "current", "doc_type": "policy"})
    assert "SELECT TOP 3" in sql and "@f0" in sql and "current" not in sql.split("WHERE", 1)[1]
    assert "status eq 'current'" in odata_filter({"status": "current"})
    assert "O''Brien" in odata_filter({"district": "O'Brien"})
    assert odata_filter({"district": None}) == "district eq null"
    assert "IS NULL" in pg_sql(3, {"district": None})
    assert "IS_NULL" in cosmos_query(3, {"district": None})
    pg = pg_sql(3, {"status": "current"})
    assert pg.count("%s") == 3  # filter value, vector, limit
    assert "current" not in pg
    with pytest.raises(ValueError):
        cosmos_query(3, {"password": "x"})
    with pytest.raises(ValueError):
        pg_sql(0, None)


def test_monthly_cost_matches_the_published_formula():
    search = monthly_cost("azure_ai_search")
    assert search["usd_month"] == round(SEARCH_USD_PER_HOUR * HOURS_PER_MONTH, 2)
    pg = monthly_cost("pgvector")
    assert pg["usd_month"] == round(PG_USD_PER_HOUR * HOURS_PER_MONTH + PG_STORAGE_GB * PG_USD_PER_GB, 2)
    assert monthly_cost("cosmos_nosql")["usd_month"] < search["usd_month"]
    assert monthly_cost("memory")["usd_month"] == 0.0


def test_live_search_requires_its_env_var():
    with pytest.raises(RuntimeError, match="AZURE_SEARCH_ENDPOINT"):
        live_search("azure_ai_search", "fee?", [0.0], 3)
    with pytest.raises(RuntimeError, match="COSMOS_ENDPOINT"):
        live_search("cosmos_nosql", "fee?", [0.0], 3)
    with pytest.raises(RuntimeError, match="PGVECTOR_DSN"):
        live_search("pgvector", "fee?", [0.0], 3)


@pytest.fixture(scope="module")
def report():
    return run_store_bench()


def test_benchmark_covers_recall_filter_and_cost(report):
    names = [row["store"] for row in report["stores"]]
    assert names == ["memory", "azure_ai_search", "cosmos_nosql", "pgvector"]
    by = {row["store"]: row for row in report["stores"]}
    for row in report["stores"]:
        assert row["p95_ms"] >= 0
        assert row["district_filter_ok"]["rate"] == 1.0
        assert row["stale_in_top3_policy_filter"]["rate"] == 0.0
        assert row["live"] is False
    # Hybrid + semantic stand-in beats exact cosine on this corpus. Status-only still leaves the FAQ.
    assert by["azure_ai_search"]["recall@3"]["rate"] > by["memory"]["recall@3"]["rate"]
    assert by["cosmos_nosql"]["recall@3"] == by["pgvector"]["recall@3"] == by["memory"]["recall@3"]
    assert by["memory"]["stale_in_top3_status_filter"]["rate"] > 0
    guide = selection_guide(report)
    assert "azure_ai_search" in guide and "cosmos_nosql" in guide and "pgvector" in guide
    assert "not measured" in guide


def test_committed_store_bench_omits_machine_local_latency(report):
    committed = json.loads(RESULTS.read_text())
    assert committed == committed_view(report)
    assert "p95_ms" not in committed["stores"][0]


def test_cosmos_items_keep_the_partition_key_and_drop_the_hash():
    from ragfail.vectorstores import cosmos_items

    chunks, emb = _tiny()
    chunks[0]["chunk_id"] = "leak#w800"
    items = cosmos_items(chunks, emb)
    assert items[0]["id"] == "leak_w800" and items[0]["doc_id"] == "a"
    assert len(items[0]["vector"]) == 3


def test_build_stores_uses_the_same_chunk_matrix():
    retriever = data.retriever("w80")
    stores = build_stores(retriever.chunks, retriever.emb, retriever.rr)
    assert len(stores) == 4
    q = next(q for q in data.questions() if q["answerable"])
    hits = stores[0].search(retriever.qemb[q["qid"]], 3, query_text=q["question"], qid=q["qid"])
    assert len(hits) == 3
