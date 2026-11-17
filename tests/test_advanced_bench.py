"""Advanced RAG bench: rewrites, Self-RAG, agentic retrieval, GraphRAG, judges."""

import json

import pytest

from ragfail import agentic_search, data, judges
from ragfail.advanced_bench import STRATEGIES, run_advanced
from ragfail.graphrag import local_chunks
from ragfail.query_rewrite import decompose_query, detect_district, hyde_document, multi_queries, step_back_query
from ragfail.scoring import CATEGORIES
from ragfail.self_rag import _supported

ROOT_RESULTS = data.ROOT / "results" / "advanced_rag.json"


def test_rewrites_are_deterministic_and_keep_the_district():
    q = "How fast do we have to get someone out to a burst main in Quillby?"
    assert detect_district(q) == "Quillby"
    assert hyde_document(q) == hyde_document(q)
    assert hyde_document(q).startswith("The policy states that ")
    assert len(multi_queries(q)) == 3
    assert "Quillby" in step_back_query(q)
    assert "leak" in step_back_query(q)
    parts = decompose_query("What is the late payment fee and the grace period in Tarnwick?")
    assert "Tarnwick" in " ".join(parts)
    assert any("billing" in p for p in parts)


def test_support_critique_rejects_a_number_the_passage_does_not_contain():
    chunk = {"text": "Crews must be on site within 3 hours.", "titled": "t. Crews must be on site within 3 hours."}
    assert _supported("3 [1]", [chunk])
    assert not _supported("9 [1]", [chunk])
    assert _supported("NOT_FOUND", [chunk])


def test_agentic_request_is_hybrid_semantic_and_guarded():
    body = agentic_search.agentic_request("What is the late payment fee in Tarnwick?")
    source = body["knowledgeSourceParams"][0]
    assert source["resultsProcessing"] == "rerank"
    assert source["kind"] == "searchIndex"
    assert source["filterAddOn"] == "status eq 'current'"
    assert body["retrievalReasoningEffort"] == {"kind": "low"}
    assert "filterAddOn" not in agentic_search.agentic_request("fee?", guards=False)["knowledgeSourceParams"][0]
    with pytest.raises(ValueError):
        agentic_search.agentic_request("fee?", effort="max")


def test_graphrag_local_search_stays_on_the_current_district_policy():
    retriever = data.retriever("w80")
    q = next(q for q in data.questions() if q["qid"] == "leak-quillby-p1_hours:p")
    chunks, route = local_chunks(q, retriever, 3)
    assert route == "local"
    assert chunks
    assert all(c["status"] == "current" and c["topic"] == "leak" for c in chunks)
    assert all(c["district"] in ("Quillby", None) for c in chunks)


@pytest.fixture(scope="module")
def report():
    return run_advanced()


def test_every_strategy_is_scored_with_the_taxonomy_and_both_judge_families(report):
    assert set(report["strategies"]) == set(STRATEGIES)
    for name, row in report["strategies"].items():
        assert sum(row["taxonomy"].values()) == report["questions"], name
        assert set(row["taxonomy"]) == set(CATEGORIES)
        assert row["judge_backend"] == "mock"
        for metric in judges.JUDGE_METRICS:
            assert row["judge"][metric] is not None, (name, metric)
            assert 0.0 <= row["judge"][metric] <= 1.0
    assert report["self_rag_retries"] > 0
    g, v = report["strategies"]["graphrag"]["judge"], report["strategies"]["vector"]["judge"]
    assert g["context_recall"] > v["context_recall"]


def test_committed_advanced_bench_matches_a_fresh_run(report):
    assert json.loads(ROOT_RESULTS.read_text()) == report


def test_live_judges_stay_off_without_the_switch(monkeypatch):
    monkeypatch.delenv("RAGFAIL_LIVE_JUDGES", raising=False)
    assert judges.live_judges_enabled() is False
    with pytest.raises(RuntimeError):
        judges.load_live_backends()


def test_injected_libraries_receive_the_ragas_and_azure_row_shape():
    seen = {}

    def ragas_eval(dataset):
        seen["ragas"] = dataset
        return {name: 0.5 for name in judges.RAGAS_METRICS}

    def azure_eval(rows):
        seen["azure"] = rows
        return {name: 3.0 for name in judges.AZURE_METRICS}

    q = data.questions()[0]
    rows = [{"question": q, "answer": "1 [1]", "chunks": [], "all_chunks": [],
             "scored": {}, "contexts_text": ["passage"], "reference": "1. gold"}]
    backend, metrics = judges.judge_rows(rows, ragas_eval=ragas_eval, azure_eval=azure_eval)
    assert backend == "ragas+azure-ai-evaluation"
    assert seen["ragas"][0]["user_input"] == q["question"]
    assert seen["ragas"][0]["retrieved_contexts"] == ["passage"]
    assert seen["azure"][0]["ground_truth"] == "1. gold"
    assert metrics["faithfulness"] == 0.5 and metrics["groundedness"] == 3.0
