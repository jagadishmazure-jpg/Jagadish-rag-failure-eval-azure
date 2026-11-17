"""RAG vs fine-tuning vs RAFT."""

import json

import pytest

from ragfail import data
from ragfail.adaptation import (
    ARMS,
    _raft_chunks,
    endpoint_accepts_token,
    finetune_payload,
    in_train,
    mock_finetune_answer,
    run_adaptation,
)
from ragfail.scoring import CATEGORIES

RESULTS = data.ROOT / "results" / "adaptation.json"


def test_train_split_is_stable_and_both_sides_are_nonempty():
    ids = sorted({q["fact_id"] for q in data.questions() if q["fact_id"]})
    assert in_train(ids[0]) == in_train(ids[0])
    assert any(in_train(i) for i in ids) and any(not in_train(i) for i in ids)


def test_finetune_payload_sends_passages_only_for_raft():
    bare = finetune_payload("How many hours?", None, "finetune")
    assert bare["passages"] == [] and bare["mode"] == "finetune"
    raft = finetune_payload("How many hours?", [{"chunk_id": "a", "titled": "within 3 hours"}], "raft")
    assert raft["passages"] == [{"chunk_id": "a", "text": "within 3 hours"}]
    with pytest.raises(ValueError):
        finetune_payload("q", None, "other")


def test_token_is_not_sent_to_a_plain_http_host():
    endpoint_accepts_token("https://adapter.example/score", "secret")
    endpoint_accepts_token("http://127.0.0.1:9/score", "secret")
    with pytest.raises(ValueError):
        endpoint_accepts_token("http://adapter.example/score", "secret")


def test_raft_drops_every_chunk_when_none_match_the_district():
    from conftest import chunk

    class OnlyOther:
        def top(self, q, setup, k):
            return [chunk(district="Ebbersley")]

    q = next(q for q in data.questions() if "Quillby" in q["question"])
    assert _raft_chunks(q, OnlyOther()) == []


def test_mock_adapter_memorises_the_old_number_and_hallucinates_held_out():
    train = next(q for q in data.questions() if q["answerable"] and q["stale_value"] is not None and in_train(q["fact_id"]))
    held = next(q for q in data.questions() if q["answerable"] and q["fact_id"] and not in_train(q["fact_id"]))
    unans = next(q for q in data.questions() if not q["answerable"])
    assert mock_finetune_answer(train) == f"{float(train['stale_value']):g}"
    assert mock_finetune_answer(held) == "9999"
    assert mock_finetune_answer(unans) == "9999"


@pytest.fixture(scope="module")
def report():
    return run_adaptation()


def test_three_arms_use_the_taxonomy_and_the_judge_names(report):
    assert set(report["arms"]) == set(ARMS)
    assert report["mode"] == "mock" and report["judge_scale"] == "unit"
    for name, row in report["arms"].items():
        assert sum(row["taxonomy"].values()) == report["questions"], name
        assert set(row["taxonomy"]) == set(CATEGORIES)
        assert row["judge_backend"] == "mock"
        assert "faithfulness" in row["judge"] and "groundedness" in row["judge"]
    rag, ft, raft = (report["arms"][n] for n in ARMS)
    assert ft["abstain_when_unanswerable"]["rate"] < rag["abstain_when_unanswerable"]["rate"]
    assert ft["stale_answers"]["rate"] > rag["stale_answers"]["rate"]
    assert ft["citation_ok"]["rate"] < rag["citation_ok"]["rate"]
    assert report["distractors_dropped"] > 0
    assert "citation_mismatch" in report["decision"]
    assert "not measured" in report["decision"]
    # Guarded RAG already removed the stale distractors, so RAFT does not pull ahead.
    assert raft["accuracy"]["rate"] == rag["accuracy"]["rate"]


def test_committed_adaptation_matches_a_fresh_run(report):
    assert json.loads(RESULTS.read_text()) == report
