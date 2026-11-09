import json
import sys

import numpy as np
import pytest
from conftest import ROOT

from ragfail import azure_search, data, service
from ragfail.corpus import INJECTIONS

sys.path.insert(0, str(ROOT / "infra/aml"))
import score  # noqa: E402


def test_index_schema_has_one_key_and_matching_vector_size():
    s = azure_search.index_schema()
    assert sum(f.get("key", False) for f in s["fields"]) == 1
    vec = next(f for f in s["fields"] if f["name"] == "vector")
    assert vec["dimensions"] == data.retriever("w80").emb.shape[1]


def test_documents_use_safe_keys_and_carry_guard_metadata():
    ch = data.chunks("w80")
    docs = azure_search.documents(ch, np.load(data.DATA / "emb/chunks-w80-titled.npy"))
    assert len({d["id"] for d in docs}) == len(ch)
    assert all("#" not in d["id"] and d["status"] in ("current", "superseded") for d in docs)
    with pytest.raises(ValueError):
        azure_search.documents(ch, np.zeros((len(ch), 10)))


def test_query_body_filters_superseded_only_when_guarded():
    assert azure_search.query_body("fee?")["filter"] == "status eq 'current'"
    assert "filter" not in azure_search.query_body("fee?", guards=False)
    assert azure_search.query_body("fee?", [0.0] * 384)["vectorQueries"][0]["k"] == 20
    with pytest.raises(ValueError):
        azure_search.query_body("fee?", [0.0] * 3)


def test_service_strips_hidden_instructions_and_skips_superseded():
    out = service.retrieve_local("What is the standard fee for a new domestic connection in Quillby?")
    assert out["instructions_stripped"] > 0 and len(out["passages"]) == 3
    assert not any(c in json.dumps(out) for _, c in INJECTIONS[:3])
    ids = {d["doc_id"]: d for d in data.corpus()}
    assert all(ids[p["doc_id"]]["status"] == "current" for p in out["passages"])


def test_service_bounds_and_cli(capsys):
    with pytest.raises(ValueError):
        service.retrieve_local("fee?", k=0)
    assert service.main(["late payment fee in Tarnwick", "-k", "2"]) == 0
    assert len(json.loads(capsys.readouterr().out)["passages"]) == 2


def test_scoring_script_rejects_bad_input():
    score.init()
    assert "error" in score.run("not json")
    assert "error" in score.run(json.dumps({"question": ""}))
    assert "error" in score.run(json.dumps({"question": "fee?", "k": True}))
    assert score.run(json.dumps({"question": "late payment fee in Tarnwick"}))["source"] == "local-bm25"
