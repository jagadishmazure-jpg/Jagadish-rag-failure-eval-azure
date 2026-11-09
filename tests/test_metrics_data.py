import hashlib
import json

import pytest
from conftest import ROOT

from ragfail.data import DataError, read_jsonl
from ragfail.metrics import mcnemar_exact, percentile, rate, wilson


def test_wilson_known_values():
    lo, hi = wilson(5, 10)
    assert round(lo, 4) == 0.2366 and round(hi, 4) == 0.7634
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_rate_and_percentile():
    assert rate(3, 4)["rate"] == 0.75
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert round(mcnemar_exact(0, 6), 4) == 0.0312
    assert mcnemar_exact(2, 9) == mcnemar_exact(9, 2)


def test_bad_jsonl_errors_never_echo_content(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"ok": 1}\nsecret-customer-text\n')
    with pytest.raises(DataError) as e:
        read_jsonl(p)
    assert "secret" not in str(e.value) and "x.jsonl:2" in str(e.value)


def test_manifest_hashes_match_the_committed_files():
    m = json.loads((ROOT / "data/emb/manifest.json").read_text())
    assert m["embed_model"] == "BAAI/bge-small-en-v1.5"
    for rel, sha in m["files"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha, rel
