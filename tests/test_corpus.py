import re
import subprocess
import sys

from conftest import ROOT

from ragfail.corpus import DISTRICTS, INJECTIONS, TOPICS


def test_corpus_size_and_mix(built):
    docs, facts, qs = built
    assert len(docs) == 43 and len(facts) == 75 and len(qs) == 182
    assert sum(d["status"] == "superseded" for d in docs) == 10
    assert sum(d["doc_type"] == "faq" for d in docs) == 1


def test_every_fact_sentence_appears_in_its_current_document(built):
    docs, facts, _ = built
    by_id = {d["doc_id"]: d for d in docs}
    for f in facts:
        assert by_id[f["doc_id"]]["status"] == "current"
        assert by_id[f["doc_id"]]["text"].count(f["sentence"]) == 1


def test_each_district_has_a_different_value_for_each_fact(built):
    _, facts, _ = built
    for topic, spec in TOPICS.items():
        for key in spec["facts"]:
            vals = [f["value"] for f in facts if f["topic"] == topic and f["key"] == key]
            assert len(vals) == len(DISTRICTS) == len(set(vals)), (topic, key)


def test_stale_and_conflicting_values_differ_from_current(built):
    _, facts, _ = built
    for f in facts:
        for other in (f["stale_value"], f["conflict_value"]):
            assert other is None or other != f["value"]
    assert sum(f["conflict_value"] is not None for f in facts) == 12


def test_injections_are_planted_where_recorded(built):
    docs, _, _ = built
    planted = [d for d in docs if d["injection_canary"]]
    assert len(planted) == 8
    for d in planted:
        assert d["injection_canary"] in d["text"]
    canaries = {c for _, c in INJECTIONS}
    assert not any(c in d["text"] for d in docs if not d["injection_canary"] for c in canaries)


def test_only_reserved_example_domains_are_used(built):
    docs, _, _ = built
    for d in docs:
        for host in re.findall(r"[\w-]+\.[a-z]{2,}\b", d["text"]):
            if "@" in host or host.split(".")[-1] in ("example",):
                continue
        assert all(h.endswith(".example") for h in re.findall(r"\b[\w-]+\.(?:com|org|net|example)\b", d["text"]))


def test_two_phrasings_per_fact_and_unanswerable_mentions_unknown_districts(built):
    _, facts, qs = built
    ans = [q for q in qs if q["answerable"]]
    assert len(ans) == 2 * len(facts)
    assert {q["style"] for q in ans} == {"lexical", "paraphrase"}
    unans = [q for q in qs if not q["answerable"]]
    assert len(unans) == 32
    assert any("Brennick" in q["question"] for q in unans)


def test_unknown_districts_do_not_exist_in_the_corpus(built):
    docs, _, _ = built
    assert not any(n in d["text"] for d in docs for n in ("Brennick", "Fallowmere"))


def test_committed_dataset_matches_the_generator():
    res = subprocess.run([sys.executable, "scripts/build_dataset.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout
