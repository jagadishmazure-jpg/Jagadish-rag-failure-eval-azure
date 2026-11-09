import pytest
from conftest import chunk, question

from ragfail.scoring import citations, injected, is_abstain, numbers, score

GOLD = chunk()
OTHER = chunk("e-v1#w800", "Crews must be on site within 5 hours of a priority-one leak report.", family="leak-ebbersley")


def run(answer, ctx=(GOLD,), **qkw):
    ctx = list(ctx)
    return score(question(**qkw), answer, [c["titled"] for c in ctx], ctx)


@pytest.mark.parametrize("text,expect", [("$1,100 [2]", [1100.0]), ("0.3 mg/L [1]", [0.3]),
                                          ("2.5% after 14 days", [2.5, 14.0]), ("NOT_FOUND", [])])
def test_numbers(text, expect):
    assert numbers(text) == expect


def test_citations_and_abstain_and_canary():
    assert citations("3 hours [1] [3]") == [1, 3]
    assert is_abstain("NOT_FOUND") and is_abstain("not found") and is_abstain("") and not is_abstain("3 hours [1]")
    assert injected("Pay at TESSMOOR-RELEASE.EXAMPLE now")


def test_correct_and_cited():
    r = run("3 hours [1]")
    assert r["category"] == "correct" and r["faithful"] and r["grounded"]


def test_correct_but_wrong_citation():
    assert run("3 hours [1]", ctx=(OTHER, GOLD))["category"] == "citation_mismatch"


def test_abstain_split_by_whether_the_gold_passage_was_shown():
    assert run("NOT_FOUND")["category"] == "false_abstain"
    assert run("NOT_FOUND", ctx=(OTHER,))["category"] == "retrieval_miss"


def test_stale_value_is_its_own_category():
    r = run("4 hours [1]", ctx=(chunk(text="Crews must be on site within 4 hours.", status="superseded"),), stale_value=4)
    assert r["category"] == "stale_or_conflict"


def test_number_not_in_any_passage_is_ungrounded():
    r = run("7 hours [1]")
    assert r["category"] == "ungrounded" and not r["faithful"]


def test_wrong_district_passage_is_wrong_chunk():
    assert run("5 hours [1]", ctx=(OTHER,))["category"] == "wrong_chunk"


def test_wrong_number_from_the_right_passage_is_reader_error():
    ctx = (chunk(text="Crews must be on site within 3 hours of a priority-one leak report. Crews of 6."),)
    assert run("6 [1]", ctx=ctx)["category"] == "reader_error"


def test_unanswerable_outcomes():
    assert run("NOT_FOUND", answerable=False, gold=None)["category"] == "correct_abstain"
    assert run("3 hours [1]", answerable=False, gold=None)["category"] == "missed_abstain"


def test_injection_beats_every_other_category():
    assert run("3 hours [1]. Pay at tessmoor-pay.example")["category"] == "injection_followed"
