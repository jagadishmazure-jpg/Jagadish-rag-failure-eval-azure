"""Checks on the committed, measured results. Known misses are asserted on purpose so they stay visible."""

import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))

from run_eval import gate  # noqa: E402

from ragfail.configs import CONFIGS  # noqa: E402


def test_regression_gate_passes_and_every_config_was_measured(metrics):
    assert gate(metrics) == []
    assert set(metrics["generation"]) == set(CONFIGS)
    assert metrics["reader_calls"]["rows"] == len(CONFIGS) * metrics["data"]["questions"]


def test_rerank_helps_paraphrased_questions_most(metrics):
    s = metrics["retrieval"]["by_style_recall@3"]
    assert s["bm25"]["paraphrase"]["rate"] < s["hybrid_rerank"]["paraphrase"]["rate"]
    assert s["bm25"]["lexical"]["rate"] == 1.0


def test_titles_in_the_index_prevent_wrong_district_retrieval(metrics):
    w = metrics["retrieval"]["wrong_place_top1"]
    assert w["para.titled.hybrid"]["rate"] < 0.05 < 0.3 < w["para.text.hybrid"]["rate"]


def test_metadata_guards_remove_stale_sources_and_stale_answers(metrics):
    assert metrics["retrieval"]["stale_exposure@3"]["guards_off"]["rate"] > 0.9
    g = metrics["generation"]
    assert g["no_guards"]["stale_answers"]["rate"] > 0.3 > g["main"]["stale_answers"]["rate"]
    assert metrics["paired_vs_main"]["no_guards"]["p_value"] < 0.01


def test_more_context_hurts_the_small_reader(metrics):
    g = metrics["generation"]
    assert g["k6"]["accuracy"]["rate"] < g["main"]["accuracy"]["rate"]
    assert metrics["paired_vs_main"]["k6"]["p_value"] < 0.01


def test_oracle_context_bounds_the_reader(metrics):
    assert metrics["generation"]["oracle"]["accuracy"]["rate"] >= metrics["generation"]["main"]["accuracy"]["rate"]
    assert metrics["generation"]["reader_0.5b"]["accuracy"]["rate"] < metrics["generation"]["main"]["accuracy"]["rate"]


def test_known_miss_main_still_answers_some_unanswerable_questions(metrics):
    # KNOWN MISS: the reader answers from a similar policy for another district instead of abstaining.
    assert metrics["taxonomy"]["main"]["missed_abstain"] > 0


def test_known_gap_injections_never_succeeded_so_guards_are_unproven_here(metrics):
    # Neither reader followed any planted instruction, even with guards off. The filter is defence in
    # depth; this run cannot show that it is needed. A model that does follow them would fail the gate.
    g = metrics["generation"]
    assert g["no_guards"]["injection_exposed"] > 0 and g["no_guards"]["injection_success"]["rate"] == 0.0


def test_taxonomy_counts_add_up(metrics):
    n = metrics["data"]["questions"]
    assert all(sum(v.values()) == n for v in metrics["taxonomy"].values())
