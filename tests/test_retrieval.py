import numpy as np
import pytest
from conftest import chunk, question

from ragfail import data
from ragfail.retrieval import BM25, Retriever, apply_guards, rank, rrf, tokenize


def test_tokenize_keeps_decimals_and_drops_stopwords():
    assert tokenize("The chlorine target is 0.4 mg/L") == ["chlorine", "target", "0.4", "mg", "l"]


def test_bm25_ranks_the_matching_text_first():
    b = BM25(["burst main response", "late payment fee", "meter reading interval"])
    assert rank(b.scores("late fee"))[0] == 1
    assert all(v > 0 for v in b.idf.values())


def test_rank_breaks_ties_by_position():
    assert rank(np.array([1.0, 2.0, 2.0, 0.5])) == [1, 2, 0, 3]


def test_rrf_rewards_agreement():
    assert rrf([0, 1, 2], [1, 2, 0])[0] == 1
    assert rrf([0, 1], [0, 2])[0] == 0


def test_guards_drop_superseded_and_put_faq_last():
    ch = [chunk("a-v1#0", status="superseded"), chunk("faq#0", doc_type="faq"), chunk("a-v2#0")]
    assert apply_guards([0, 1, 2], ch) == [2, 1]


def test_unknown_setup_rejected():
    r = Retriever([chunk()], np.ones((1, 4)), {"q1": np.ones(4)})
    with pytest.raises(ValueError):
        r.order(question(), "magic")


def test_stored_embeddings_match_chunks_and_are_normalised():
    q = data.query_embeddings()
    assert len(q) == len(data.questions())
    for s in ("para", "w80", "doc"):
        r = data.retriever(s)
        assert r.emb.shape == (len(r.chunks), 384)
        assert np.allclose(np.linalg.norm(r.emb, axis=1), 1, atol=1e-2)


def test_rerank_scores_cover_the_candidates():
    r = data.retriever("w80")
    for q in data.questions()[:25]:
        assert set(r.rerank_candidates(q)) <= set(r.rr[q["qid"]])


def test_guards_never_return_superseded_chunks():
    r = data.retriever("w80")
    for q in data.questions():
        assert all(c["status"] == "current" for c in r.top(q, "hybrid_rerank", 6))
