import pytest

from ragfail import data
from ragfail.chunking import STRATEGIES, chunk_doc, gold_chunks


def test_each_paragraph_lands_in_exactly_one_chunk():
    doc = data.corpus()[0]
    paras = doc["text"].split("\n\n")[1:]
    for s in STRATEGIES:
        joined = "\n\n".join(c["text"] for c in chunk_doc(doc, s))
        assert all(joined.count(p) == 1 for p in paras), s


def test_chunk_sizes_follow_the_strategy():
    for c in data.chunks("w80"):
        assert len(c["text"].split()) <= 80 or "\n\n" not in c["text"]
    assert len(data.chunks("para")) > len(data.chunks("w80")) > len(data.chunks("doc"))


def test_titled_text_carries_the_title_and_metadata():
    for c in data.chunks("para")[:20]:
        assert c["titled"].startswith(c["title"]) and c["text"] in c["titled"]
        assert "district" not in c["text"] or c["district"] in c["titled"]


def test_every_answerable_question_has_exactly_one_gold_chunk_per_strategy():
    for s in STRATEGIES:
        ch = data.chunks(s)
        for q in data.questions():
            assert len(gold_chunks(ch, q)) == (1 if q["answerable"] else 0), (s, q["qid"])


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        chunk_doc(data.corpus()[0], "sentence")
