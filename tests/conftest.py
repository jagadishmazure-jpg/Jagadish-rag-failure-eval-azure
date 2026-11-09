import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def metrics():
    return json.loads((ROOT / "results/metrics.json").read_text())


@pytest.fixture(scope="session")
def built():
    from ragfail.corpus import build_corpus, build_questions

    docs, facts = build_corpus()
    return docs, facts, build_questions(facts)


def chunk(cid="d-v1#w800", text="Crews must be on site within 3 hours of a priority-one leak report.", **kw):
    base = {"chunk_id": cid, "doc_id": cid.split("#")[0], "family": "leak-quillby", "topic": "leak",
            "district": "Quillby", "status": "current", "doc_type": "policy", "effective": "2026-01-15",
            "title": "Leak response standard (Quillby district)", "text": text, "titled": "T. " + text}
    return {**base, **kw}


def question(**kw):
    base = {"qid": "q1", "question": "How fast in Quillby?", "answerable": True, "gold": 3, "gold_doc": "d-v1",
            "gold_sentence": "Crews must be on site within 3 hours of a priority-one leak report.",
            "family": "leak-quillby", "stale_value": None, "kinds": ["plain"], "style": "lexical"}
    return {**base, **kw}
