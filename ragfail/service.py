"""Guarded retrieval service: the part of the pipeline that ships.

Offline it uses BM25 over the committed corpus (no model weights needed). In Azure it calls Azure AI
Search (see azure_search.py). Either way the same two guards run: only current, authoritative
documents first, and hidden instructions are stripped before any text reaches a model.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import data
from .guards import check_question, strip_instructions
from .retrieval import apply_guards, rank

DEFAULT_K = 3


def retrieve_local(question: str, k: int = DEFAULT_K) -> dict:
    q = check_question(question)
    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")
    r = data.retriever("w80")
    order = apply_guards(rank(r.bm25.scores(q)), r.chunks)[:k]
    return package([r.chunks[i] for i in order], "local-bm25")


def package(chunks: list[dict], source: str) -> dict:
    passages, stripped = [], 0
    for c in chunks:
        text, n = strip_instructions(c["titled"])
        stripped += n
        passages.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "effective": c["effective"], "text": text})
    return {"source": source, "passages": passages, "instructions_stripped": stripped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Guarded retrieval over the Tessmoor Water corpus.")
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    a = ap.parse_args(argv)
    print(json.dumps(retrieve_local(a.question, a.k), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
