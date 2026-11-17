"""Self-RAG critique loop, offline.

Self-RAG retrieves, then critiques the passage (relevant?), the answer (supported?) and the decision
to abstain (useful?). This loop does the same three checks with code, at most one retry:

* relevant: the best passage shares at least two content words with the question,
* supported: every number in the answer appears in the cited passage,
* useful: an abstention is retried once on the keyword query when a numbered sentence still overlaps.

The retry uses the multi-query ranking. There is no second model.
"""

from __future__ import annotations

import numpy as np

from .extractive import extractive_answer
from .query_rewrite import multi_queries
from .retrieval import apply_guards, rank, rrf, tokenize
from .scoring import citations, numbers


def _supported(answer: str, chunks: list[dict]) -> bool:
    if "NOT_FOUND" in answer.upper().replace(" ", "_"):
        return True
    cites, nums = citations(answer), numbers(answer)
    if not cites or not nums:
        return False
    for cite in cites:
        if not 1 <= cite <= len(chunks):
            return False
        have = set(numbers(chunks[cite - 1]["text"])) | set(numbers(chunks[cite - 1].get("titled") or ""))
        if any(n not in have for n in nums):
            return False
    return True


def _overlap(question: str, chunks: list[dict]) -> int:
    q = set(tokenize(question))
    best = 0
    for chunk in chunks:
        toks = set(tokenize(chunk.get("titled") or chunk.get("text") or ""))
        best = max(best, len(q & toks))
    return best


def _multi_chunks(q: dict, retriever, k: int) -> list[dict]:
    rankings = [rank(retriever.bm25.scores(sub)) for sub in multi_queries(q["question"])]
    rankings.append(rank(retriever.emb @ retriever.qemb[q["qid"]].astype(np.float32)))
    order = apply_guards(rrf(*rankings), retriever.chunks)
    return [retriever.chunks[i] for i in order[:k]]


def self_rag(q: dict, retriever, k: int = 3) -> tuple[str, list[dict], list[str]]:
    """Return ``(answer, chunks shown, critique notes)``."""
    notes: list[str] = []
    chunks = retriever.top(q, "hybrid_rerank", k)
    if _overlap(q["question"], chunks) < 2:
        notes.append("isrel_retry")
        chunks = _multi_chunks(q, retriever, k)
    answer = extractive_answer(q["question"], chunks)
    if not _supported(answer, chunks):
        notes.append("issup_abstain")
        return "NOT_FOUND", chunks, notes
    if answer == "NOT_FOUND" and _overlap(q["question"], chunks) >= 1:
        notes.append("isuse_retry")
        # One more pass: keyword chunks only, same reader. If it still abstains, keep NOT_FOUND.
        chunks = _multi_chunks(q, retriever, k)
        answer = extractive_answer(q["question"], chunks)
        if not _supported(answer, chunks):
            notes.append("issup_abstain")
            return "NOT_FOUND", chunks, notes
    notes.append("accept")
    return answer, chunks, notes
