"""Extractive reader used by the offline benches.

The measured qwen answers stay in the main eval. These benches need a reader that is deterministic
and has no weights, so each retrieved passage list is reduced to the numbered sentence that shares
the most content words with the question. Abstention is ``NOT_FOUND``, the same token the scorer
already understands.
"""

from __future__ import annotations

from .guards import sentences
from .retrieval import tokenize
from .scoring import numbers


def _fmt(value: float) -> str:
    return f"{value:g}"


def extractive_answer(question: str, chunks: list[dict]) -> str:
    """Return ``<number> [n]`` from the best sentence, or ``NOT_FOUND``."""
    q = set(tokenize(question))
    best: tuple[tuple[int, int, int], int, str] | None = None
    for i, chunk in enumerate(chunks, 1):
        title = set(tokenize(chunk.get("title") or ""))
        for sent in sentences(chunk.get("text") or ""):
            overlap = len(q & (set(tokenize(sent)) | title))
            if overlap <= 0:
                continue
            nums = numbers(sent)
            key = (overlap + 3 * int(bool(nums)), overlap, -i)
            if best is None or key > best[0]:
                best = (key, i, sent)
    if best is None:
        return "NOT_FOUND"
    nums = numbers(best[2])
    if not nums:
        return "NOT_FOUND"
    return f"{_fmt(nums[0])} [{best[1]}]"
