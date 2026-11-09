"""Retrievers: BM25 (computed here), dense (from stored embeddings), hybrid (reciprocal rank fusion)
and hybrid plus a cross-encoder rerank (from stored scores). Plus the two metadata guards.

Dense embeddings and rerank scores were computed once on CPU by `scripts/embed_and_rerank.py` and are
committed, so every retrieval number replays exactly in CI without model weights.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = set("a an the of in on at to for and or is are be by with what how many much does do we our "
           "it its this that from as within any each after before".split())
RRF_K = 60
RERANK_DEPTH = 20


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        for i, (d, ln) in enumerate(zip(self.docs, self.lens)):
            s = 0.0
            for t in q:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out[i] = s
        return out


def rank(scores: np.ndarray) -> list[int]:
    """Indices by descending score; ties broken by index so results are deterministic."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def rrf(*rankings: list[int], k: int = RRF_K) -> list[int]:
    s: dict[int, float] = {}
    for r in rankings:
        for pos, i in enumerate(r):
            s[i] = s.get(i, 0.0) + 1.0 / (k + pos + 1)
    return sorted(s, key=lambda i: (-s[i], i))


def apply_guards(order: list[int], chunks: list[dict]) -> list[int]:
    """Metadata guards: drop superseded versions, then move non-authoritative sources (the FAQ) last."""
    live = [i for i in order if chunks[i]["status"] == "current"]
    return [i for i in live if chunks[i]["doc_type"] == "policy"] + [i for i in live if chunks[i]["doc_type"] != "policy"]


class Retriever:
    """All four setups over one chunk list."""

    SETUPS = ("bm25", "dense", "hybrid", "hybrid_rerank")

    def __init__(self, chunks: list[dict], chunk_emb: np.ndarray | None, query_emb: dict[str, np.ndarray] | None,
                 rerank_scores: dict[str, dict[str, float]] | None = None, field: str = "titled"):
        self.chunks = chunks
        self.bm25 = BM25([c[field] for c in chunks])
        self.emb = None if chunk_emb is None else chunk_emb.astype(np.float32)
        self.qemb = query_emb or {}
        self.rr = rerank_scores or {}
        self.index = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    def order(self, q: dict, setup: str, guards: bool = True) -> list[int]:
        if setup == "bm25":
            out = rank(self.bm25.scores(q["question"]))
        elif setup == "dense":
            out = rank(self.emb @ self.qemb[q["qid"]].astype(np.float32))
        elif setup in ("hybrid", "hybrid_rerank"):
            out = rrf(rank(self.bm25.scores(q["question"])), rank(self.emb @ self.qemb[q["qid"]].astype(np.float32)))
            if setup == "hybrid_rerank":
                scores = self.rr[q["qid"]]
                head = [i for i in out if self.chunks[i]["chunk_id"] in scores][:RERANK_DEPTH * 2]
                head = sorted(head, key=lambda i: (-scores[self.chunks[i]["chunk_id"]], i))
                out = head + [i for i in out if i not in set(head)]
        else:
            raise ValueError(f"unknown setup: {setup}")
        return apply_guards(out, self.chunks) if guards else out

    def top(self, q: dict, setup: str, k: int, guards: bool = True) -> list[dict]:
        return [self.chunks[i] for i in self.order(q, setup, guards)[:k]]

    def rerank_candidates(self, q: dict) -> list[str]:
        """Chunk ids the cross-encoder must score: hybrid top RERANK_DEPTH with and without guards."""
        a = self.order(q, "hybrid", guards=True)[:RERANK_DEPTH]
        b = self.order(q, "hybrid", guards=False)[:RERANK_DEPTH]
        return sorted({self.chunks[i]["chunk_id"] for i in a + b})
