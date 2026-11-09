"""The generation configurations compared in the eval. Each one changes a single thing from `main`."""

from __future__ import annotations

MAIN = {"strategy": "w80", "setup": "hybrid_rerank", "k": 3, "guards": True, "oracle": False,
        "reader": "qwen2.5-1.5b"}

CONFIGS: dict[str, dict] = {
    "main": {},
    "bm25": {"setup": "bm25"},
    "dense": {"setup": "dense"},
    "hybrid": {"setup": "hybrid"},
    "k1": {"k": 1},
    "k6": {"k": 6},
    "chunk_para": {"strategy": "para"},
    "chunk_doc": {"strategy": "doc"},
    "no_guards": {"guards": False},
    "oracle": {"oracle": True},
    "reader_0.5b": {"reader": "qwen2.5-0.5b"},
}


def config(name: str) -> dict:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name}")
    return {**MAIN, **CONFIGS[name], "name": name}


def context_for(q: dict, cfg: dict, retriever) -> list[dict]:
    """Chunks shown to the reader. Oracle: the gold chunk alone (answerable) or main retrieval (unanswerable)."""
    if cfg["oracle"] and q["answerable"]:
        from .chunking import gold_chunks

        gold = gold_chunks(retriever.chunks, q)
        return [c for c in retriever.chunks if c["chunk_id"] in gold][:1]
    return retriever.top(q, cfg["setup"], cfg["k"], guards=cfg["guards"])
