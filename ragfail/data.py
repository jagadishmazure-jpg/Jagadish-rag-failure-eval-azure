"""Load the committed corpus, questions, embeddings and rerank scores."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .chunking import STRATEGIES, chunk_corpus
from .retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class DataError(ValueError):
    """Bad data file. Messages name the file and line, never the content."""


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise DataError(f"{path.name}:{n}: not valid JSON") from None
        if not isinstance(row, dict):
            raise DataError(f"{path.name}:{n}: expected an object")
        out.append(row)
    return out


@lru_cache(maxsize=1)
def corpus() -> list[dict]:
    return read_jsonl(DATA / "corpus.jsonl")


@lru_cache(maxsize=1)
def questions() -> list[dict]:
    return read_jsonl(DATA / "questions.jsonl")


def chunks(strategy: str) -> list[dict]:
    return chunk_corpus(corpus(), strategy)


@lru_cache(maxsize=1)
def query_embeddings() -> dict[str, np.ndarray]:
    ids = json.loads((DATA / "emb" / "queries.json").read_text())
    mat = np.load(DATA / "emb" / "queries.npy")
    if len(ids) != len(mat):
        raise DataError("queries.npy and queries.json differ in length")
    return dict(zip(ids, mat))


@lru_cache(maxsize=None)
def retriever(strategy: str, field: str = "titled") -> Retriever:
    if strategy not in STRATEGIES:
        raise DataError(f"unknown strategy {strategy}")
    ch = chunks(strategy)
    emb = np.load(DATA / "emb" / f"chunks-{strategy}-{field}.npy")
    if len(emb) != len(ch):
        raise DataError(f"chunks-{strategy}-{field}.npy does not match the corpus; rerun embed_and_rerank.py")
    rr_path = DATA / "rerank" / f"{strategy}.json"
    rr = json.loads(rr_path.read_text()) if field == "titled" and rr_path.exists() else None
    return Retriever(ch, emb, query_embeddings(), rr, field=field)
