"""Graph retrieval shaped like Microsoft GraphRAG local search, built from the corpus itself.

GraphRAG extracts entities with a model, then walks the entity graph to text units. This corpus is
synthetic, so the entities (district, topic, document status) are already fields. Local search links
those entities in the question and returns current policy chunks of that document. It does not run
the GraphRAG extractor. Set ``GRAPHRAG_QUERY_URL`` to call a GraphRAG query endpoint instead.
"""

from __future__ import annotations

import os

from .query_rewrite import detect_district, detect_topic


def live_graphrag_enabled() -> bool:
    return bool(os.environ.get("GRAPHRAG_QUERY_URL"))


def graphrag_request(question: str) -> dict:
    return {"query": question, "method": "local"}


def local_chunks(q: dict, retriever, k: int = 3) -> tuple[list[dict], str]:
    """Return ``(chunks, route)``. ``route`` is ``local`` or ``vector_fallback`` when linking finds nothing."""
    district, topic = detect_district(q["question"]), detect_topic(q["question"])
    chunks = retriever.chunks
    pool = []
    for i, c in enumerate(chunks):
        if c["status"] != "current" or c["doc_type"] != "policy":
            continue
        if district and c["district"] not in (district, None):
            continue
        if topic and c["topic"] != topic:
            continue
        if district is None and topic is None:
            continue
        pool.append(i)
    if not pool:
        return retriever.top(q, "dense", k), "vector_fallback"
    scores = retriever.bm25.scores(q["question"])
    pool.sort(key=lambda i: (-scores[i], i))
    return [chunks[i] for i in pool[:k]], "local"
