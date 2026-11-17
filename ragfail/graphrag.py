"""Graph retrieval shaped like Microsoft GraphRAG local search, built from the corpus itself.

GraphRAG extracts entities with a model, then walks the entity graph to text units. This corpus is
synthetic, so the entities (district, topic, document status) are already fields. Local search links
those entities in the question and returns current policy chunks of that document. It does not run
the GraphRAG extractor. Set ``GRAPHRAG_QUERY_URL`` to call a GraphRAG query endpoint instead.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .query_rewrite import detect_district, detect_topic


def live_graphrag_enabled() -> bool:
    return bool(os.environ.get("GRAPHRAG_QUERY_URL"))


def graphrag_request(question: str) -> dict:
    return {"query": question, "method": "local"}


def local_chunks(q: dict, retriever, k: int = 3) -> tuple[list[dict], str]:
    """Return ``(chunks, route)``. ``route`` is ``local``, ``live``, or ``vector_fallback``."""
    if live_graphrag_enabled():
        return live_graph_chunks(q["question"], k), "live"
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


def chunks_from_graphrag(payload: dict, k: int) -> list[dict]:
    rows = payload.get("chunks") or payload.get("context") or []
    out = []
    for row in rows:
        if isinstance(row, str):
            text, chunk_id = row, f"live-{len(out)}"
            title = ""
        else:
            text = row.get("text") or row.get("content") or ""
            chunk_id = str(row.get("chunk_id") or row.get("id") or f"live-{len(out)}")
            title = str(row.get("title") or "")
        if not text:
            continue
        out.append({"chunk_id": chunk_id, "doc_id": str((row.get("doc_id") if isinstance(row, dict) else "") or ""),
                    "title": title, "text": text, "titled": text, "status": "current", "doc_type": "policy",
                    "district": row.get("district") if isinstance(row, dict) else None,
                    "topic": str(row.get("topic") or "") if isinstance(row, dict) else ""})
    if not out:
        raise RuntimeError("GraphRAG query returned no chunks")
    return out[:k]


def live_graph_chunks(question: str, k: int = 3) -> list[dict]:
    url = os.environ["GRAPHRAG_QUERY_URL"]
    req = urllib.request.Request(url, data=json.dumps(graphrag_request(question)).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - the caller's query endpoint
        payload = json.loads(resp.read().decode())
    return chunks_from_graphrag(payload, k)
