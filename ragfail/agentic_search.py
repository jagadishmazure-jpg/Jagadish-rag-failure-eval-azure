"""Azure AI Search agentic retrieval: query planning, hybrid search, semantic ranker.

The live call is ``POST /knowledgebases/{name}/retrieve?api-version=2026-08-01-preview``. The
knowledge source runs hybrid (keyword + vector) queries and reranks with the semantic ranker
(``resultsProcessing: rerank``). The metadata guard is the OData ``filterAddOn`` ``status eq 'current'``.

Offline, the same three steps run locally: decomposed subqueries, BM25 fused with the stored dense
vector, then the committed cross-encoder scores standing in for the semantic ranker.
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from .query_rewrite import decompose_query
from .retrieval import apply_guards, rank, rrf

API_VERSION = "2026-08-01-preview"
KNOWLEDGE_SOURCE = "tessmoor-policies"


def live_agentic_enabled() -> bool:
    return bool(os.environ.get("AZURE_SEARCH_ENDPOINT") and os.environ.get("AZURE_SEARCH_KNOWLEDGE_BASE"))


def agentic_request(question: str, *, knowledge_source: str = KNOWLEDGE_SOURCE, guards: bool = True,
                    effort: str = "low") -> dict:
    if effort not in ("minimal", "low", "medium"):
        raise ValueError("effort must be minimal, low or medium")
    source = {
        "knowledgeSourceName": knowledge_source,
        "kind": "searchIndex",
        "resultsProcessing": "rerank",
        "includeReferences": True,
    }
    if guards:
        source["filterAddOn"] = "status eq 'current'"
    return {
        "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
        "outputMode": "extractiveData",
        "retrievalReasoningEffort": {"kind": effort},
        "includeActivity": True,
        "knowledgeSourceParams": [source],
    }


def agentic_order(q: dict, retriever) -> list[int]:
    """Subqueries (the planner) fused with the dense query, then cross-encoder rerank (the ranker)."""
    rankings = [rank(retriever.bm25.scores(sub)) for sub in decompose_query(q["question"])]
    rankings.append(rank(retriever.emb @ retriever.qemb[q["qid"]].astype(np.float32)))
    fused = rrf(*rankings)
    scores = retriever.rr.get(q["qid"], {})
    head = [i for i in fused if retriever.chunks[i]["chunk_id"] in scores][:40]
    head = sorted(head, key=lambda i: (-scores[retriever.chunks[i]["chunk_id"]], i))
    return head + [i for i in fused if i not in set(head)]


def agentic_chunks(q: dict, retriever, k: int = 3, guards: bool = True) -> list[dict]:
    if live_agentic_enabled():
        return live_agentic_chunks(q["question"], k, guards=guards)
    order = agentic_order(q, retriever)
    if guards:
        order = apply_guards(order, retriever.chunks)
    return [retriever.chunks[i] for i in order[:k]]


def chunks_from_agentic(payload: dict, k: int) -> list[dict]:
    """Map a knowledge-base retrieve response onto the chunk shape the reader expects."""
    refs = payload.get("references") or []
    out = []
    for ref in refs:
        text = ref.get("content") or ref.get("text") or ""
        if not text:
            continue
        out.append({
            "chunk_id": str(ref.get("id") or ref.get("chunk_id") or f"live-{len(out)}"),
            "doc_id": str(ref.get("doc_id") or ref.get("docKey") or ""),
            "title": str(ref.get("title") or ""),
            "text": text,
            "titled": text,
            "status": "current",
            "doc_type": "policy",
            "district": ref.get("district"),
            "topic": str(ref.get("topic") or ""),
        })
    if not out:
        raise RuntimeError("agentic retrieval returned no references")
    return out[:k]


def live_agentic_chunks(question: str, k: int = 3, guards: bool = True) -> list[dict]:
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
    kb = os.environ["AZURE_SEARCH_KNOWLEDGE_BASE"]
    url = f"{endpoint}/knowledgebases/{kb}/retrieve?api-version={API_VERSION}"
    body = agentic_request(question, guards=guards)
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {_search_token()}"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - the caller's search endpoint
        payload = json.loads(resp.read().decode())
    return chunks_from_agentic(payload, k)


def _search_token() -> str:
    token = os.environ.get("AZURE_SEARCH_TOKEN")
    if token:
        return token
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise RuntimeError("agentic retrieval needs AZURE_SEARCH_TOKEN or azure-identity") from None
    return DefaultAzureCredential().get_token("https://search.azure.com/.default").token
