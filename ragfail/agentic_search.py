"""Azure AI Search agentic retrieval: query planning, hybrid search, semantic ranker.

The live call is ``POST /knowledgebases/{name}/retrieve?api-version=2026-08-01-preview``. The
knowledge source runs hybrid (keyword + vector) queries and reranks with the semantic ranker
(``resultsProcessing: rerank``). The metadata guard is the OData ``filterAddOn`` ``status eq 'current'``.

Offline, the same three steps run locally: decomposed subqueries, BM25 fused with the stored dense
vector, then the committed cross-encoder scores standing in for the semantic ranker.
"""

from __future__ import annotations

import os

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
    order = agentic_order(q, retriever)
    if guards:
        order = apply_guards(order, retriever.chunks)
    return [retriever.chunks[i] for i in order[:k]]
