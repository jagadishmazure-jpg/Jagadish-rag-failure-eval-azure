"""Azure AI Search mapping for the corpus. WRITTEN, NOT RUN against Azure.

The pure functions (index schema, documents, query body) are tested offline. `upload` and `search`
need `pip install azure-search-documents azure-identity` and an Azure AI Search service created by
infra/; they use Entra ID (DefaultAzureCredential), never API keys.
"""

from __future__ import annotations

import re

import numpy as np

INDEX = "tessmoor-policies"
DIMS = 384  # BAAI/bge-small-en-v1.5, the same model the local eval measured
SAFE_KEY = re.compile(r"[^A-Za-z0-9_\-=]")


def doc_key(chunk_id: str) -> str:
    """Azure AI Search keys allow letters, digits, _ - = only."""
    return SAFE_KEY.sub("_", chunk_id)


def index_schema(dims: int = DIMS) -> dict:
    s = lambda name, **kw: {"name": name, "type": "Edm.String", **kw}  # noqa: E731
    return {
        "name": INDEX,
        "fields": [
            s("id", key=True, filterable=True),
            s("chunk_id", filterable=True, retrievable=True),
            s("doc_id", filterable=True, facetable=True),
            s("title", searchable=True),
            s("text", searchable=True, analyzer="en.microsoft"),
            s("status", filterable=True),
            s("doc_type", filterable=True),
            s("district", filterable=True, facetable=True),
            s("topic", filterable=True, facetable=True),
            {"name": "effective", "type": "Edm.DateTimeOffset", "filterable": True, "sortable": True},
            {"name": "vector", "type": "Collection(Edm.Single)", "searchable": True, "retrievable": False,
             "dimensions": dims, "vectorSearchProfile": "hnsw-cosine"},
        ],
        "vectorSearch": {
            "algorithms": [{"name": "hnsw", "kind": "hnsw", "hnswParameters": {"metric": "cosine", "m": 4,
                                                                             "efConstruction": 400, "efSearch": 500}}],
            "profiles": [{"name": "hnsw-cosine", "algorithm": "hnsw"}],
        },
        "semantic": {"configurations": [{"name": "default", "prioritizedFields": {
            "titleField": {"fieldName": "title"}, "prioritizedContentFields": [{"fieldName": "text"}]}}]},
    }


def documents(chunks: list[dict], emb: np.ndarray) -> list[dict]:
    if len(chunks) != len(emb) or emb.shape[1] != DIMS:
        raise ValueError("embeddings do not match the chunks or the index dimensions")
    return [{"id": doc_key(c["chunk_id"]), "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"],
             "text": c["titled"], "status": c["status"], "doc_type": c["doc_type"], "district": c["district"],
             "topic": c["topic"], "effective": f"{c['effective']}T00:00:00Z",
             "vector": [round(float(x), 6) for x in e]} for c, e in zip(chunks, emb)]


def query_body(question: str, query_vector: list[float] | None = None, guards: bool = True,
               candidates: int = 20) -> dict:
    """Keyword (BM25) query, plus a vector query when a vector is given (hybrid, fused by the service),
    re-ranked by the semantic ranker."""
    body = {"search": question, "top": candidates, "queryType": "semantic", "semanticConfiguration": "default",
            "select": "chunk_id,doc_id,title,text,status,doc_type,district,topic,effective"}
    if query_vector is not None:
        if len(query_vector) != DIMS:
            raise ValueError("query vector has the wrong dimensions")
        body["vectorQueries"] = [{"kind": "vector", "vector": query_vector, "fields": "vector", "k": candidates}]
    if guards:
        body["filter"] = "status eq 'current'"
    return body  # the caller reorders policy-first (retrieval.apply_guards) and keeps the top k


def upload(endpoint: str, chunks: list[dict], emb: np.ndarray) -> None:  # pragma: no cover - needs Azure
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import SearchIndex

    cred = DefaultAzureCredential()
    SearchIndexClient(endpoint, cred).create_or_update_index(SearchIndex.from_dict(index_schema()))
    client = SearchClient(endpoint, INDEX, cred)
    docs = documents(chunks, emb)
    for i in range(0, len(docs), 500):
        client.upload_documents(docs[i:i + 500])


def search(endpoint: str, question: str, k: int = 3) -> list[dict]:  # pragma: no cover - needs Azure
    """Query the index and return chunk records, policy documents first, top k."""
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient

    body = query_body(question)
    client = SearchClient(endpoint, INDEX, DefaultAzureCredential())
    hits = list(client.search(search_text=body["search"], top=body["top"], query_type=body["queryType"],
                              semantic_configuration_name=body["semanticConfiguration"], filter=body.get("filter"),
                              select=body["select"].split(",")))
    rows = [{**h, "titled": h["text"]} for h in hits]
    rows = [r for r in rows if r["doc_type"] == "policy"] + [r for r in rows if r["doc_type"] != "policy"]
    return rows[:k]
