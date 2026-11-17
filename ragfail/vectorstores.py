"""Pluggable vector stores: in-memory, Azure AI Search, Cosmos DB for NoSQL, PostgreSQL pgvector.

The default for each Azure store is an in-process exact-cosine mock with that store's filter rules.
Live clients run only when their environment variable is set, and they are imported lazily.

Estimated monthly cost uses the Search price already cited in ``docs/deployment.md`` ($0.101 per hour,
Basic, eastus2, 2026-11-09) and labeled estimates for Cosmos DB provisioned throughput and PostgreSQL
Flexible Server Burstable B1ms (2026-11-17). Check the retail prices API before enabling the stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .retrieval import BM25, rank, rrf

HOURS_PER_MONTH = 730
SEARCH_USD_PER_HOUR = 0.101
COSMOS_USD_PER_100_RU_HOUR = 0.008
COSMOS_RU_PER_S = 400
COSMOS_STORAGE_GB = 1.0
COSMOS_USD_PER_GB = 0.25
PG_USD_PER_HOUR = 0.017
PG_STORAGE_GB = 32
PG_USD_PER_GB = 0.115

STORE_NAMES = ("memory", "azure_ai_search", "cosmos_nosql", "pgvector")
FILTER_FIELDS = ("status", "doc_type", "district", "topic", "doc_id")


@dataclass
class Hit:
    chunk_id: str
    score: float
    chunk: dict


def monthly_cost(kind: str) -> dict:
    """Always-on estimate for the IaC sku of one store. ``memory`` is free."""
    if kind == "memory":
        return {"usd_month": 0.0, "sku": "in-process", "breakdown": {"compute": 0.0}}
    if kind == "azure_ai_search":
        compute = round(SEARCH_USD_PER_HOUR * HOURS_PER_MONTH, 2)
        return {"usd_month": compute, "sku": "basic, 1 replica, semantic ranker free",
                "breakdown": {"search_basic": compute, "semantic_ranker": 0.0}}
    if kind == "cosmos_nosql":
        compute = round((COSMOS_RU_PER_S / 100) * COSMOS_USD_PER_100_RU_HOUR * HOURS_PER_MONTH, 2)
        storage = round(COSMOS_STORAGE_GB * COSMOS_USD_PER_GB, 2)
        return {"usd_month": round(compute + storage, 2), "sku": f"provisioned {COSMOS_RU_PER_S} RU/s, diskANN",
                "breakdown": {"throughput": compute, "storage": storage}}
    if kind == "pgvector":
        compute = round(PG_USD_PER_HOUR * HOURS_PER_MONTH, 2)
        storage = round(PG_STORAGE_GB * PG_USD_PER_GB, 2)
        return {"usd_month": round(compute + storage, 2), "sku": "Flexible Server B1ms, 32 GiB, pgvector HNSW",
                "breakdown": {"compute": compute, "storage": storage}}
    raise ValueError(f"unknown store {kind}")


def _mask(chunks: list[dict], flt: dict | None) -> np.ndarray:
    if not flt:
        return np.ones(len(chunks), dtype=bool)
    unknown = set(flt) - set(FILTER_FIELDS)
    if unknown:
        raise ValueError(f"unsupported filter field: {sorted(unknown)[0]}")
    keep = np.ones(len(chunks), dtype=bool)
    for key, value in flt.items():
        keep &= np.array([c.get(key) == value for c in chunks])
    return keep


def _cosine_order(emb: np.ndarray, vector: np.ndarray, mask: np.ndarray) -> tuple[list[int], np.ndarray]:
    v = vector.astype(np.float32)
    norm = float(np.linalg.norm(v))
    if norm:
        v = v / norm
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    e = emb / np.maximum(norms, 1e-9)
    scores = e @ v
    scores = np.where(mask, scores, -np.inf)
    return rank(scores), scores


class MemoryStore:
    """Exact cosine. The baseline the other mocks are measured against."""

    name = "memory"
    kind = "memory"
    ann = "exact"
    live = False

    def __init__(self, chunks: list[dict], emb: np.ndarray):
        self.chunks, self.emb = chunks, emb.astype(np.float32)

    def search(self, vector: np.ndarray, k: int, *, query_text: str = "", flt: dict | None = None,
               qid: str | None = None) -> list[Hit]:
        order, scores = _cosine_order(self.emb, vector, _mask(self.chunks, flt))
        return [Hit(self.chunks[i]["chunk_id"], float(scores[i]), self.chunks[i]) for i in order[:k] if scores[i] > -np.inf]


class AzureSearchStore:
    """Hybrid (BM25 + vector) and a cross-encoder stand-in for the semantic ranker.

    Live queries go to ``AZURE_SEARCH_ENDPOINT`` through ``azure_search.search`` when no vector is
    required, and through ``query_body`` (hybrid + semantic) when the endpoint is set. The mock is
    what the benchmark runs in CI.
    """

    name = "azure_ai_search"
    kind = "azure_ai_search"
    ann = "exact+bm25+cross-encoder"
    live = False

    def __init__(self, chunks: list[dict], emb: np.ndarray, rerank: dict | None = None):
        self.chunks, self.emb = chunks, emb.astype(np.float32)
        self.bm25 = BM25([c["titled"] for c in chunks])
        self.rerank = rerank or {}

    def search(self, vector: np.ndarray, k: int, *, query_text: str = "", flt: dict | None = None,
               qid: str | None = None) -> list[Hit]:
        mask = _mask(self.chunks, flt)
        dense, scores = _cosine_order(self.emb, vector, mask)
        if query_text:
            lexical = [i for i in rank(self.bm25.scores(query_text)) if mask[i]]
            fused = [i for i in rrf(lexical, dense) if mask[i]]
        else:
            fused = dense
        if qid and qid in self.rerank:
            known = self.rerank[qid]
            head = [i for i in fused if self.chunks[i]["chunk_id"] in known][:40]
            head = sorted(head, key=lambda i: (-known[self.chunks[i]["chunk_id"]], i))
            fused = head + [i for i in fused if i not in set(head)]
        return [Hit(self.chunks[i]["chunk_id"], float(scores[i]), self.chunks[i]) for i in fused[:k]]


class CosmosStore:
    """Vector-only search with the filter in the predicate, matching a ``VectorDistance`` WHERE query.

    Live SQL is built by ``cosmos_query`` and sent when ``COSMOS_ENDPOINT`` is set.
    """

    name = "cosmos_nosql"
    kind = "cosmos_nosql"
    ann = "exact (live IaC: diskANN)"
    live = False

    def __init__(self, chunks: list[dict], emb: np.ndarray):
        self.chunks, self.emb = chunks, emb.astype(np.float32)

    def search(self, vector: np.ndarray, k: int, *, query_text: str = "", flt: dict | None = None,
               qid: str | None = None) -> list[Hit]:
        order, scores = _cosine_order(self.emb, vector, _mask(self.chunks, flt))
        return [Hit(self.chunks[i]["chunk_id"], float(scores[i]), self.chunks[i]) for i in order[:k] if scores[i] > -np.inf]


class PgVectorStore:
    """Same exact cosine as Cosmos, with a SQL WHERE filter. Live path uses ``PGVECTOR_DSN``."""

    name = "pgvector"
    kind = "pgvector"
    ann = "exact (live: HNSW cosine)"
    live = False

    def __init__(self, chunks: list[dict], emb: np.ndarray):
        self.chunks, self.emb = chunks, emb.astype(np.float32)

    def search(self, vector: np.ndarray, k: int, *, query_text: str = "", flt: dict | None = None,
               qid: str | None = None) -> list[Hit]:
        order, scores = _cosine_order(self.emb, vector, _mask(self.chunks, flt))
        return [Hit(self.chunks[i]["chunk_id"], float(scores[i]), self.chunks[i]) for i in order[:k] if scores[i] > -np.inf]


def cosmos_query(k: int, flt: dict | None) -> str:
    """Parameterized-looking Cosmos SQL. ``k`` is an integer literal because TOP cannot be a parameter.
    Filter values are not interpolated; callers bind them as ``@f0`` ... beside ``@vector``."""
    if not isinstance(k, int) or not 1 <= k <= 50:
        raise ValueError("k must be an integer from 1 to 50")
    where = []
    if flt:
        for i, key in enumerate(sorted(flt)):
            if key not in FILTER_FIELDS:
                raise ValueError(f"unsupported filter field: {key}")
            where.append(f"c.{key} = @f{i}")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return (f"SELECT TOP {k} c.chunk_id, c.doc_id, c.title, c.text, c.status, c.doc_type, c.district, c.topic, "
            f"VectorDistance(c.vector, @vector) AS score FROM c{clause} "
            "ORDER BY VectorDistance(c.vector, @vector)")


def pg_sql(k: int, flt: dict | None) -> str:
    """pgvector query. Values are bound by the caller (``%s``), never interpolated."""
    if not isinstance(k, int) or not 1 <= k <= 50:
        raise ValueError("k must be an integer from 1 to 50")
    where = []
    if flt:
        for key in sorted(flt):
            if key not in FILTER_FIELDS:
                raise ValueError(f"unsupported filter field: {key}")
            where.append(f"{key} = %s")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return ("SELECT chunk_id, doc_id, title, text, status, doc_type, district, topic "
            f"FROM chunks{clause} ORDER BY embedding <=> %s::vector LIMIT %s")


def live_store_env() -> dict[str, str | None]:
    return {
        "azure_ai_search": os.environ.get("AZURE_SEARCH_ENDPOINT"),
        "cosmos_nosql": os.environ.get("COSMOS_ENDPOINT"),
        "pgvector": os.environ.get("PGVECTOR_DSN"),
    }


def build_stores(chunks: list[dict], emb: np.ndarray, rerank: dict | None = None) -> list:
    """Mock stores. A set env var does not silently switch the benchmark to the network; call
    ``live_search`` from a script that opts in."""
    return [
        MemoryStore(chunks, emb),
        AzureSearchStore(chunks, emb, rerank),
        CosmosStore(chunks, emb),
        PgVectorStore(chunks, emb),
    ]


def odata_filter(flt: dict | None) -> str | None:
    """Equality filter in the OData shape Azure AI Search expects. Values are escaped, never joined raw."""
    if not flt:
        return None
    parts = []
    for key in sorted(flt):
        if key not in FILTER_FIELDS:
            raise ValueError(f"unsupported filter field: {key}")
        value = str(flt[key]).replace("'", "''")
        parts.append(f"{key} eq '{value}'")
    return " and ".join(parts)


def live_search(kind: str, query_text: str, vector: list[float], k: int, flt: dict | None = None) -> list[dict]:
    """One live query. Raises if the env var or the client library is missing."""
    env = live_store_env()
    if kind == "azure_ai_search":
        if not env["azure_ai_search"]:
            raise RuntimeError("AZURE_SEARCH_ENDPOINT is not set")
        from .azure_search import search

        return search(env["azure_ai_search"], query_text, k=k, query_vector=vector, guards=False,
                      filter_expr=odata_filter(flt))
    if kind == "cosmos_nosql":
        if not env["cosmos_nosql"]:
            raise RuntimeError("COSMOS_ENDPOINT is not set")
        try:
            from azure.cosmos import CosmosClient
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise RuntimeError('Cosmos live search needs azure-cosmos and azure-identity. pip install -e ".[stores]"') from None
        sql = cosmos_query(k, flt)
        client = CosmosClient(env["cosmos_nosql"], credential=DefaultAzureCredential())
        container = client.get_database_client(os.environ["COSMOS_DATABASE"]).get_container_client(os.environ["COSMOS_CONTAINER"])
        params = [{"name": "@vector", "value": vector}]
        params += [{"name": f"@f{i}", "value": flt[key]} for i, key in enumerate(sorted(flt or {}))]
        return list(container.query_items(sql, parameters=params, enable_cross_partition_query=True))
    if kind == "pgvector":
        if not env["pgvector"]:
            raise RuntimeError("PGVECTOR_DSN is not set")
        try:
            import psycopg
        except ImportError:
            raise RuntimeError('pgvector live search needs psycopg. pip install -e ".[stores]"') from None
        sql = pg_sql(k, flt)
        params = [flt[key] for key in sorted(flt or {})]
        params.append("[" + ",".join(str(float(x)) for x in vector) + "]")
        params.append(k)
        with psycopg.connect(env["pgvector"]) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    raise ValueError(f"unknown store {kind}")
