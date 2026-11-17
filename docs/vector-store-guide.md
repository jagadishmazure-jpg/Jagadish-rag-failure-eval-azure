# Vector store selection guide

Four stores share one interface (`search(vector, k, query_text, filter)`):

| Store | Mock | Live | IaC |
|---|---|---|---|
| In-memory exact cosine | always | — | — |
| Azure AI Search, hybrid + cross-encoder stand-in for the semantic ranker | default | `AZURE_SEARCH_ENDPOINT` | `infra/vectorstores` and the original `infra/terraform` service |
| Azure Cosmos DB for NoSQL, `VectorDistance`, filter in the WHERE | default | `COSMOS_ENDPOINT`, `COSMOS_DATABASE`, `COSMOS_CONTAINER` | diskANN, 384 dimensions, 400 RU/s, keys off |
| PostgreSQL + pgvector | default | `PGVECTOR_DSN` (Azure, or `docker compose up -d`) | Flexible Server B1ms, extension `VECTOR`, password auth off |

`pip install -e ".[stores]"` adds `azure-cosmos` and `psycopg` for the live clients. The benchmark does
not open a connection unless you call `vectorstores.live_search`.

Metadata filtering is part of the score. `status eq current` still leaves the FAQ in the top 3, because
the FAQ is current. `status = current AND doc_type = policy` removes every superseded and FAQ chunk
(`stale_pol` 0). A district equality filter returns only that district.

Cost is an always-on estimate, not a bill. Search reuses the $0.101 per hour Basic price from
[deployment.md](deployment.md) (eastus2, 2026-11-09). Cosmos DB is $0.008 per 100 RU/s per hour at 400
RU/s plus $0.25 per GB. PostgreSQL is $0.017 per hour for Burstable B1ms plus $0.115 per GiB for 32 GiB
(estimates, 2026-11-17). Check the retail prices API before enabling `infra/vectorstores`.

The selection rule: among the three Azure stores whose policy-filtered recall@3 is within 0.01 of the
best, pick the lowest estimated monthly cost. The mock uses exact cosine, so Cosmos (diskANN) and
pgvector (HNSW) are not showing approximate-index loss. In-process p95 is measured and then dropped
from `results/vector_stores.json`, because it depends on the machine. Live service p95 is not measured.

<!-- run: python scripts/run_store_bench.py --summary -->
```text
mode: mock. Exact cosine in-process, except Azure AI Search which also fuses BM25 and the committed cross-encoder.
p95_ms is measured around the in-process call and is not printed here (it depends on the machine).
Live service p95 and live recall: not measured. Set AZURE_SEARCH_ENDPOINT, COSMOS_ENDPOINT or PGVECTOR_DSN.

store            recall@3   status   policy stale_off stale_pol      usd
memory             0.8733   0.8867   0.9000    0.9118    0.0000     0.00
azure_ai_search    0.9733   0.9733   0.9800    0.9559    0.0000    73.73
cosmos_nosql       0.8733   0.8867   0.9000    0.9118    0.0000    23.61
pgvector           0.8733   0.8867   0.9000    0.9118    0.0000    16.09

Store ids: azure_ai_search, cosmos_nosql, pgvector. memory is the free baseline and is not a deployment choice.
Policy-filtered recall@3 spread across the three Azure stores on this mock: 0.0800.
Among stores within 0.01 of the best policy-filtered recall (0.9800), the lowest estimated monthly cost is azure_ai_search at $73.73 (basic, 1 replica, semantic ranker free).
The mock searches exact cosine, so Cosmos DB (diskANN) and pgvector (HNSW) are not showing approximate-index loss. Re-measure recall@k and service p95 with COSMOS_ENDPOINT, PGVECTOR_DSN and AZURE_SEARCH_ENDPOINT before choosing.
Choose Azure AI Search when the workload needs hybrid keyword plus vector plus the semantic ranker, or the guarded retrieval service is already pointed at it.
Choose Azure Cosmos DB for NoSQL when the documents already live as items and the filter must sit in the same VectorDistance query.
Choose Azure Database for PostgreSQL with pgvector when you want SQL predicates and joins, the lowest always-on estimate in this table, and a local docker-compose database for development.
In-process p95 is measured and returned as p95_ms. It is machine-local and is not a service latency. Live service p95 is not measured.
```
<!-- /run -->

Local pgvector:

```bash
docker compose up -d
# PGVECTOR_DSN=postgresql://tessmoor:local-dev-only@localhost:5432/tessmoor
```

The compose password is a laptop default. The Azure server does not accept a password.
