# ADR 0006: Advanced benches default to deterministic stand-ins

**Status:** accepted

**Context.** HyDE, multi-query, step-back, decomposition, Self-RAG, agentic retrieval, GraphRAG, RAGAS,
the Azure AI Evaluation SDK, and a fine-tuned adapter all call a model or a cloud store. The rest of
this repo replays in CI with no network and no weights.

**Decision.** Each of those techniques has a pure-Python stand-in that uses the committed corpus,
embeddings and cross-encoder scores. The live call is a separate function, and it runs only when its
environment variable is set. The committed result tables are the mock run. Live numbers are left as
"not measured".

**Consequences.** CI stays offline and the tables replay. A mock row is not a measurement of Azure AI
Search, Cosmos DB, PostgreSQL, RAGAS or a QLoRA adapter. The guides say so next to the numbers.
