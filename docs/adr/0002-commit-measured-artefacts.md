# ADR 0002: Commit embeddings, rerank scores and reader answers

**Status:** accepted

**Context.** CI runners have no model weights and should not download gigabytes on every push.

**Decision.** Run the embedding model, the cross-encoder and the readers once on CPU, commit their outputs
with a hash manifest, and replay them in CI. BM25 is recomputed live. The eval also checks that every
stored answer's passages still equal what the current retriever returns, so code and data cannot drift apart.

**Consequences.** Any change to the corpus, chunking, retrieval or prompt needs a re-run of
`scripts/embed_and_rerank.py` and `scripts/run_reader.py` (about an hour on 8 cores).
