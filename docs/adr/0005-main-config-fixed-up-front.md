# ADR 0005: Fix the main configuration before measuring

**Status:** accepted

**Context.** With 182 questions, picking the best of many configurations and reporting its score would
overstate it.

**Decision.** The main configuration (80-word chunks with titles, hybrid retrieval with cross-encoder
rerank, top 3, guards on, 1.5B reader) was set before any answers were generated. Other configurations
change one thing each and are compared with paired tests.

**Consequences.** Main may not be the best on every metric; the report shows where it is not.
