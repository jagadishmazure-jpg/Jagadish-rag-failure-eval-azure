# Advanced RAG bench

Same 182 questions, same ordered failure taxonomy (`ragfail/scoring.py`). The reader in this bench is
extractive (the sentence in the retrieved chunks with the most content-word overlap), so the run needs
no model weights. The measured qwen answers stay in the main eval.

| Strategy | What the mock does | Live switch |
|---|---|---|
| HyDE | Hypothetical policy sentence, embedded as a BM25-weighted average of chunk vectors, fused with the original query | `RAGFAIL_LIVE_LLM=1` plus Azure OpenAI |
| Multi-query | Original question, a keyword query, and a short paraphrase, fused with RRF | same |
| Step-back | A higher-level question that still names the district | same |
| Decomposition | Entity query (topic + district) plus any coordinated clauses | same |
| Self-RAG | Relevance, support and usefulness checks, one retry | none (the critique is code) |
| Agentic retrieval | Subqueries, hybrid (BM25 + dense), cross-encoder stand-in for the semantic ranker, `status eq 'current'` | `AZURE_SEARCH_ENDPOINT` and `AZURE_SEARCH_KNOWLEDGE_BASE` |
| GraphRAG | Local search over district, topic and document status | `GRAPHRAG_QUERY_URL` |
| Vector RAG | Dense cosine, the comparison arm for GraphRAG | the store bench |

Agentic retrieval's live request is `POST /knowledgebases/{name}/retrieve?api-version=2026-08-01-preview`
with `resultsProcessing: rerank` (hybrid query + semantic ranker) and `filterAddOn: status eq 'current'`.
See `ragfail/agentic_search.py`.

Judges: the mock fills the RAGAS names (faithfulness, answer relevancy, context precision, context
recall) and the Azure AI Evaluation names (groundedness, relevance, retrieval) from the code scorer.
`RAGFAIL_LIVE_JUDGES=1` calls the libraries and needs `pip install -e ".[judges]"`,
`AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT`. Offline scores are 0-1. The Azure evaluators
return 1-5 on the live path.

The extractive reader answers most unanswerable questions from a nearby policy, so `missed_abstain` is
the top category for every strategy. Read accuracy and context recall for the retrieval comparison.

<!-- run: python scripts/run_advanced_bench.py --summary -->
```text
mode: mock. Judges: RAGAS and Azure AI Evaluation metric names, scored by code. Set RAGFAIL_LIVE_JUDGES=1 (and install the [judges] extra) to call the libraries.
questions: 182. Reader: extractive. k=3.

strategy         accuracy  abstain    faith  ctx_rec ctx_prec    stale  top_failure
hybrid_rerank      0.7933   0.0625   1.0000   0.9800   0.3267   0.0000  missed_abstain (30)
hyde               0.7533   0.0938   1.0000   0.9000   0.3000   0.0000  missed_abstain (29)
multi_query        0.7667   0.0625   1.0000   0.9067   0.3022   0.0000  missed_abstain (30)
step_back          0.7533   0.0938   1.0000   0.9467   0.3156   0.0000  missed_abstain (29)
decomposition      0.7333   0.0625   1.0000   0.9333   0.3111   0.0000  missed_abstain (30)
self_rag           0.7933   0.0625   1.0000   0.9800   0.3267   0.0000  missed_abstain (30)
agentic            0.7933   0.0625   1.0000   0.9800   0.3267   0.0000  missed_abstain (30)
graphrag           0.7600   0.0625   1.0000   0.9733   0.4467   0.0000  missed_abstain (30)
vector             0.7533   0.0625   1.0000   0.9000   0.3000   0.0000  missed_abstain (30)

GraphRAG context_recall 0.9733 (local 176, vector fallback 6) vs vector RAG 0.9000.
Self-RAG questions with a critique retry: 8.
Agentic retrieval: offline planner + hybrid + cross-encoder stand-in for the semantic ranker. Live: AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KNOWLEDGE_BASE.
Live Azure columns for this bench are not measured yet.
```
<!-- /run -->
