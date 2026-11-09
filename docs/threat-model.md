# Threat model

A RAG system turns **documents into answers**. Its risks are documents that lie (stale, conflicting or
planted text), answers that are not backed by the documents, and the system answering when it should
say "I don't know".

## STRIDE

| Threat | Example | Control | Where |
|---|---|---|---|
| Spoofing | A document pretends to be a system message ("SYSTEM OVERRIDE ...") | Instruction filter strips it; system prompt says passages are data; eval measures attack success | `guards.strip_instructions`, `reader.SYSTEM_GUARDED` |
| Tampering | An old version or an old FAQ contradicts current policy | Superseded versions filtered by metadata; non-policy sources ranked last | `retrieval.apply_guards`, Search filter `status eq 'current'` |
| Repudiation | "Where did this answer come from?" | Answers must cite a passage; citation checked by code | `scoring.score` |
| Information disclosure | Injected text asks staff to send card numbers to an outside address | Canary detection in every answer; CI gate on injection success | `scoring.injected`, `scripts/run_eval.py` |
| Denial of service | Very long questions | Question size cap | `guards.check_question` |
| Elevation of privilege | Endpoint writes to the index | Endpoint identity has Search Index Data Reader only; API keys disabled | `infra/` |

## OWASP LLM Top 10 (where it fits)

| Threat | Control or measurement |
|---|---|
| Prompt injection, indirect (LLM01) | Four planted instructions; filter + system prompt; success rate measured with and without guards. One politely worded injection passes the filter on purpose (known miss). |
| Sensitive information disclosure (LLM02) | Synthetic data only; canary addresses use the reserved `.example` domain |
| Supply chain (LLM03) | Reader weights pinned by SHA-256; embedding and rerank files hashed in a manifest; Python deps and actions pinned |
| Data and model poisoning (LLM04) | Stale and conflicting sources measured; metadata guards |
| Improper output handling (LLM05) | Answers are short strings with citations; nothing is executed |
| Misinformation (LLM09) | Grounding, citation and abstention checks by code; unanswerable questions in every run |
| Vector and embedding weaknesses (LLM08) | Wrong-district retrieval measured with and without titles in the indexed text |

## Out of scope

Real customer data, multi-tenant access control on documents, and attacks on the embedding model itself.
