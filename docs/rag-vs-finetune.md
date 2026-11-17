# RAG vs fine-tuning vs RAFT

Three arms, one taxonomy, the same RAGAS and Azure AI Evaluation metric names as the advanced bench.

| Arm | Offline behavior | Live switch |
|---|---|---|
| RAG | Guarded hybrid + rerank, extractive reader | none (this is the local pipeline) |
| Fine-tune | Mock of a QLoRA adapter. About 70% of facts, by SHA-256, are memorised. Facts that have a superseded value were learned from the old number. Everything else, including unanswerable questions, returns `9999`. No passages, so a right number is `citation_mismatch`. | `FINETUNE_ENDPOINT` (optional `FINETUNE_TOKEN`) |
| RAFT | Same retrieval as RAG, then drop chunks from the wrong district, then the extractive reader. The payload for the live endpoint includes the passages. | same endpoint, `mode: raft` |

The mock is not a trained adapter. It is a fixed behavior you can replace by pointing
`FINETUNE_ENDPOINT` at a service that hosts the adapter (a QLoRA merge running elsewhere). The request
body is `{"mode", "question", "passages"}` and the response is `{"answer"}`. Passages are empty for
the fine-tune arm.

<!-- run: python scripts/run_adaptation.py --summary -->
```text
mode: mock. Train facts: 46. Held-out facts: 29.
Mock QLoRA adapter. Trained facts are answered from memory. Facts with a superseded value were learned from the old number. Held-out and unanswerable questions return 9999.

arm        accuracy  abstain    stale    cited  top_failure
rag          0.7933   0.0625   0.0000   0.8933  missed_abstain (30)
finetune     0.3067   0.0000   0.6765   0.0000  ungrounded (58)
raft         0.7933   0.0625   0.0000   0.8933  missed_abstain (30)

Unanswerable questions: the fine-tuned arm abstains less often than RAG. Prefer RAG or RAFT when staff can ask for something the corpus does not contain. Policy changes: the fine-tuned arm repeats superseded numbers it was trained on. Prefer RAG with the metadata guard when a fact can change after training. RAFT matches guarded RAG here. The district filter dropped 157 distractor chunks, and the metadata guard had already removed superseded text, so the score did not move. Prefer plain RAG until a trained reader has to face distractors the guard does not catch. Citations: a fine-tuned answer has no passage, so a correct number is citation_mismatch in the taxonomy. Use that arm only when a citation is not required. Point FINETUNE_ENDPOINT at a served QLoRA adapter to replace the mock fine-tune and RAFT answers. Live rows are not measured.
```
<!-- /run -->

How to read the decision text the script prints:

- If the fine-tuned arm abstains less often, prefer RAG or RAFT when staff can ask something the corpus does not contain.
- If it repeats superseded numbers, prefer RAG with the metadata guard when facts change after training.
- If RAFT beats guarded RAG on accuracy or abstention, prefer RAFT when you can train on the question, the gold passage and distractors.
- If RAFT only ties, the guard already removed the stale text and the district filter did not move the score. Prefer plain RAG until a trained reader has to face distractors the guard does not catch.
- A fine-tuned answer with no passage cannot satisfy the citation check. Use that arm when a citation is not required.
