# Implementation guide: running this failure eval on your own RAG system

1. **Write a small corpus with traps, or tag your own.** You need: near-duplicate documents, a superseded
   version, a conflicting informal source, and a document with a planted instruction (use a canary
   string on a reserved domain such as `.example`). Keep answers checkable by code.
2. **Write questions in two styles.** The document's own words and the way users really ask. The gap
   between them is the most useful retrieval number you will get.
3. **Add unanswerable questions** close to real ones. Expect "should abstain" to be the weakest metric.
4. **Measure retrieval first.** recall@k for each retriever, chunk size, with and without titles in
   the indexed text. Fix retrieval before blaming the model.
5. **Score answers by code where you can.** Correct, grounded (every number appears in the passages),
   cited passage contains the answer, abstained, canary present. Use an LLM judge only for open text,
   and calibrate it against people.
6. **Classify every failure** with one ordered taxonomy (see `ragfail/scoring.py`) so a fix can be
   aimed at one category.
7. **Turn on guards one at a time** and re-measure: metadata filter for superseded documents, source
   ranking, instruction filter, a system prompt that treats passages as data.
8. **Gate CI** on the main configuration: recall floor, accuracy floor, abstention floor, injection and
   stale-answer ceilings. Commit measured answers so the gate replays without a GPU.
9. **For Azure**, mirror the guards in Azure AI Search (filter on `status`, a field for the source type)
   and re-run the same questions against the cloud pipeline before switching traffic. See
   [deployment.md](deployment.md).
