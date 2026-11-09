# Data card

**What it is.** A synthetic policy library for a fictional water utility, Tessmoor Water, plus questions
with exact answers. Written by code in `ragfail/corpus.py` (seeded, deterministic). No real company,
person or document is used.

| Part | Count | Notes |
|---|---|---|
| Current policy documents | 32 | 6 policies x 5 districts, plus 2 company-wide |
| Superseded versions | 10 | older values, metadata `status: superseded`, still in the index |
| Old customer FAQ | 1 | 12 answers that contradict current policy |
| Documents with a hidden instruction | 8 | 4 attack texts; canaries use the reserved `.example` domain |
| Facts | 75 | one sentence each, one numeric answer each |
| Questions | 182 | 150 answerable (two phrasings per fact: the document's words, and a staff member's words) and 32 unanswerable |

**Why it looks like this.** Near-identical policies for five districts make "right policy, wrong
district" a real risk. Policy sentences do not repeat the district name, as in real documents, so the
title has to carry it. Unanswerable questions are close to real ones (a district that does not exist,
a fee that is not listed) so that a model has something tempting to answer from.

**Measured artefacts** (all on an 8-core CPU, committed):

- `data/emb/*.npy`: BAAI/bge-small-en-v1.5 embeddings (384 dims, float16) of every chunk and question.
- `data/rerank/*.json`: cross-encoder/ms-marco-MiniLM-L6-v2 scores for each question's candidates.
- `data/answers.jsonl`: every reader answer, with the passages shown, tokens and seconds.
- `data/emb/manifest.json`: model names, versions, timings and SHA-256 of every file.

**Limits.** English only, short documents, numeric answers only, one domain. Two phrasings of the same
fact are not independent samples.
