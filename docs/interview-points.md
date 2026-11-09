# Interview points

**30-second version.** I built a harness that plants RAG failure modes in a document library and
measures each one. It covers near-duplicate policies for five districts, superseded versions still in
the index, a contradicting FAQ, hidden instructions and unanswerable questions. Two small open models
answered 2,002 times on CPU, and code scores every answer, so there is no judge to argue with. The main
setup finds the right passage 98.0%<!-- m:retrieval/grid/w80.titled.hybrid_rerank/recall@3/rate --> of the
time and answers 86.7%<!-- m:generation/main/accuracy/rate --> correctly. Every failure lands in one
category, so each fix has an owner.

**Design choices I would defend**

- *Plant the failures, don't wait for them.* A synthetic corpus with known traps gives exact answers and
  lets me switch each trap on and off.
- *Measure once, replay in CI.* Embeddings, rerank scores and answers are committed with a hash
  manifest. The eval checks that every stored answer's passages still match the live retriever, so code
  and data cannot drift apart.
- *One ordered taxonomy.* An answer that is both stale and wrong counts once, in the bucket you would
  fix first.
- *Metadata guards before prompt tricks.* Dropping superseded documents by metadata cut stale answers
  from 50.0%<!-- m:generation/no_guards/stale_answers/rate --> to zero. Asking a small model to "prefer
  the newest" would not be reliable.
- *Main setup fixed up front.* Main is not the best on every metric (one passage or paragraph chunks
  score higher), and the report says so.

**Things that went wrong and what I did**

- My first instruction filter re-joined every paragraph after splitting it into sentences, which turned
  "0.4" into "0. 4" in the passages the model saw. The model then answered "0. 3 mg/L". I caught it in
  the raw answers, made the filter leave clean paragraphs byte-identical, added a test for that, and
  re-ran all 2,002 answers.
- The first prompt showed an example answer ("14 days [2]"), and the 1.5B model copied it for most
  questions. I replaced it with worked examples on an unrelated topic, plus a test that the examples
  contain no corpus content.
- No planted instruction ever worked, even with guards off. I report that as "guards unproven here",
  keep a test for it, and keep the CI ceiling at zero.

**What I would do next**

- Run the same questions against the Azure AI Search + Foundry pipeline that the IaC describes.
- Attack harder: more injection styles, injections in titles, and a reader that is known to follow them.
- Open-ended answers scored by a judge calibrated against human labels.
