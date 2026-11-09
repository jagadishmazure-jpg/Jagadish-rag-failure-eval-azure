# ADR 0001: Score answers by code, not by an LLM judge

**Status:** accepted

**Context.** Faithfulness and correctness are usually judged by another LLM. Judges drift, are biased by
answer length and position, and cost money.

**Decision.** Every fact has one numeric answer and one gold sentence. Correctness, grounding, citation
and abstention are checked by code (`ragfail/scoring.py`).

**Consequences.** Exact and repeatable, but only for short factual answers. Open-ended answers would need
a judge calibrated against human labels (Planned).
