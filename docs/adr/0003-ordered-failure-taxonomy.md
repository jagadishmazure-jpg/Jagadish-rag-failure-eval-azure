# ADR 0003: One ordered failure taxonomy

**Status:** accepted

**Context.** One answer can fail in several ways at once (for example a wrong number that is also from a
stale document). Counting it in several buckets hides what to fix first.

**Decision.** Each answer gets exactly one category, the first that matches: injection followed, correct,
citation mismatch, false abstain, retrieval miss, stale or conflict, ungrounded, wrong chunk, reader
error, missed abstain (and correct abstain for unanswerable questions).

**Consequences.** Totals add up and categories map to owners (retrieval, data hygiene, prompt, model).
The order is a choice; it is in code and tested.
