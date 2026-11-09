# ADR 0004: Guards use metadata, not the model, for stale and conflicting sources

**Status:** accepted

**Context.** Asking the reader to "prefer the newest document" relies on a small model reading dates.

**Decision.** Drop superseded chunks by metadata before ranking, and rank non-policy sources (the FAQ)
after policy ones. In Azure AI Search this is a filter (`status eq 'current'`) plus a client-side reorder.

**Consequences.** Cheap and reliable when metadata is right; useless when it is wrong. Document owners
must keep `status` and `doc_type` accurate.
