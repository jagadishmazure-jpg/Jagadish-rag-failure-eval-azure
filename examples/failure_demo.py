"""Walk through four failure modes with real, measured reader answers (from data/answers.jsonl).

Run: python examples/failure_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragfail import data  # noqa: E402
from ragfail.configs import config, context_for  # noqa: E402
from ragfail.scoring import score  # noqa: E402

qs = {q["qid"]: q for q in data.questions()}
answers = {(r["config"], r["qid"]): r for r in data.read_jsonl(data.DATA / "answers.jsonl")}


def category(name: str, qid: str) -> str:
    cfg, q, row = config(name), qs[qid], answers[(name, qid)]
    return score(q, row["answer"], row["shown"], context_for(q, cfg, data.retriever(cfg["strategy"])))["category"]


def first(pred) -> str:
    """The first question (in file order) that shows the failure: picked by rule, not by hand."""
    return next(q["qid"] for q in data.questions() if pred(q))


CASES = [
    ("Hidden instruction in a document (the readers ignored it even without guards)", "connect-quillby-fee:l",
     ["no_guards", "main"]),
    ("Superseded version still in the index",
     first(lambda q: "stale" in q["kinds"] and category("no_guards", q["qid"]) == "stale_or_conflict"),
     ["no_guards", "main"]),
    ("Paraphrased question, keyword search",
     first(lambda q: q["style"] == "paraphrase" and category("bm25", q["qid"]) == "retrieval_miss"
           and category("main", q["qid"]) == "correct"), ["bm25", "main"]),
    ("Question the corpus cannot answer", first(lambda q: not q["answerable"] and category("main", q["qid"]) == "missed_abstain"),
     ["main"]),
]

for title, qid, cfgs in CASES:
    q = qs[qid]
    gold = "should abstain" if not q["answerable"] else f"gold {q['gold']:g}"
    print(f"[{title}] {q['question']} ({gold})")
    for name in cfgs:
        cfg = config(name)
        row = answers[(name, qid)]
        ctx = context_for(q, cfg, data.retriever(cfg["strategy"]))
        s = score(q, row["answer"], row["shown"], ctx)
        print(f"  {name:<10} passages {', '.join(row['context'])}")
        print(f"  {'':<10} answer {row['answer']!r} -> {s['category']}"
              + (f" (guard stripped {row['stripped']} sentence(s))" if row["stripped"] else ""))
    print()
