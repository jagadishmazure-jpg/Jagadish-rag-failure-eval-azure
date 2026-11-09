"""Write the synthetic corpus and questions to data/. Deterministic; CI checks the files match."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragfail.corpus import build_corpus, build_questions  # noqa: E402


def render() -> dict[str, str]:
    docs, facts = build_corpus()
    qs = build_questions(facts)
    dump = lambda rows: "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)  # noqa: E731
    return {"corpus.jsonl": dump(docs), "questions.jsonl": dump(qs)}


def main() -> int:
    check = "--check" in sys.argv
    bad = []
    for name, text in render().items():
        p = ROOT / "data" / name
        if check:
            if not p.exists() or p.read_text(encoding="utf-8") != text:
                bad.append(name)
        else:
            p.write_text(text, encoding="utf-8")
    if bad:
        print("out of date:", ", ".join(bad))
        return 1
    print("dataset", "matches" if check else "written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
