"""Advanced RAG bench. Offline by default.

    python scripts/run_advanced_bench.py --summary
    python scripts/run_advanced_bench.py --write
    python scripts/run_advanced_bench.py --check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragfail.advanced_bench import run_advanced, summary_lines  # noqa: E402
from ragfail.reports import atomic_json  # noqa: E402

OUT = ROOT / "results" / "advanced_rag.json"


def main() -> int:
    report = run_advanced()
    if "--check" in sys.argv:
        committed = json.loads(OUT.read_text())
        if committed != report:
            print("results/advanced_rag.json does not match a fresh mock run. Re-run with --write.")
            return 1
        print("advanced rag bench matches results/advanced_rag.json")
        return 0
    if "--write" in sys.argv:
        atomic_json(OUT, report)
    print("\n".join(summary_lines(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
