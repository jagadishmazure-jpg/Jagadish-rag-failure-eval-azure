"""Vector store benchmark. Offline exact-cosine mocks by default.

    python scripts/run_store_bench.py --summary
    python scripts/run_store_bench.py --write
    python scripts/run_store_bench.py --check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragfail.reports import atomic_json  # noqa: E402
from ragfail.store_bench import committed_view, run_store_bench, summary_lines  # noqa: E402

OUT = ROOT / "results" / "vector_stores.json"


def main() -> int:
    report = run_store_bench()
    view = committed_view(report)
    if "--check" in sys.argv:
        committed = json.loads(OUT.read_text())
        if committed != view:
            print("results/vector_stores.json does not match a fresh mock run. Re-run with --write.")
            return 1
        print("vector store bench matches results/vector_stores.json")
        return 0
    if "--write" in sys.argv:
        atomic_json(OUT, view)
    print("\n".join(summary_lines(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
