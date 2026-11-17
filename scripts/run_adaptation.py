"""RAG vs fine-tuning vs RAFT. The fine-tuned arm is a mock unless FINETUNE_ENDPOINT is set.

    python scripts/run_adaptation.py --summary
    python scripts/run_adaptation.py --write
    python scripts/run_adaptation.py --check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragfail.adaptation import run_adaptation, summary_lines  # noqa: E402

OUT = ROOT / "results" / "adaptation.json"


def main() -> int:
    report = run_adaptation()
    if "--check" in sys.argv:
        committed = json.loads(OUT.read_text())
        if committed != report:
            print("results/adaptation.json does not match a fresh mock run. Re-run with --write.")
            return 1
        print("adaptation bench matches results/adaptation.json")
        return 0
    if "--write" in sys.argv:
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\n".join(summary_lines(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
