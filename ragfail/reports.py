"""Write a benchmark report without truncating the previous file first."""

from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    """Write JSON, flush it, then replace the destination. A crash leaves the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
