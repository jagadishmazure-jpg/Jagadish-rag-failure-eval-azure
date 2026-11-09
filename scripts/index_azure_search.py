"""Build the Azure AI Search index from the committed chunks and embeddings. WRITTEN, NOT RUN.

Run only by the gated deploy workflow:  python scripts/index_azure_search.py https://<service>.search.windows.net
Needs: pip install azure-search-documents azure-identity, and the Search Index Data Contributor role.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragfail import azure_search, data  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].startswith("https://"):
        print(__doc__)
        return 2
    chunks = data.chunks("w80")
    emb = np.load(data.DATA / "emb" / "chunks-w80-titled.npy").astype(np.float32)
    azure_search.upload(sys.argv[1], chunks, emb)
    print(f"indexed {len(chunks)} chunks into {azure_search.INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
