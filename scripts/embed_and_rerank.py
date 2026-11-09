"""Compute dense embeddings and cross-encoder rerank scores on CPU, once, and store them in data/.

Needs: pip install torch sentence-transformers (CPU). Not needed for tests or the eval, which replay
the stored files. Models: BAAI/bge-small-en-v1.5 (MIT) and cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragfail import data  # noqa: E402
from ragfail.chunking import STRATEGIES  # noqa: E402
from ragfail.retrieval import Retriever  # noqa: E402

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.manual_seed(0)
    torch.set_num_threads(8)
    out = data.DATA / "emb"
    out.mkdir(exist_ok=True)
    (data.DATA / "rerank").mkdir(exist_ok=True)
    enc = SentenceTransformer(EMBED_MODEL, device="cpu")
    qs = data.questions()
    t = time.time()
    qmat = enc.encode([QUERY_PREFIX + q["question"] for q in qs], normalize_embeddings=True, batch_size=32)
    np.save(out / "queries.npy", qmat.astype(np.float16))
    (out / "queries.json").write_text(json.dumps([q["qid"] for q in qs]))
    timings = {"queries": round(time.time() - t, 2)}
    for s in STRATEGIES:
        ch = data.chunks(s)
        for field in ("titled", "text"):
            t = time.time()
            m = enc.encode([c[field] for c in ch], normalize_embeddings=True, batch_size=32)
            np.save(out / f"chunks-{s}-{field}.npy", m.astype(np.float16))
            timings[f"chunks-{s}-{field}"] = round(time.time() - t, 2)
    data.query_embeddings.cache_clear()
    ce = CrossEncoder(RERANK_MODEL, device="cpu")
    for s in STRATEGIES:
        ch = data.chunks(s)
        r = Retriever(ch, np.load(out / f"chunks-{s}-titled.npy"), data.query_embeddings())
        idx = {c["chunk_id"]: c for c in ch}
        pairs, keys = [], []
        for q in qs:
            for cid in r.rerank_candidates(q):
                pairs.append((q["question"], idx[cid]["titled"]))
                keys.append((q["qid"], cid))
        t = time.time()
        sc = ce.predict(pairs, batch_size=64)
        timings[f"rerank-{s}"] = round(time.time() - t, 2)
        res: dict[str, dict[str, float]] = {}
        for (qid, cid), v in zip(keys, sc):
            res.setdefault(qid, {})[cid] = round(float(v), 4)
        (data.DATA / "rerank" / f"{s}.json").write_text(json.dumps(res, sort_keys=True))
        timings[f"rerank-{s}-pairs"] = len(pairs)
    manifest = {"embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL, "query_prefix": QUERY_PREFIX,
                "device": "cpu (8 threads)", "torch": torch.__version__, "seconds": timings,
                "files": {p.relative_to(ROOT).as_posix(): sha(p)
                          for p in sorted((data.DATA / "emb").glob("*.np*")) + sorted((data.DATA / "rerank").glob("*.json"))}}
    (data.DATA / "emb" / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(timings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
