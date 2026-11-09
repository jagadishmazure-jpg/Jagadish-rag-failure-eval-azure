"""Run the reader models over every question and configuration, on CPU, and store raw answers.

Needs llama-cpp-python==0.3.36 and the GGUF files from fetch_models.sh. Identical prompts are run once
and reused across configurations (greedy decoding makes them identical). Output: data/answers.jsonl.
Resumable: rows already in the file are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragfail import data  # noqa: E402
from ragfail.configs import CONFIGS, config, context_for  # noqa: E402
from ragfail.reader import build_messages  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", type=Path, default=data.DATA / "answers.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    a = ap.parse_args()
    from llama_cpp import Llama

    spec = json.loads((ROOT / "scripts" / "models.json").read_text())
    samp = spec["sampling"]
    files = {m["label"]: m["file"] for m in spec["readers"]}
    done, cache = set(), {}
    if a.out.exists():
        for r in data.read_jsonl(a.out):
            done.add((r["config"], r["qid"]))
            cache[r["prompt_sha"]] = r
    models: dict[str, Llama] = {}
    with a.out.open("a", encoding="utf-8") as fh:
        for name in a.configs.split(","):
            cfg = config(name)
            r = data.retriever(cfg["strategy"])
            if cfg["reader"] not in models:
                models[cfg["reader"]] = Llama(str(a.models_dir / files[cfg["reader"]]), n_ctx=samp["n_ctx"],
                                              n_threads=samp["n_threads"], seed=samp["seed"], verbose=False)
            llm = models[cfg["reader"]]
            for q in data.questions()[: a.limit or None]:
                if (name, q["qid"]) in done:
                    continue
                ctx = context_for(q, cfg, r)
                msgs, shown, stripped = build_messages(q["question"], ctx, guarded=cfg["guards"])
                sha = hashlib.sha256((cfg["reader"] + json.dumps(msgs)).encode()).hexdigest()
                if sha in cache:
                    prev = cache[sha]
                    out, ptok, ctok, secs, reused = prev["answer"], prev["prompt_tokens"], prev["completion_tokens"], prev["latency_s"], True
                else:
                    t = time.perf_counter()
                    res = llm.create_chat_completion(messages=msgs, temperature=samp["temperature"],
                                                     max_tokens=samp["max_tokens"])
                    secs = round(time.perf_counter() - t, 3)
                    out = res["choices"][0]["message"]["content"].strip()
                    ptok, ctok, reused = res["usage"]["prompt_tokens"], res["usage"]["completion_tokens"], False
                row = {"config": name, "qid": q["qid"], "reader": cfg["reader"], "answer": out,
                       "context": [c["chunk_id"] for c in ctx], "shown": shown, "stripped": stripped,
                       "prompt_tokens": ptok, "completion_tokens": ctok, "latency_s": secs,
                       "reused": reused, "prompt_sha": sha}
                cache[sha] = row
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            print(name, "done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
