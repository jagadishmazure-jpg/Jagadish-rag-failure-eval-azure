"""Split documents into chunks. Three strategies, compared in the eval.

* ``para``: one paragraph per chunk (small, precise, loses context),
* ``w80``: paragraphs packed up to 80 words,
* ``doc``: the whole document, split only above 300 words (large, carries everything).

Chunks never split a sentence. ``titled`` text puts the document title in front of the chunk, which is
what the retrievers index; the untitled text is kept to measure how much the title matters.
"""

from __future__ import annotations

STRATEGIES = {"para": 1, "w80": 80, "doc": 300}


def _pack(paras: list[str], limit: int) -> list[str]:
    if limit == 1:
        return paras
    out, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and n + w > limit:
            out.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        out.append("\n\n".join(cur))
    return out


def chunk_doc(doc: dict, strategy: str) -> list[dict]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunking strategy: {strategy}")
    header, *paras = doc["text"].split("\n\n")
    out = []
    for i, body in enumerate(_pack(paras, STRATEGIES[strategy])):
        out.append({"chunk_id": f"{doc['doc_id']}#{strategy}{i}", "doc_id": doc["doc_id"], "strategy": strategy,
                    "family": doc["family"], "topic": doc["topic"], "district": doc["district"],
                    "status": doc["status"], "doc_type": doc["doc_type"], "effective": doc["effective"],
                    "title": doc["title"], "header": header, "text": body,
                    "titled": f"{doc['title']}. {header}\n{body}"})
    return out


def chunk_corpus(docs: list[dict], strategy: str) -> list[dict]:
    return [c for d in docs for c in chunk_doc(d, strategy)]


def gold_chunks(chunks: list[dict], question: dict) -> set[str]:
    """Chunks of the current gold document that contain the gold sentence."""
    if not question["answerable"]:
        return set()
    return {c["chunk_id"] for c in chunks
            if c["doc_id"] == question["gold_doc"] and question["gold_sentence"] in c["text"]}
