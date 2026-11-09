"""Azure ML / Microsoft Foundry scoring script for the guarded retrieval service. WRITTEN, NOT RUN in Azure.

init() builds the BM25 index from the committed corpus (no pickle). If SEARCH_ENDPOINT is set, run()
queries Azure AI Search instead, with the endpoint's managed identity (Search Index Data Reader). run() accepts {"question": "...", "k": 3}
and returns guarded passages: current documents first, hidden instructions stripped. Answer generation is
left to the caller's model deployment, which should get only these passages.
"""

from __future__ import annotations

import json
import logging
import os

from ragfail import azure_search, data, service

log = logging.getLogger("ragfail.score")


def init() -> None:
    data.retriever("w80")
    log.info("retriever ready")


def run(raw_data: str) -> dict:
    try:
        payload = json.loads(raw_data)
    except (TypeError, json.JSONDecodeError):
        return {"error": "body must be JSON like {\"question\": \"...\"}"}
    q = payload.get("question") if isinstance(payload, dict) else None
    k = payload.get("k", service.DEFAULT_K) if isinstance(payload, dict) else service.DEFAULT_K
    if not isinstance(q, str) or not q.strip():
        return {"error": "field 'question' (non-empty string) is required"}
    if not isinstance(k, int) or isinstance(k, bool):
        return {"error": "field 'k' must be an integer"}
    try:
        endpoint = os.environ.get("SEARCH_ENDPOINT")
        if endpoint:  # pragma: no cover - needs Azure
            out = service.package(azure_search.search(endpoint, service.check_question(q), k), "azure-ai-search")
        else:
            out = service.retrieve_local(q, k)
    except ValueError as e:
        return {"error": str(e)}
    log.info("returned %d passages, stripped %d instruction sentences", len(out["passages"]), out["instructions_stripped"])
    return out
