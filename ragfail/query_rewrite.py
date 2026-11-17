"""Deterministic query rewrites for the advanced RAG bench.

HyDE, multi-query, step-back and decomposition are usually written by a model. Offline, the same
four shapes are produced from the question text and the corpus vocabulary, so a run replays in CI.
Set ``RAGFAIL_LIVE_LLM=1`` and the Azure OpenAI variables to replace ``rewrite_live`` with a model.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from .corpus import DISTRICTS
from .retrieval import tokenize

# Content words that identify a policy. Ambiguous words (fee, hours, days) are left out on purpose:
# they occur in more than one policy, so they would link the question to the wrong entity.
TOPIC_WORDS: dict[str, frozenset[str]] = {
    "leak": frozenset({"leak", "leaks", "burst", "crew", "crews", "technician", "technicians", "priority"}),
    "billing": frozenset({"late", "overdue", "arrears", "instalment", "installment", "grace", "payment"}),
    "connect": frozenset({"connection", "connections", "hook", "quote", "builder"}),
    "quality": frozenset({"chlorine", "disinfectant", "samples", "sample", "sampling"}),
    "outage": frozenset({"outage", "interruption", "interruptions", "compensation", "credit"}),
    "meter": frozenset({"meter", "meters", "estimated", "guessed"}),
    "complaints": frozenset({"complaint", "complaints"}),
    "privacy": frozenset({"hardship", "callback", "retention", "records"}),
}

_AND = re.compile(r"\band\b", re.I)


def detect_district(question: str) -> str | None:
    toks = set(tokenize(question))
    hits = [d for d in DISTRICTS if d.lower() in toks]
    return hits[0] if hits else None


def detect_topic(question: str) -> str | None:
    toks = set(tokenize(question))
    scores = {topic: len(toks & words) for topic, words in TOPIC_WORDS.items()}
    best = max(scores, key=lambda t: (scores[t], t))
    return best if scores[best] else None


def hyde_document(question: str) -> str:
    """A hypothetical policy sentence. Dense retrieval embeds this instead of the short question."""
    words = tokenize(question)
    body = " ".join(words) if words else "policy"
    return f"The policy states that {body}."


def multi_queries(question: str) -> list[str]:
    """Original question, a keyword query, and a short 'required' paraphrase."""
    words = tokenize(question)
    keyword = " ".join(words) if words else question
    required = "required " + " ".join(words[:8]) if words else question
    return list(dict.fromkeys([question.strip(), keyword, required]))


def step_back_query(question: str) -> str:
    """A higher-level question that keeps the district. Dropping the district on this corpus
    retrieves the same policy for the wrong place."""
    district, topic = detect_district(question), detect_topic(question)
    if topic and district:
        return f"What does the {topic} policy require in {district}?"
    if topic:
        return f"What does the {topic} policy require?"
    return "What does the policy require?"


def decompose_query(question: str) -> list[str]:
    """Split a coordinated question, and always add an entity query (topic + district) when both are known."""
    parts = [question.strip()]
    if _AND.search(question):
        for piece in _AND.split(question):
            piece = piece.strip(" ?.")
            if len(tokenize(piece)) >= 2:
                parts.append(piece)
    district, topic = detect_district(question), detect_topic(question)
    if topic and district:
        parts.append(f"{topic} {district}")
    elif topic:
        parts.append(topic)
    return list(dict.fromkeys(p for p in parts if p))


def live_llm_enabled() -> bool:
    return os.environ.get("RAGFAIL_LIVE_LLM") == "1"


def offline_rewrites(question: str, kind: str) -> list[str]:
    """The deterministic queries the bench fuses. Live mode replaces the generated lines only."""
    if kind == "hyde":
        return [hyde_document(question)]
    if kind == "multi_query":
        return multi_queries(question)
    if kind == "step_back":
        return [question, step_back_query(question)]
    if kind == "decomposition":
        return decompose_query(question)
    raise ValueError(f"unknown rewrite: {kind}")


def live_rewrites(question: str, kind: str) -> list[str]:
    if not live_llm_enabled():
        return offline_rewrites(question, kind)
    text = chat_completion(rewrite_request(question, kind))
    lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
    if kind == "hyde":
        return [text.strip() or hyde_document(question)]
    if kind == "step_back":
        return [question, (lines[0] if lines else step_back_query(question))]
    return list(dict.fromkeys([question, *lines])) or offline_rewrites(question, kind)


def chat_completion(body: dict) -> str:
    """One Azure OpenAI chat call. Raises if the endpoint or a credential is missing."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    if not endpoint or not deployment:
        raise RuntimeError("RAGFAIL_LIVE_LLM=1 needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT")
    version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=_openai_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - the caller's Azure endpoint
        payload = json.loads(resp.read().decode())
    try:
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Azure OpenAI response has no message content") from None


def _openai_headers() -> dict[str, str]:
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if key:
        return {"api-key": key, "Content-Type": "application/json"}
    token = os.environ.get("AZURE_OPENAI_TOKEN")
    if not token:
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise RuntimeError("RAGFAIL_LIVE_LLM=1 needs AZURE_OPENAI_API_KEY or azure-identity") from None
        token = DefaultAzureCredential().get_token("https://cognitiveservices.azure.com/.default").token
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def rewrite_request(question: str, kind: str) -> dict:
    """Chat-completions body for the live rewrite. ``kind`` is hyde, multi_query, step_back or decomposition."""
    if kind not in ("hyde", "multi_query", "step_back", "decomposition"):
        raise ValueError(f"unknown rewrite: {kind}")
    instructions = {
        "hyde": "Write one hypothetical policy sentence that would answer the question. Do not name a number.",
        "multi_query": "Write three search queries, one per line, that would find the policy sentence.",
        "step_back": "Write one broader question that still names the place, if the question names one.",
        "decomposition": "Write one search query per line, splitting the question into its parts.",
    }
    return {
        "messages": [
            {"role": "system", "content": instructions[kind]},
            {"role": "user", "content": question},
        ]
    }
