"""RAGAS and Azure AI Evaluation scoring, with a deterministic stand-in.

The live path (``RAGFAIL_LIVE_JUDGES=1``) calls the RAGAS library and the Azure AI Evaluation SDK.
Both need an Azure OpenAI deployment and are not installed in CI. The default path scores the same
row shape (question, answer, contexts, reference) with the code scorer, using the metric names those
libraries publish:

* RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
* Azure AI Evaluation: groundedness, relevance, retrieval

Offline, every metric is on a 0-1 scale. The Azure evaluators return 1-5 when the SDK runs; the live
path records that mean under the same key and sets ``judge_scale`` to ``azure-1-to-5``.
"""

from __future__ import annotations

import os

from .chunking import gold_chunks
from .retrieval import tokenize
from .scoring import is_abstain

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
AZURE_METRICS = ("groundedness", "relevance", "retrieval")
JUDGE_METRICS = RAGAS_METRICS + AZURE_METRICS


def live_judges_enabled() -> bool:
    return os.environ.get("RAGFAIL_LIVE_JUDGES") == "1"


def ragas_row(question: str, answer: str, contexts: list[str], reference: str) -> dict:
    """One example in the RAGAS 0.2 field names."""
    return {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": reference,
    }


def azure_eval_row(question: str, answer: str, contexts: list[str], reference: str) -> dict:
    """One example in the column names the Azure evaluators map onto."""
    return {"question": question, "answer": answer, "context": "\n".join(contexts), "ground_truth": reference}


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return round(sum(xs) / len(xs), 4)


def mock_judge_rows(rows: list[dict]) -> dict:
    """Code proxy for one strategy. Each row needs question, answer, contexts (chunk dicts) and the
    ``score()`` result. Context precision and recall are averaged over answerable questions only."""
    buckets = {name: [] for name in JUDGE_METRICS}
    for row in rows:
        q, answer, chunks, scored = row["question"], row["answer"], row["chunks"], row["scored"]
        abst = is_abstain(answer)
        faithful = 1.0 if abst or scored["faithful"] else 0.0
        if abst and not q["answerable"]:
            relevancy = 1.0
        elif abst or not scored["correct"]:
            q_toks, a_toks = set(tokenize(q["question"])), set(tokenize(answer))
            relevancy = (len(q_toks & a_toks) / len(q_toks | a_toks)) if q_toks and a_toks else 0.0
        else:
            relevancy = 1.0
        grounded = 1.0 if abst or scored["grounded"] else 0.0
        buckets["faithfulness"].append(faithful)
        buckets["answer_relevancy"].append(relevancy)
        buckets["groundedness"].append(grounded)
        buckets["relevance"].append(relevancy)
        if not q["answerable"]:
            continue
        gold = gold_chunks(row["all_chunks"], q)
        got = [c["chunk_id"] for c in chunks]
        hit = 1.0 if any(cid in gold for cid in got) else 0.0
        prec = (sum(cid in gold for cid in got) / len(got)) if got else 0.0
        buckets["context_precision"].append(prec)
        buckets["context_recall"].append(hit)
        buckets["retrieval"].append(hit)
    return {name: _mean(vals) for name, vals in buckets.items()}


def load_live_backends():
    """Import RAGAS and Azure AI Evaluation, or raise with the install hint. No network."""
    missing = []
    try:
        import ragas  # noqa: F401
    except ImportError:
        missing.append("ragas")
    try:
        import azure.ai.evaluation  # noqa: F401
    except ImportError:
        missing.append("azure-ai-evaluation")
    if missing:
        raise RuntimeError(
            "RAGFAIL_LIVE_JUDGES=1 needs " + " and ".join(missing) + '. Install with: pip install -e ".[judges]"'
        )
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not endpoint or not deployment:
        raise RuntimeError("RAGFAIL_LIVE_JUDGES=1 needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT")
    return endpoint, deployment


def live_judge_rows(rows: list[dict], *, ragas_eval=None, azure_eval=None) -> dict:
    """Score with the two libraries. Tests inject ``ragas_eval`` and ``azure_eval``; otherwise both
    SDKs are imported and called. A failure is raised, never swapped for the mock."""
    dataset = [ragas_row(r["question"]["question"], r["answer"], r["contexts_text"], r["reference"]) for r in rows]
    azure_rows = [azure_eval_row(r["question"]["question"], r["answer"], r["contexts_text"], r["reference"]) for r in rows]
    if ragas_eval is None or azure_eval is None:
        load_live_backends()
        ragas_eval = ragas_eval or _ragas_evaluate
        azure_eval = azure_eval or _azure_evaluate
    ragas_scores = ragas_eval(dataset)
    azure_scores = azure_eval(azure_rows)
    out = {name: ragas_scores[name] for name in RAGAS_METRICS}
    out.update({name: azure_scores[name] for name in AZURE_METRICS})
    return out


def _ragas_evaluate(dataset: list[dict]) -> dict:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    result = evaluate(
        EvaluationDataset.from_list(dataset),
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
    )
    scores = result.scores if hasattr(result, "scores") else dict(result)
    # ragas aggregates to a list of per-row dicts or a summary dict depending on version.
    if isinstance(scores, list):
        return {name: _mean([float(row[name]) for row in scores if row.get(name) is not None]) for name in RAGAS_METRICS}
    return {name: round(float(scores[name]), 4) for name in RAGAS_METRICS}


def _azure_model_config() -> dict:
    return {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    }


def _azure_evaluate(rows: list[dict]) -> dict:
    import json
    import tempfile

    from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator, RetrievalEvaluator, evaluate

    cfg = _azure_model_config()
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        path = fh.name
    try:
        result = evaluate(
            data=path,
            evaluators={
                "groundedness": GroundednessEvaluator(cfg),
                "relevance": RelevanceEvaluator(cfg),
                "retrieval": RetrievalEvaluator(cfg),
            },
            evaluator_config={
                "groundedness": {"column_mapping": {"query": "${data.question}", "context": "${data.context}",
                                                    "response": "${data.answer}"}},
                "relevance": {"column_mapping": {"query": "${data.question}", "response": "${data.answer}"}},
                "retrieval": {"column_mapping": {"query": "${data.question}", "context": "${data.context}",
                                                 "ground_truth": "${data.ground_truth}"}},
            },
        )
    finally:
        os.remove(path)
    metrics = result.get("metrics", result)
    return {name: round(float(metrics[name]), 4) for name in AZURE_METRICS}


def judge_rows(rows: list[dict], *, ragas_eval=None, azure_eval=None) -> tuple[str, dict]:
    """Return ``(backend, metrics)``. Backend is ``mock`` unless the live switch is on."""
    if live_judges_enabled() or ragas_eval or azure_eval:
        return "ragas+azure-ai-evaluation", live_judge_rows(rows, ragas_eval=ragas_eval, azure_eval=azure_eval)
    return "mock", mock_judge_rows(rows)
