"""Pure, testable helpers for the demo's Metrics view.

Discovers and summarises the committed/local eval JSON (``eval_rag*.json`` from
``scripts.eval_rag`` and ``eval_ragas*.json`` from ``scripts.eval_ragas``) so the
Streamlit page keeps only the rendering layer. No Streamlit import here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_RETRIEVAL = "retrieval"
_RAGAS = "ragas"
_UNKNOWN = "unknown"


def classify_eval(payload: dict[str, Any]) -> str:
    """Tell a retrieval-recall run from a RAGAS run by its distinguishing key."""
    if "recall_at_k" in payload:
        return _RETRIEVAL
    if "faithfulness" in payload:
        return _RAGAS
    return _UNKNOWN


def discover_eval_files(directory: str | Path) -> list[Path]:
    """Return eval JSON files in ``directory`` (``eval_rag*``/``eval_ragas*``), sorted by name."""
    base = Path(directory)
    if not base.is_dir():
        return []
    files = set(base.glob("eval_rag*.json")) | set(base.glob("eval_ragas*.json"))
    return sorted(files, key=lambda p: p.name)


def load_eval_file(path: str | Path) -> dict[str, Any]:
    """Load one eval JSON, tagging it with its source name and detected kind."""
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"_name": Path(path).name, "_kind": classify_eval(payload), **payload}


def embedder_label(config: dict[str, Any]) -> str:
    """Human label for the embedder a run used, from its recorded config.

    A local bundle path means the fine-tuned embedder; otherwise the model name;
    ``unknown`` for runs written before provenance was recorded.
    """
    if config.get("embedding_local_path"):
        return f"fine-tuned ({Path(str(config['embedding_local_path'])).name})"
    if config.get("embedding_model"):
        return str(config["embedding_model"])
    return "unknown"


def retrieval_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per retrieval run: recall@1/3/5, MRR, embedder, ablation flags."""
    rows: list[dict[str, Any]] = []
    for p in payloads:
        if p.get("_kind") != _RETRIEVAL:
            continue
        recall = p.get("recall_at_k", {})
        config = p.get("config", {})
        rows.append(
            {
                "Run": p.get("_name", "—"),
                "Embedder": embedder_label(config),
                "Rerank": config.get("rerank"),
                "Focused query": config.get("focused_query"),
                "recall@1": recall.get("1"),
                "recall@3": recall.get("3"),
                "recall@5": recall.get("5"),
                "MRR": p.get("mrr"),
                "Queries": p.get("evaluated_queries"),
            }
        )
    return rows


def ragas_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per RAGAS run: faithfulness / answer-relevancy / refusal rate."""
    rows: list[dict[str, Any]] = []
    for p in payloads:
        if p.get("_kind") != _RAGAS:
            continue
        answered = p.get("answered_only", {})
        rows.append(
            {
                "Run": p.get("_name", "—"),
                "Scored": p.get("scored_answers"),
                "Refusal rate": p.get("refusal_rate"),
                "Faithfulness": p.get("faithfulness"),
                "Answer relevancy": p.get("answer_relevancy"),
                "Faithfulness (answered)": answered.get("faithfulness"),
                "Relevancy (answered)": answered.get("answer_relevancy"),
            }
        )
    return rows
