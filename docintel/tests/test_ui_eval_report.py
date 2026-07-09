"""Unit tests for the demo Metrics view's pure eval-report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from docintel.ui.eval_report import (
    classify_eval,
    discover_eval_files,
    embedder_label,
    load_eval_file,
    ragas_rows,
    retrieval_rows,
)

_RETRIEVAL = {
    "evaluated_queries": 1253,
    "recall_at_k": {"1": 0.21, "3": 0.41, "5": 0.51},
    "mrr": 0.39,
    "recall_at_max_k_by_category": {"Governing Law": 1.0},
    "config": {"rerank": True, "focused_query": True, "embedding_local_path": "/m/bge-small-cuad"},
}
_RAGAS = {
    "scored_answers": 40,
    "refusal_rate": 0.375,
    "faithfulness": 0.66,
    "answer_relevancy": 0.49,
    "answered_only": {"faithfulness": 0.66, "answer_relevancy": 0.71},
}


def test_classify_eval_distinguishes_kinds() -> None:
    assert classify_eval(_RETRIEVAL) == "retrieval"
    assert classify_eval(_RAGAS) == "ragas"
    assert classify_eval({"something": 1}) == "unknown"


def test_embedder_label_prefers_local_bundle_then_model_then_unknown() -> None:
    assert (
        embedder_label({"embedding_local_path": "/m/bge-small-cuad"})
        == "fine-tuned (bge-small-cuad)"
    )
    assert embedder_label({"embedding_model": "BAAI/bge-small-en-v1.5"}) == "BAAI/bge-small-en-v1.5"
    assert embedder_label({}) == "unknown"


def test_discover_and_load_eval_files(tmp_path: Path) -> None:
    (tmp_path / "eval_rag_finetuned.json").write_text(json.dumps(_RETRIEVAL), encoding="utf-8")
    (tmp_path / "eval_ragas_new.json").write_text(json.dumps(_RAGAS), encoding="utf-8")
    (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")

    files = discover_eval_files(tmp_path)
    assert [p.name for p in files] == ["eval_rag_finetuned.json", "eval_ragas_new.json"]

    loaded = load_eval_file(files[0])
    assert loaded["_name"] == "eval_rag_finetuned.json"
    assert loaded["_kind"] == "retrieval"


def test_discover_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert discover_eval_files(tmp_path / "nope") == []


def test_retrieval_rows_projects_recall_and_embedder() -> None:
    payloads = [{"_name": "r.json", "_kind": "retrieval", **_RETRIEVAL}]
    rows = retrieval_rows(payloads)
    assert rows[0]["recall@5"] == 0.51
    assert rows[0]["MRR"] == 0.39
    assert rows[0]["Embedder"] == "fine-tuned (bge-small-cuad)"


def test_ragas_rows_projects_scores_and_ignores_retrieval() -> None:
    payloads = [
        {"_name": "g.json", "_kind": "ragas", **_RAGAS},
        {"_name": "r.json", "_kind": "retrieval", **_RETRIEVAL},
    ]
    rows = ragas_rows(payloads)
    assert len(rows) == 1
    assert rows[0]["Faithfulness"] == 0.66
    assert rows[0]["Relevancy (answered)"] == 0.71
