"""seqeval entity-level metrics for CORD token classification.

Produces the overall precision/recall/F1/accuracy plus a per-field F1 for
every CORD category, ready to log to MLflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

_IGNORE_INDEX = -100


def align_predictions(
    predictions: Any,
    label_ids: Any,
    id2label: Mapping[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Argmax predictions, drop ``-100`` positions, map ids to BIO strings."""
    preds = np.asarray(predictions).argmax(axis=-1)
    labels = np.asarray(label_ids)

    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []
    for pred_row, label_row in zip(preds, labels, strict=True):
        true_seq: list[str] = []
        pred_seq: list[str] = []
        for pred_id, label_id in zip(pred_row, label_row, strict=True):
            if int(label_id) == _IGNORE_INDEX:
                continue
            true_seq.append(id2label[int(label_id)])
            pred_seq.append(id2label[int(pred_id)])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    return true_labels, pred_labels


def compute_seqeval_metrics(
    predictions: Any,
    label_ids: Any,
    id2label: Mapping[int, str],
) -> dict[str, float]:
    """Return overall + per-field entity-level metrics."""
    true_labels, pred_labels = align_predictions(predictions, label_ids, id2label)

    metrics: dict[str, float] = {
        "precision": float(precision_score(true_labels, pred_labels)),
        "recall": float(recall_score(true_labels, pred_labels)),
        "f1": float(f1_score(true_labels, pred_labels)),
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
    }

    report: dict[str, Any] = classification_report(
        true_labels, pred_labels, output_dict=True, zero_division=0
    )
    for field, scores in report.items():
        if field in {"micro avg", "macro avg", "weighted avg"}:
            continue
        if isinstance(scores, Mapping):
            metrics[f"f1_{field}"] = float(scores["f1-score"])
    return metrics
