"""Tests for seqeval-based KIE metrics."""

from __future__ import annotations

import numpy as np

from docintel.kie.metrics import align_predictions, compute_seqeval_metrics

ID2LABEL = {0: "O", 1: "B-menu.nm", 2: "I-menu.nm", 3: "B-total.total_price"}


def test_align_predictions_drops_ignored_index() -> None:
    # logits over 4 classes for 3 tokens; second token is masked with -100.
    predictions = np.array([[[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 0, 9]]], dtype=float)
    label_ids = np.array([[0, -100, 3]])
    true_labels, pred_labels = align_predictions(predictions, label_ids, ID2LABEL)
    assert true_labels == [["O", "B-total.total_price"]]
    assert pred_labels == [["O", "B-total.total_price"]]


def test_compute_metrics_perfect_prediction() -> None:
    predictions = np.array([[[0, 9, 0, 0], [0, 0, 9, 0]]], dtype=float)  # B-menu.nm, I-menu.nm
    label_ids = np.array([[1, 2]])
    metrics = compute_seqeval_metrics(predictions, label_ids, ID2LABEL)
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_menu.nm"] == 1.0


def test_compute_metrics_includes_per_field_keys() -> None:
    predictions = np.array([[[0, 9, 0, 0], [0, 0, 9, 0]]], dtype=float)
    label_ids = np.array([[1, 2]])
    metrics = compute_seqeval_metrics(predictions, label_ids, ID2LABEL)
    assert "f1_menu.nm" in metrics
    assert all(isinstance(value, float) for value in metrics.values())
