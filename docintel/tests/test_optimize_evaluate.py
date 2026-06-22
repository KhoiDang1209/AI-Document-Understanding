"""Tests for model-agnostic F1 evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from docintel.optimize.evaluate import evaluate_model


def test_evaluate_model_perfect_predictions_scores_f1_one() -> None:
    # Two examples; labels use -100 to mark ignored subword positions.
    encoded = [
        {"labels": [0, 1, 1]},
        {"labels": [0, 1, -100]},
    ]
    id2label = {0: "O", 1: "B-menu.nm"}

    def run_logits(sample: Mapping[str, Any]) -> list[list[float]]:
        # Emit logits whose argmax equals the (clamped) label at each position.
        logits: list[list[float]] = []
        for label in sample["labels"]:
            target = 0 if label in (0, -100) else 1
            row = [0.0, 0.0]
            row[target] = 9.0
            logits.append(row)
        return logits

    metrics = evaluate_model(run_logits, encoded, id2label)
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
