"""Evaluate any model (PyTorch or ONNX) over encoded examples and score F1.

The model is injected as ``run_logits`` — a callable mapping one encoded
example to its per-token logits — so the same evaluation drives every config.
Metric computation is delegated to ``kie.metrics`` (no duplication).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from docintel.kie.metrics import compute_seqeval_metrics


def collect_predictions(
    run_logits: Callable[[Mapping[str, Any]], Any],
    encoded: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any]:
    """Run ``run_logits`` over each example; stack logits and label ids."""
    predictions: list[Any] = []
    label_ids: list[Any] = []
    for sample in encoded:
        logits = np.asarray(run_logits(sample), dtype=np.float32)
        predictions.append(logits)
        label_ids.append(np.asarray(sample["labels"]))
    return np.stack(predictions), np.stack(label_ids)


def evaluate_model(
    run_logits: Callable[[Mapping[str, Any]], Any],
    encoded: Sequence[Mapping[str, Any]],
    id2label: Mapping[int, str],
) -> dict[str, float]:
    """Compute seqeval F1 (+ per-field) for one model over encoded examples."""
    predictions, label_ids = collect_predictions(run_logits, encoded)
    return compute_seqeval_metrics(predictions, label_ids, id2label)
