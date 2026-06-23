"""Tests for the Prometheus metrics module and /metrics endpoint."""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from docintel.api.metrics import build_metrics, record_extraction
from docintel.schema import Document, ValidationReport


def _doc(*, ok: bool, confidences: dict[str, float]) -> Document:
    return Document(
        id="d1",
        currency="IDR",
        field_confidence=confidences,
        validation=ValidationReport(ok=ok),
        created_at="2026-06-23T00:00:00+00:00",
    )


def test_record_extraction_observes_confidences_and_outcome() -> None:
    registry = CollectorRegistry()
    metrics = build_metrics(registry)

    record_extraction(metrics, _doc(ok=True, confidences={"total": 0.9, "subtotal": 0.8}))

    assert registry.get_sample_value("docintel_kie_field_confidence_count") == 2.0
    assert registry.get_sample_value("docintel_validation_total", {"outcome": "ok"}) == 1.0


def test_record_extraction_counts_failed_outcome() -> None:
    registry = CollectorRegistry()
    metrics = build_metrics(registry)

    record_extraction(metrics, _doc(ok=False, confidences={}))

    assert registry.get_sample_value("docintel_validation_total", {"outcome": "failed"}) == 1.0
    # No confidences observed -> histogram count is 0 (metric still present).
    assert registry.get_sample_value("docintel_kie_field_confidence_count") == 0.0
