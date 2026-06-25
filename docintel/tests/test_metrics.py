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


from fastapi.testclient import TestClient  # noqa: E402

from docintel.api.main import create_app  # noqa: E402


def test_metrics_endpoint_exposes_custom_and_http_metrics() -> None:
    with TestClient(create_app()) as client:
        client.get("/health")  # generate one HTTP sample
        resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    # Custom metric is registered at app build time, so it is always present.
    assert "docintel_kie_field_confidence" in body
    # Instrumentator default HTTP metric.
    assert "http_request_duration_seconds" in body


from typing import Any  # noqa: E402

from tests.test_extract import _make_stubbed_app, _png_bytes  # noqa: E402


def test_extract_records_validation_metric(tmp_path: Any) -> None:
    from docintel.config import Settings

    app = _make_stubbed_app(Settings(sqlite_path=str(tmp_path / "db.sqlite")))
    with TestClient(app) as client:
        assert (
            client.post(
                "/extract", files={"file": ("r.png", _png_bytes(), "image/png")}
            ).status_code
            == 200
        )
        body = client.get("/metrics").text
    app.dependency_overrides.clear()
    assert "docintel_validation_total{outcome=" in body


def test_record_contract_extraction_counts_clauses() -> None:
    from docintel.api.metrics import build_metrics, record_contract_extraction
    from docintel.contracts.schema import ContractDocument, ExtractedClause

    registry = CollectorRegistry()
    metrics = build_metrics(registry)
    doc = ContractDocument(
        id="c1",
        source="ocr",
        clauses=[
            ExtractedClause(
                clause_type="Parties",
                answer_text="Acme",
                char_start=0,
                char_end=4,
                confidence=0.8,
            ),
        ],
        derived={"Parties": ["Acme"]},
        page_count=2,
        created_at="2026-06-25T00:00:00+00:00",
    )
    record_contract_extraction(metrics, doc)
    assert registry.get_sample_value("docintel_contract_clauses_total", {"source": "ocr"}) == 1.0
