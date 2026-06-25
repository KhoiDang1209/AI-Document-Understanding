"""Prometheus metrics for the DocIntel API.

Custom domain metrics are bound to a caller-supplied ``CollectorRegistry`` so each
FastAPI app instance owns its own metrics. This keeps the test suite — which builds
the app many times — free of duplicate-registration errors on the global registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram

from docintel.contracts.schema import ContractDocument
from docintel.schema import Document

_CONFIDENCE_BUCKETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class Metrics:
    """Custom DocIntel metrics bound to a single registry."""

    kie_field_confidence: Histogram
    validation_total: Counter
    contract_clause_confidence: Histogram
    contract_clause_total: Counter


def build_metrics(registry: CollectorRegistry) -> Metrics:
    """Create the custom metrics against ``registry`` (one set per app instance)."""
    return Metrics(
        kie_field_confidence=Histogram(
            "docintel_kie_field_confidence",
            "Per-field KIE confidence observed on /extract.",
            buckets=_CONFIDENCE_BUCKETS,
            registry=registry,
        ),
        validation_total=Counter(
            "docintel_validation",
            "Documents processed by /extract, labelled by validation outcome.",
            labelnames=("outcome",),
            registry=registry,
        ),
        contract_clause_confidence=Histogram(
            "docintel_contract_clause_confidence",
            "Per-clause extraction confidence observed on /contracts/extract.",
            buckets=_CONFIDENCE_BUCKETS,
            registry=registry,
        ),
        contract_clause_total=Counter(
            "docintel_contract_clauses",
            "Clauses extracted on /contracts/extract, labelled by ingestion source.",
            labelnames=("source",),
            registry=registry,
        ),
    )


def record_extraction(metrics: Metrics, document: Document) -> None:
    """Record one extracted document: field confidences + validation outcome."""
    for value in document.field_confidence.values():
        metrics.kie_field_confidence.observe(value)
    outcome = "ok" if document.validation.ok else "failed"
    metrics.validation_total.labels(outcome=outcome).inc()


def record_contract_extraction(metrics: Metrics, doc: ContractDocument) -> None:
    """Record one extracted contract: clause confidences + clause count by source."""
    for clause in doc.clauses:
        metrics.contract_clause_confidence.observe(clause.confidence)
    metrics.contract_clause_total.labels(source=doc.source).inc(len(doc.clauses))
