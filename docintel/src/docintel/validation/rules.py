"""Validation rules over a Document: hard errors and soft warnings.

Validation annotates the response; it never blocks. ``ok`` is false only when
at least one hard error is present.
"""

from __future__ import annotations

from docintel.config import Settings
from docintel.schema import Document, ValidationIssue, ValidationReport


def _reconciliation(document: Document, tolerance: float) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if document.total is not None and document.subtotal is not None:
        expected = document.subtotal + (document.tax or 0.0) + (document.service or 0.0)
        if abs(expected - document.total) > tolerance:
            issues.append(
                ValidationIssue(
                    rule="reconciliation",
                    severity="error",
                    message=f"subtotal+tax+service ({expected}) != total ({document.total})",
                    field="total",
                )
            )
    prices = [item.price for item in document.line_items if item.price is not None]
    if document.subtotal is not None and prices:
        items_sum = sum(prices)
        if abs(items_sum - document.subtotal) > tolerance:
            issues.append(
                ValidationIssue(
                    rule="reconciliation",
                    severity="error",
                    message=f"sum(line items) ({items_sum}) != subtotal ({document.subtotal})",
                    field="subtotal",
                )
            )
    return issues


def _required_fields(document: Document) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if document.total is None:
        issues.append(
            ValidationIssue(
                rule="required_fields", severity="error", message="total is missing", field="total"
            )
        )
    if not document.line_items:
        issues.append(
            ValidationIssue(
                rule="required_fields",
                severity="error",
                message="no line items extracted",
                field="line_items",
            )
        )
    return issues


def _low_confidence(document: Document, threshold: float) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field, score in document.field_confidence.items():
        if score < threshold:
            issues.append(
                ValidationIssue(
                    rule="low_confidence",
                    severity="warning",
                    message=f"{field} confidence {score:.2f} below {threshold}",
                    field=field,
                )
            )
    for index, item in enumerate(document.line_items):
        if item.confidence < threshold:
            issues.append(
                ValidationIssue(
                    rule="low_confidence",
                    severity="warning",
                    message=f"line item {index} confidence {item.confidence:.2f} below {threshold}",
                    field=f"line_items[{index}]",
                )
            )
    return issues


def _number_sanity(document: Document) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    money = {
        "subtotal": document.subtotal,
        "tax": document.tax,
        "service": document.service,
        "total": document.total,
    }
    for field, value in money.items():
        if value is not None and value < 0:
            issues.append(
                ValidationIssue(
                    rule="number_sanity",
                    severity="warning",
                    message=f"{field} is negative ({value})",
                    field=field,
                )
            )
    for field in document.unparsed_fields:
        issues.append(
            ValidationIssue(
                rule="number_sanity",
                severity="warning",
                message=f"{field} had text that could not be parsed as money",
                field=field,
            )
        )
    return issues


def validate(document: Document, settings: Settings) -> ValidationReport:
    """Run all rules; collect errors/warnings; set ``ok`` from error count."""
    errors: list[ValidationIssue] = []
    errors.extend(_reconciliation(document, settings.validation_tolerance))
    errors.extend(_required_fields(document))
    warnings: list[ValidationIssue] = []
    warnings.extend(_low_confidence(document, settings.confidence_threshold))
    warnings.extend(_number_sanity(document))
    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)
