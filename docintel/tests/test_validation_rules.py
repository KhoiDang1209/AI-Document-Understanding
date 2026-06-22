"""Tests for the validation rule engine."""

from __future__ import annotations

from docintel.config import Settings
from docintel.schema import Document, LineItem
from docintel.validation.rules import validate


def _doc(**kw: object) -> Document:
    base: dict[str, object] = {"id": "x", "currency": "IDR", "created_at": "t"}
    base.update(kw)
    return Document(**base)  # type: ignore[arg-type]


def test_clean_receipt_has_no_errors() -> None:
    doc = _doc(
        line_items=[LineItem(name="a", price=600.0, confidence=0.9)],
        subtotal=600.0,
        tax=400.0,
        total=1000.0,
    )
    report = validate(doc, Settings())
    assert report.ok is True
    assert report.errors == []


def test_reconciliation_mismatch_is_an_error() -> None:
    doc = _doc(
        line_items=[LineItem(name="a", price=600.0, confidence=0.9)],
        subtotal=600.0,
        tax=400.0,
        total=9999.0,
    )
    report = validate(doc, Settings())
    assert report.ok is False
    assert any(e.rule == "reconciliation" for e in report.errors)


def test_missing_total_and_no_items_are_errors() -> None:
    report = validate(_doc(), Settings())
    assert report.ok is False
    rules = {e.rule for e in report.errors}
    assert "required_fields" in rules


def test_low_confidence_is_a_warning_not_an_error() -> None:
    doc = _doc(
        line_items=[LineItem(name="a", price=600.0, confidence=0.1)],
        subtotal=600.0,
        total=600.0,
    )
    report = validate(doc, Settings())
    assert report.ok is True
    assert any(w.rule == "low_confidence" for w in report.warnings)


def test_negative_and_unparsed_are_warnings() -> None:
    doc = _doc(
        line_items=[LineItem(name="a", price=600.0, confidence=0.9)],
        subtotal=600.0,
        total=-5.0,
        unparsed_fields=["tax"],
    )
    report = validate(doc, Settings())
    rules = {w.rule for w in report.warnings}
    assert "number_sanity" in rules
