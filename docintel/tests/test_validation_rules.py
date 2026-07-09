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


def test_unparsed_field_is_a_warning() -> None:
    doc = _doc(
        line_items=[LineItem(name="a", price=600.0, confidence=0.9)],
        subtotal=600.0,
        total=600.0,
        unparsed_fields=["tax"],
    )
    report = validate(doc, Settings())
    assert report.ok is True
    assert any(w.rule == "number_sanity" for w in report.warnings)


def test_partial_line_item_prices_do_not_trigger_subtotal_mismatch() -> None:
    # One line item's price was not extracted (None). Summing only the extracted
    # prices under-counts and would spuriously fail the subtotal check as a hard
    # error, so line-item reconciliation must be skipped unless every item is priced.
    doc = _doc(
        line_items=[
            LineItem(name="a", price=600.0, confidence=0.9),
            LineItem(name="b", price=None, confidence=0.9),
        ],
        subtotal=1000.0,
        total=1000.0,
    )
    report = validate(doc, Settings())
    assert report.ok is True
    assert not any(e.rule == "reconciliation" for e in report.errors)
