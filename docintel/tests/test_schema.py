"""Tests for the Document schema and field normalizers."""

from __future__ import annotations

import pytest

from docintel.schema import Document, ValidationReport, detect_currency, parse_money


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.000", 10000.0),
        ("10,000", 10000.0),
        ("Rp 25.000", 25000.0),
        ("1.234.567", 1234567.0),
        ("12,50", 12.5),
        ("", None),
        ("n/a", None),
    ],
)
def test_parse_money(raw: str, expected: float | None) -> None:
    assert parse_money(raw) == expected


def test_detect_currency_from_symbols() -> None:
    assert detect_currency(["Rp", "10.000"], default="USD") == "IDR"
    assert detect_currency(["$5.00"], default="IDR") == "USD"
    assert detect_currency(["plain", "text"], default="IDR") == "IDR"


def test_document_defaults_are_safe() -> None:
    doc = Document(id="abc", currency="IDR", created_at="2026-06-22T00:00:00+00:00")
    assert doc.line_items == []
    assert doc.total is None
    assert isinstance(doc.validation, ValidationReport)
    assert doc.validation.ok is True
