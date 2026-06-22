"""Canonical Document schema and field normalizers for served KIE output.

The curated receipt shape distills the ~30 CORD sub-labels into a clean
domain model. CORD has no merchant/store-name label, so none appears here.
Money strings are receipt-style integers with '.'/',' thousands separators.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

_CURRENCY_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("rp", "IDR"),
    ("idr", "IDR"),
    ("₩", "KRW"),
    ("원", "KRW"),
    ("krw", "KRW"),
    ("$", "USD"),
    ("usd", "USD"),
    ("€", "EUR"),
    ("eur", "EUR"),
)


class WordPrediction(BaseModel):
    """One OCR word with the BIO label the KIE model assigned to it."""

    text: str
    box: tuple[int, int, int, int] = Field(
        description="Pixel box [x_min, y_min, x_max, y_max], top-left origin."
    )
    label: str
    confidence: float


class LineItem(BaseModel):
    """One purchased item: name, quantity, unit price, line price."""

    name: str | None = None
    qty: float | None = None
    unit_price: float | None = None
    price: float | None = None
    confidence: float = 0.0


class ValidationIssue(BaseModel):
    """A single validation finding against a Document."""

    rule: str
    severity: Literal["error", "warning"]
    message: str
    field: str | None = None


class ValidationReport(BaseModel):
    """Aggregated validation outcome; ``ok`` is false when any error is present."""

    ok: bool = True
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)


class Document(BaseModel):
    """Structured, validated extraction result for one receipt image."""

    id: str
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    service: float | None = None
    total: float | None = None
    currency: str
    field_confidence: dict[str, float] = Field(default_factory=dict)
    unparsed_fields: list[str] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    created_at: str


def parse_money(raw: str) -> float | None:
    """Parse a receipt money string to a float, or ``None`` if it has no digits.

    Strips currency symbols/letters; treats a trailing ``.``/``,`` followed by
    exactly two digits as a decimal separator, otherwise all separators are
    thousands separators (the common CORD case, e.g. ``"10.000"`` -> 10000.0).
    """
    cleaned = re.sub(r"[^\d.,]", "", raw)
    if not re.search(r"\d", cleaned):
        return None
    decimal_match = re.search(r"[.,](\d{2})$", cleaned)
    if decimal_match and len(re.sub(r"[.,]", "", cleaned)) > 2:
        integer = re.sub(r"[.,]", "", cleaned[: decimal_match.start()])
        return float(f"{integer}.{decimal_match.group(1)}")
    return float(re.sub(r"[.,]", "", cleaned))


def detect_currency(texts: Iterable[str], default: str) -> str:
    """Infer a currency code from any currency symbol found in ``texts``."""
    for text in texts:
        lowered = text.lower()
        for symbol, code in _CURRENCY_SYMBOLS:
            if symbol in lowered:
                return code
    return default
