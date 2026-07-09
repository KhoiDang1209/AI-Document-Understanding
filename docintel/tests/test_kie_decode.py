"""Tests for decoding word predictions into a Document."""

from __future__ import annotations

import pytest

from docintel.kie.decode import build_document
from docintel.schema import WordPrediction


def _w(text: str, label: str, conf: float = 0.9) -> WordPrediction:
    return WordPrediction(text=text, box=(0, 0, 1, 1), label=label, confidence=conf)


def test_groups_two_line_items_by_b_menu_nm() -> None:
    preds = [
        _w("Coke", "B-menu.nm"),
        _w("2", "B-menu.cnt"),
        _w("3.000", "B-menu.price"),
        _w("Rice", "B-menu.nm"),
        _w("1", "B-menu.cnt"),
        _w("5.000", "B-menu.price"),
        _w("8.000", "B-sub_total.subtotal_price"),
        _w("8.000", "B-total.total_price"),
    ]
    doc = build_document(preds, default_currency="IDR")
    assert [i.name for i in doc.line_items] == ["Coke", "Rice"]
    assert doc.line_items[0].qty == 2.0
    assert doc.line_items[0].price == 3000.0
    assert doc.subtotal == 8000.0
    assert doc.total == 8000.0
    assert doc.currency == "IDR"
    assert doc.line_items[0].confidence == pytest.approx(0.9)
    assert doc.field_confidence["subtotal"] == pytest.approx(0.9)


def test_multiword_name_joins_and_outside_is_ignored() -> None:
    preds = [
        _w("Fried", "B-menu.nm"),
        _w("Rice", "I-menu.nm"),
        _w("--", "O"),
        _w("5.000", "B-menu.price"),
    ]
    doc = build_document(preds, default_currency="IDR")
    assert doc.line_items[0].name == "Fried Rice"
    assert doc.line_items[0].price == 5000.0


def test_two_scalar_spans_pick_highest_confidence_not_concatenated() -> None:
    # The model tagged two separate numbers as subtotal; the real one (503.000)
    # has higher confidence than the spurious one (52.815). They must not be
    # glued into 50300052815 — the highest-confidence span wins.
    preds = [
        _w("Coke", "B-menu.nm"),
        _w("503.000", "B-sub_total.subtotal_price", conf=0.95),
        _w("52.815", "B-sub_total.subtotal_price", conf=0.40),
    ]
    doc = build_document(preds, default_currency="IDR")
    assert doc.subtotal == 503000.0
    assert doc.field_confidence["subtotal"] == pytest.approx(0.95)


def test_scalar_span_joins_continuation_tokens() -> None:
    # A single value split across B-/I- (e.g. OCR breaking 1.591.600 apart) is
    # one span and must still be joined into a single number.
    preds = [
        _w("Coke", "B-menu.nm"),
        _w("1.591", "B-total.total_price"),
        _w("600", "I-total.total_price"),
    ]
    doc = build_document(preds, default_currency="IDR")
    assert doc.total == 1591600.0


def test_orphan_menu_field_before_first_name_is_not_a_phantom_item() -> None:
    # A menu sub-field (here a price) tagged before any B-menu.nm has no line
    # item to attach to. It must be dropped, not opened as a nameless row — that
    # phantom "first line item" is what the Phase 4 report otherwise reads as a
    # model-quality failure.
    preds = [
        _w("5.000", "B-menu.price"),
        _w("Coke", "B-menu.nm"),
        _w("3.000", "B-menu.price"),
    ]
    doc = build_document(preds, default_currency="IDR")
    assert [i.name for i in doc.line_items] == ["Coke"]
    assert doc.line_items[0].price == 3000.0


def test_unparseable_total_recorded() -> None:
    preds = [
        _w("Coke", "B-menu.nm"),
        _w("3.000", "B-menu.price"),
        _w("oops", "B-total.total_price"),
    ]
    doc = build_document(preds, default_currency="IDR")
    assert doc.total is None
    assert "total" in doc.unparsed_fields
