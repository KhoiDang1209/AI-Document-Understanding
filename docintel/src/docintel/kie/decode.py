"""Decode per-word BIO predictions into a curated receipt Document.

Walks words in reading order: a new line item begins at each ``B-menu.nm``;
menu sub-fields attach to the current item; ``sub_total.*``/``total.*`` map to
scalar fields. Money strings are parsed with ``schema.parse_money``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from docintel.schema import Document, LineItem, WordPrediction, detect_currency, parse_money


@dataclass
class _ScalarSpan:
    """One contiguous BIO span of words tagged for a single scalar field."""

    texts: list[str] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)

    @property
    def mean_conf(self) -> float:
        return sum(self.confs) / len(self.confs) if self.confs else 0.0


_LINE_FIELDS = {
    "menu.nm": "name",
    "menu.cnt": "qty",
    "menu.unitprice": "unit_price",
    "menu.price": "price",
}
_SCALAR_FIELDS = {
    "sub_total.subtotal_price": "subtotal",
    "sub_total.tax_price": "tax",
    "sub_total.service_price": "service",
    "total.total_price": "total",
}


def _category(label: str) -> str | None:
    """Strip the ``B-``/``I-`` prefix; return None for the outside label ``O``."""
    if label == "O" or "-" not in label:
        return None
    return label.split("-", 1)[1]


def _collect_spans(
    predictions: Sequence[WordPrediction],
) -> tuple[list[dict[str, object]], dict[str, list[_ScalarSpan]], list[str]]:
    """Group words into line-item rows and per-scalar-field BIO spans.

    Each ``B-`` starts a new scalar span; ``I-`` continuations extend the field's
    most recent span. Keeping spans separate (instead of concatenating all tokens)
    prevents two distinct numbers mislabelled as the same field from being glued.
    """
    items: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    scalar_spans: dict[str, list[_ScalarSpan]] = {}
    all_texts: list[str] = []

    for pred in predictions:
        all_texts.append(pred.text)
        category = _category(pred.label)
        if category is None:
            continue
        if category == "menu.nm" and pred.label.startswith("B-"):
            current = {"_words": {}, "_conf": []}
            items.append(current)
        if category in _LINE_FIELDS:
            if current is None:
                current = {"_words": {}, "_conf": []}
                items.append(current)
            field_name = _LINE_FIELDS[category]
            words = current["_words"]
            words.setdefault(field_name, []).append(pred.text)  # type: ignore[attr-defined]
            current["_conf"].append(pred.confidence)  # type: ignore[attr-defined]
        elif category in _SCALAR_FIELDS:
            field_name = _SCALAR_FIELDS[category]
            spans = scalar_spans.setdefault(field_name, [])
            if pred.label.startswith("B-") or not spans:
                spans.append(_ScalarSpan())
            spans[-1].texts.append(pred.text)
            spans[-1].confs.append(pred.confidence)
    return items, scalar_spans, all_texts


def _select_scalar_span(spans: list[_ScalarSpan]) -> _ScalarSpan | None:
    """Choose one span for a scalar field: highest-confidence parseable span.

    Falls back to the highest-confidence span overall when none parse as money,
    so the unparsed-field warning and its confidence are still recorded.
    """
    if not spans:
        return None
    parseable = [s for s in spans if parse_money(" ".join(s.texts)) is not None]
    candidates = parseable or spans
    return max(candidates, key=lambda s: s.mean_conf)


def build_document(
    predictions: Sequence[WordPrediction],
    default_currency: str,
) -> Document:
    """Assemble a Document from word predictions (id/validation set by caller)."""
    items, scalar_spans, all_texts = _collect_spans(predictions)

    line_items: list[LineItem] = []
    for raw in items:
        words: dict[str, list[str]] = raw["_words"]  # type: ignore[assignment]
        confs: list[float] = raw["_conf"]  # type: ignore[assignment]
        name = " ".join(words["name"]) if "name" in words else None
        line_items.append(
            LineItem(
                name=name,
                qty=parse_money(" ".join(words["qty"])) if "qty" in words else None,
                unit_price=(
                    parse_money(" ".join(words["unit_price"])) if "unit_price" in words else None
                ),
                price=parse_money(" ".join(words["price"])) if "price" in words else None,
                confidence=sum(confs) / len(confs) if confs else 0.0,
            )
        )

    scalars: dict[str, float | None] = {}
    field_confidence: dict[str, float] = {}
    unparsed_fields: list[str] = []
    for field_name, spans in scalar_spans.items():
        chosen = _select_scalar_span(spans)
        if chosen is None:
            continue
        joined = " ".join(chosen.texts)
        value = parse_money(joined)
        scalars[field_name] = value
        field_confidence[field_name] = chosen.mean_conf
        if value is None and joined.strip():
            unparsed_fields.append(field_name)

    return Document(
        id="",
        line_items=line_items,
        subtotal=scalars.get("subtotal"),
        tax=scalars.get("tax"),
        service=scalars.get("service"),
        total=scalars.get("total"),
        currency=detect_currency(all_texts, default_currency),
        field_confidence=field_confidence,
        unparsed_fields=unparsed_fields,
        created_at="",
    )
