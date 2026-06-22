"""Decode per-word BIO predictions into a curated receipt Document.

Walks words in reading order: a new line item begins at each ``B-menu.nm``;
menu sub-fields attach to the current item; ``sub_total.*``/``total.*`` map to
scalar fields. Money strings are parsed with ``schema.parse_money``.
"""

from __future__ import annotations

from collections.abc import Sequence

from docintel.schema import Document, LineItem, WordPrediction, detect_currency, parse_money

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
) -> tuple[list[dict[str, object]], dict[str, list[str]], dict[str, list[float]], list[str]]:
    """Group words into line-item rows and scalar text buckets (raw, unparsed)."""
    items: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    scalar_text: dict[str, list[str]] = {}
    scalar_conf: dict[str, list[float]] = {}
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
            field = _LINE_FIELDS[category]
            words = current["_words"]
            words.setdefault(field, []).append(pred.text)  # type: ignore[attr-defined]
            current["_conf"].append(pred.confidence)  # type: ignore[attr-defined]
        elif category in _SCALAR_FIELDS:
            field = _SCALAR_FIELDS[category]
            scalar_text.setdefault(field, []).append(pred.text)
            scalar_conf.setdefault(field, []).append(pred.confidence)
    return items, scalar_text, scalar_conf, all_texts


def build_document(
    predictions: Sequence[WordPrediction],
    default_currency: str,
) -> Document:
    """Assemble a Document from word predictions (id/validation set by caller)."""
    items, scalar_text, scalar_conf, all_texts = _collect_spans(predictions)

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
    for field, texts in scalar_text.items():
        joined = " ".join(texts)
        value = parse_money(joined)
        scalars[field] = value
        scalar_confs = scalar_conf[field]
        field_confidence[field] = sum(scalar_confs) / len(scalar_confs)
        if value is None and joined.strip():
            unparsed_fields.append(field)

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
