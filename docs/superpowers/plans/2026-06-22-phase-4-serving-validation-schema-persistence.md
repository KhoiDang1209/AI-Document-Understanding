# Phase 4 — Serving + Validation + Schema + Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the registered ONNX-INT8 LayoutLMv3 model into `/extract` so it returns a schema-validated, structured `Document` with explicit validation flags, persist every result (image → MinIO, metadata → SQLite), and serve `GET /documents/{id}`.

**Architecture:** Six new seams, each one file with one responsibility: a pure Pydantic `schema` (Document/LineItem/validation types + money/currency normalizers), a pure `validation/rules` engine, a pure `kie/decode` (word predictions → Document), a heavy `kie/backend` (pull INT8 from MLflow, run raw `onnxruntime.InferenceSession`), `storage` (SQLite + MinIO via boto3), and the rewired API (`/extract` + `/documents/{id}`). Heavy libraries (onnxruntime, transformers, mlflow, boto3) are imported **inside functions** so modules load cheaply and pure seams stay unit-testable — mirroring `kie/train.py` and `optimize/*`.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, ONNX Runtime, transformers (LayoutLMv3Processor), MLflow registry, SQLite (stdlib), MinIO via boto3 S3 client, pytest, ruff, mypy strict.

## Global Constraints

- Python `>=3.12`; full type hints; mypy `strict` must pass over `src`.
- Prefer functional components over classes; Pydantic `BaseModel` for data contracts; frozen dataclasses allowed as plain data containers.
- No hardcoded constants — all knobs live in `Settings`.
- Served model: pull `cord-layoutlmv3-onnx-int8` (via `Settings.kie_onnx_registered_model_name`) from MLflow and run it via **raw `onnxruntime.InferenceSession` feeding all four inputs** (`input_ids`, `attention_mask`, `bbox` as int64; `pixel_values` as float32). **Do not** use the Optimum wrapper — it silently drops `bbox`/`pixel_values` (Phase 3 deviation).
- The LayoutLMv3 **processor is not fine-tuned**: load it from `Settings.kie_model_name` (`microsoft/layoutlmv3-base`) with `apply_ocr=False`. `id2label` comes from the INT8 model's `config.json`.
- **CORD has no merchant/store-name label** — the schema has no `merchant` field.
- Boxes are normalized to LayoutLMv3's 0–1000 space via the existing `docintel.kie.dataset.normalize_box` (do not duplicate).
- Validation **annotates, never blocks**: `/extract` always returns `200` with fields + `validation` flags, even when `validation.ok` is `false`.
- Heavy libs (onnxruntime, transformers, mlflow, boto3) imported **inside functions**; typed as `Any` where untyped.
- Minimal changes: reuse `kie/dataset.normalize_box`, `kie/labels`, `optimize/export.download_registered_model`; do not refactor unrelated code.
- Every commit message ends with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Environment: `uv sync --extra dev --extra kie --extra serve` (kie → mlflow + boto3; serve → onnxruntime + transformers). The heavy backend path is integration-verified by a real `/extract` run with `docker compose up -d mlflow minio`, not unit-tested — mirroring Phase 2/3.

## File Structure

| File | Responsibility |
|---|---|
| `src/docintel/schema.py` | Pydantic `Document`, `LineItem`, `ValidationIssue`, `ValidationReport`, `WordPrediction`; `parse_money`, `detect_currency`. |
| `src/docintel/validation/rules.py` | `validate(document, settings) -> ValidationReport` + the four rule functions. |
| `src/docintel/kie/decode.py` | `build_document(predictions, image_height, default_currency) -> Document` (reading-order line-item grouping). |
| `src/docintel/kie/backend.py` | `KIEBackend` Protocol, `LayoutLMv3OnnxBackend`, pure seam `words_from_token_logits`. |
| `src/docintel/storage/db.py` | SQLite metadata: `init_db`, `save_document`, `get_document`. |
| `src/docintel/storage/objects.py` | MinIO/S3 image bytes: `make_s3_client`, `ensure_bucket`, `put_image`, `get_image`. |
| `src/docintel/api/routes/extract.py` | (rewire) image → preprocess → OCR → KIE → decode → validate → persist → `Document`. |
| `src/docintel/api/routes/documents.py` | `GET /documents/{id}`, `GET /documents/{id}/image`. |
| `src/docintel/api/main.py` | (modify) register `documents` router; init `app.state` slots. |
| `src/docintel/config.py` | (modify) add Phase 4 settings. |
| `pyproject.toml` | (modify) add `serve` extra; extend mypy overrides. |

---

## Task 1: Schema + normalizers (`schema.py`) + settings + `serve` extra

**Files:**
- Create: `src/docintel/schema.py`
- Create: `tests/test_schema.py`
- Modify: `src/docintel/config.py` (add Phase 4 settings)
- Modify: `pyproject.toml` (`serve` extra + mypy overrides)

**Interfaces:**
- Produces:
  - `WordPrediction(BaseModel){ text: str, box: tuple[int,int,int,int], label: str, confidence: float }` — `box` is pixel `[x_min,y_min,x_max,y_max]`; `label` is a BIO string (e.g. `"B-menu.nm"`, `"O"`).
  - `LineItem(BaseModel){ name: str|None=None, qty: float|None=None, unit_price: float|None=None, price: float|None=None, confidence: float=0.0 }`
  - `ValidationIssue(BaseModel){ rule: str, severity: Literal["error","warning"], message: str, field: str|None=None }`
  - `ValidationReport(BaseModel){ ok: bool=True, errors: list[ValidationIssue]=[], warnings: list[ValidationIssue]=[] }`
  - `Document(BaseModel){ id: str, line_items: list[LineItem]=[], subtotal: float|None=None, tax: float|None=None, service: float|None=None, total: float|None=None, currency: str, field_confidence: dict[str,float]={}, unparsed_fields: list[str]=[], validation: ValidationReport=ValidationReport(), created_at: str }`
  - `parse_money(raw: str) -> float | None`
  - `detect_currency(texts: Iterable[str], default: str) -> str`
  - New settings: `kie_onnx_model_version: str="1"`, `sqlite_path: str="data/docintel.db"`, `minio_bucket: str="documents"`, `minio_secure: bool=False`, `validation_tolerance: float=1.0`, `confidence_threshold: float=0.5`, `default_currency: str="IDR"`.

- [ ] **Step 1: Add the `serve` extra and mypy overrides to `pyproject.toml`**

In `[project.optional-dependencies]`, after the `optimize` extra, add:
```toml
serve = [
    "onnxruntime>=1.16",
    "transformers>=4.40,<5",
]
```
Extend the mypy overrides `module` list (currently ends `"matplotlib.*"`) to also include `"boto3.*", "botocore.*", "transformers.*"`:
```toml
module = ["datasets.*", "huggingface_hub.*", "cv2.*", "doctr.*", "PIL.*", "seqeval.*", "mlflow.*", "optimum.*", "onnx.*", "onnxruntime.*", "matplotlib.*", "boto3.*", "botocore.*", "transformers.*"]
```

- [ ] **Step 2: Add the Phase 4 settings**

In `src/docintel/config.py`, after the `kie_onnx_registered_model_name` line, add:
```python
    # Serving + persistence (Phase 4)
    kie_onnx_model_version: str = "1"
    sqlite_path: str = "data/docintel.db"
    minio_bucket: str = "documents"
    minio_secure: bool = False
    validation_tolerance: float = 1.0
    confidence_threshold: float = 0.5
    default_currency: str = "IDR"
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_schema.py`:
```python
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
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.schema'`.

- [ ] **Step 5: Implement `schema.py`**

Create `src/docintel/schema.py`:
```python
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
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_schema.py -v`
Expected: all parametrized `parse_money` cases + currency + document tests pass.

- [ ] **Step 7: Lint + type-check**

Run: `uv run ruff check src/docintel/schema.py tests/test_schema.py && uv run mypy src/docintel/schema.py src/docintel/config.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/docintel/config.py src/docintel/schema.py tests/test_schema.py
git commit -m "feat(schema): add Document schema, money/currency normalizers, serve extra

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Validation rule engine (`validation/rules.py`)

**Files:**
- Create: `src/docintel/validation/rules.py`
- Create: `tests/test_validation_rules.py`

**Interfaces:**
- Consumes: `Document`, `LineItem`, `ValidationIssue`, `ValidationReport` from `docintel.schema`; `docintel.config.Settings`.
- Produces: `validate(document: Document, settings: Settings) -> ValidationReport`
  - **Hard (errors):** reconciliation (`|subtotal + (tax or 0) + (service or 0) - total| <= tolerance`, and `|sum(line_items.price) - subtotal| <= tolerance`); required fields (`total` not None, ≥ 1 line item).
  - **Soft (warnings):** low confidence (any `field_confidence[k] < threshold` or any line item `confidence < threshold`); number sanity (any negative money value; any `unparsed_fields` entry).
  - `ok` is `False` iff any error is present.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validation_rules.py`:
```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_validation_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.validation.rules'`.

- [ ] **Step 3: Implement `rules.py`**

Create `src/docintel/validation/rules.py`:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_validation_rules.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/docintel/validation/rules.py tests/test_validation_rules.py && uv run mypy src/docintel/validation/rules.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/docintel/validation/rules.py tests/test_validation_rules.py
git commit -m "feat(validation): add rule engine (reconciliation, required, confidence, sanity)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: KIE decode — word predictions → Document (`kie/decode.py`)

**Files:**
- Create: `src/docintel/kie/decode.py`
- Create: `tests/test_kie_decode.py`

**Interfaces:**
- Consumes: `WordPrediction`, `Document`, `LineItem`, `parse_money`, `detect_currency` from `docintel.schema`.
- Produces: `build_document(predictions: Sequence[WordPrediction], default_currency: str) -> Document`
  - `id=""`, `created_at=""`, `validation=ValidationReport()` are placeholders set later by the API.
  - Walks predictions in input (reading) order. A new `LineItem` starts at each `B-menu.nm`; `menu.nm`/`menu.cnt`/`menu.unitprice`/`menu.price` fields attach to the current item (text joined for multi-word fields). Scalars map: `sub_total.subtotal_price`→`subtotal`, `sub_total.tax_price`→`tax`, `sub_total.service_price`→`service`, `total.total_price`→`total`.
  - `field_confidence` holds mean confidence per populated scalar field; each `LineItem.confidence` is the mean over its words. A scalar/line field whose joined text is non-empty but `parse_money` returns `None` is recorded in `unparsed_fields`.
  - `currency = detect_currency(all word texts, default_currency)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kie_decode.py`:
```python
"""Tests for decoding word predictions into a Document."""

from __future__ import annotations

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


def test_unparseable_total_recorded() -> None:
    preds = [
        _w("Coke", "B-menu.nm"),
        _w("3.000", "B-menu.price"),
        _w("oops", "B-total.total_price"),
    ]
    doc = build_document(preds, default_currency="IDR")
    assert doc.total is None
    assert "total" in doc.unparsed_fields
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_kie_decode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.kie.decode'`.

- [ ] **Step 3: Implement `decode.py`**

Create `src/docintel/kie/decode.py`:
```python
"""Decode per-word BIO predictions into a curated receipt Document.

Walks words in reading order: a new line item begins at each ``B-menu.nm``;
menu sub-fields attach to the current item; ``sub_total.*``/``total.*`` map to
scalar fields. Money strings are parsed with ``schema.parse_money``.
"""

from __future__ import annotations

from collections.abc import Sequence

from docintel.schema import Document, LineItem, WordPrediction, detect_currency, parse_money

_LINE_FIELDS = {"menu.nm": "name", "menu.cnt": "qty", "menu.unitprice": "unit_price", "menu.price": "price"}
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
            words = current["_words"]  # type: ignore[assignment]
            words.setdefault(field, []).append(pred.text)  # type: ignore[union-attr]
            current["_conf"].append(pred.confidence)  # type: ignore[union-attr]
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
                unit_price=parse_money(" ".join(words["unit_price"])) if "unit_price" in words else None,
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
        confs = scalar_conf[field]
        field_confidence[field] = sum(confs) / len(confs)
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_kie_decode.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/docintel/kie/decode.py tests/test_kie_decode.py && uv run mypy src/docintel/kie/decode.py`
Expected: clean. (If mypy objects to the `dict[str, object]` row containers, keep the `# type: ignore` comments shown; they are the minimal localized suppressions.)

- [ ] **Step 6: Commit**

```bash
git add src/docintel/kie/decode.py tests/test_kie_decode.py
git commit -m "feat(kie): decode word predictions into curated receipt Document

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: KIE backend — ONNX inference (`kie/backend.py`)

The model load + `InferenceSession.run` path is heavy (onnxruntime, transformers, mlflow) and is imported inside functions; it is **integration-verified** by the Task 6 / controller `/extract` run, not unit-tested. The pure token→word aggregation seam **is** unit-tested.

**Files:**
- Create: `src/docintel/kie/backend.py`
- Create: `tests/test_kie_backend.py`

**Interfaces:**
- Consumes: `WordPrediction` from `docintel.schema`; `docintel.kie.dataset.normalize_box`; `docintel.optimize.export.download_registered_model`; `docintel.config.Settings`; `docintel.pipeline.types.OCRResult`.
- Produces:
  - `words_from_token_logits(logits: Any, word_ids: Sequence[int | None], words: Sequence[str], boxes_pixel: Sequence[tuple[int,int,int,int]], id2label: Mapping[int, str]) -> list[WordPrediction]` — for each word, use its **first** subword token; label = argmax; confidence = softmax max. (pure, tested)
  - `class KIEBackend(Protocol)`: `predict(self, ocr: OCRResult) -> list[WordPrediction]`.
  - `LayoutLMv3OnnxBackend` with `@classmethod load(cls, settings: Settings, tracking_uri: str | None = None) -> LayoutLMv3OnnxBackend` and `predict(self, ocr: OCRResult) -> list[WordPrediction]`.

- [ ] **Step 1: Write the failing test (pure seam)**

Create `tests/test_kie_backend.py`:
```python
"""Tests for the pure token->word aggregation seam."""

from __future__ import annotations

import numpy as np

from docintel.kie.backend import words_from_token_logits


def test_first_subword_token_drives_word_label() -> None:
    # 2 words; word 0 -> tokens 1,2 ; word 1 -> token 3. Token 0 is a special token.
    id2label = {0: "O", 1: "B-menu.nm", 2: "I-menu.nm"}
    # logits shape (seq=4, num_labels=3)
    logits = np.array(
        [
            [9.0, 0.0, 0.0],  # special (word_id None)
            [0.0, 9.0, 0.0],  # word 0 first token -> B-menu.nm
            [0.0, 0.0, 9.0],  # word 0 second token (ignored)
            [9.0, 0.0, 0.0],  # word 1 -> O
        ],
        dtype=np.float32,
    )
    word_ids = [None, 0, 0, 1]
    preds = words_from_token_logits(
        logits, word_ids, ["Coke", "x"], [(0, 0, 1, 1), (2, 2, 3, 3)], id2label
    )
    assert [p.label for p in preds] == ["B-menu.nm", "O"]
    assert preds[0].text == "Coke"
    assert 0.99 < preds[0].confidence <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_kie_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.kie.backend'`.

- [ ] **Step 3: Implement `backend.py`**

Create `src/docintel/kie/backend.py`:
```python
"""Serve the registered ONNX-INT8 LayoutLMv3 KIE model on CPU.

The quantized graph is pulled from MLflow and run via a raw
``onnxruntime.InferenceSession`` feeding all four inputs (input_ids,
attention_mask, bbox as int64; pixel_values as float32) — the Optimum wrapper
silently drops bbox/pixel_values (Phase 3 deviation). The processor is the
un-fine-tuned base processor (apply_ocr=False); id2label comes from the model
config. Heavy libraries are imported inside functions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from docintel.config import Settings
from docintel.kie.dataset import normalize_box
from docintel.pipeline.types import OCRResult
from docintel.schema import WordPrediction

_ONNX_INPUTS = ("input_ids", "attention_mask", "bbox", "pixel_values")


def _softmax_max(row: Any) -> tuple[int, float]:
    exp = np.exp(row - np.max(row))
    probs = exp / exp.sum()
    best = int(np.argmax(probs))
    return best, float(probs[best])


def words_from_token_logits(
    logits: Any,
    word_ids: Sequence[int | None],
    words: Sequence[str],
    boxes_pixel: Sequence[tuple[int, int, int, int]],
    id2label: Mapping[int, str],
) -> list[WordPrediction]:
    """Reduce per-token logits to one prediction per word (first subword wins)."""
    array = np.asarray(logits)
    seen: set[int] = set()
    predictions: list[WordPrediction] = []
    for position, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen or word_id >= len(words):
            continue
        seen.add(word_id)
        label_id, confidence = _softmax_max(array[position])
        predictions.append(
            WordPrediction(
                text=words[word_id],
                box=boxes_pixel[word_id],
                label=id2label[label_id],
                confidence=confidence,
            )
        )
    return predictions


class KIEBackend(Protocol):
    """Maps an OCR result to per-word BIO predictions."""

    def predict(self, ocr: OCRResult) -> list[WordPrediction]: ...


class LayoutLMv3OnnxBackend:
    """ONNX-INT8 LayoutLMv3 token classifier served via onnxruntime."""

    def __init__(self, session: Any, processor: Any, id2label: Mapping[int, str]) -> None:
        self._session = session
        self._processor = processor
        self._id2label = id2label

    @classmethod
    def load(
        cls,
        settings: Settings,
        tracking_uri: str | None = None,
    ) -> LayoutLMv3OnnxBackend:
        """Pull the INT8 model from MLflow and build a ready-to-serve backend."""
        import onnxruntime as ort
        from transformers import LayoutLMv3Processor

        from docintel.optimize.export import download_registered_model

        dest = Path(settings.data_dir) / "models" / settings.kie_onnx_registered_model_name
        bundle = download_registered_model(
            settings.kie_onnx_registered_model_name,
            settings.kie_onnx_model_version,
            dest,
            tracking_uri or settings.mlflow_tracking_uri,
        )
        onnx_path = next(Path(bundle).rglob("*quantized*.onnx"), None) or next(
            Path(bundle).rglob("*.onnx")
        )
        config = json.loads((onnx_path.parent / "config.json").read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in config["id2label"].items()}
        session = ort.InferenceSession(str(onnx_path))
        processor = LayoutLMv3Processor.from_pretrained(settings.kie_model_name, apply_ocr=False)
        return cls(session, processor, id2label)

    def predict(self, ocr: OCRResult) -> list[WordPrediction]:
        """Run OCR words through the ONNX KIE graph and label each word."""
        words = [w.text for w in ocr.words]
        boxes_pixel = [tuple(w.bbox) for w in ocr.words]
        if not words:
            return []
        boxes_1000 = [
            normalize_box(list(box), ocr.image_width, ocr.image_height) for box in boxes_pixel
        ]
        encoding = self._processor(
            ocr_image_placeholder(ocr),
            text=words,
            boxes=boxes_1000,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        feeds = {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
            "bbox": encoding["bbox"].astype(np.int64),
            "pixel_values": encoding["pixel_values"].astype(np.float32),
        }
        logits = self._session.run(None, feeds)[0][0]
        word_ids = encoding.word_ids(batch_index=0)
        return words_from_token_logits(
            logits, word_ids, words, boxes_pixel, self._id2label  # type: ignore[arg-type]
        )


def ocr_image_placeholder(ocr: OCRResult) -> Any:
    """Build a white RGB PIL image of the OCR's dimensions for the processor.

    The KIE model attends to ``pixel_values``; serving feeds the page geometry
    via boxes and a blank canvas of the correct size (the real pixels are not
    re-decoded here — the API passes the decoded image in Task 6).
    """
    from PIL import Image

    return Image.new("RGB", (ocr.image_width, ocr.image_height), "white")
```

**Note for the implementer:** in Task 6 the API has the *decoded* image already, so the real `predict` should accept it. Adjust the signature there: `predict(self, ocr: OCRResult, image: "PIL.Image") -> ...` is preferable to the blank placeholder. To keep this task's pure seam testable and avoid a forward dependency, this task ships the placeholder; **Task 6, Step 3** replaces `ocr_image_placeholder(ocr)` with the real PIL image passed from the route and updates the `KIEBackend.predict` signature to `predict(self, ocr, image)`. Record this in the SDD ledger.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_kie_backend.py -v`
Expected: 1 passed.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/docintel/kie/backend.py tests/test_kie_backend.py && uv run mypy src/docintel/kie/backend.py`
Expected: clean (heavy libs are in the mypy override list; `Any` returns are acceptable).

- [ ] **Step 6: Commit**

```bash
git add src/docintel/kie/backend.py tests/test_kie_backend.py
git commit -m "feat(kie): add LayoutLMv3 ONNX-INT8 serving backend + token aggregation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Persistence — SQLite + MinIO (`storage/db.py`, `storage/objects.py`)

**Files:**
- Create: `src/docintel/storage/db.py`
- Create: `src/docintel/storage/objects.py`
- Create: `tests/test_storage_db.py`
- Create: `tests/test_storage_objects.py`

**Interfaces:**
- Produces (`db.py`):
  - `init_db(path: str) -> None` — create the `documents` table if absent.
  - `save_document(path: str, document: Document, image_key: str) -> None` — upsert by id; stores `document.model_dump_json()`.
  - `get_document(path: str, document_id: str) -> tuple[Document, str] | None` — returns `(Document, image_key)` or None.
- Produces (`objects.py`):
  - `make_s3_client(settings: Settings) -> Any` (boto3 S3 client against the MinIO endpoint).
  - `ensure_bucket(client: Any, bucket: str) -> None`
  - `put_image(client: Any, bucket: str, key: str, data: bytes, content_type: str) -> None`
  - `get_image(client: Any, bucket: str, key: str) -> bytes | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_storage_db.py`:
```python
"""SQLite metadata persistence round-trip."""

from __future__ import annotations

from pathlib import Path

from docintel.schema import Document, LineItem
from docintel.storage.db import get_document, init_db, save_document


def _doc() -> Document:
    return Document(
        id="doc1",
        line_items=[LineItem(name="Coke", price=3000.0, confidence=0.9)],
        subtotal=3000.0,
        total=3000.0,
        currency="IDR",
        created_at="2026-06-22T00:00:00+00:00",
    )


def test_save_then_get_round_trips(tmp_path: Path) -> None:
    db = str(tmp_path / "docintel.db")
    init_db(db)
    save_document(db, _doc(), image_key="doc1.png")
    fetched = get_document(db, "doc1")
    assert fetched is not None
    document, image_key = fetched
    assert document.id == "doc1"
    assert document.line_items[0].name == "Coke"
    assert image_key == "doc1.png"


def test_get_unknown_returns_none(tmp_path: Path) -> None:
    db = str(tmp_path / "docintel.db")
    init_db(db)
    assert get_document(db, "missing") is None
```

Create `tests/test_storage_objects.py`:
```python
"""MinIO/S3 object helpers, exercised against a fake client."""

from __future__ import annotations

from typing import Any

from docintel.storage.objects import ensure_bucket, get_image, put_image


class _FakeS3:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.buckets: set[str] = set()

    def head_bucket(self, Bucket: str) -> None:  # noqa: N803
        if Bucket not in self.buckets:
            raise RuntimeError("missing")

    def create_bucket(self, Bucket: str) -> None:  # noqa: N803
        self.buckets.add(Bucket)

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:  # noqa: N803
        self.store[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if (Bucket, Key) not in self.store:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(self.store[(Bucket, Key)])}


def test_ensure_bucket_creates_when_missing() -> None:
    client = _FakeS3()
    ensure_bucket(client, "documents")
    assert "documents" in client.buckets


def test_put_then_get_round_trips() -> None:
    client = _FakeS3()
    ensure_bucket(client, "documents")
    put_image(client, "documents", "a.png", b"bytes", "image/png")
    assert get_image(client, "documents", "a.png") == b"bytes"


def test_get_missing_returns_none() -> None:
    client = _FakeS3()
    ensure_bucket(client, "documents")
    assert get_image(client, "documents", "nope.png") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_storage_db.py tests/test_storage_objects.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.storage.db'`.

- [ ] **Step 3: Implement `db.py`**

Create `src/docintel/storage/db.py`:
```python
"""SQLite persistence for extracted Document metadata (stdlib sqlite3, no ORM)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from docintel.schema import Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    document_json TEXT NOT NULL,
    image_key TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def _connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_db(path: str) -> None:
    """Create the documents table if it does not exist."""
    with _connect(path) as conn:
        conn.execute(_SCHEMA)


def save_document(path: str, document: Document, image_key: str) -> None:
    """Upsert one document's metadata by id."""
    with _connect(path) as conn:
        conn.execute(
            "INSERT INTO documents (id, document_json, image_key, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "document_json=excluded.document_json, image_key=excluded.image_key",
            (document.id, document.model_dump_json(), image_key, document.created_at),
        )


def get_document(path: str, document_id: str) -> tuple[Document, str] | None:
    """Return ``(Document, image_key)`` for an id, or None if absent."""
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT document_json, image_key FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    if row is None:
        return None
    return Document.model_validate_json(row[0]), row[1]
```

- [ ] **Step 4: Implement `objects.py`**

Create `src/docintel/storage/objects.py`:
```python
"""MinIO object storage for uploaded images via a boto3 S3 client."""

from __future__ import annotations

from typing import Any

from docintel.config import Settings


def make_s3_client(settings: Settings) -> Any:
    """Build a boto3 S3 client pointed at the MinIO endpoint."""
    import boto3

    scheme = "https" if settings.minio_secure else "http"
    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )


def ensure_bucket(client: Any, bucket: str) -> None:
    """Create ``bucket`` if it does not already exist."""
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:  # noqa: BLE001 — head raises for missing/forbidden; create is the recovery
        client.create_bucket(Bucket=bucket)


def put_image(client: Any, bucket: str, key: str, data: bytes, content_type: str) -> None:
    """Store image bytes under ``key``."""
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def get_image(client: Any, bucket: str, key: str) -> bytes | None:
    """Fetch image bytes for ``key``; return None if the object is absent."""
    from botocore.exceptions import ClientError

    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError:
        return None
    data: bytes = response["Body"].read()
    return data
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_storage_db.py tests/test_storage_objects.py -v`
Expected: 5 passed.

- [ ] **Step 6: Lint + type-check**

Run: `uv run ruff check src/docintel/storage/ tests/test_storage_db.py tests/test_storage_objects.py && uv run mypy src/docintel/storage/db.py src/docintel/storage/objects.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/docintel/storage/db.py src/docintel/storage/objects.py tests/test_storage_db.py tests/test_storage_objects.py
git commit -m "feat(storage): add SQLite metadata + MinIO image persistence

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Wire the pipeline into the API (`extract.py`, `documents.py`, `main.py`)

**Files:**
- Modify: `src/docintel/api/routes/extract.py` (full pipeline + persist; returns `Document`)
- Create: `src/docintel/api/routes/documents.py` (`GET /documents/{id}`, `GET /documents/{id}/image`)
- Modify: `src/docintel/kie/backend.py` (Step 3: real image into `predict`)
- Modify: `src/docintel/api/main.py` (register router; init state)
- Modify: `tests/test_extract.py` (update to the new `Document` response)
- Create: `tests/test_documents.py`

**Interfaces:**
- Consumes: `KIEBackend`, `LayoutLMv3OnnxBackend` (Task 4), `build_document` (Task 3), `validate` (Task 2), storage (Task 5), `Document` (Task 1).
- Produces:
  - `get_kie_backend(request) -> KIEBackend` (lazy `app.state.kie_backend`, mirrors `get_ocr_engine`).
  - `get_s3_client(request) -> Any` (lazy `app.state.s3_client`).
  - `POST /extract -> Document` (status 200 always; persists every call).
  - `GET /documents/{id} -> Document` (404 if absent).
  - `GET /documents/{id}/image` -> streamed image bytes (404 if absent).

- [ ] **Step 1: Update `kie/backend.py` so `predict` takes the real image**

In `src/docintel/kie/backend.py`: change the Protocol and method to accept the decoded image, and delete `ocr_image_placeholder`.
```python
class KIEBackend(Protocol):
    """Maps an OCR result + page image to per-word BIO predictions."""

    def predict(self, ocr: OCRResult, image: Any) -> list[WordPrediction]: ...
```
In `LayoutLMv3OnnxBackend.predict`, change the signature to `def predict(self, ocr: OCRResult, image: Any) -> list[WordPrediction]:` and replace `ocr_image_placeholder(ocr)` with `image`. Remove the `ocr_image_placeholder` function. (The image is the BGR `np.ndarray` from `cv2`; the LayoutLMv3 image processor accepts a numpy HWC array. If RGB is required, the route converts with `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` before calling — see Step 4.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_documents.py`:
```python
"""Tests for /extract (full pipeline) and /documents retrieval, fully stubbed."""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docintel.api.main import create_app
from docintel.api.routes.extract import get_kie_backend, get_ocr_engine, get_s3_client
from docintel.config import Settings, get_settings
from docintel.pipeline.types import OCRResult, OCRWord
from docintel.schema import WordPrediction


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    return buf.getvalue()


def _ocr() -> OCRResult:
    return OCRResult(
        text="Coke 3.000",
        words=[
            OCRWord(text="Coke", bbox=(0, 0, 4, 2), confidence=0.9),
            OCRWord(text="3.000", bbox=(0, 3, 4, 5), confidence=0.9),
        ],
        confidence=0.9,
        image_width=16,
        image_height=16,
    )


class _FakeBackend:
    def predict(self, ocr: OCRResult, image: Any) -> list[WordPrediction]:
        return [
            WordPrediction(text="Coke", box=(0, 0, 4, 2), label="B-menu.nm", confidence=0.9),
            WordPrediction(text="3.000", box=(0, 3, 4, 5), label="B-menu.price", confidence=0.9),
        ]


class _FakeS3:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.buckets: set[str] = {"documents"}

    def head_bucket(self, Bucket: str) -> None:  # noqa: N803
        return None

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:  # noqa: N803
        self.store[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if (Bucket, Key) not in self.store:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        class _Body:
            def __init__(self, d: bytes) -> None:
                self._d = d

            def read(self) -> bytes:
                return self._d

        return {"Body": _Body(self.store[(Bucket, Key)])}


@pytest.fixture
def client(tmp_path: Any) -> Iterator[TestClient]:
    app = create_app()
    s3 = _FakeS3()
    app.dependency_overrides[get_ocr_engine] = lambda: lambda image: _ocr()
    app.dependency_overrides[get_kie_backend] = lambda: _FakeBackend()
    app.dependency_overrides[get_s3_client] = lambda: s3
    app.dependency_overrides[get_settings] = lambda: Settings(
        sqlite_path=str(tmp_path / "db.sqlite")
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_extract_returns_document_and_persists(client: TestClient) -> None:
    resp = client.post("/extract", files={"file": ("r.png", _png(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["line_items"][0]["name"] == "Coke"
    assert body["line_items"][0]["price"] == 3000.0
    assert "validation" in body
    doc_id = body["id"]

    got = client.get(f"/documents/{doc_id}")
    assert got.status_code == 200
    assert got.json()["id"] == doc_id

    img = client.get(f"/documents/{doc_id}/image")
    assert img.status_code == 200
    assert img.content == _png() or len(img.content) > 0


def test_get_unknown_document_404(client: TestClient) -> None:
    assert client.get("/documents/nope").status_code == 404
    assert client.get("/documents/nope/image").status_code == 404
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_documents.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_kie_backend'`.

- [ ] **Step 4: Rewrite `extract.py`**

Replace `src/docintel/api/routes/extract.py` with:
```python
"""The /extract endpoint: image upload -> OCR -> KIE -> validate -> persist -> Document."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from docintel.config import Settings, get_settings
from docintel.kie.backend import KIEBackend, LayoutLMv3OnnxBackend
from docintel.kie.decode import build_document
from docintel.pipeline.ocr import Image, OCREngine, load_doctr_engine
from docintel.pipeline.preprocess import preprocess
from docintel.schema import Document
from docintel.storage.db import init_db, save_document
from docintel.storage.objects import ensure_bucket, make_s3_client, put_image
from docintel.validation.rules import validate

logger = logging.getLogger("docintel.api.extract")
router = APIRouter(tags=["pipeline"])

_ACCEPTED_TYPES = {"image/png": "png", "image/jpeg": "jpg"}


def get_ocr_engine(request: Request) -> OCREngine:
    """Return the process-wide docTR engine, loading it once on first use."""
    engine: OCREngine | None = getattr(request.app.state, "ocr_engine", None)
    if engine is None:
        engine = load_doctr_engine(get_settings())
        request.app.state.ocr_engine = engine
    return engine


def get_kie_backend(request: Request) -> KIEBackend:
    """Return the process-wide KIE backend, loading it once on first use."""
    backend: KIEBackend | None = getattr(request.app.state, "kie_backend", None)
    if backend is None:
        backend = LayoutLMv3OnnxBackend.load(get_settings())
        request.app.state.kie_backend = backend
    return backend


def get_s3_client(request: Request) -> Any:
    """Return the process-wide MinIO/S3 client, building it once on first use."""
    client = getattr(request.app.state, "s3_client", None)
    if client is None:
        client = make_s3_client(get_settings())
        request.app.state.s3_client = client
    return client


@router.post("/extract", response_model=Document, summary="Extract structured fields from an image")
async def extract(
    file: UploadFile,
    settings: Settings = Depends(get_settings),  # noqa: B008
    engine: OCREngine = Depends(get_ocr_engine),  # noqa: B008
    backend: KIEBackend = Depends(get_kie_backend),  # noqa: B008
    s3: Any = Depends(get_s3_client),  # noqa: B008
) -> Document:
    """Run the full pipeline, persist the result, and return a validated Document."""
    if file.content_type not in _ACCEPTED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type!r}; use PNG or JPEG.",
        )
    data = await file.read()
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
        )
    image: Image | None = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # type: ignore[assignment]
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the uploaded bytes as an image.",
        )
    if settings.preprocess_enabled:
        image = preprocess(image, settings)

    start = time.perf_counter()
    ocr = engine(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictions = backend.predict(ocr, rgb)
    document = build_document(predictions, settings.default_currency)
    document = document.model_copy(
        update={
            "id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    document = document.model_copy(update={"validation": validate(document, settings)})
    latency_ms = (time.perf_counter() - start) * 1000

    image_key = f"{document.id}.{_ACCEPTED_TYPES[file.content_type]}"
    ensure_bucket(s3, settings.minio_bucket)
    put_image(s3, settings.minio_bucket, image_key, data, file.content_type)
    init_db(settings.sqlite_path)
    save_document(settings.sqlite_path, document, image_key)

    logger.info(
        "extract.complete",
        extra={
            "document_id": document.id,
            "latency_ms": round(latency_ms, 2),
            "line_items": len(document.line_items),
            "validation_ok": document.validation.ok,
        },
    )
    return document
```

- [ ] **Step 5: Create `documents.py`**

Create `src/docintel/api/routes/documents.py`:
```python
"""Retrieval endpoints for persisted documents."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response

from docintel.api.routes.extract import get_s3_client
from docintel.config import Settings, get_settings
from docintel.schema import Document
from docintel.storage.db import get_document
from docintel.storage.objects import get_image

router = APIRouter(tags=["documents"])


def _lookup(settings: Settings, document_id: str) -> tuple[Document, str]:
    found = get_document(settings.sqlite_path, document_id)
    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return found


@router.get("/documents/{document_id}", response_model=Document, summary="Retrieve a document")
def read_document(
    document_id: str,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Document:
    """Return a previously extracted document by id."""
    document, _ = _lookup(settings, document_id)
    return document


@router.get("/documents/{document_id}/image", summary="Retrieve a document's source image")
def read_document_image(
    document_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
    s3: Any = Depends(get_s3_client),  # noqa: B008
) -> Response:
    """Stream the stored source image for a document by id."""
    _, image_key = _lookup(settings, document_id)
    data = get_image(s3, settings.minio_bucket, image_key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")
    media_type = "image/png" if image_key.endswith(".png") else "image/jpeg"
    return Response(content=data, media_type=media_type)
```

- [ ] **Step 6: Register the router and init state in `main.py`**

In `src/docintel/api/main.py`: add `documents` to the import — `from docintel.api.routes import documents, extract, health`. In `lifespan`, after `app.state.ocr_engine = None`, add:
```python
    app.state.kie_backend = None
    app.state.s3_client = None
```
In `create_app`, after `app.include_router(extract.router)`, add `app.include_router(documents.router)`.

- [ ] **Step 7: Update the existing `tests/test_extract.py`**

The old tests assert an `OCRResult` body. Update `_stub_result`-based tests so `/extract` is driven with stubbed OCR **and** KIE + S3 + temp DB, matching `tests/test_documents.py`'s override fixture. Concretely: in `tests/test_extract.py`, add `get_kie_backend` and `get_s3_client` overrides (a no-op KIE backend returning `[]` and the `_FakeS3` from `test_documents`), point `sqlite_path` at a tmp file, and change `test_extract_returns_ocr_result` to assert the `Document` shape:
```python
def test_extract_returns_document(client: TestClient) -> None:
    resp = client.post("/extract", files={"file": ("r.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "id" in body and "validation" in body and body["currency"]
```
Keep the 415/400/422/413/preprocess tests, adding the new overrides to each app instance they build. (DRY: lift the override wiring into the module's `client` fixture and a helper that builds a stubbed app.)

- [ ] **Step 8: Run the API tests to verify they pass**

Run: `uv run pytest tests/test_documents.py tests/test_extract.py -v`
Expected: all pass.

- [ ] **Step 9: Full suite + lint + types**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src`
Expected: all green; the slow OCR test stays deselected.

- [ ] **Step 10: Commit**

```bash
git add src/docintel/api/ src/docintel/kie/backend.py tests/test_extract.py tests/test_documents.py
git commit -m "feat(api): wire KIE pipeline into /extract; add /documents retrieval

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Controller Final Integration (after Task 6 — proves the deliverable)

The unit tests cover the pure seams with everything stubbed; the heavy ONNX path is proven by a real run on the laptop with services up. This is **not** a subagent task.

0. Sync the env: `uv sync --extra dev --extra kie --extra serve` (from `docintel/`).
1. Bring up services: `docker compose up -d mlflow minio` (the INT8 model must already be registered as `cord-layoutlmv3-onnx-int8` — Phase 3 did this).
2. Start the API against host services:
   ```bash
   DOCINTEL_MLFLOW_TRACKING_URI=http://localhost:5000 \
   DOCINTEL_MINIO_ENDPOINT=localhost:9000 \
   DOCINTEL_AWS_S3_ENDPOINT=http://localhost:9000 \
     uv run uvicorn docintel.api.main:app --port 8000
   ```
   (Set `MLFLOW_S3_ENDPOINT_URL`/`AWS_*` as the repo's compose expects for MLflow artifact pulls — mirror Phase 3's run.)
3. POST a CORD receipt image:
   ```bash
   curl -sS -F "file=@<some-cord-receipt>.png;type=image/png" http://localhost:8000/extract | tee /tmp/doc.json
   ```
   Verify the JSON has `line_items`, `subtotal`/`total`, and a `validation` block. Capture `id`.
4. Retrieve it: `curl -sS http://localhost:8000/documents/<id>` returns the same document; `curl -sS http://localhost:8000/documents/<id>/image -o /tmp/out.png` returns the image. Confirm a row exists in SQLite and an object in the MinIO `documents` bucket.
5. Record results (a sample request/response, any gotchas) in the SDD ledger for the report.

If the first KIE load is slow (model download), that is expected once; it is cached under `data/models/`.

---

## Report

On completion, add `docs/phases/phase4/report_phase4.md` summarizing what was built, the end-to-end verification, key decisions (curated schema, no merchant, validation-annotates-never-blocks, boto3-against-MinIO), and deviations. Flip the Phase 4 row in `docs/phases/README.md` to ✅.

## Done When

- `/extract` returns a schema-validated `Document` (curated receipt fields + explicit `validation` flags) produced by the MLflow-registered ONNX-INT8 model via raw `onnxruntime.InferenceSession`.
- Every `/extract` persists image (MinIO) + metadata (SQLite); `GET /documents/{id}` and `/documents/{id}/image` retrieve them; unknown ids 404.
- ruff, ruff-format, mypy-strict, and the full pytest suite are green.

## Self-Review (completed by plan author)

- **Spec coverage:** `KIEBackend`+`LayoutLMv3OnnxBackend` → Task 4; wire KIE into `/extract` → Task 6; Pydantic `Document` + normalizers → Task 1; rule engine (hard/soft + confidence) → Task 2; SQLite + MinIO + id → Task 5/6; `GET /documents/{id}` (+image) → Task 6; decode/grouping → Task 3. All spec sections map to a task.
- **Type consistency:** `WordPrediction` defined in `schema.py` (Task 1), consumed by decode (Task 3) and produced by backend (Task 4); `build_document(predictions, default_currency)` signature matches its call in Task 6; `predict(ocr, image)` finalized in Task 6 Step 1 and used in the route Step 4 and the fake backend in tests. `validate(document, settings)` consistent across Tasks 2 and 6.
- **Placeholder scan:** no TBD/TODO; every code step ships full code. The one forward-dependency (image into `predict`) is explicitly resolved in Task 6 Step 1 rather than left vague.
