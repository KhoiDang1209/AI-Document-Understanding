# Phase 4 Design — Serving + Validation + Structured Schema + Persistence

**Date:** 2026-06-22
**Status:** Approved (design); ready for implementation planning.
**Brief:** [`docs/phases/phase4/phase4.md`](../../phases/phase4/phase4.md) · [`docs/plan.md` §Phase 4](../../plan.md)

## Goal

A trustworthy, persisted, retrievable end-to-end response on CPU. `/extract` runs the
full pipeline (image → preprocess → OCR → KIE → decode → validation), returns
schema-validated structured fields with explicit validation flags from the
MLflow-registered ONNX-INT8 model, and persists every result so documents are
retrievable by id.

## Context (what already exists)

- Phases 0–3 deliver: OCR `/extract` (currently returns raw `OCRResult`), a fine-tuned
  LayoutLMv3 KIE model, and an optimized artifact **`cord-layoutlmv3-onnx-int8` v1**
  registered in MLflow (~650 ms p50 on laptop CPU, 98.4% of fp32 F1).
- The KIE model is **not yet wired into serving** — Phase 4 does that.
- Phase 3 proved the ONNX graph must be run via **raw `onnxruntime.InferenceSession`**
  feeding all four inputs; the Optimum wrapper silently drops `bbox` / `pixel_values`.
- OCR engine is lazy-loaded once onto `app.state` (`api/routes/extract.py:23`) — the KIE
  backend follows the same pattern.

## Decisions (locked during brainstorming)

1. **Schema scope: curated receipt shape** — not the full ~30 CORD sub-labels and not a
   generic entity list. The KIE labels are distilled into a clean domain `Document`.
   **CORD has no merchant/store-name category** (labels are only `menu.*`, `sub_total.*`,
   `total.*`, `void_menu.*`), so the schema has **no `merchant` field** — the model cannot
   produce one. A merchant heuristic (e.g. top-of-receipt text) is a possible later
   advancement, out of scope here.
2. **Persistence: SQLite (metadata) + MinIO (image)** as briefed — demonstrates
   object-storage skills; serving path depends on MinIO being up.
3. **Validation enforces all four rule families** (two hard, two soft) and **annotates,
   never blocks** — fields always return; a failing receipt is `200` with `validation.ok=false`.
4. **`/extract` becomes a breaking change** — returns `Document`, not `OCRResult`. Intended
   per the brief; OCR-only output is not separately preserved.
5. **`GET /documents/{id}/image`** is included as a minor addition beyond the 6 listed tasks.

## Architecture / new modules

```
api/routes/extract.py     (rewired)  image → preprocess → OCR → KIE → decode → validate → persist → Document
api/routes/documents.py   (new)      GET /documents/{id}, GET /documents/{id}/image
kie/backend.py            (new)      KIEBackend Protocol + LayoutLMv3OnnxBackend
kie/decode.py             (new)      token BIO predictions → entity spans → curated receipt Document
schema.py                 (new)      Pydantic Document, LineItem, Money + field normalizers
validation/rules.py       (new)      the four rule families → ValidationReport (errors/warnings)
storage/db.py             (new)      SQLite metadata (stdlib sqlite3, no ORM)
storage/objects.py        (new)      MinIO image put/get (minio client)
```

## Components

### `kie/backend.py`
- `KIEBackend` — `Protocol` with `predict(words, boxes, image) -> list[WordPrediction]`
  (per-word label + confidence).
- `LayoutLMv3OnnxBackend` — pulls `cord-layoutlmv3-onnx-int8` from MLflow once (lazy, on
  `app.state`, same as the OCR engine). Uses the LayoutLMv3 processor to build the four
  inputs (`input_ids`, `attention_mask`, `bbox` int64, `pixel_values` float32) from OCR
  words + boxes **normalized to 0–1000** by image dims. Runs **raw
  `onnxruntime.InferenceSession`**; per-token logits → argmax labels mapped back to words
  via `word_ids`. Reuses `kie.labels` / processor logic from Phase 2.

### `kie/decode.py`
- Word-level BIO predictions → contiguous entity spans → curated `Document`.
- **Line-item grouping** is the hard part (which `menu.nm` pairs with which `menu.price`):
  walk word predictions in OCR reading order and **start a new line item at each
  `B-menu.nm`**, attaching the row's `menu.cnt`/`menu.unitprice`/`menu.price` to the current
  item — simpler and more deterministic than y-clustering, and unit-testable.
  `sub_total.*` / `total.*` map to scalar fields.
- Per-field confidence = mean of its constituent token confidences.

### `schema.py`
- `Document{ id, line_items[], subtotal, tax, service, total, currency, validation, created_at }`
  (no `merchant` — CORD has no such label).
- `LineItem{ name, qty, unit_price, price, confidence }`
- Money is represented as a parsed `float | None` via a `parse_money` normalizer (CORD
  prices are integer strings with `.`/`,` thousands separators, e.g. `"10.000"`); currency
  inferred from symbols, default `default_currency` in `Settings`.

### `validation/rules.py`
- Pure functions, each `(Document) -> list[Issue]`; `validate(document) -> ValidationReport`.
- **Hard (errors):**
  - Reconciliation — `|sum(line_items.price) + tax − total| ≤ tolerance`; `subtotal == sum(items)`.
  - Required fields — `total` parseable; ≥ 1 line item.
- **Soft (warnings):**
  - Low confidence — any field below `confidence_threshold`.
  - Money / number sanity — unparseable money strings, negative total, mixed currency symbols.
- Tolerance + threshold live in `Settings`. (Date sanity is N/A — CORD receipts have no date.)

### `storage/`
- `db.py` — stdlib `sqlite3`, table
  `documents(id, fields_json, validation_json, created_at, image_key)`; `save_document`,
  `get_document`. No ORM.
- `objects.py` — `minio` client; bucket `documents`, key `{id}.png`; `put_image`, `get_image`.

## Data flow & API contracts

- **`POST /extract`** → generate `id` (uuid4) → run full pipeline → **always persist**
  (image → MinIO, metadata → SQLite) → return `Document` (fields + `validation` flags + `id`).
  Validation never blocks; failing receipts return `200` with `validation.ok=false`.
- **`GET /documents/{id}`** → join SQLite metadata, reconstruct `Document`; `404` if unknown.
- **`GET /documents/{id}/image`** → stream the stored image from MinIO; `404` if unknown.

## Settings (new)
`sqlite_path`, `minio_bucket`, `validation_tolerance`, `confidence_threshold`,
`default_currency`. All env-driven (`DOCINTEL_` prefix), no hardcoded constants.

## Testing
- Unit-test the pure seams: decode grouping, each validation rule, `parse_money` normalizer,
  schema construction, SQLite round-trip (temp db), MinIO via a stubbed client.
- The heavy ONNX path is integration-verified by a real `/extract` run on a CORD sample,
  mirroring Phase 3's approach.

## Dependencies
New `serve` extra: `minio` (plus `onnxruntime` + `transformers`, already in `optimize`).
SQLite is stdlib.

## Done when
`/extract` returns schema-validated structured fields with explicit validation flags, pulled
from the MLflow-registered ONNX-INT8 model; documents retrievable by id.
