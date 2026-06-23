# Phase 4 — Serving + Validation + Structured Schema + Persistence

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **✅ Complete** — see [report](report_phase4.md).

**Goal:** A trustworthy, persisted, retrievable end-to-end response on CPU.

## Research 🔬
- [x] Canonical output schema (fields, types, currency/date normalization).
- [x] Validation rules worth enforcing (line items reconcile to total; date/currency sanity).

## Tasks
- [x] 💻 `KIEBackend` interface + `LayoutLMv3OnnxBackend` (pulls registered model from MLflow).
- [x] 💻 Wire KIE into `/extract`: image → preprocess → OCR → KIE → validation → fields.
- [x] Pydantic `Document` schema + field normalizers.
- [x] Rule engine: hard failures vs soft warnings, with confidence flags.
- [x] Persist metadata (SQLite) + image (MinIO), linked by document id.
- [x] `GET /documents/{id}` retrieval.

## Done when 📦
- [x] `/extract` returns schema-validated structured fields with explicit validation flags, pulled from the MLflow-registered ONNX INT8 model; documents retrievable by id.

## Report
On completion, add `report_phase4.md` to this folder.
