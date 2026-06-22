# Phase 4 — Serving + Validation + Structured Schema + Persistence

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Next** (current focus).

**Goal:** A trustworthy, persisted, retrievable end-to-end response on CPU.

## Research 🔬
- [ ] Canonical output schema (fields, types, currency/date normalization).
- [ ] Validation rules worth enforcing (line items reconcile to total; date/currency sanity).

## Tasks
- [ ] 💻 `KIEBackend` interface + `LayoutLMv3OnnxBackend` (pulls registered model from MLflow).
- [ ] 💻 Wire KIE into `/extract`: image → preprocess → OCR → KIE → validation → fields.
- [ ] Pydantic `Document` schema + field normalizers.
- [ ] Rule engine: hard failures vs soft warnings, with confidence flags.
- [ ] Persist metadata (SQLite) + image (MinIO), linked by document id.
- [ ] `GET /documents/{id}` retrieval.

## Done when 📦
- [ ] `/extract` returns schema-validated structured fields with explicit validation flags, pulled from the MLflow-registered ONNX INT8 model; documents retrievable by id.

## Report
On completion, add `report_phase4.md` to this folder.
