# Phase 4 — Validation + Structured Schema + Persistence

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started**.

**Goal:** Trustworthy, persisted, retrievable output.

## Research 🔬
- [ ] Canonical output schema (fields, types, currency/date normalization).
- [ ] Validation rules worth enforcing (line items reconcile to total, date/currency sanity).

## Tasks
- [ ] Pydantic `Document` schema + field normalizers.
- [ ] Rule engine: hard failures vs soft warnings, with confidence.
- [ ] Persist metadata (SQLite) + image (MinIO), linked by document id.
- [ ] `GET /documents/{id}` retrieval.

## Done when 📦
- [ ] Every response is schema-validated with explicit validation flags; documents retrievable by id.

## Report
On completion, add `report_phase4.md` to this folder.
