# Phase 2 — KIE Fine-tune (LayoutLMv3) + MLflow

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **✅ Complete**.

**Goal:** The flagship build-time step — fine-tune the extraction model on Colab and make the run reproducible and tracked.

## Research 🔬
- [x] LayoutLMv3 fine-tuning recipe for CORD (token classification).
- [x] Label schema mapping (CORD fields → output JSON schema).

## Tasks
- [x] ☁️ Colab notebook: dataset prep, fine-tune LayoutLMv3.
- [x] ☁️ Log params, metrics, and artifacts to MLflow; register the chosen model.
- [x] ☁️ Push the model artifact to MinIO.
- [x] Evaluate: KIE F1 per field on the CORD test split, logged to MLflow.

## Done when 📦
- [x] A fine-tuned LayoutLMv3 is registered in MLflow with F1 recorded and reproducible from the notebook.

## Report
See [`report_phase2.md`](report_phase2.md).
