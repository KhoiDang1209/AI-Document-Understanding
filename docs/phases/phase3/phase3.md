# Phase 3 — KIE: Fine-tune LayoutLMv3 → ONNX INT8

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started**.

**Goal:** The flagship — extract structured fields, trained on Colab, served locally on CPU.

## Research 🔬
- [ ] LayoutLMv3 fine-tuning recipe for CORD (token classification).
- [ ] Label schema mapping (CORD fields → output JSON schema).
- [ ] ONNX export + INT8 quantization; accuracy/latency trade-off.

## Tasks
- [ ] ☁️ Colab notebook: dataset prep, fine-tune LayoutLMv3, log to MLflow.
- [ ] ☁️ Export ONNX + INT8 quantize; register in MLflow + push to MinIO.
- [ ] 💻 `KIEBackend` interface + `LayoutLMv3OnnxBackend` (pulls registered model).
- [ ] 💻 Wire KIE into `/extract`: image → layout → OCR → KIE → fields.
- [ ] Evaluate: KIE F1 per field on CORD test split.

## Done when 📦
- [ ] `/extract` returns structured fields; model pulled from MLflow, runs as ONNX INT8 on CPU.
- [ ] F1 recorded and reproducible from the notebook.

## Report
On completion, add `report_phase3.md` to this folder.
