# Phase 1 — OCR Baseline + `/extract`

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started**.

**Goal:** The thinnest end-to-end slice — image in, raw OCR text + boxes out.

## Research 🔬
- [ ] PaddleOCR vs docTR on CPU: accuracy on receipts, latency/page, ONNX support, footprint.
- [ ] Image input contract (upload vs base64; size limits).

## Tasks
- [ ] `OCREngine` interface + chosen pretrained implementation.
- [ ] OpenCV preprocessing (deskew, denoise, resize) behind a toggle.
- [ ] `POST /extract` → `{text, boxes, confidence}` (typed response model).
- [ ] Unit tests on sample images; latency logged.

## Done when 📦
- [ ] `/extract` returns OCR results for a sample receipt; engine swappable via config.

## Report
On completion, add `report_phase1.md` to this folder.
