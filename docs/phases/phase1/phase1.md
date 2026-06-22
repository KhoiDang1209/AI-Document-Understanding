# Phase 1 — OCR Baseline + `/extract`

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **✅ Complete**.

**Goal:** The thinnest end-to-end slice — image in, raw OCR text + boxes out.

## Research 🔬
- [x] PaddleOCR vs docTR on CPU: accuracy on receipts, latency/page, ONNX support, footprint.
- [x] Image input contract (upload vs base64; size limits).

## Tasks
- [x] `OCREngine` interface + chosen pretrained implementation.
- [x] OpenCV preprocessing (deskew, denoise, resize) behind a toggle.
- [x] `POST /extract` → `{text, boxes, confidence}` (typed response model).
- [x] Unit tests on sample images; latency logged.

## Done when 📦
- [x] `/extract` returns OCR results for a sample receipt; engine swappable via config.

## Report
See [`report_phase1.md`](report_phase1.md).
