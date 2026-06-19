# Phase 2 — Layout Detection + Preprocessing

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started**.

**Goal:** Add document-region structure ahead of OCR/KIE.

## Research 🔬
- [ ] DocLayout-YOLO weights/license; class taxonomy vs needs.
- [ ] ONNX export + CPU inference latency.
- [ ] Quantify whether layout improves downstream KIE (spike).

## Tasks
- [ ] `LayoutDetector` interface + DocLayout-YOLO (ONNX Runtime).
- [ ] Integrate: preprocess → layout → region-scoped OCR.
- [ ] Visual debug output (boxes overlaid) for inspection.

## Done when 📦
- [ ] `/extract` returns labeled regions + bounding boxes; layout runs as ONNX on CPU within budget.

## Report
On completion, add `report_phase2.md` to this folder.
