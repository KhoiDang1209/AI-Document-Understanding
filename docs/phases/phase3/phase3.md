# Phase 3 — Optimization: ONNX + INT8 + Benchmark

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **✅ Complete** — see [`docs/benchmark.md`](../../benchmark.md).

**Goal:** The inference-speedup story — export, quantize, and quantify the trade-off.

## Research 🔬
- [x] ONNX export path for LayoutLMv3; INT8 quantization options in ONNX Runtime.
- [x] Metric definitions: F1, CER/WER, p50/p95 latency, throughput, model size.

## Tasks
- [x] ☁️/💻 Export the registered model to ONNX; INT8 quantize.
- [x] Benchmark harness: fixed sample set, warm-up, repeated runs.
- [x] Compare fp32 vs INT8 (F1 / latency / throughput / size); log to MLflow.
- [x] `docs/benchmark.md` with tables + plots.

## Done when 📦
- [x] A reproducible benchmark report justifies the served ONNX INT8 artifact.

## Outcome
On laptop CPU the served **ONNX INT8** model is **3.0× faster** (p50 1934 → 650 ms), **2.9× throughput**, and **4× smaller** (480 → 121 MB) than PyTorch fp32, retaining **98.4% of F1**. Full tables and plots in [`docs/benchmark.md`](../../benchmark.md).

## Report
See [`report_phase3.md`](report_phase3.md) (full benchmark in [`docs/benchmark.md`](../../benchmark.md)).
