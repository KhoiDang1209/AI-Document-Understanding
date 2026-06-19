# Phase 5 — Optimization + Benchmark Report

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started**.

**Goal:** A rigorous engineering write-up comparing backends and configurations.

## Research 🔬
- [ ] Quantization variants; ONNX Runtime threading/session options on CPU.
- [ ] Metric definitions: F1, CER/WER, p50/p95 latency, throughput, model size.

## Tasks
- [ ] Benchmark harness: fixed sample set, warm-up, repeated runs.
- [ ] Compare fp32 vs INT8; LayoutLMv3 vs optional LLM KIE.
- [ ] `docs/benchmark.md` with tables + plots; log to MLflow.

## Done when 📦
- [ ] Reproducible benchmark report covering accuracy / latency / size trade-offs.

## Report
On completion, add `report_phase5.md` to this folder.
