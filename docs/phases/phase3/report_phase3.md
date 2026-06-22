# Phase 3 Report — Optimization: ONNX + INT8 + Benchmark

**Status:** ✅ Complete and verified
**Location:** `docintel/src/docintel/optimize/`
**Date:** 2026-06-22

Phase 3 is the inference-speedup story: export the registered LayoutLMv3 KIE model to ONNX, apply dynamic INT8 quantization, and quantify the accuracy/latency/size trade-off in a reproducible benchmark — entirely on the laptop CPU, which is the serving target. The full benchmark with tables and plots is in [`../../benchmark.md`](../../benchmark.md); this report summarizes what was built and why. See [`phase3.md`](phase3.md) for the phase brief.

---

## 1. What Was Built

### Optimize package (`src/docintel/optimize/`)
- **`config.py`** — frozen `BenchmarkConfig` (source model name/version, sample size, warmup/repeat counts, thread count, quant type, seed) with `from_settings` / `with_overrides`.
- **`export.py`** — `download_registered_model()` (pull the `cord-layoutlmv3` bundle from MLflow) and `export_to_onnx()` (Hugging Face Optimum `ORTModelForTokenClassification`, `export=True`).
- **`quantize.py`** — `quantize_dynamic_int8()` via `ORTQuantizer` + `AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)` (no calibration data needed).
- **`evaluate.py`** — model-agnostic F1 evaluation that reuses `kie.metrics.compute_seqeval_metrics`, so all three configs are scored by the same yardstick as Phase 2.
- **`benchmark.py`** — `latency_stats` (nearest-rank p50/p95), `measure_latency` (warmup + repeats), `dir_size_mb`.
- **`report.py`** — `ConfigResult`, markdown table + matplotlib plots, `write_report`.
- **`run_benchmark.py`** — the `docintel-benchmark-kie` CLI orchestrator: download → export → quantize → build samples → score + time the three configs → write `docs/benchmark.md` + plots → log to MLflow and register the INT8 artifact.

### Tests (`tests/`)
- `test_optimize_config.py`, `test_optimize_benchmark.py`, `test_optimize_report.py`, `test_optimize_evaluate.py`, `test_optimize_export.py`, `test_optimize_run_benchmark.py` — cover the pure seams (config, latency math, report rendering, the MLflow flatten/slugify helpers); the heavy export/quantize/inference path is integration-verified by an actual run.

### Packaging
- Added the `optimize` extra: `optimum[onnxruntime]`, `onnx`, `matplotlib`, and **`transformers>=4.40,<5`**. Registered the `docintel-benchmark-kie` console script.

---

## 2. Results

Evaluated on 50 CORD `test` receipts (latency: 3 warmup runs discarded, 5 timed repeats), laptop CPU:

| Config | F1 | p50 (ms) | p95 (ms) | Throughput (doc/s) | Size (MB) |
|--------|-----|----------|----------|--------------------|-----------|
| torch-fp32 | 0.8449 | 1934.2 | 2379.9 | 0.50 | 480.5 |
| onnx-fp32 | 0.8449 | 1214.7 | 1401.4 | 0.78 | 480.8 |
| onnx-int8 | 0.8315 | 650.2 | 852.0 | 1.46 | 121.4 |

**Served ONNX INT8 vs PyTorch fp32:** 3.0× faster p50, 2.9× throughput, 4× smaller, retaining **98.4% of F1** (1.3-point drop). ONNX Runtime alone gives ~1.6×; dynamic INT8 adds a further ~1.9×. The onnx-fp32 F1 *exactly* equals torch-fp32 F1, confirming the export is faithful.

The INT8 artifact is registered as **`cord-layoutlmv3-onnx-int8` v1** for Phase 4 serving.

---

## 3. Verification

| Check | Command | Result |
|-------|---------|--------|
| Lint / format / types | `ruff check . && ruff format --check . && mypy src` | clean |
| Tests | `uv run --extra dev --extra train --extra optimize --extra kie pytest` | 56 passed |
| End-to-end benchmark | `docintel-benchmark-kie` (laptop CPU) | `docs/benchmark.md` + plots + MLflow run + registered INT8 model |

---

## 4. Key Decisions

1. **Three-config benchmark (torch-fp32 / onnx-fp32 / onnx-int8)** — separates the runtime gain from the quantization gain instead of reporting a single conflated speedup.
2. **Dynamic INT8 (no calibration)** — simplest quantization that delivers the size/latency win without a calibration dataset; the F1 cost (1.3 points) is small and measured.
3. **Reuse `kie.metrics` for scoring** — identical metric to Phase 2 training, so the numbers are directly comparable.
4. **Run ONNX via raw `onnxruntime.InferenceSession`** — see Deviations; required for LayoutLMv3's multi-input graph.

---

## 5. Deviations / Gotchas

- **Optimum requires `transformers<5`** — installing Optimum forced transformers 5.x → 4.57.6. Pinned `transformers>=4.40,<5` in the `optimize` extra; the Phase 2 model (saved under tf5.x) loads fine under tf4.x.
- **Optimum's `ORTModelForTokenClassification.__call__` silently drops LayoutLMv3's `bbox` / `pixel_values`**, so the ONNX graph rejects the call. The benchmark runs the ONNX graph directly via `onnxruntime.InferenceSession`, feeding all four inputs (`input_ids`, `attention_mask`, `bbox` as int64; `pixel_values` as float32). **Phase 4 serving must do the same — not use the Optimum wrapper.**
- **The `optimize` workflow also needs the `kie` extra** (MLflow) — sync `dev train optimize kie` together.

---

## 6. Phase 3 Checklist (from `plan.md`)

- [x] ONNX export path for LayoutLMv3; INT8 quantization options in ONNX Runtime.
- [x] Metric definitions: F1, p50/p95 latency, throughput, model size.
- [x] Export the registered model to ONNX; INT8 quantize.
- [x] Benchmark harness: fixed sample set, warm-up, repeated runs.
- [x] Compare fp32 vs INT8 (F1 / latency / throughput / size); log to MLflow.
- [x] `docs/benchmark.md` with tables + plots.
- [x] **Done when:** a reproducible benchmark report justifies the served ONNX INT8 artifact.

---

## 7. Next: Phase 4 — Serving + Validation + Schema + Persistence

- `LayoutLMv3OnnxBackend` pulls `cord-layoutlmv3-onnx-int8` from MLflow and runs it via raw `onnxruntime.InferenceSession` (all four inputs).
- Wire KIE into `/extract`; add a Pydantic `Document` schema + field validation.
- Persist results (SQLite + MinIO) with `GET /documents/{id}`.
