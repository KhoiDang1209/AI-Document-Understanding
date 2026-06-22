# Phase 3 — Optimization: ONNX + INT8 + Benchmark (Design)

**Date:** 2026-06-22
**Phase:** 3 (core MLOps spine)
**Status:** Approved design, ready for implementation plan
**Predecessor:** Phase 2 registered `cord-layoutlmv3` v1 in MLflow (LayoutLMv3-base fine-tuned on CORD, eval F1 0.881).

---

## Goal

Produce the inference-speedup story for the registered LayoutLMv3 model: export it to
ONNX, quantize to INT8, and quantify the trade-off with a reproducible benchmark, so a
served ONNX-INT8 artifact is justified by data and ready for Phase 4 to consume.

## Scope decisions

- **Runs entirely on the laptop CPU.** ONNX export, dynamic INT8 quantization, and the
  benchmark need no GPU, and the laptop CPU *is* the real serving target. No Colab, no
  manual notebook handoff (unlike Phase 2). The controller drives it end-to-end on the
  laptop; the user approves.
- **Export/quantize toolchain: Hugging Face Optimum + ONNX Runtime** (over raw
  `torch.onnx.export`). Purpose-built for transformers, minimal code, and correctly wires
  LayoutLMv3's four inputs (`input_ids`, `bbox`, `attention_mask`, `pixel_values`). ONNX
  Runtime is the serving runtime in Phase 4 regardless.
- **Dynamic INT8** (weights quantized, activations dynamic) over static INT8 — standard for
  transformer encoders on CPU, needs no calibration dataset (YAGNI).
- **Accuracy metric is F1** (reusing Phase 2's seqeval). The roadmap's CER/WER are OCR
  metrics and do not apply to this token-classification model — explicitly excluded.

## Architecture

New package `src/docintel/optimize/` (parallel to `kie/`), one responsibility per file:

| File | Responsibility |
|---|---|
| `config.py` | `BenchmarkConfig` frozen dataclass (sample size, warmup runs, repeats, thread count, quant type) + `from_settings` + `with_overrides`. No hardcoded constants. |
| `export.py` | Download the registered model artifacts from MLflow; load with transformers; Optimum ONNX export (fp32). |
| `quantize.py` | Dynamic INT8 quantization via `ORTQuantizer`. |
| `evaluate.py` | Run an ONNX or torch model over the CORD test split → F1, **reusing** `kie/dataset.py` (encode) and `kie/metrics.py` (seqeval). No metric logic duplicated. |
| `benchmark.py` | Pure timing harness: warmup + repeated runs → `LatencyStats` (p50/p95/mean latency, throughput); `dir_size_mb`. |
| `report.py` | Render markdown tables + matplotlib plots into `docs/benchmark.md`. |
| `run_benchmark.py` | CLI `docintel-benchmark-kie`: orchestrates export → quantize → evaluate → benchmark → report; logs to MLflow; registers the served artifact. |

Heavy libraries (optimum, onnxruntime, transformers, datasets, matplotlib) are imported
**inside functions**, so the modules load cheaply and the pure helpers stay unit-testable
without the heavy stack — the same pattern Phase 2 used for `kie/train.py`.

## Data flow

```
MLflow registry (cord-layoutlmv3 v1)
  → download bundle/model + processor + label_map        [export.py]
  → ONNX fp32 (Optimum export)                           [export.py]
  → ONNX INT8 (dynamic quantization)                     [quantize.py]
  → F1 per variant on CORD test split                    [evaluate.py]
  → latency / throughput per variant on a fixed sample   [benchmark.py]
  → docs/benchmark.md (tables + plots)                   [report.py]
  → log metrics + artifacts to MLflow,                   [run_benchmark.py]
    register cord-layoutlmv3-onnx-int8
```

## What we compare (3 configs)

1. **PyTorch fp32** — the trained model as-is (baseline, torch on CPU).
2. **ONNX fp32** — Optimum export, run on ONNX Runtime.
3. **ONNX INT8** — dynamic-quantized, run on ONNX Runtime. **← the served artifact.**

Three configs (not two) so the report attributes the speedup cleanly: PyTorch→ONNX is the
runtime gain; ONNX fp32→INT8 is the quantization gain.

## Metrics per config

**Evaluation split:** the CORD `test` split, held out and identical across all three
configs, so the fp32→INT8 F1 delta is apples-to-apples. The absolute F1 may differ slightly
from Phase 2's reported 0.881 (that was the training-time eval split); what matters here is
the same split for every config.

- **Accuracy:** F1 (+ precision / recall / accuracy) via seqeval, on the CORD test split.
- **Latency:** p50 / p95 / mean per document (warmup runs discarded, N repeats).
- **Throughput:** documents/second.
- **Size:** artifact size on disk (MB).

**Acceptance (soft):** INT8 should be meaningfully faster while retaining F1 within a small
delta (≤ 1.5 F1 points vs PyTorch fp32). This is flagged in the report, not a hard gate —
the deliverable is the honest trade-off, not a pass/fail.

## MLflow logging + Phase 4 handoff

One benchmark run logged to a new experiment `cord-kie-benchmark`:

- **Params:** source model name + version, sample size, warmup, repeats, thread count, quant type.
- **Metrics:** flattened per config (e.g. `f1_torch_fp32`, `p95_ms_onnx_int8`,
  `throughput_onnx_int8`, `size_mb_onnx_int8`).
- **Artifacts:** `docs/benchmark.md` + plot PNGs.
- **Register the served artifact:** the ONNX-INT8 model is registered under a new registry
  name **`cord-layoutlmv3-onnx-int8`** (new `Settings.kie_onnx_registered_model_name`), so
  Phase 4's `LayoutLMv3OnnxBackend` pulls it by name. fp32 and INT8 stay distinct registered
  models.

## Dependencies

New `optimize` extra in `pyproject.toml`: `optimum[onnxruntime]`, `onnx`, `matplotlib`.
Reuses the existing `train` extra for transformers / datasets / seqeval. The workflow runs
under `uv sync --extra dev --extra train --extra optimize` (`dev` so pytest resolves inside
`.venv` rather than falling through to the Anaconda base — a known repo gotcha). mypy
overrides extended with `optimum.*`, `onnx.*`, `onnxruntime.*`, `matplotlib.*`. ONNX Runtime
graduates to a core runtime dependency in Phase 4 when serving lands.

## Testing

Mirrors Phase 2's split between pure (CI) and heavy (manual integration) code.

**Unit-tested in CI** (pure, fast, no heavy model):
- `benchmark.py` — `LatencyStats` math (p50 / p95 / mean / throughput on known inputs); the
  warmup + repeat loop calls a fake `run_fn` exactly `warmup + repeats` times; `dir_size_mb`.
- `report.py` — markdown table rendering from a metrics dict.
- `config.py` — `BenchmarkConfig` defaults, `from_settings`, `with_overrides`.

**Not in CI** (heavy — real export / quantize / eval): imported inside functions, exercised
by the actual laptop benchmark run that produces the real `docs/benchmark.md`. This is Phase
3's integration step (analogous to Phase 2's Colab run), driven by the controller on the
laptop.

## Error handling

- MLflow unreachable → explicit message to run `docker compose up -d mlflow`.
- Registered model / version missing → clear error naming what was searched.
- ONNX export / opset failure → surfaced, not swallowed.
- Fair benchmark: fixed seed, configurable `torch.set_num_threads`, warmup runs discarded.

## Key risk (verify first)

**Optimum × transformers 5.x × LayoutLMv3 export.** Phase 2 resolved `transformers 5.12.1`,
which is very new; Optimum's ONNX exporter historically tracks transformers 4.x. If Optimum
does not yet support 5.x for LayoutLMv3, the export fails.

**Mitigation:** the first implementation task is a *toolchain smoke-export* (a minimal
forward → ONNX) to confirm the stack works before building the harness. If it fails, the
fallback is pinning transformers for the optimize path, or dropping to raw
`torch.onnx.export` for the export step only (Approach B). De-risk up front, not late.

## Done when

- A reproducible `docs/benchmark.md` compares PyTorch fp32 vs ONNX fp32 vs ONNX INT8 on
  F1 / latency / throughput / size, with tables and plots.
- An MLflow run records those metrics and artifacts.
- `cord-layoutlmv3-onnx-int8` is registered in the MLflow model registry for Phase 4 to
  serve.

## Out of scope (Phase 4+)

- Wiring the ONNX model into `/extract` serving (Phase 4).
- The canonical output schema, validation rules, and persistence (Phase 4).
- Static INT8 / calibration, GPU benchmarking, or alternative quantization schemes
  (not needed for the CPU serving story).
