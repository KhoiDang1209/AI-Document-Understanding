# Phase 2 Report — KIE Fine-tune (LayoutLMv3) + MLflow

**Status:** ✅ Complete and verified
**Location:** `docintel/` (training on Colab GPU; registration on the laptop)
**Date:** 2026-06-22

Phase 2 is the flagship build-time step: fine-tune LayoutLMv3 for key-information extraction on the CORD receipt dataset, make the run reproducible and tracked in MLflow, and register the resulting model so later phases can pull it by name. Training runs on Colab GPU; the bundle is imported and registered on the laptop. See [`phase2.md`](phase2.md) for the phase brief and [`../../plan.md`](../../plan.md) for the full roadmap.

---

## 1. What Was Built

### KIE package (`src/docintel/kie/`)
- **`config.py`** — `TrainingConfig` and KIE settings (model name, dataset, epochs, batch size, learning rate, MLflow experiment/registry names). No hardcoded constants.
- **`labels.py`** — data-derived CORD BIO label schema: the label set is built from the dataset's own categories, yielding **59 labels** (`O` + 29 CORD field categories × `B-`/`I-`). `build_label_maps()` returns the `id2label` / `label2id` pair used everywhere downstream.
- **`dataset.py`** — `parse_cord_example()` (CORD `ground_truth` JSON → words, boxes, BIO tags) and `encode_example()` (words/boxes/tags + image → LayoutLMv3 processor features). Fixes the B-/I- assignment on an empty first word and passes the image to the processor.
- **`metrics.py`** — `compute_seqeval_metrics()`: entity-level overall F1/precision/recall plus per-field breakdown via seqeval.
- **`train.py`** — `LayoutLMv3` training builder (Hugging Face `Trainer`) and the bundle saver (model + processor + `label_map.json`).
- **`import_run.py`** — `docintel-import-kie` CLI: imports the Colab-produced bundle, logs it to the local MLflow server, and registers it as `cord-layoutlmv3`. Forces UTF-8 stdout so non-UTF-8 Windows consoles don't crash on MLflow's emoji.

### Colab notebook (`notebooks/`)
- Orchestrates dataset prep → LayoutLMv3 fine-tune → MLflow logging end to end on GPU, producing a downloadable bundle (`cord-layoutlmv3-bundle.zip`).

### Tests (`tests/`)
- `test_kie_config.py`, `test_kie_labels.py`, `test_kie_dataset.py`, `test_kie_metrics.py`, `test_kie_train.py`, `test_kie_import.py` — cover label-schema derivation, CORD parsing/encoding, seqeval metrics, the training-builder wiring, and the import/registration seam.

### Packaging
- Added `train` and `kie` optional extras (transformers, accelerate, seqeval, mlflow). `mlflow` is pinned `<3` to match the v2.16.2 compose server; `accelerate` is required by the `transformers` `Trainer`.

---

## 2. Results

Fine-tuned on the CORD `train` split, evaluated on the CORD `test` split (Colab GPU):

| Metric | Value |
|--------|-------|
| F1 (entity-level, seqeval) | **0.881** |
| Precision | 0.865 |
| Recall | 0.897 |

The bundle was imported on the laptop and **`cord-layoutlmv3` v1 is registered** in the local MLflow registry (run `bd474d9f`, 59-label token classifier). A CPU smoke test confirmed forward inference produces logits of shape `(1, seq_len, 59)`.

---

## 3. Verification

| Check | Command | Result |
|-------|---------|--------|
| Lint / format / types | `ruff check . && ruff format --check . && mypy src` | clean |
| Tests | `uv run --extra dev --extra kie --extra train pytest` | pass |
| Training | Colab notebook (GPU) | F1 0.881 logged to MLflow |
| Registration | `docintel-import-kie` (host, `DOCINTEL_MLFLOW_TRACKING_URI=http://localhost:5000`) | `cord-layoutlmv3` v1 registered |
| Inference smoke | CPU forward pass | logits `(1, seq, 59)` |

---

## 4. Key Decisions

1. **LayoutLMv3 token classification on CORD** — layout-aware model matches the receipt KIE task; CORD ships the word boxes and field labels needed for supervised fine-tuning.
2. **Data-derived BIO schema** — the label set comes from the dataset rather than a hand-maintained constant, so it can't drift from the data.
3. **Colab trains, laptop serves** — GPU fine-tuning happens on Colab; the registered artifact is pulled to the CPU laptop, matching the project's hardware constraint.
4. **MLflow model registry as the handoff** — later phases consume the model by registry name/version, decoupling them from training mechanics.

---

## 5. Deviations / Deferred

- **MinIO not used in practice** — the compose MLflow server uses a local artifact root (`/mlflow/artifacts`); the planned MinIO push was unnecessary for a single-host setup and deferred.
- **Cross-platform import bug fixed in place** — the import CLI crashed on Windows `cp1252` consoles (MLflow's 🏃 emoji) *after* registering the model; fixed by reconfiguring stdout to UTF-8 (`9046411`).

None of these block Phase 3.

---

## 6. Phase 2 Checklist (from `plan.md`)

- [x] LayoutLMv3 fine-tuning recipe for CORD (token classification).
- [x] Label schema mapping (CORD fields → BIO labels → output schema).
- [x] Colab notebook: dataset prep, fine-tune LayoutLMv3.
- [x] Log params, metrics, and artifacts to MLflow; register the chosen model.
- [x] Evaluate: KIE F1 on the CORD test split, logged to MLflow.
- [x] **Done when:** a fine-tuned LayoutLMv3 is registered in MLflow with F1 recorded and reproducible from the notebook.

---

## 7. Next: Phase 3 — Optimization: ONNX + INT8 + Benchmark

- Export the registered model to ONNX; INT8 quantize.
- Benchmark fp32 vs ONNX vs INT8 (F1 / latency / throughput / size).
- Produce a reproducible `docs/benchmark.md`.
