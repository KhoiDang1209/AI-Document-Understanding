# Phase 2 — KIE Fine-tune (LayoutLMv3) + MLflow Design

**Status:** Approved (brainstorming) — ready for implementation planning
**Date:** 2026-06-21
**Phase:** 2 (CORE — the MLOps spine)
**Depends on:** Phase 0 (Settings, MLflow + MinIO docker-compose services, `download_data.py`),
Phase 1 (the `pipeline/` package, established functional/typed/tested conventions)

## Goal

The flagship build-time step: **fine-tune LayoutLMv3 on CORD for key-information
extraction (KIE), and make the run reproducible and tracked.** The deliverable is a
*fine-tuned model registered in the local MLflow with an F1 number*, reproducible from a
committed Colab notebook — **not** a served endpoint (serving arrives in Phase 4).

**Done when:** a fine-tuned `layoutlmv3-base` is registered in the local docker-compose
MLflow (artifacts in MinIO), with overall + per-field **F1 logged**, the whole run is
reproducible from the committed notebook, and a CPU smoke test confirms the registered
model loads and predicts.

## The Hardware Reality (why this phase has an unusual shape)

The laptop has **no GPU**; training runs on **Colab GPU**. MLflow and MinIO live in
**local docker-compose** on the laptop, which is **not publicly reachable**, so Colab
cannot push to `mlflow:5000` / `minio:9000` during a run. This phase is therefore
**human-in-the-loop and split across the Colab/laptop boundary** — a deliberate
train-on-Colab / serve-on-laptop split, made into an explicit, scripted MLOps handoff.

## Decisions (locked in brainstorming)

1. **Task = token classification on the full CORD schema.** Use CORD's native ~30-class
   field set as-is (BIO-tagged). It is the published benchmark, so our F1 is directly
   comparable to the LayoutLMv3 literature (~96% F1) — a legible CV claim — and the dataset
   already carries these labels (no manual relabeling). Mapping these fields to a cleaner
   canonical output JSON is deferred to Phase 4.
2. **Model = `microsoft/layoutlmv3-base`** (not `-large`). Fits Colab GPU memory, and the
   Phase 4 serving target is **CPU**, where base is far more viable and Phase 3 quantizes
   it. Large buys ~1 F1 point at multiples of the cost.
3. **Tracking/registry bridge = Colab file-store + a committed laptop-side import script.**
   Colab logs params/metrics/model to its **own local MLflow file-store** (`./mlruns` on
   the Colab VM) and saves a self-contained model bundle. The user downloads the bundle; a
   committed laptop-side import script (`docintel-import-kie`) ingests it into the local
   docker-compose MLflow + MinIO and **registers** the model. No third-party account; the
   local registry stays the single source of truth that Phase 4 serves from.
4. **Thin notebook + logic in `src/docintel/kie/`.** The real logic — label schema, CORD →
   features/tokenization with box+label alignment, the seqeval metric wrapper, the training
   builder, and the import/register script — lives as typed, unit-tested functions in
   `src/docintel/kie/`. The Colab notebook `pip install`s the repo (from the Phase 2 branch)
   and only orchestrates. Pure pieces are CPU-testable on the laptop; only the GPU train
   loop runs on Colab.
5. **No serving / no API work this phase.** No `/extract` change, no endpoint to expose
   (so no ngrok). The "test" is the F1 metric plus a CPU load-and-predict smoke check.
6. **Training hyperparameters live in a typed `TrainingConfig` dataclass** (named defaults,
   overridable from the notebook) — not scattered literals. Service-relevant values
   (`kie_model_name`, `kie_registered_model_name`) go into `Settings` because the Phase 4
   serving side needs them.

## Workflow Shape (human-in-the-loop)

```
1. I build (laptop, CPU)      src/docintel/kie/* + unit tests + import script + the notebook
                              file.  Built and CPU-tested via subagent-driven development.
2. You run (Colab, GPU)       open notebook → pip install repo@branch → load CORD →
                              build features → fine-tune → log to ./mlruns (file-store) →
                              evaluate F1 → zip a model bundle → download it.
3. We close the loop          run `docintel-import-kie <bundle>` → ingest into local MLflow
   (laptop, CPU)              + MinIO + register; CPU smoke test loads + predicts.
```

Implication for cadence: subagent-driven development covers the **buildable-on-laptop**
artifacts (steps 1). The Colab run (step 2) is the user's manual step. Step 3 is run
together. The branch is merged once F1 is logged, the model is registered, and the run is
reproducible.

## Architecture & Modules

KIE work lives under the existing `kie/` package (currently an empty stub). The `pipeline/`,
`validation/`, and `storage/` packages are left untouched this phase.

```
docintel/src/docintel/
  kie/
    labels.py       # CORD label schema: field list, BIO tags, id2label/label2id
    config.py       # TrainingConfig dataclass (epochs, lr, batch size, seed, ...)
    dataset.py      # CORD example -> {words, boxes(0-1000), ner_tags}; processor tokenize + label align
    metrics.py      # seqeval entity-level F1: overall + per-field
    train.py        # build HF Trainer, fine-tune, log to MLflow, save bundle (Colab-invoked)
    import_run.py   # laptop-side: bundle -> local MLflow + MinIO + Model Registry
  scripts/
    (entry point)   # docintel-import-kie -> kie.import_run:main
  config.py         # + kie_model_name, kie_registered_model_name
docintel/notebooks/
  phase2_kie_layoutlmv3.ipynb   # thin Colab orchestrator
```

### `kie/labels.py` — the label contract

A fixed, code-level schema (not env config): the canonical list of CORD field names, the
derived BIO tag list (`O`, `B-<field>`, `I-<field>`), and the `id2label` / `label2id`
mappings. Pure data + small helpers. This is the contract Phase 3 (ONNX export) and Phase 4
(serving) reuse, so it is the single definition of "what the model predicts."

### `kie/dataset.py` — CORD → LayoutLMv3 features (functional)

Pure functions that convert a CORD example into model features:

- extract `words` and word-level `boxes` from the CORD annotation;
- normalize boxes to LayoutLMv3's `0–1000` integer coordinate space using the image size;
- assign each word its BIO `ner_tag` id;
- tokenize with `LayoutLMv3Processor`, **aligning labels to subword tokens** — label only
  the first subword of each word, `-100` for continuation subwords and special tokens.

No I/O or global state; unit-tested on a tiny synthetic CORD-shaped fixture (CPU).

> **Research risk (first plan task):** `naver-clova-ix/cord-v2` (what `download_data.py`
> fetches) is in **Donut format** (image + a `ground_truth` JSON), not pre-tagged
> token-classification format. Its `valid_line` entries *do* carry word-level quad boxes +
> field labels, so `words/boxes/ner_tags` are derivable — **but the exact parse needs
> care**, and a cleaner pre-formatted CORD variant may exist on HF. Confirming the dataset
> source and writing/validating the converter (against a small fixture) is the first task in
> the implementation plan; the rest of the modules depend on its output shape.

### `kie/metrics.py` — evaluation

A `compute_metrics`-style function wrapping `seqeval` for **entity-level** precision /
recall / F1, returning both the overall scores and **per-field** F1 (the benchmark
numbers). Unit-tested on toy predicted/gold sequences with a known expected F1.

### `kie/config.py` — `TrainingConfig`

A typed dataclass holding hyperparameters with sensible defaults (epochs, learning rate,
train/eval batch size, weight decay, warmup, seed, eval/save strategy). The notebook may
override fields. Keeps "no hardcoded constants" satisfied without abusing env vars for
experiment knobs.

### `kie/train.py` — the training builder (Colab-invoked)

Builds the Hugging Face `Trainer` from a `TrainingConfig`, runs the fine-tune, logs params
+ metrics + the model artifact to MLflow (the Colab file-store), records reproducibility
params (git SHA, dataset revision, seed), evaluates on the CORD **test** split, and saves a
**self-contained bundle**: model weights + `LayoutLMv3Processor` + label maps + an `MLmodel`
descriptor + a `metrics.json`. The function is import-safe on CPU (heavy libs imported
inside it); the actual training executes on Colab.

### `kie/import_run.py` — the laptop-side handoff

`docintel-import-kie <bundle-dir>`: reads the downloaded bundle and, using the tracking URI
+ MinIO creds from `Settings`, creates/locates an MLflow experiment, logs the run's params
+ metrics, uploads the model artifacts (to MinIO via MLflow's artifact store), and
**registers** the model under `settings.kie_registered_model_name`. Deterministic; CPU;
unit-tested against a temporary MLflow file-store (no live MinIO needed in the test).

## Configuration (no hardcoded constants)

Added to `Settings` (env prefix `DOCINTEL_`), with defaults, mirrored into `.env.example`:

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `kie_model_name` | `str` | `microsoft/layoutlmv3-base` | base checkpoint to fine-tune / load |
| `kie_registered_model_name` | `str` | `cord-layoutlmv3` | MLflow Model Registry name |

Training hyperparameters live in `TrainingConfig` (code), not env — they are experiment
config, not service config.

## Testing Strategy (all on the laptop, CPU)

- **`tests/test_kie_labels.py`** — label list integrity (no dups, BIO well-formed),
  `id2label`/`label2id` round-trip.
- **`tests/test_kie_dataset.py`** — a synthetic CORD-shaped example yields the expected
  `words`, `0–1000`-normalized `boxes`, and `ner_tags`; subword alignment puts `-100` on
  continuation subwords and special tokens.
- **`tests/test_kie_metrics.py`** — toy predicted vs gold tag sequences produce a known
  overall and per-field F1.
- **`tests/test_kie_import.py`** — `import_run` logs params/metrics and registers a model
  against a temporary MLflow file-store (artifact store pointed at a temp dir; no live
  MinIO).

The **real fine-tune and the real-CORD parse** are exercised by the Colab notebook run
itself (the phase's integration test). Anything that downloads weights/data or needs a GPU
is out of the fast unit suite; if a CPU-only "tiny real data" check is added it carries the
existing `slow` marker.

## Dependencies

Two distinct needs, because the work spans the Colab/laptop boundary:

**Colab `train` extra** (optional group; used by the notebook on Colab, and by any
CPU unit test light enough to exercise the training/feature code):

- `transformers` — LayoutLMv3 model + `Trainer` + processor
- `datasets` — CORD loading (already used by `download_data.py`'s `data` extra)
- `seqeval` — entity-level F1
- `Pillow` — image handling for box normalization (already present from Phase 1)

**Laptop side** (needed to run `docintel-import-kie` and its unit test — not part of the
Colab training extra): the **`mlflow` client** plus an S3-compatible artifact dependency
(e.g. `boto3`) so MLflow can push artifacts to MinIO. Exact group placement (a small
`kie` extra vs. a base dependency) is resolved in the plan; the constraint is that
`test_kie_import.py` can run on the laptop without the heavy `train` deps.

`transformers`/`torch` are heavy; the unit tests that need them are kept minimal and may be
marked `slow` if import cost is significant. The CPU serving image is **not** changed this
phase.

## Reproducibility

- Fixed `seed` in `TrainingConfig`; set across `random`/`numpy`/`torch`.
- The notebook pins its install (repo at the Phase 2 branch + pinned `train` deps) and
  records the **git SHA** and **dataset revision** as MLflow params.
- CORD is obtained through the existing `download_data.py` path (HF `naver-clova-ix/cord-v2`),
  with the revision pinned in the notebook.

## Out of Scope (Phase 2)

- Serving / `/extract` integration / any API or endpoint change (Phase 4).
- ngrok / exposing endpoints (nothing to serve yet).
- ONNX export + INT8 quantization (Phase 3).
- Canonical output JSON schema + validation + persistence + `/documents/{id}` (Phase 4).
- The full LLM-KIE backend (advancement A5) and layout detection (advancement A1).
- Any GPU usage on the laptop; any change to the CPU serving Docker image.

## Cross-Phase Notes

- `kie/labels.py` (the label contract) and `kie/dataset.py` (feature shape) are reused by
  Phase 3 (ONNX export of this exact model) and Phase 4 (serving + mapping fields to the
  canonical schema).
- The `import_run.py` handoff establishes the pattern Phase 4 relies on: the served model is
  pulled from the **local** MLflow registry / MinIO, populated by this phase.
- Carried-forward Phase 1 `.gitignore` note: if this phase writes processed artifacts under
  `data/processed/`, mirror the `data/raw/` DVC-pointer negation. (Phase 2 itself writes no
  processed dataset to the repo — the bundle is produced on Colab and imported, not
  committed — so no action is required now.)
```
