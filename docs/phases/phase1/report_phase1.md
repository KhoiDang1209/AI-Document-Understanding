# Phase 1 Report — OCR Baseline + `/extract`

**Status:** ✅ Complete and verified
**Location:** `docintel/`
**Date:** 2026-06-18

Phase 1 delivers the thinnest end-to-end slice of the pipeline: an image goes in over HTTP and raw OCR text, per-word boxes, and confidence come out. It introduces a swappable `OCREngine` interface with a pretrained docTR backend, optional OpenCV preprocessing, and the `POST /extract` endpoint. See [`phase1.md`](phase1.md) for the phase brief and [`../../plan.md`](../../plan.md) for the full roadmap.

---

## 1. What Was Built

### Pipeline package (`src/docintel/pipeline/`)
- **`types.py`** — Pydantic domain models `OCRWord` (text, pixel `bbox` `[x_min, y_min, x_max, y_max]`, confidence) and `OCRResult` (text, words, mean confidence, image dimensions). These models are returned directly as the `/extract` response, so the pipeline contract and the API contract are one and the same.
- **`ocr.py`** — `OCREngine` `Protocol` (a callable `Image -> OCRResult`) plus `load_doctr_engine()`, which loads docTR's pretrained `ocr_predictor` once and maps its `Document.export()` into an `OCRResult`. The engine is built behind a factory so a future engine can be swapped in via config without touching callers.
- **`preprocess.py`** — optional OpenCV preprocessing (deskew, denoise, resize) behind the `preprocess_enabled` toggle, capped by `preprocess_max_dim`.

### API (`src/docintel/api/routes/extract.py`)
- **`POST /extract`** — accepts a PNG/JPEG `UploadFile`, validates content type (415 on unsupported), enforces `max_upload_mb` (413 on oversize), decodes via OpenCV (400 on undecodable bytes), optionally preprocesses, runs OCR, and logs latency / word count / mean confidence as structured fields.
- **Lazy engine loading** — `get_ocr_engine` loads the docTR predictor once and caches it on `app.state`, so model weights are read on first request rather than at import time.

### Configuration (`src/docintel/config.py`)
- `ocr_engine: Literal["doctr"] = "doctr"` — engine selector (one backend today, interface ready for more).
- `preprocess_enabled: bool = False`, `preprocess_max_dim: int = 2000`, `max_upload_mb: float = 10.0`.

### Tests (`tests/`)
- `test_types.py` — OCR domain-model contracts.
- `test_ocr.py` — engine-mapping logic (`export` dict → `OCRResult`) without loading real weights.
- `test_ocr_doctr.py` — docTR backend (slow-marked; exercises the real predictor).
- `test_preprocess.py` — deskew/denoise/resize behavior and the enabled/disabled toggle.
- `test_extract.py` — `/extract` happy path plus 415 / 413 / 400 error contracts, using a stub engine via dependency override.

### Packaging & Docker
- Added OCR runtime deps and a `slow` pytest marker (`pyproject.toml`).
- `Dockerfile` installs CPU-only torch and bakes the docTR weights into the image so the first request doesn't pull them over the network.

---

## 2. Verification

All commands run locally on Python 3.12 (Windows, CPU).

| Check | Command | Result |
|-------|---------|--------|
| Lint | `ruff check .` | All checks passed |
| Format | `ruff format --check .` | clean |
| Type check | `mypy src` | Success: no issues |
| Tests (fast) | `pytest -m "not slow"` | pass |
| Tests (incl. docTR) | `pytest` | pass (docTR weights loaded) |
| Endpoint | `POST /extract` with a sample receipt | returns text + word boxes + confidence |

---

## 3. Key Decisions

1. **docTR over PaddleOCR** — cleaner Python packaging, straightforward CPU install, and a pretrained predictor that needs no extra config to produce competitive receipt OCR.
2. **`OCREngine` as a `Protocol`, not a base class** — keeps the project's functional style; an engine is just a callable, swappable via `ocr_engine` config.
3. **Pipeline models *are* the API models** — one Pydantic contract for both, avoiding a redundant serialization layer at this stage.
4. **Preprocessing off by default** — docTR handles raw images well; the deskew/denoise/resize path is opt-in (`preprocess_enabled`) so it can be measured before being switched on.
5. **Lazy, cached engine on `app.state`** — fast import and test collection; weights load once on first request.

---

## 4. Deviations / Deferred

- **Single engine implemented** — the interface is engine-agnostic but only docTR ships; PaddleOCR was evaluated and not adopted.
- **OCR-quality metrics (CER/WER) not formalized** — Phase 1 proves the slice works end to end; quantitative accuracy is measured against the KIE task in later phases.

None of these block Phase 2.

---

## 5. Phase 1 Checklist (from `plan.md`)

- [x] PaddleOCR vs docTR on CPU — docTR chosen.
- [x] Image input contract (PNG/JPEG upload; `max_upload_mb` size limit).
- [x] `OCREngine` interface + pretrained docTR implementation.
- [x] OpenCV preprocessing (deskew, denoise, resize) behind a toggle.
- [x] `POST /extract` → `{text, words, confidence, image_width, image_height}` (typed response model).
- [x] Unit tests on sample images; latency logged.
- [x] **Done when:** `/extract` returns OCR results for a sample receipt; engine swappable via config.

---

## 6. Next: Phase 2 — KIE Fine-tune (LayoutLMv3) + MLflow

- Fine-tune LayoutLMv3 on CORD (token classification) on Colab GPU.
- Track params/metrics/artifacts in MLflow and register the chosen model.
- Report KIE F1 per field on the CORD test split.
