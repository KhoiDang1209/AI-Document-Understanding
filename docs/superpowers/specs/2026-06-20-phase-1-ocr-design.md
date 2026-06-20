# Phase 1 — OCR Baseline + `/extract` Design

**Status:** Approved (brainstorming) — ready for implementation planning
**Date:** 2026-06-20
**Phase:** 1 (CORE — the MLOps spine)
**Depends on:** Phase 0 (FastAPI scaffold, Settings, JSON logging, docker-compose `api`/`mlflow`/`minio`)

## Goal

The thinnest end-to-end slice of the pipeline: **an image goes in, raw OCR (`text`,
`boxes`, `confidence`) comes out** of `POST /extract`, served on CPU. Accuracy is *not*
the gate this phase (KIE F1 arrives in Phase 2); ease, CPU-friendliness, swappability, and
a clean test story are what matter.

**Done when:** `POST /extract` returns OCR results for a sample receipt image, and the OCR
engine is swappable via configuration.

## Decisions (locked in brainstorming)

1. **OCR engine = docTR** (`python-doctr`, PyTorch-CPU backend), behind an `OCREngine`
   interface so it stays swappable. Chosen over PaddleOCR (heavier/fiddlier on Windows) and
   Tesseract (already on the author's CV, weak on receipts). docTR's documented ONNX-export
   path aligns with the project's later optimization theme.
2. **Single `POST /extract` that matures.** Phase 1 returns `{text, words, confidence}`; in
   Phase 4 the same endpoint's response deepens to structured fields (OCR moves internal).
   One endpoint, matches `proposal.md`. No external consumers yet, so the Phase 4 response
   change is acceptable.
3. **Input contract = multipart upload, PNG/JPEG only**, with a configurable size cap.
   PDF and base64-JSON inputs are deferred (YAGNI) until a later phase needs them.
4. **Preprocessing = minimal module behind a toggle, default OFF.** A `preprocess()`
   (deskew + denoise + resize) exists and is unit-tested, but the default `/extract` path is
   docTR-native so the baseline stays clean and attributable.
5. **OCR runs docTR's native CPU inference** in Phase 1 — no custom ONNX export yet (ONNX is
   the Phase 3 KIE story; the `OCREngine` interface keeps an ONNX docTR backend a drop-in
   later).
6. **Word-level boxes in absolute pixel coordinates** (docTR's relative coords are converted
   using the decoded image dimensions).

## Architecture & Modules

OCR work lives under the existing `pipeline/` package (per `plan.md`'s structure). The
`kie/`, `validation/`, and `storage/` stubs are left untouched this phase.

```
docintel/src/docintel/
  pipeline/
    types.py        # Pydantic models: BBox, OCRWord, OCRResult (reused as the API response)
    preprocess.py   # preprocess(image, settings) -> image  (deskew + denoise + resize)
    ocr.py          # OCREngine Protocol + load_doctr_engine(settings) -> OCREngine
  api/
    routes/extract.py   # POST /extract route + the engine dependency
    main.py             # lifespan builds the docTR engine once, stores it on app.state
  config.py             # + ocr_engine, preprocess_enabled, max_upload_mb
```

### The `OCREngine` interface (functional)

Per the CLAUDE.md "functional over classes" standard, the engine is a callable, not a
class hierarchy:

```python
class OCREngine(Protocol):
    def __call__(self, image: np.ndarray) -> OCRResult: ...

def load_doctr_engine(settings: Settings) -> OCREngine:
    predictor = ocr_predictor(pretrained=True)   # loads weights ONCE
    def _run(image: np.ndarray) -> OCRResult:
        ...  # run predictor, map docTR output -> OCRResult
    return _run
```

- The heavy model is constructed **once at app startup** (FastAPI lifespan) and stored on
  `app.state.ocr_engine`; it is never rebuilt per request.
- Swapping engines later = a different factory function returning the same `OCREngine`
  shape (e.g. an ONNX docTR backend, PaddleOCR, or an LLM-OCR).
- A test stub is simply a plain function with the `OCREngine` shape — no model download.

## Data Flow — `POST /extract`

```
multipart file (UploadFile)
 → validate content-type ∈ {image/png, image/jpeg}        else 415 Unsupported Media Type
 → enforce size ≤ DOCINTEL_MAX_UPLOAD_MB                   else 413 Payload Too Large
 → decode bytes → np.ndarray (BGR/RGB)                     else 400 Bad Request (undecodable)
 → if settings.preprocess_enabled: image = preprocess(image, settings)   (default off)
 → result = engine(image)                                 (app-singleton docTR engine)
 → log {latency_ms, word_count, mean_confidence} (structured JSON)
 → 200 OCRResult
```

Missing file field is handled by FastAPI as **422 Unprocessable Entity** (standard
validation). Latency is logged on every request, including error paths where OCR ran.

## Response Contract

`OCRResult` (the `pipeline/types.py` Pydantic model) is returned directly as the `/extract`
response — one model, no duplication between domain and API layers.

```jsonc
{
  "text": "STARBUCKS\nTOTAL 12.99\n...",          // full concatenated document text
  "words": [
    { "text": "TOTAL", "bbox": [120, 880, 240, 922], "confidence": 0.98 }
  ],
  "confidence": 0.94,        // mean of word confidences (0.0 if no words found)
  "image_width": 1200,
  "image_height": 1600       // so pixel bboxes are interpretable
}
```

- `bbox` = `[x_min, y_min, x_max, y_max]` integer **pixels**, top-left origin.
- `words` is the plan's "boxes," at **word** granularity.
- `text` is built by joining recognized lines/words in reading order with newlines/spaces.
- Empty image (no text detected) → `words: []`, `text: ""`, `confidence: 0.0`, dims set.

## Configuration (no hardcoded constants)

Added to `Settings` (env prefix `DOCINTEL_`), all with defaults; mirrored into
`.env.example`:

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `ocr_engine` | `Literal["doctr"]` | `"doctr"` | selects the OCR backend (swappable) |
| `preprocess_enabled` | `bool` | `false` | toggles the OpenCV preprocessing step |
| `max_upload_mb` | `float` | `10` | maximum accepted upload size (megabytes) |

## Preprocessing Module

`preprocess(image: np.ndarray, settings: Settings) -> np.ndarray` — applied only when
`preprocess_enabled` is true. Steps:

1. **Deskew** — estimate the dominant text skew angle (min-area-rect over thresholded
   foreground) and rotate to correct it.
2. **Denoise** — `cv2.fastNlMeansDenoising` (grayscale) to reduce scan speckle.
3. **Resize** — cap the longest side (downscale only) to keep OCR latency bounded; preserve
   aspect ratio.

The function is pure (input ndarray → output ndarray), independently unit-testable, and has
no I/O or global state. When the toggle is off it is never called (the route passes the
decoded image straight to the engine).

## Error Handling

| Condition | Status | Notes |
|---|---|---|
| content-type not png/jpeg | 415 | checked before reading the whole body where possible |
| body larger than `max_upload_mb` | 413 | enforced on the read bytes |
| bytes not a decodable image | 400 | `cv2.imdecode` / PIL returns nothing |
| no `file` field | 422 | FastAPI request validation |

All are returned as typed FastAPI/JSON error responses (`{"detail": ...}`); none leak
stack traces. Latency for any request that reached OCR is logged.

## Testing Strategy

- **`tests/test_extract.py`** (fast, no model) — overrides the engine dependency with a
  stub `OCREngine` function and asserts:
  - 200 + correct response shape for a valid PNG/JPEG;
  - 415 for a non-image content-type;
  - 413 for an over-cap upload;
  - 400 for undecodable bytes;
  - 422 when the file field is missing.
- **`tests/test_preprocess.py`** (fast) — a synthetic skewed image asserts: deskew reduces
  the measured skew angle; resize caps the longest side; the function returns an
  `np.ndarray`; (and that the route leaves the image untouched when the toggle is off).
- **`tests/test_ocr_doctr.py`** (`@pytest.mark.slow`, may download weights) — PIL-generates
  an image containing known text (e.g. `TOTAL 12.99`), runs the **real** docTR engine, and
  asserts the recognized `text` contains that token (case-insensitive) with
  `confidence > 0`. Excluded from the default fast run via the `slow` marker; CI runs it in
  a separate, cache-backed step.

No dataset images are bundled (avoids licensing and network in unit tests); the slow test
generates its own deterministic fixture.

## Dependencies & Docker

New runtime dependencies (main, not dev):

- `python-doctr[torch]` — OCR engine (pulls `torch`, `torchvision` CPU wheels)
- `opencv-python-headless` — preprocessing + image decode (headless: no GUI libs for Docker)
- `numpy` — array interchange
- `Pillow` — image handling / test fixture generation
- `python-multipart` — required by FastAPI for multipart file uploads

The `slow` pytest marker is registered in `pyproject.toml` (`[tool.pytest.ini_options]`)
and excluded from the default run (e.g. `addopts = "-m 'not slow'"`).

**Docker:** torch-CPU is a large wheel, so the `api` image grows by a few hundred MB
(expected and acceptable for a CPU inference service). The Dockerfile adds a build step that
**pre-downloads the docTR model weights** (instantiating `ocr_predictor(pretrained=True)`
during build, caching under `~/.cache/doctr`) so the running container needs no network at
startup or first request.

## Out of Scope (Phase 1)

- PDF and base64-JSON inputs (deferred until needed).
- Custom ONNX export / quantization of the OCR engine (Phase 3 theme; interface keeps it a
  drop-in later).
- KIE, validation, persistence, `/documents/{id}` (Phases 2+ / 4).
- Returning latency in the response body (it is logged, per the plan's "latency logged").
- Batching, async model pools, GPU paths.

## Cross-Phase Notes

- The `OCREngine` factory pattern established here is the template for the Phase 4
  `KIEBackend` interface.
- Carried-forward Phase 0 item to honor when processed data first appears (Phase 1/2):
  `docintel/.gitignore` tracks `.dvc` pointers under `data/raw/` but not `data/processed/` —
  mirror that negation if/when this phase writes processed artifacts. (Phase 1 writes no
  processed data, so no action required now.)
