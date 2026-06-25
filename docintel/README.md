# DocIntel (application)

Implementation of the DocIntel Document AI system. See the repository-root docs
for context: `../README.md`, `../proposal.md`, `../plan.md`.

## Requirements

- Python 3.12+
- Docker + Docker Compose (for the service stack)

## Local development

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

ruff check .            # lint
ruff format --check .   # format check
mypy src                # type check
pytest                  # tests

# Run the API
uvicorn docintel.api.main:app --reload
# -> http://localhost:8000/health   |   docs at /docs
```

## Service stack

```bash
cp .env.example .env
docker compose up --build
```

Brings up the API (`:8000`), MLflow (`:5000`), and MinIO (`:9000`, console
`:9001`). Qdrant (GraphRAG) and the monitoring stack arrive in later phases.

## Datasets (build-time)

```bash
pip install -e ".[data]"
docintel-download-data --dataset cord    # or: sroie  -> data/raw/<dataset>
```

Track the downloaded data with DVC (pointers are versioned in git, the data
itself goes to the DVC remote):

```bash
dvc add docintel/data/raw/cord       # creates docintel/data/raw/cord.dvc
dvc push                             # pushes to the configured local remote
git add docintel/data/raw/cord.dvc   # raw data stays ignored by data/raw/** rules
```

## Layout

```
src/docintel/
  api/         FastAPI app + routes
  pipeline/    preprocess -> layout -> OCR -> KIE -> validation (Phase 1+)
  kie/         Key Information Extraction backends (Phase 3+)
  contracts/   Contract Intelligence extraction (C1+)
  rag/         Retrieval-Augmented Generation (Phase 6+)
  validation/  schema + rule engine (Phase 4+)
  storage/     metadata + artifact persistence (Phase 4+)
  scripts/     operational scripts (data download, ...)
  config.py    settings (env-driven)
  logging_config.py  structured JSON logging
tests/         unit/integration tests
infra/         Kubernetes manifests, monitoring config (later phases)
notebooks/     Colab training/optimization notebooks (build-time)
```

## Contract Intelligence (C1)

The C1 pipeline extracts legal clauses from contract PDFs using a fine-tuned
DeBERTa-v3-base extractive-QA model (CUAD, 41 clause types) served as a
dynamic-INT8 ONNX model on CPU.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/contracts/extract` | Upload a contract PDF; returns a `ContractDocument` with extracted clauses |
| `GET` | `/contracts/{id}` | Retrieve a previously extracted `ContractDocument` by id |

### Dual-path ingestion

`POST /contracts/extract` automatically selects the ingestion path:

- **Digital** — PDF has an extractable text layer (PyMuPDF `get_text`). Fast, no OCR required.
- **OCR** — PDF is scanned or image-only; pages are rasterized and run through the docTR OCR engine.

The `ContractDocument.source` field records which path was used (`"digital"` or `"ocr"`).

### 41 clause types

The model answers one CUAD-style question per clause category, covering: Document Name,
Parties, Agreement Date, Effective/Expiration Date, Renewal Term, Governing Law,
Non-Compete, Exclusivity, IP Ownership, License Grant, Cap on Liability, Warranty
Duration, Insurance, and 28 more — see `src/docintel/contracts/questions.py` for the
full list.

### Local model override

To bypass the MLflow registry and load a bundle from disk, set:

```bash
DOCINTEL_CONTRACT_ONNX_LOCAL_PATH=/path/to/cuad-extractor-onnx-int8
```

This mirrors the `DOCINTEL_KIE_ONNX_LOCAL_PATH` escape hatch for the KIE model and is
the recommended path for laptop CPU serving (no MLflow / MinIO required at startup).

### Build-time notebooks (Colab GPU)

| Notebook | Purpose |
|----------|---------|
| `notebooks/cuad_finetune.ipynb` | Fine-tune DeBERTa-v3-base on CUAD; register `cuad-extractor` |
| `notebooks/cuad_onnx_export.ipynb` | Export fp32 → INT8 ONNX; eval F1/ANLS/AUPR/CER; register `cuad-extractor-onnx-int8` |
