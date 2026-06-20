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
