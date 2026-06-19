# Phase 0 Report — Foundations & Environment

**Status:** ✅ Complete and verified
**Location:** `docintel/`
**Date:** 2026-06-17

Phase 0 establishes the engineering foundation for DocIntel: a clean, typed, tested Python 3.12 package; a runnable FastAPI service with a health endpoint; a Docker Compose stack for the backing services; structured logging; env-driven configuration; quality tooling; a dataset-download command; and CI. See [`phase0.md`](phase0.md) for the phase brief and [`../../plan.md`](../../plan.md) for the full roadmap.

---

## 1. What Was Built

### Application package (`src/docintel/`)
- **`config.py`** — `Settings` via `pydantic-settings`, env-prefixed `DOCINTEL_`, `.env` support. No hardcoded constants; one cached accessor `get_settings()`.
- **`logging_config.py`** — dependency-free `JsonFormatter` + `configure_logging()`. Machine-parseable logs from day one (Loki-ready); promotes `extra={...}` fields; routes uvicorn loggers through the same handler.
- **`api/main.py`** — application factory `create_app()` + ASGI `app`. Lifespan configures logging and emits `service.startup` / `service.shutdown` events.
- **`api/routes/health.py`** — `GET /health` returning a typed `HealthResponse` (status, service, version, environment).
- **`pipeline/`, `kie/`, `rag/`, `validation/`, `storage/`** — package stubs with docstrings marking the phase each is implemented in.
- **`scripts/download_data.py`** — CLI (`docintel-download-data`) to fetch CORD / SROIE from the Hugging Face Hub into `data/raw/`. The `datasets` import is deferred so `--help` works without the optional extra.

### Tests (`tests/`)
- `conftest.py` — `client` fixture (FastAPI `TestClient`).
- `test_health.py` — `/health` contract + `/openapi.json` served.
- `test_config.py` — settings defaults + env override.

### Packaging & tooling
- **`pyproject.toml`** — hatchling build, src layout, runtime deps (FastAPI, uvicorn, pydantic, pydantic-settings) and optional groups `dev` (pytest, httpx, ruff, mypy, pre-commit) and `data` (datasets, huggingface-hub, Pillow, dvc). Configures ruff (lint+isort), mypy (strict), pytest. Exposes the `docintel-download-data` console script.
- **`Dockerfile`** — `python:3.12-slim`, CPU-only, layer-cached install, container `HEALTHCHECK`, uvicorn entrypoint.
- **`docker-compose.yml`** — `api`, `mlflow` (sqlite backend + volume), `qdrant`, `minio`; named volumes; pinned image tags.
- **`.pre-commit-config.yaml`** — ruff, ruff-format, mypy, standard hygiene hooks.
- **`.dockerignore`, `.gitignore`, `.env.example`, `README.md`.**
- **CI** — `.github/workflows/ci.yml` (repo root) runs install → ruff → format check → mypy → pytest on push/PR, scoped to `docintel/`.

---

## 2. Directory Layout

```
docintel/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── report_phase0.md
├── .pre-commit-config.yaml
├── .dockerignore / .gitignore / .env.example
├── src/docintel/
│   ├── __init__.py            (__version__)
│   ├── config.py
│   ├── logging_config.py
│   ├── api/{main.py, routes/health.py}
│   ├── pipeline/ kie/ rag/ validation/ storage/   (stubs)
│   └── scripts/download_data.py
├── tests/{conftest.py, test_health.py, test_config.py}
├── data/ notebooks/ infra/    (.gitkeep)
└── (.venv/  — local, git-ignored)
```

---

## 3. Verification

All commands run locally on Python 3.12.3 (Windows).

| Check | Command | Result |
|-------|---------|--------|
| Dependency install | `pip install -e ".[dev]"` | exit 0 |
| App import | `python -c "from docintel.api.main import app"` | `app import OK: DocIntel` |
| Lint | `ruff check .` | All checks passed |
| Format | `ruff format --check .` | 18 files already formatted |
| Type check | `mypy src` | Success: no issues found in 14 source files |
| Tests | `pytest` | **4 passed** |
| Compose syntax | `docker compose config` | exit 0 (valid) |

Reproduce:

```bash
cd docintel
python -m venv .venv && .venv\Scripts\Activate.ps1   # or: source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy src && pytest
```

Run the API:

```bash
uvicorn docintel.api.main:app --reload   # http://localhost:8000/health, docs at /docs
```

---

## 4. Key Decisions

1. **Project nested in `docintel/`** — the application is a self-contained package separate from the repo-root planning docs.
2. **`src/` layout + hatchling** — avoids import-shadowing, forces installed-package testing.
3. **Dependency-free JSON logging** — no logging library lock-in; structured output ready for Loki in Phase 8.
4. **Optional dependency groups** — `data` and `dev` tooling stay out of the runtime Docker image (smaller, faster build).
5. **`mypy --strict` from the start** — cheap to maintain now, expensive to retrofit later.
6. **Pinned compose image tags** — reproducible service stack.

---

## 5. Deviations / Deferred

- **`docker compose up` not executed** — compose file validated via `docker compose config`; images are not pulled/built here (heavy, network-bound). First real run happens when Phase 1 adds pipeline logic.
- **DVC not initialized** — `dvc` is not installed locally (it is in the `data` extra). DVC `init` belongs at the git repo root; deferred to when the first dataset is actually pulled. Tracking intent is recorded in `.gitignore` (`data/raw/`, `data/processed/`).
- **Datasets not downloaded** — `download_data.py` is implemented but not run (network + size). Executed at the start of Phase 1/3 when needed.

None of these block Phase 1.

---

## 6. Phase 0 Checklist (from `plan.md`)

- [x] Project structure (`api`, `pipeline`, `kie`, `rag`, `validation`, `storage`, `tests`, `notebooks`, `infra`, `data`)
- [x] Dependency management via `pyproject.toml`; Python 3.12 pinned
- [x] `Dockerfile` (CPU) + `docker-compose.yml` (api, mlflow, minio, qdrant)
- [x] FastAPI skeleton with `GET /health`
- [x] Config via Pydantic `BaseSettings`; structured JSON logging
- [x] Tooling: ruff + format + mypy + pre-commit; pytest running
- [x] Data download script (CORD/SROIE)
- [x] Basic CI workflow (lint, format, type, test)
- [~] DVC tracking — deferred (see §5)
- [x] **Done when:** `/health` works; lint/type/test pass locally and in CI

---

## 7. Next: Phase 1 — OCR Baseline + `/extract`

- Spike PaddleOCR vs docTR on CPU (accuracy, latency, ONNX support).
- Define `OCREngine` interface + first implementation.
- OpenCV preprocessing module.
- `POST /extract` returning OCR text + boxes + confidence.
- First real `docker compose up` end-to-end run.
