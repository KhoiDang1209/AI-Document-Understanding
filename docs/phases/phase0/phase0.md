# Phase 0 — Foundations & Environment

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **✅ Complete** — see [`report_phase0.md`](report_phase0.md).

**Goal:** A clean repository that runs `docker compose up`, exposes a stub API, and has data and experiment tracking wired in.

## Research 🔬
- [ ] Confirm CORD + SROIE access, license, and format (HF `datasets` vs raw download).
- [ ] Decide dependency tooling (uv / poetry / pip-tools) for Python 3.12.
- [ ] Finalize repository layout.

## Tasks
- [ ] Project structure: `src/docintel/{pipeline,kie,rag,api,validation,storage}`, `tests/`, `notebooks/`, `infra/`, `data/`.
- [ ] Dependency management via `pyproject.toml`; pin Python 3.12.
- [ ] `Dockerfile` (CPU base) + `docker-compose.yml`: `api`, `mlflow`, `minio`, `qdrant`.
- [ ] FastAPI skeleton with `GET /health`.
- [ ] Configuration via Pydantic `BaseSettings`; structured JSON logging.
- [ ] Tooling: ruff + format + mypy + pre-commit; `pytest` running.
- [ ] Data download script (CORD/SROIE) → `data/raw/`; DVC init + tracking.

## Done when 📦
- [ ] `docker compose up` starts the service stack; `/health` returns 200.
- [ ] Lint, type-check, and tests pass locally and in a basic CI workflow.

## Outcome
Implemented under [`docintel/`](../../../docintel/). Full write-up, verification table, decisions, and deferred items in [`report_phase0.md`](report_phase0.md).
