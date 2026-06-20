# Phase 0 — Foundations Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the existing Phase 0 scaffold to its "done when" gate — scope-aligned to the MLOps core, with DVC initialized, an `.env.example` present, and the `docker compose` stack verified to serve `/health` 200.

**Architecture:** The scaffold (FastAPI app, Pydantic settings, JSON logging, tests, CI, Dockerfile, compose) already exists and passes lint/type/test. This plan removes leftover Qdrant references (Qdrant is a GraphRAG *advancement*, not core), adds the missing developer/data-versioning artifacts, and proves the stack runs end-to-end. No new application features.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2 / pydantic-settings, pytest, ruff, mypy, Docker Compose (api + mlflow + minio), DVC.

## Global Constraints

- Python: `requires-python = ">=3.12"`; target version `py312`. (verbatim from `pyproject.toml`)
- Full type hints on all functions; `mypy` runs in `strict` mode over `src`. (verbatim from `pyproject.toml [tool.mypy]`)
- Prefer functional components over classes; keep functions small and focused. (CLAUDE.md Coding Standard)
- No hardcoded constants — configuration lives in settings/env. (CLAUDE.md Coding Standard)
- Minimal changes — touch only what the task requires; do not refactor unrelated code. (CLAUDE.md Minimal Changes)
- **Commit standard:** Never commit automatically. Before every commit, explain what changed, summarize affected files, and get the user's confirmation. The `git commit` step in each task below is gated on that confirmation. (CLAUDE.md Commit Standard)
- Core compose services are exactly `api`, `mlflow`, `minio`. Prometheus/Grafana/Loki arrive in Phase 5; Qdrant arrives with the GraphRAG advancement (A2). (`docs/proposal.md` §5, `docs/plan.md`)
- All shell paths below assume the worktree root `D:\AI Document Understanding\.claude\worktrees\phase-0-foundations`. Python tooling runs from the `docintel/` subdirectory via the local venv: `docintel\.venv\Scripts\<tool>.exe` (PowerShell).

---

### Task 1: Trim Qdrant from the core

Qdrant belongs to the GraphRAG advancement (A2), not the Phase 0 core. Remove it from the compose stack, settings, and docs, and lock the decision with a compose regression test.

**Files:**
- Create: `docintel/tests/test_compose.py`
- Modify: `docintel/pyproject.toml` (add `pyyaml` to the `dev` extra)
- Modify: `docintel/docker-compose.yml` (remove the `qdrant` service and its `depends_on`/`environment` references)
- Modify: `docintel/src/docintel/config.py` (remove the `qdrant_url` field)
- Modify: `docintel/src/docintel/api/routes/health.py:27-28` (drop "Qdrant" from the docstring)
- Modify: `docintel/README.md` (remove Qdrant + `/ask` from the core stack description)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `docintel/tests/test_compose.py::test_core_services` guards the service set `{"api", "mlflow", "minio"}` for all later tasks/phases.

- [ ] **Step 1: Add `pyyaml` to the dev extra**

In `docintel/pyproject.toml`, under `[project.optional-dependencies]`, add `pyyaml` to the `dev` list:

```toml
dev = [
    "pytest>=8.2",
    "httpx>=0.27",
    "ruff>=0.5",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "pyyaml>=6.0",
]
```

Then install it into the venv (PowerShell, from the worktree root):

```powershell
cd docintel; .\.venv\Scripts\python.exe -m pip install --quiet -e ".[dev]"
```

- [ ] **Step 2: Write the failing compose test**

Create `docintel/tests/test_compose.py`:

```python
"""Guards the core docker-compose service set against scope drift."""

from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker-compose.yml"

# Core = MLOps spine only. Qdrant (GraphRAG) and Prometheus/Grafana/Loki
# (observability) are deliberately out of the Phase 0 core stack.
EXPECTED_SERVICES = {"api", "mlflow", "minio"}


def test_core_services() -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert set(compose["services"]) == EXPECTED_SERVICES
```

- [ ] **Step 3: Run the test to verify it fails**

```powershell
cd docintel; .\.venv\Scripts\pytest.exe tests/test_compose.py -v
```

Expected: FAIL — `AssertionError` because `services` still contains `qdrant` (set is `{api, mlflow, qdrant, minio}`).

- [ ] **Step 4: Remove the `qdrant` service from compose**

In `docintel/docker-compose.yml`: delete the entire `qdrant:` service block, remove `- qdrant` from the `api` service's `depends_on`, and remove the `DOCINTEL_QDRANT_URL: http://qdrant:6333` line from the `api` service's `environment`. The `api` `depends_on` becomes:

```yaml
    depends_on:
      - mlflow
      - minio
```

and the `api` `environment` becomes:

```yaml
    environment:
      DOCINTEL_MLFLOW_TRACKING_URI: http://mlflow:5000
      DOCINTEL_MINIO_ENDPOINT: minio:9000
```

Also delete the `qdrant-data:` entry under the top-level `volumes:` key.

- [ ] **Step 5: Remove `qdrant_url` from settings**

In `docintel/src/docintel/config.py`, delete the line:

```python
    qdrant_url: str = "http://qdrant:6333"
```

- [ ] **Step 6: Drop Qdrant from the health docstring**

In `docintel/src/docintel/api/routes/health.py`, change the readiness note in the `health()` docstring from:

```python
    Readiness checks for backing services (MLflow, Qdrant, MinIO) are added in
    later phases as those dependencies come online.
```

to:

```python
    Readiness checks for backing services (MLflow, MinIO) are added in later
    phases as those dependencies come online.
```

- [ ] **Step 7: Update the README core stack description**

In `docintel/README.md`, in the "Service stack" section, replace the line:

```
Brings up the API (`:8000`), MLflow (`:5000`), Qdrant (`:6333`), and MinIO
(`:9000`, console `:9001`).
```

with:

```
Brings up the API (`:8000`), MLflow (`:5000`), and MinIO (`:9000`, console
`:9001`). Qdrant (GraphRAG) and the monitoring stack arrive in later phases.
```

- [ ] **Step 8: Run the full quality gate**

```powershell
cd docintel; .\.venv\Scripts\ruff.exe check . ; .\.venv\Scripts\ruff.exe format --check . ; .\.venv\Scripts\mypy.exe src ; .\.venv\Scripts\pytest.exe -q
```

Expected: ruff "All checks passed!", format clean, mypy "Success", pytest all passing (now 5 tests incl. `test_core_services`).

- [ ] **Step 9: Commit** (after user confirmation per Commit Standard)

```bash
git add docintel/pyproject.toml docintel/docker-compose.yml docintel/src/docintel/config.py docintel/src/docintel/api/routes/health.py docintel/README.md docintel/tests/test_compose.py
git commit -m "refactor(core): remove Qdrant from Phase 0 core stack

Qdrant belongs to the GraphRAG advancement (A2). Drop the service,
the qdrant_url setting, and doc references; add a compose test that
locks the core service set to {api, mlflow, minio}."
```

---

### Task 2: Add `.env.example`

The README instructs `cp .env.example .env`, but the file does not exist. Add it, derived from `Settings`, and guard it against drift.

**Files:**
- Create: `docintel/.env.example`
- Create: `docintel/tests/test_env_example.py`

**Interfaces:**
- Consumes: `docintel.config.Settings` (fields define the valid env keys); the `env_prefix="DOCINTEL_"` from `Settings.model_config`.
- Produces: `docintel/.env.example` consumed by `docker-compose.yml`'s `env_file: - .env` and by local developers.

- [ ] **Step 1: Write the failing drift test**

Create `docintel/tests/test_env_example.py`:

```python
"""Ensures .env.example keys map to real Settings fields (no drift)."""

from __future__ import annotations

from pathlib import Path

from docintel.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"
PREFIX = "DOCINTEL_"


def _env_keys() -> list[str]:
    keys: list[str] = []
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.append(line.split("=", 1)[0].strip())
    return keys


def test_env_example_exists() -> None:
    assert ENV_EXAMPLE.is_file()


def test_env_keys_are_valid_settings_fields() -> None:
    valid = set(Settings.model_fields)
    for key in _env_keys():
        assert key.startswith(PREFIX), f"{key} missing {PREFIX} prefix"
        field = key[len(PREFIX) :].lower()
        assert field in valid, f"{key} is not a Settings field"
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd docintel; .\.venv\Scripts\pytest.exe tests/test_env_example.py -v
```

Expected: FAIL — `test_env_example_exists` fails because `.env.example` does not exist.

- [ ] **Step 3: Create `.env.example`**

Create `docintel/.env.example` (values mirror the compose defaults; secrets are placeholders):

```dotenv
# Copy to .env and adjust. All keys use the DOCINTEL_ prefix.
DOCINTEL_ENVIRONMENT=local
DOCINTEL_LOG_LEVEL=INFO
DOCINTEL_HOST=0.0.0.0
DOCINTEL_PORT=8000

# Backing services (defaults target the docker-compose network)
DOCINTEL_MLFLOW_TRACKING_URI=http://mlflow:5000
DOCINTEL_MINIO_ENDPOINT=minio:9000
DOCINTEL_MINIO_ACCESS_KEY=minioadmin
DOCINTEL_MINIO_SECRET_KEY=minioadmin

# Filesystem
DOCINTEL_DATA_DIR=data
```

- [ ] **Step 4: Run the test to verify it passes**

```powershell
cd docintel; .\.venv\Scripts\pytest.exe tests/test_env_example.py -v
```

Expected: PASS — both tests green. (`.env` itself stays gitignored; `.env.example` is tracked.)

- [ ] **Step 5: Commit** (after user confirmation per Commit Standard)

```bash
git add docintel/.env.example docintel/tests/test_env_example.py
git commit -m "feat(config): add .env.example with Settings-drift guard"
```

---

### Task 3: Initialize DVC for data versioning

`plan.md` Phase 0 calls for "DVC init + tracking." Initialize DVC, configure a local remote, and make the `.dvc` pointer-file workflow work despite the blanket `data/` gitignore. The actual dataset download (multi-hundred-MB) is documented, not executed here.

**Files:**
- Create: `.dvc/` (via `dvc init`), `.dvcignore`
- Create: `docintel/data/.gitkeep`
- Modify: `.gitignore` (allow `.dvc` pointer files and `.gitkeep` under `data/`)
- Modify: `docintel/README.md` (document the DVC data workflow)

**Interfaces:**
- Consumes: `docintel-download-data` CLI (from `docintel/src/docintel/scripts/download_data.py`) which writes to `<data_dir>/raw/<dataset>`.
- Produces: a configured DVC repo with a `localremote`; the documented `dvc add` workflow for later phases.

- [ ] **Step 1: Confirm DVC is available**

DVC ships in the `data` extra. Install and check (PowerShell, from worktree root):

```powershell
cd docintel; .\.venv\Scripts\python.exe -m pip install --quiet -e ".[data]"; .\.venv\Scripts\dvc.exe --version
```

Expected: a version string (e.g. `3.x.y`).

- [ ] **Step 2: Initialize DVC at the worktree root**

DVC must live at the git root (the worktree root), not in `docintel/`. Run from the worktree root:

```powershell
.\docintel\.venv\Scripts\dvc.exe init
```

Expected: creates `.dvc/` and `.dvcignore`; prints "Initialized DVC repository."

- [ ] **Step 3: Configure a local DVC remote**

```powershell
.\docintel\.venv\Scripts\dvc.exe remote add -d localremote .dvc/storage
```

Expected: `.dvc/config` now contains a default remote named `localremote`.

- [ ] **Step 4: Allow `.dvc` pointer files through gitignore**

The root `.gitignore` ignores `data/` wholesale, which would also hide DVC's tracked pointer files. Append negation rules so pointers and the keep-file are versioned while raw data stays ignored. Add to the end of `.gitignore`:

```gitignore
# DVC: track pointer files and keep-files, not the raw data they point to
!docintel/data/
docintel/data/*
!docintel/data/.gitkeep
!docintel/data/**/
!docintel/data/**/*.dvc
```

Then create the keep-file:

```powershell
New-Item -ItemType File docintel\data\.gitkeep
```

- [ ] **Step 5: Verify the gitignore negation works**

```powershell
cd "D:\AI Document Understanding\.claude\worktrees\phase-0-foundations"; git check-ignore docintel/data/.gitkeep; echo "exit: $LASTEXITCODE"
```

Expected: prints nothing and `exit: 1` (i.e. `.gitkeep` is NOT ignored). Also confirm a sample pointer path is tracked:

```powershell
git check-ignore docintel/data/raw/cord.dvc; echo "exit: $LASTEXITCODE"
```

Expected: `exit: 1` (a future `cord.dvc` pointer would be tracked).

- [ ] **Step 6: Document the data workflow in the README**

In `docintel/README.md`, replace the "Datasets (build-time)" section body with:

````markdown
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
git add docintel/data/raw/cord.dvc docintel/data/raw/.gitignore
```
````

- [ ] **Step 7: Run the quality gate (no code changed, but confirm green)**

```powershell
cd docintel; .\.venv\Scripts\pytest.exe -q
```

Expected: all tests pass (DVC adds no Python code; this confirms nothing broke).

- [ ] **Step 8: Commit** (after user confirmation per Commit Standard)

```bash
git add .dvc .dvcignore .gitignore docintel/data/.gitkeep docintel/README.md
git commit -m "chore(data): initialize DVC with local remote and pointer tracking"
```

---

### Task 4: Verify the stack end-to-end (Phase 0 "done when" gate)

Prove `docker compose up` brings the core stack online and `/health` returns 200. This is the Phase 0 completion gate; it requires Docker Desktop running.

**Files:** none (verification only).

**Interfaces:**
- Consumes: `docintel/docker-compose.yml` (now `api`, `mlflow`, `minio`), the `api` `Dockerfile`, and the `GET /health` route.
- Produces: a confirmed-runnable core stack — the precondition for Phase 1.

- [ ] **Step 1: Validate the compose file statically**

```powershell
cd docintel; docker compose config --quiet; echo "exit: $LASTEXITCODE"
```

Expected: `exit: 0` and no error output (compose syntax + service references resolve).

- [ ] **Step 2: Prepare the env file**

```powershell
cd docintel; Copy-Item .env.example .env -Force
```

Expected: `.env` created (gitignored).

- [ ] **Step 3: Build and start the stack**

```powershell
cd docintel; docker compose up -d --build
```

Expected: `api`, `mlflow`, and `minio` containers reach "Started"/healthy. First run pulls the `mlflow` and `minio` images and builds the `api` image.

- [ ] **Step 4: Verify `/health` returns 200**

Allow a few seconds for startup, then:

```powershell
(Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing).StatusCode
```

Expected: `200`. Inspect the body to confirm identity:

```powershell
(Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing).Content
```

Expected: JSON containing `"status":"ok"`, `"service":"DocIntel"`.

- [ ] **Step 5: Tear the stack down**

```powershell
cd docintel; docker compose down
```

Expected: all three containers removed; named volumes (`mlflow-data`, `minio-data`) persist.

- [ ] **Step 6: No commit**

This task changes no tracked files (`.env` is gitignored). Nothing to commit. Record the verified result (status code + container list) in the Phase 0 report if one is maintained (`docs/phases/phase0/report_phase0.md`).

---

## Notes / Out of scope

- The empty `docintel/src/docintel/rag/` package is a leftover from the pre-refactor scope (RAG is advancement A2). It is harmless and left untouched here to keep the diff minimal; it returns to use in A2.
- Actual dataset download + `dvc add`/`dvc push` of CORD/SROIE is documented in Task 3 but not executed (large download); it is exercised naturally in Phase 1/2 when the data is first needed.
- Adding Prometheus/Grafana/Loki to compose is Phase 5, not here.
