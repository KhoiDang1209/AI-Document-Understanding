# docintel (application)

The DocIntel application: FastAPI service, pipeline, and model code. See the
[repository root README](../README.md) for the full project overview, and
[`../docs/`](../docs/) for architecture, the runbook, and per-phase reports.

## Local development

```bash
uv sync --all-extras          # full env (uv sync replaces the env; sync ALL extras for the test suite)
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest                 # one slow real-OCR test is deselected by default

uvicorn docintel.api.main:app --reload
# -> http://localhost:8000/health   |   docs at /docs
```

## Service stack

```bash
cp .env.example .env
docker compose up -d --build
```

See [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md) for the full stand-up + demo walkthrough, including pointing the API at local model bundles and enabling the optional LLM/tracing.

## Datasets (build-time)

```bash
uv sync --extra data
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
  pipeline/    preprocess -> layout -> OCR -> KIE -> validation
  kie/         key-information extraction (LayoutLMv3, ONNX-INT8)
  contracts/   contract clause extraction (CUAD, DeBERTa-v3 QA)
  rag/         hybrid retrieval + generate-or-degrade
  graph/       Neo4j store, Cypher templates, router
  agent/       LangGraph orchestration
  ui/          Streamlit demo (Extract / Ask / Agent / Graph / Metrics)
  validation/  schema + rule engine
  storage/     metadata + artifact persistence
  scripts/     eval scripts, data download
  config.py    settings (env-driven)
  logging_config.py  structured JSON logging
tests/         unit/integration tests
monitoring/    Prometheus / Loki / Promtail / Grafana config (provisioned)
notebooks/     Colab training / fine-tuning / optimization notebooks (build-time)
```

## Build-time notebooks (Colab GPU)

| Notebook | Purpose |
|---|---|
| `notebooks/cuad_finetune.ipynb` | Fine-tune DeBERTa-v3-base on CUAD; register `cuad-extractor` |
| `notebooks/cuad_onnx_export.ipynb` | Export fp32 → INT8 ONNX; eval F1/ANLS/AUPR/CER; register `cuad-extractor-onnx-int8` |
| `notebooks/cuad_embed_finetune.ipynb` | Fine-tune the `bge-small-en-v1.5` retrieval embedder on CUAD |

## Contract Intelligence agent

The agent (`POST /agent`) chains extraction, RAG, and GraphRAG behind a single
call — a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine
(`route → retrieve/graph_query → generate → critique`, one bounded retry). It
adds no new model, only composition. Tracing is via a self-hosted
[Langfuse](https://langfuse.com) instance when configured, and is strictly
best-effort — a tracing error is logged and swallowed, never breaking a
request. See [`../docs/architecture.md`](../docs/architecture.md) for the full
flow and degradation matrix.

```bash
curl -X POST http://localhost:8000/agent \
  -H 'Content-Type: application/json' \
  -d '{"task": "Which contracts expire within 90 days?"}'
```
