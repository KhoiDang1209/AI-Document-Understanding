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

## Contract Intelligence Agent (C4)

The C4 agent chains the C1–C3 capabilities behind a single `POST /agent` call. It is a
[LangGraph](https://langchain-ai.github.io/langgraph/) state machine that orchestrates the
existing tools — it adds no new model, only composition.

### Endpoint

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/agent` | Run the agent on a compound task; returns an `AgentResponse` (answer + citations + trace id) |

### Flow

```
route → retrieve | graph_query → generate → critique ⤶ (bounded retry)
```

- **route** — reuses C3's rule-based router (`graph` for expiration/renewal questions, else `vector`).
- **retrieve** — C3 graph query for graph routes, C2 vector retrieval otherwise.
- **generate** — reuses C2's `generate_or_degrade` to produce a grounded answer from the citations.
- **critique** — if no citations were found and the retry budget remains, fall back to vector
  retrieval and retry once. The pass count is capped by `DOCINTEL_AGENT_MAX_RETRIES` (default `1`),
  so the graph cannot loop unbounded.

### Degraded path

Graceful degradation is mandatory: when the LLM endpoint (`DOCINTEL_LLM_BASE_URL`) is unset or
unreachable, `/agent` still returns HTTP 200 with the deterministic tool outputs gathered so far,
`answer: null`, and `status: "degraded"`. The retrieval and routing steps run without an LLM.

### Tracing (self-hosted Langfuse)

Every node is traced to a self-hosted [Langfuse](https://langfuse.com) instance when configured.
Tracing is strictly best-effort — a tracing error is logged and swallowed, never breaking a request.

```bash
docker compose up langfuse           # starts Langfuse + its Postgres
export DOCINTEL_LANGFUSE_HOST=http://localhost:3000
export DOCINTEL_LANGFUSE_PUBLIC_KEY=pk-...
export DOCINTEL_LANGFUSE_SECRET_KEY=sk-...
```

When both keys are set, the returned `trace_id` points at the run in the Langfuse UI; otherwise it
is `null` and the agent runs untraced.

### Example

```bash
curl -X POST http://localhost:8000/agent \
  -H 'Content-Type: application/json' \
  -d '{"task": "Which contracts expire within 90 days?"}'
```
