<div align="center">

# DocIntel — Document AI & Contract Intelligence Platform

**Turn document images and contract PDFs into validated, structured, queryable data — served on CPU, fully observable, and backed by a reproducible MLOps lifecycle.**

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![ONNX Runtime](https://img.shields.io/badge/inference-ONNX%20Runtime-005CED)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2)
![Prometheus](https://img.shields.io/badge/metrics-Prometheus-E6522C)
![Grafana](https://img.shields.io/badge/dashboards-Grafana-F46800)
![Docker](https://img.shields.io/badge/packaging-Docker-2496ED)
[![CI](https://github.com/KhoiDang1209/AI-Document-Understanding/actions/workflows/ci.yml/badge.svg)](https://github.com/KhoiDang1209/AI-Document-Understanding/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## Overview

**Problem.** Structured data locked in document images and PDFs — receipts, contracts — is expensive to get at. Off-the-shelf OCR returns raw text, not validated fields; asking questions of a pile of contracts still means reading them.

**Solution.** DocIntel is one typed, tested, reproducible codebase with two products built on a shared engineering spine:

1. **Document extraction pipeline** — a receipt/invoice image goes in; validated, schema-checked JSON (line items, totals, confidence, reconciliation flags) comes out.
2. **Contract Intelligence platform** — a contract PDF goes in; the system extracts 41 CUAD clause types, then answers natural-language questions over the corpus with hybrid vector retrieval, a knowledge graph, and an orchestrating agent — each with citations.

Both are served on CPU. The only optional, GPU-bound component is the generative LLM used for contract Q&A — without it, the system degrades gracefully to citations-only rather than failing.

**Features**

- ONNX-INT8 key-information extraction (LayoutLMv3, fine-tuned on CORD) — 3× faster than fp32, 4× smaller, 98.4% of the F1
- Pydantic + rule-based validation that annotates rather than blocks (a receipt that fails reconciliation still returns `200` with the specific issues)
- Contract clause extraction (CUAD, DeBERTa-v3 extractive QA) with dual-path ingestion (digital text layer or OCR fallback)
- Hybrid retrieval (BM25 + dense, CUAD-fine-tuned embedder) with grounded, citation-backed answers
- A Neo4j knowledge graph for structured questions (e.g. renewal/expiration dates) with a rule-based vector/graph router
- A LangGraph agent that composes retrieval + generation with a bounded retry and best-effort tracing
- Full MLOps lifecycle: MLflow tracking + registry, DVC-tracked datasets, reproducible Colab fine-tuning
- Full observability: Prometheus metrics, Loki logs, provisioned Grafana dashboards — no manual setup
- A Streamlit demo UI (Extract / Ask / Agent / Graph / Metrics) and a scripted end-to-end walkthrough

## Architecture

![End-to-End Architecture](docs/End-to-End-Pipeline.png)

```
Document extraction pipeline                 Contract Intelligence platform
  Image → Preprocess (OpenCV)                  PDF → Extract (DeBERTa-v3 QA, ONNX-INT8)
        → OCR (docTR)                                ├─→ Vector RAG (hybrid BM25 + dense, rerank)
        → KIE (LayoutLMv3, ONNX-INT8)                 ├─→ GraphRAG (Neo4j date/renewal templates)
        → Decode (BIO → fields)                       └─→ Agent (LangGraph: route → retrieve → generate → critique)
        → Validation (Pydantic + rules)
        → Persist (SQLite + MinIO)
```

| Component | Responsibility |
|---|---|
| **Preprocess / OCR** | Deskew/denoise (OpenCV); word-level text + boxes + confidence (docTR) |
| **KIE** | Token classification (LayoutLMv3) → candidate fields, served as ONNX-INT8 |
| **Contract extraction** | Extractive QA (DeBERTa-v3) over 41 CUAD clause types, dual-path (digital / OCR) ingestion |
| **Validation** | Pydantic schema + business rules (reconciliation, required fields, confidence thresholds) |
| **RAG / GraphRAG** | Hybrid BM25+dense retrieval into Qdrant; Neo4j graph for structured queries; rule-based router |
| **Agent** | LangGraph state machine orchestrating retrieval + generation for compound tasks |
| **Serving** | FastAPI, async, OpenAPI docs, pulls registered models from MLflow at startup |
| **Persistence** | SQLite (metadata), MinIO (source images/artifacts), Qdrant (vectors), Neo4j (graph) |
| **Observability** | Prometheus (metrics) + Loki (logs) + Grafana (dashboards), provisioned from in-repo config |

Full reference architecture: [`docs/pipeline.md`](docs/pipeline.md). Contract Intelligence flow + degradation matrix: [`docs/architecture.md`](docs/architecture.md).

## Tech Stack

| Layer | Technology |
|---|---|
| API & serving | Python 3.12, FastAPI, Uvicorn, Pydantic v2 |
| Vision / OCR | OpenCV, docTR |
| Information extraction | LayoutLMv3 (fine-tuned on CORD), DeBERTa-v3 (fine-tuned on CUAD) |
| Retrieval | Qdrant (vector), Neo4j (graph), BM25 + `bge-small-en-v1.5` (fastembed, fine-tuned) |
| Agent orchestration | LangGraph, Langfuse (tracing) |
| Inference runtime | ONNX Runtime (dynamic INT8) |
| MLOps & storage | MLflow, MinIO, DVC |
| Observability | Prometheus, Grafana, Loki, Promtail |
| Packaging & CI | Docker, docker-compose, GitHub Actions |
| Tooling | uv, ruff, mypy (strict), pytest |

## Project Structure

```
.
├── docintel/                 # The application + its stack
│   ├── src/docintel/
│   │   ├── pipeline/         # preprocess → layout → OCR → KIE → validation
│   │   ├── kie/               # key-information extraction (LayoutLMv3, ONNX-INT8)
│   │   ├── contracts/         # contract clause extraction (DeBERTa-v3 QA)
│   │   ├── rag/                # hybrid retrieval + generate-or-degrade
│   │   ├── graph/              # Neo4j store, Cypher templates, router
│   │   ├── agent/              # LangGraph orchestration
│   │   ├── ui/                  # Streamlit demo
│   │   ├── api/                  # FastAPI app + routes
│   │   └── scripts/               # eval + data-download scripts
│   ├── tests/                 # unit & integration tests (pytest)
│   ├── monitoring/            # Prometheus / Loki / Promtail / Grafana config (provisioned)
│   ├── notebooks/             # Colab training / fine-tuning / benchmarking
│   ├── docker-compose.yml     # the service stack
│   ├── Dockerfile             # CPU-only runtime image
│   └── pyproject.toml         # deps + extras (serve, kie, contracts, rag, graph, agent, dev)
├── docs/                      # architecture, runbook, phase reports, benchmarks
├── models/                    # local model bundles (git-ignored, kept off git)
└── CLAUDE.md                  # engineering standards for this repo
```

## Requirements

- Docker + Docker Compose (recommended — brings up the full stack), **or**
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) for local, dependency-by-dependency development

No GPU is required to run or serve the system. A GPU is only needed to reproduce the Colab fine-tuning notebooks, and optionally to self-host the generative LLM for contract Q&A.

## Installation

```bash
git clone https://github.com/KhoiDang1209/AI-Document-Understanding.git
cd AI-Document-Understanding/docintel
cp .env.example .env
docker compose up -d --build      # first build is slow: torch + baked docTR weights
```

This brings up the API, MLflow, MinIO, and the Prometheus/Loki/Grafana stack (add Qdrant/Neo4j/Langfuse for the Contract Intelligence surfaces — see [`docs/RUNBOOK.md`](docs/RUNBOOK.md)).

The fine-tuned models are too large for git and are kept locally (git-ignored) under `models/`. Point the API at them directly with `DOCINTEL_KIE_ONNX_LOCAL_PATH` / `DOCINTEL_CONTRACT_ONNX_LOCAL_PATH`, or register them in MLflow first — see [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Usage

```bash
curl http://localhost:8000/health
curl -F "file=@receipt.png;type=image/png" http://localhost:8000/extract
curl -F "file=@contract.pdf;type=application/pdf" http://localhost:8000/contracts/extract
curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "When does this agreement expire?"}'
```

Interactive demo:

```bash
streamlit run src/docintel/ui/app.py       # Extract / Ask / Agent / Graph / Metrics views
docintel-demo                              # scripted end-to-end HTTP walkthrough
```

Open the API docs at **http://localhost:8000/docs**, the Grafana dashboard at **http://localhost:3000 → DocIntel**, and Prometheus targets at **http://localhost:9090/targets**.

Tear down with `docker compose down` (add `-v` to wipe volumes).

## API

| Method & path | Purpose |
|---|---|
| `POST /extract` | Document image → validated, structured `Document` JSON |
| `GET /documents/{id}` | Retrieve a previously extracted document |
| `GET /documents/{id}/image` | Retrieve the stored source image |
| `POST /contracts/extract` | Contract PDF → extracted clauses; auto-indexes into RAG + graph |
| `GET /contracts/{id}` | Retrieve a previously extracted contract |
| `POST /ask` | Grounded answer over indexed contracts (graph or vector route), with citations |
| `POST /agent` | LangGraph agent over a compound task |
| `GET /health` | Liveness check |
| `GET /metrics` | Prometheus exposition (HTTP + custom KIE/validation metrics) |
| `GET /docs` | Interactive Swagger UI |

The generative LLM (`/ask`, `/agent`) is optional and the only GPU-bound component; without it both endpoints still return `200` with citations-only output (`status: "degraded"`). See [`docs/architecture.md`](docs/architecture.md) for the full degradation matrix.

## AI / ML

| | Document extraction | Contract Intelligence |
|---|---|---|
| **Dataset** | CORD (primary), SROIE (benchmark) | CUAD (41 clause types, 510 contracts) |
| **Model** | LayoutLMv3 (token classification) | DeBERTa-v3-base (extractive QA) + `bge-small-en-v1.5` (fine-tuned embedder) |
| **Training** | Fine-tuned on Colab GPU, tracked in MLflow | Fine-tuned on Colab GPU, tracked in MLflow |
| **Optimization** | ONNX export → dynamic INT8 quantization | ONNX export → dynamic INT8 quantization |
| **Evaluation** | Field F1 / precision / recall / latency / size — see [`docs/benchmark.md`](docs/benchmark.md) | Retrieval recall@k / MRR + RAGAS answer quality — see below |

**KIE optimization:** ONNX-INT8 is **3.0× faster** (p50 1934 → 650 ms), **2.9× higher throughput**, and **4× smaller** (480 → 121 MB) than the fp32 baseline, retaining **98.4% of F1** (0.8449 → 0.8315).

**Retrieval fine-tuning** (40 CUAD contracts, seed 0, 1,253 queries):

| Stack | Recall@1 | Recall@5 | Recall@30 | MRR |
|---|---|---|---|---|
| Stock (full stack) | 0.206 | 0.494 | 0.816 | 0.373 |
| **Fine-tuned, no reranker** | **0.316** | **0.745** | **0.949** | **0.528** |

Recall@5 improved **0.494 → 0.745 (+51% relative)**. A notable, honestly-reported finding: the generic reranker helps the stock embedder but *harms* the fine-tuned ordering (0.745 → 0.513) — production runs with rerank off. RAGAS answer-quality deltas were within noise at this sample size; the retrieval eval is the load-bearing evidence. Full write-up: [`docs/phases/c2-embed-finetune/report_c2_embed_finetune.md`](docs/phases/c2-embed-finetune/report_c2_embed_finetune.md).

## Testing

```bash
cd docintel
uv sync --all-extras                     # full env (uv sync replaces the env; sync ALL extras for the test suite)
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest                            # one slow real-OCR test is deselected by default
```

CI (GitHub Actions) runs lint, format, type-check, and the full test suite on every push/PR, plus a separate job that stands up a live Neo4j for the GraphRAG parity test.

## Deployment

`docker compose up` (from `docintel/`) brings up the full stack — all ports published to `localhost`:

| Service | URL / port | Role |
|---|---|---|
| API | http://localhost:8000 ([`/docs`](http://localhost:8000/docs)) | FastAPI app — pipeline + `/metrics` |
| MLflow | http://localhost:5000 | Experiment tracking + model registry |
| MinIO | http://localhost:9000 (S3) · :9001 (console) | Object storage for source images/artifacts |
| Qdrant | http://localhost:6333 | Vector store for contract retrieval |
| Neo4j | http://localhost:7474 (browser) · :7687 (bolt) | Knowledge graph for structured contract queries |
| Prometheus | http://localhost:9090 | Scrapes the API `/metrics` every 15s |
| Loki + Promtail | http://localhost:3100 (internal) | Log aggregation, tailed from Docker container stdout |
| Grafana | http://localhost:3000 | Dashboards over Prometheus + Loki |
| Langfuse (optional) | http://localhost:3000 | Agent tracing |

Containers demonstrate deployability; running a public always-on service is intentionally out of scope. A Kubernetes packaging pass (manifests validated on `kind`) is on the roadmap below.

## Limitations

- **LLM-backed answers are optional and intermittent.** The generative LLM is self-hosted on Colab GPU behind an ngrok tunnel — the only non-CPU, non-always-on dependency. `/ask` and `/agent` are designed to degrade gracefully rather than assume it's up.
- **RAGAS answer-quality deltas are within noise** at the current sample size (5 contracts, 40 questions); the retrieval recall/MRR eval is the stronger evidence and is reported instead of an overclaimed answer-quality number.
- **The generic cross-encoder reranker hurts the fine-tuned retrieval ordering** (recall@5 0.745 → 0.513) and is disabled by default; a CUAD-tuned reranker is the natural next step, not yet built.
- **Layout detection is not yet implemented** — OCR runs over the full page rather than region-scoped crops.

## Roadmap

- [x] Document extraction pipeline: OCR → KIE → validation → persistence, served on CPU
- [x] MLOps spine: MLflow tracking/registry, ONNX/INT8 optimization + benchmark, Docker packaging
- [x] Full observability: Prometheus + Loki + Grafana, provisioned out of the box
- [x] Contract clause extraction (CUAD) with dual-path (digital/OCR) ingestion
- [x] Hybrid vector RAG with a fine-tuned embedder, GraphRAG, and a LangGraph agent
- [ ] Layout detection (DocLayout-YOLO) ahead of OCR, for region-scoped extraction
- [ ] Kubernetes packaging (manifests validated on `kind`)
- [ ] On-demand LLM-based KIE backend (QLoRA-tuned, encoder-vs-LLM benchmark)
- [ ] CUAD-tuned reranker to recover the generic reranker's regression

Detailed per-phase build history: [`docs/phases/README.md`](docs/phases/README.md).

## Documentation

| Document | Purpose |
|---|---|
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | The whole story — what was built, results, and links |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Stand up the full stack and walk the demo, incl. troubleshooting |
| [`docs/pipeline.md`](docs/pipeline.md) | End-to-end reference architecture |
| [`docs/architecture.md`](docs/architecture.md) | Contract Intelligence flow & degradation matrix |
| [`docs/benchmark.md`](docs/benchmark.md) | ONNX / INT8 accuracy · latency · size benchmark |
| [`docs/plan.md`](docs/plan.md) · [`docs/proposal.md`](docs/proposal.md) · [`docs/research.md`](docs/research.md) | Roadmap, design decisions, feasibility |
| [`docs/phases/README.md`](docs/phases/README.md) | Per-phase build history and completion reports |

## About

An AI-engineering portfolio project built to demonstrate the full applied-ML lifecycle — data → fine-tuning → optimization → serving → MLOps → observability → RAG → agents — on one reproducible codebase, rather than chasing a single accuracy number.

## License

[MIT](LICENSE) © Khoi Dang
