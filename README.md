<div align="center">

# DocIntel — Production-Grade Document AI

**A reference implementation of an end-to-end Document Understanding system: turn invoice/receipt images into validated, structured, queryable data.**

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![ONNX Runtime](https://img.shields.io/badge/inference-ONNX%20Runtime-005CED)
![Docker](https://img.shields.io/badge/packaging-Docker-2496ED)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2)

</div>

---

## Overview

DocIntel ingests a document image and returns clean, schema-validated JSON — then lets you ask natural-language questions about your documents. It is built as a **modular, testable, reproducible system**, not a notebook prototype.

The purpose of this repository is to demonstrate the **full applied-AI engineering lifecycle** to a professional standard:

> data engineering → fine-tuning → optimization → serving → retrieval (RAG) → agent orchestration → MLOps → observability.

The emphasis is on **engineering quality and breadth** — composable services, clean interfaces, reproducible experiments, and operational tooling — rather than on chasing a single accuracy number or operating a hosted service.

## What This Project Demonstrates

| Domain | Demonstrated by |
|---|---|
| **Data engineering** | Dataset ingestion (CORD / SROIE), preprocessing, versioning (DVC) |
| **Model fine-tuning** | LayoutLMv3 for Key Information Extraction; optional QLoRA on Qwen2.5-3B |
| **Model optimization** | ONNX export, INT8 quantization, accuracy/latency/size benchmarking |
| **Inference serving** | FastAPI service, pluggable model backends, request batching |
| **Retrieval-Augmented Generation** | Embedding index over extracted data, grounded Q&A, RAGAS evaluation |
| **Agent orchestration** | LangGraph multi-step agent (extract → validate → retrieve → answer) |
| **MLOps** | MLflow tracking & model registry, artifact storage, reproducible pipelines |
| **Observability** | Prometheus + Grafana + Loki metrics/logs; Langfuse LLM tracing |
| **Engineering hygiene** | Typed, tested, containerized, CI-checked, documented |

## Architecture

![End-to-End Architecture](End-to-End-Pipeline.png)

```
Image
 → Preprocess (OpenCV)
 → Layout detection (DocLayout-YOLO, ONNX)
 → OCR (PaddleOCR / docTR, ONNX)
 → Key Information Extraction (LayoutLMv3, ONNX INT8)
 → Validation (Pydantic + rules)
 → Structured JSON  ──►  Index (embeddings → Qdrant)
                                    │
                                    └─► Q&A / Agent (RAG + LLM)
```

See **[`pipeline.md`](pipeline.md)** for the full reference architecture.

## Tech Stack

| Layer | Technology |
|---|---|
| API & serving | Python 3.12, FastAPI, Uvicorn, Pydantic v2 |
| Vision / OCR | OpenCV, DocLayout-YOLO, PaddleOCR / docTR |
| Document understanding (KIE) | LayoutLMv3 (default); Qwen2.5-3B / Vintern-1B (optional) |
| Inference runtime | ONNX Runtime (INT8) |
| Retrieval & agents | sentence-transformers, Qdrant, LangGraph, RAGAS |
| MLOps & storage | MLflow, MinIO, DVC |
| Observability | Prometheus, Grafana, Loki, Langfuse |
| Packaging & orchestration | Docker, docker-compose, Kubernetes (kind) |
| CI/CD | GitHub Actions |

## Compute Model

Development and CPU-capable inference run **locally**; GPU-bound work runs on **Google Colab Pro**. This mirrors a real MLOps separation of concerns:

- **Build-time (Colab, batch GPU):** training, fine-tuning, ONNX export, quantization, benchmarking.
- **Run-time (local, CPU services):** the API, pipeline inference, RAG, agents, and the full observability stack.

Models are trained on Colab, registered in MLflow, and pulled locally as optimized ONNX artifacts. Containers and Kubernetes manifests are included to demonstrate deployability — running a public, always-on service is intentionally **out of scope**. See **[`research.md`](research.md)** for the rationale.

## Repository Structure

```
.
├── src/docintel/        # Application code (pipeline, kie, rag, api, validation, storage)
├── notebooks/           # Colab training / fine-tuning / benchmarking
├── infra/               # docker-compose, Kubernetes manifests, monitoring config
├── tests/               # Unit & integration tests
├── data/                # Datasets (DVC-tracked, git-ignored)
├── pipeline.md          # Reference architecture
├── research.md          # Engineering environment & feasibility analysis
├── proposal.md          # System design
└── plan.md              # Phased build roadmap
```

## Getting Started

> Prerequisites: Docker + Docker Compose.

```bash
git clone <repo-url>
cd docintel
docker compose up
```

This starts the API and supporting services (MLflow, MinIO, Qdrant, monitoring). The interactive API docs are available at `http://localhost:8000/docs`.

## Documentation

| Document | Purpose |
|---|---|
| [`pipeline.md`](pipeline.md) | End-to-end reference architecture |
| [`research.md`](research.md) | Engineering environment, feasibility, datasets, scope |
| [`proposal.md`](proposal.md) | Concrete system design and tech-stack decisions |
| [`plan.md`](plan.md) | Phased build roadmap with checklists |

---

<div align="center">
<sub>A reference implementation built to demonstrate end-to-end, production-grade AI engineering.</sub>
</div>
