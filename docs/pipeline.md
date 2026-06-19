# Reference Architecture

This document describes the end-to-end architecture of the **DocIntel** Document AI
system. It is the north-star reference; for the concrete, scoped implementation see
[`proposal.md`](proposal.md), for the engineering environment see
[`research.md`](research.md), and for the phased build order see [`plan.md`](plan.md).

> **Goal:** Transform a document image (invoice / receipt) into **structured, validated
> JSON**, served behind an API — and demonstrate the full **MLOps lifecycle** around it:
> train, track, register, optimize, serve, and monitor.

The system is organized in two layers:

- **Core** — the MLOps spine. An end-to-end pipeline with a measurable business metric
  (field-level F1 + CPU latency). This is built first and is the project's headline.
- **Advancements** — capabilities layered on once the core is green: layout detection,
  GraphRAG, agents, and Kubernetes.

---

## The MLOps Loop (the spine)

The core is best read as a single closed loop, split across two environments:

```
            build-time (Colab GPU, batch)                run-time (local CPU, service)
  ┌─────────────────────────────────────────┐   ┌──────────────────────────────────────┐
  │ CORD/SROIE → fine-tune LayoutLMv3 (KIE)  │   │  Image                                │
  │   → log run + metrics ──► MLflow         │   │   → Preprocess        (OpenCV)        │
  │   → export ONNX + INT8 quantize          │   │   → OCR               (pretrained)    │
  │   → benchmark (F1 / latency / size)      │   │   → KIE               (ONNX INT8)     │
  │   → register model ─────► MLflow Registry│   │   → Validation        (Pydantic+rules)│
  │   → push artifact ──────► MinIO          │   │   → Structured JSON                   │
  └────────────────────────────┬────────────┘   │   → Persist     (SQLite + MinIO)      │
                               │                 └───────────────┬──────────────────────┘
                               │  pull registered ONNX artifact  │
                               └────────────────────────────────►│
                                                                  ▼
                                       FastAPI  /extract · /documents/{id} · /health
                                                                  │
                                       Prometheus ─► Grafana ; logs ─► Loki
```

The clean boundary between **build-time** (GPU, ephemeral, batch) and **run-time** (CPU,
persistent, service-oriented) is the core MLOps story.

---

## A. Build-Time — Train, Track, Register (Colab GPU)

Offline stage that produces the model the run-time pipeline serves.

| Step | Component | Purpose |
|------|-----------|---------|
| Datasets | **CORD** (primary), **SROIE** (benchmark) | Receipt KIE training & evaluation |
| Information extraction | **LayoutLMv3** (token classification) | Fine-tuned to extract key fields |
| Experiment tracking | **MLflow** | Log runs, params, metrics |
| Model registry | **MLflow Model Registry** | Versioned, promotable models |
| Artifact storage | **MinIO** | ONNX artifacts + sample images |

Every training run is logged to MLflow; the chosen model is registered and its ONNX
artifact pushed to MinIO. Build steps are scripted in notebooks so an ephemeral Colab
session is fully reproducible.

---

## B. Optimization — ONNX + INT8 (Colab → artifact)

The inference-speedup story, and a first-class deliverable:

1. **Export to ONNX** from the fine-tuned PyTorch checkpoint.
2. **INT8 quantization** via ONNX Runtime.
3. **Benchmark** fp32 vs INT8 — field F1, p50/p95 latency, throughput, model size — logged
   to MLflow and written up in `docs/benchmark.md`.

The benchmark quantifies the accuracy/latency/size trade-off and justifies the served
artifact.

---

## C. Run-Time — Inference Pipeline (local CPU)

Each stage is a **swappable module behind an interface**.

### 1. Preprocess — OpenCV
Deskew, denoise, resize behind a config toggle. *Output:* normalized image.

### 2. OCR — pretrained (PaddleOCR or docTR → ONNX)
Pretrained engine, not trained from scratch. *Output:* text + boxes + confidence.

### 3. Key Information Extraction — LayoutLMv3 (ONNX INT8)
The fine-tuned model, pulled from the MLflow registry, run on CPU via ONNX Runtime.
*Output:* candidate fields with confidence. The backend sits behind a `KIEBackend`
interface so alternative backends can be added later without touching the pipeline.

### 4. Validation — Pydantic + rule engine
Schema validation plus business rules (line items reconcile to total; date/currency
sanity). Hard failures vs. soft warnings, each flagged. *Output:* validated `Document`.

### 5. Structured JSON + Persistence
Canonical JSON (`company`, `date`, `address`, `total`, line items, each with confidence).
Metadata persisted to **SQLite**, source image to **MinIO**, linked by document id.

---

## D. Serving — FastAPI (CPU)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/extract` | Document image → validated structured JSON |
| GET | `/documents/{id}` | Retrieve a previously extracted document |
| GET | `/health` | Health / readiness |

Async, automatic OpenAPI docs, request/response validation. The service pulls the
registered ONNX model at startup.

---

## E. Monitoring & Observability

| Tool | Role |
|------|------|
| **Prometheus** | Metrics scrape (`/metrics`) |
| **Grafana** | Dashboards |
| **Loki** | Structured-log aggregation |

**Monitored signals:** request count / throughput, p50/p95 latency, error rate, resource
usage, and **model signals** (KIE confidence distribution). Demonstrated locally via the
compose stack — not operated as a hosted service.

---

## F. Packaging — Docker Compose

`docker compose up` brings the whole core online: `api`, `mlflow`, `minio`, `prometheus`,
`grafana`, `loki`. Reproducible from a clean checkout. CI (GitHub Actions) runs lint,
type-check, tests, and image build.

---

## Component Summary — Core

| Layer | Technologies |
|-------|--------------|
| Data & training | CORD, SROIE, LayoutLMv3 (fine-tuned) |
| Tracking & registry | MLflow, MLflow Model Registry |
| Optimization | ONNX, INT8 quantization, benchmarking |
| Run-time pipeline | OpenCV → pretrained OCR → KIE (ONNX INT8) → Pydantic validation |
| Storage | MinIO (artifacts/images), SQLite (metadata) |
| Serving | FastAPI, ONNX Runtime |
| Observability | Prometheus, Grafana, Loki |
| Packaging / CI | Docker, docker-compose, GitHub Actions |

---

## Advancements (after the core is green)

These extend the system but are explicitly **out of the core**. Each is built only once
the MLOps spine is complete and demonstrable. See [`plan.md`](plan.md) for sequencing.

| Advancement | Adds | Notes |
|-------------|------|-------|
| **Layout detection** | DocLayout-YOLO → ONNX, inserted before OCR | region-scoped OCR/KIE; CV competency |
| **GraphRAG over extracted data** | embeddings + graph/vector store; `POST /ask` | retrieval over the structured corpus; RAGAS eval |
| **Agent orchestration** | LangGraph; `POST /agent`; Langfuse tracing | multi-step: extract → validate → retrieve → answer |
| **Kubernetes** | manifests validated on `kind` | Deployment, Service, Ingress, HPA |
| **On-demand LLM KIE backend** | QLoRA Qwen2.5-3B behind `KIEBackend` | encoder-vs-LLM benchmark; on-demand GPU |

---

## Consumers & Integrations

The API is consumed by web apps, automation pipelines, and third-party systems — the same
contract whether the request hits the core `/extract` or an advancement endpoint.
