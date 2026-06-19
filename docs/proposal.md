# System Design — DocIntel

This document defines the concrete system design for **DocIntel**: a document-understanding
service that turns invoice/receipt images into validated structured JSON, built to
demonstrate the full **MLOps lifecycle** around a fine-tuned model.

It commits to **one** implementation of the [reference architecture](pipeline.md), within
the [engineering environment](research.md) (local CPU run-time + Colab GPU build-time),
and follows the build order in [`plan.md`](plan.md). Where it diverges from the reference,
the reasoning is noted in §7.

---

## 1. The System

DocIntel is built in two layers. The **core** is the MLOps spine and is delivered first;
**advancements** extend it once the core is green.

**Core endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `POST /extract` | Image → validated structured JSON (the pipeline) |
| `GET /documents/{id}` | Retrieve a previously extracted document |
| `GET /health` | Health / readiness |

**Advancement endpoints (later):** `POST /ask` (GraphRAG), `POST /agent` (LangGraph).

What makes this a complete demonstration of applied AI engineering is not endpoint count —
it is the **lifecycle**: a model fine-tuned on GPU, tracked and registered, optimized to
ONNX INT8, served on CPU, and monitored, all reproducible from a clean checkout.

---

## 2. Run-Time Pipeline (local · CPU · Docker)

```
Image
 → 1. Preprocess       (OpenCV: deskew, denoise, resize)
 → 2. OCR              (PaddleOCR or docTR, pretrained → ONNX)   text + boxes + confidence
 → 3. KIE              (LayoutLMv3 fine-tuned → ONNX INT8)       field extraction
 → 4. Validation       (Pydantic + rule engine)                 totals reconcile, formats
 → 5. Structured JSON  (persisted: SQLite metadata + MinIO images)
```

Each stage is a **swappable module behind an interface**. The KIE backend is pluggable
behind a `KIEBackend` interface — LayoutLMv3 (default) today, an LLM backend later — with
no change to the surrounding pipeline. Layout detection is an advancement (§8), not part
of the core path.

---

## 3. Build-Time Pipeline (Colab Pro · GPU · batch)

```
CORD / SROIE dataset
 → fine-tune LayoutLMv3 (KIE)            → MLflow run + metrics
 → export to ONNX + INT8 quantization    → artifact
 → benchmark (F1 / CER / latency / size) → docs/benchmark.md + MLflow
 → register in MLflow + push to MinIO
```

The run-time environment pulls the registered ONNX artifact and serves it. This
build-time / run-time separation is the core MLOps story.

---

## 4. Tech Stack by Layer (core)

| Layer | Choice |
|---|---|
| Language / API | Python 3.12, FastAPI + Uvicorn, Pydantic v2 |
| Preprocess | OpenCV, Pillow |
| OCR | PaddleOCR or docTR (pretrained) → ONNX |
| KIE | LayoutLMv3, fine-tuned, ONNX INT8 |
| Validation | Pydantic models + rule engine |
| Experiment tracking / registry | MLflow + MLflow Model Registry |
| Optimization | ONNX Runtime, INT8 quantization, benchmark harness |
| Storage | SQLite (metadata), MinIO (images / artifacts) |
| Data / model versioning | Git + DVC |
| Packaging | Docker + docker-compose |
| Observability | Prometheus + Grafana + Loki |
| CI/CD | GitHub Actions |

Advancement-only technologies (DocLayout-YOLO, embeddings/graph store, LangGraph,
Langfuse, Kubernetes, Qwen) are listed in §8.

---

## 5. Service Topology (docker-compose, core)

| Service | Role |
|---|---|
| `api` | FastAPI app — `/extract`, `/documents/{id}`, `/health`, `/metrics` |
| `mlflow` | Experiment tracking + model registry |
| `minio` | S3-compatible object storage (images, model artifacts) |
| `prometheus` + `grafana` + `loki` | Metrics, dashboards, logs |

`docker compose up` brings the entire core online locally. Advancements add services
(e.g. a vector/graph store) to the same compose file.

---

## 6. Skill Coverage (core)

The core is weighted toward **MLOps**, the competency this project exists to demonstrate.

| Competency | Delivered by |
|---|---|
| **MLOps lifecycle** | MLflow tracking + registry; build/run separation; reproducible artifact promotion |
| Model fine-tuning | LayoutLMv3 fine-tune on CORD (token classification) |
| Model optimization | ONNX export + INT8 quantization + benchmark report |
| Data engineering / preprocessing | CORD/SROIE ingestion, OpenCV preprocessing |
| Inference serving | FastAPI, async, pluggable KIE backend, CPU ONNX Runtime |
| Evaluation | KIE F1 per field, OCR CER/WER, latency/throughput, model size |
| Storage / persistence | MinIO artifacts/images, SQLite metadata |
| Observability | Prometheus + Grafana + Loki |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |

Retrieval, agents, layout, and Kubernetes are covered by the advancements (§8).

---

## 7. Key Design Decisions

1. **MLOps is the spine, not a side feature.** The headline is the lifecycle around a
   fine-tuned model — train → track → register → optimize → serve → monitor — with a
   reproducible build/run boundary. Everything in the core serves that story.
2. **Use pretrained OCR (PaddleOCR / docTR) rather than training MixNet + PARSeq from
   scratch.** Reproducing an OCR engine adds little to a lifecycle-focused portfolio;
   effort is concentrated on the KIE fine-tune and on optimization/serving. This is the
   main divergence from [`pipeline.md`](pipeline.md)'s full reference.
3. **Default KIE = LayoutLMv3 (ONNX INT8).** Portable, fast, CPU-friendly. The backend is
   pluggable so an LLM backend can be added later (§8) without pipeline changes.
4. **Layout detection is deferred.** OCR → KIE is enough for the core spine; DocLayout-YOLO
   is an advancement that improves region scoping later.
5. **Deployability is demonstrated, not operated.** Docker (core) and Kubernetes
   (advancement) artifacts prove the capability; running an always-on hosted service is
   out of scope.
6. **Reproducibility first.** Docker, DVC, and MLflow let a reviewer reconstruct and run
   the system end-to-end.

See [`research.md`](research.md) §4 for the full scope boundaries.

---

## 8. Advancements (after the core)

Built only once the core is green and demonstrable. Each is pluggable and additive:

1. **Layout detection** — DocLayout-YOLO → ONNX, inserted before OCR for region-scoped
   extraction.
2. **GraphRAG over extracted data** — embed/structure the extracted JSON into a
   graph/vector store; `POST /ask` returns grounded answers with citations; RAGAS eval.
3. **Agent orchestration** — LangGraph graph (extract → validate → retrieve → answer);
   `POST /agent`; Langfuse tracing on every node.
4. **Kubernetes** — Deployment, Service, Ingress, HPA validated on a local `kind` cluster.
5. **On-demand LLM KIE backend** — QLoRA Qwen2.5-3B behind the `KIEBackend` interface;
   extends the benchmark with an encoder-vs-LLM comparison.
