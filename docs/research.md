# Engineering Environment & Feasibility

This document records the engineering environment behind DocIntel, the rationale for the
build-time / run-time split, the datasets used, and the deliberate scope boundaries.

The objective of DocIntel is to demonstrate **end-to-end, production-grade AI engineering
with MLOps at the center**. Decisions here optimize for that goal — a reproducible model
lifecycle, latency engineering, and maintainability — rather than for operating a hosted
service. The project is organized as a **core** (the MLOps spine, built first) plus
**advancements** (built only after the core is green); see [`plan.md`](plan.md).

---

## 1. Compute Model

DocIntel uses two complementary environments, mirroring a real MLOps separation of concerns:

| Environment | Role | Used for |
|-------------|------|----------|
| **Local (CPU)** | Run-time | API, pipeline inference, observability, packaging |
| **Google Colab Pro (GPU, batch)** | Build-time | Training, fine-tuning, ONNX export, quantization, benchmarking |

Models are trained on Colab, logged and registered in MLflow, exported to optimized
ONNX/INT8 artifacts, and pulled locally for CPU inference. This is the project's core
MLOps story: a clean boundary between **build-time** (GPU, ephemeral, batch) and
**run-time** (CPU, persistent, service-oriented).

**Colab Pro characteristics (build-time):** access to T4 / L4 / (occasionally) A100 GPUs,
high-RAM runtimes, and longer sessions — sufficient for fine-tuning LayoutLMv3 and (later)
QLoRA on ≤3B models. Sessions are ephemeral, so all build steps are scripted and
reproducible from notebooks. Where an advancement needs a GPU at run-time (e.g. an LLM KIE
backend), it is served **on-demand** from Colab via a tunnel (ngrok), never as an
always-on service.

**Local characteristics (run-time):** CPU inference via ONNX Runtime. This makes latency
engineering — quantization, batching, caching — a first-class concern rather than an
afterthought, which is itself a valuable thing to demonstrate.

---

## 2. Component Feasibility

### Core

| Stage | Component | Build-time (Colab) | Run-time (local CPU) |
|-------|-----------|--------------------|----------------------|
| Preprocess | OpenCV | — | CPU |
| OCR | PaddleOCR / docTR (pretrained) | — | CPU / ONNX inference |
| KIE | LayoutLMv3 | Fine-tune + quantize | ONNX INT8 inference |
| Optimization | ONNX, INT8, ONNX Runtime | Export & quantize | Benchmark |
| Validation | Pydantic + rules | — | CPU |
| Tracking / registry | MLflow | Log + register | Pull artifact |
| Storage | MinIO, SQLite | Push artifact | CPU |
| Serving | FastAPI | — | CPU |
| Observability | Prometheus / Grafana / Loki | — | CPU |
| Packaging / CI | Docker, docker-compose, GitHub Actions | — | CPU |

### Advancements

| Stage | Component | Build-time (Colab) | Run-time |
|-------|-----------|--------------------|----------|
| Layout | DocLayout-YOLO | Fine-tune / export | ONNX inference (CPU) |
| GraphRAG | embeddings + graph/vector store | — | CPU |
| Agents | LangGraph + Langfuse | — | CPU |
| LLM KIE (optional) | Qwen2.5-3B (QLoRA) | Fine-tune | On-demand (Colab + ngrok) |
| Orchestration | Kubernetes (kind) | — | CPU |

Every run-time component in the core is CPU-friendly; every GPU-dependent step is confined
to build-time on Colab, or to an on-demand advancement.

---

## 3. Datasets

All datasets are public and free to use for research/portfolio purposes.

| Dataset | Task | Used in | Notes |
|---------|------|---------|-------|
| **CORD** | Receipt KIE (30+ fields) | **Core** | Primary KIE dataset — rich labels, manageable size |
| **SROIE** | Receipt OCR + KIE (4 fields) | **Core** | Classic benchmark |
| **FUNSD** | Form understanding | Advancement | Entity + relation extraction |
| **XFUND** | Multilingual forms | Advancement | FUNSD's multilingual counterpart |
| **DocVQA** | Document QA | Advancement | Feeds the GraphRAG / Q&A layer |
| **DocLayNet / PubLayNet** | Layout detection | Advancement | Layout training |
| **RVL-CDIP** | Document classification | Advancement | Optional classification stage |

**Core set:** CORD (rich KIE) + SROIE (benchmark). Everything else enters with an
advancement.

---

## 4. Scope Boundaries

To keep the project focused on a high-quality MLOps lifecycle, the following are
intentionally **out of the core**, each with the approach used instead:

| Out of core | Approach used instead |
|--------------|-----------------------|
| Operating an always-on public GPU service | Containerized, deployable artifacts; GPU inference on-demand (Colab + ngrok) |
| Training OCR (MixNet/PARSeq) from scratch | Pretrained PaddleOCR / docTR; effort concentrated on the KIE fine-tune |
| Layout detection in the core path | Deferred to an advancement (DocLayout-YOLO) |
| RAG / agents in the core | Deferred to advancements (GraphRAG, LangGraph) |
| Production GPU-autoscaling Kubernetes cluster | Manifests validated on a local `kind` cluster (advancement) |
| Large-scale multi-GPU throughput benchmarking | Representative single-environment benchmarks with documented latency/accuracy/size curves |
| Vendor-specific accelerated runtimes (e.g. TensorRT) | ONNX Runtime with INT8 quantization (portable, demonstrates the optimization skill) |

These boundaries are about **focus and sequencing**, not capability — the deferred items
return as advancements once the core ships.

---

## 5. AI-Engineering Skill Coverage

| Competency | Delivered by | Layer |
|------------|--------------|-------|
| **MLOps** | MLflow tracking + registry, build/run separation, reproducible promotion, DVC | Core |
| Model fine-tuning | LayoutLMv3 fine-tune on CORD | Core |
| Model optimization | ONNX export + INT8 + benchmark report | Core |
| Data engineering / preprocessing | CORD/SROIE ingestion, OpenCV preprocessing | Core |
| Inference serving | FastAPI, async, pluggable KIE backend, CPU ONNX Runtime | Core |
| Evaluation | KIE F1, OCR CER/WER, latency/throughput, model size | Core |
| Observability | Prometheus, Grafana, Loki | Core |
| Containerization | Docker, docker-compose | Core |
| CI/CD | GitHub Actions | Core |
| Layout analysis (CV) | DocLayout-YOLO | Advancement |
| Retrieval-Augmented Generation | GraphRAG + RAGAS | Advancement |
| Agent orchestration | LangGraph + Langfuse | Advancement |
| Kubernetes orchestration | Manifests on `kind` | Advancement |
| Cost / latency trade-off analysis | Encoder-vs-LLM KIE benchmark | Advancement |

The core already covers the competencies this project exists to add to the CV — MLOps,
fine-tuning, optimization, serving, and observability. Advancements broaden the surface
(RAG/agents/CV/K8s), which the CV already evidences elsewhere.

---

## 6. Architectural Decisions

1. **MLOps lifecycle is the centerpiece.** Colab for training/optimization; local CPU
   services for run-time; MLflow as the tracking + registry backbone connecting them.
2. **Explicit build-time / run-time separation**, documented as a first-class design
   decision and made reproducible via notebooks, Docker, DVC, and MLflow.
3. **Default KIE = LayoutLMv3 (ONNX INT8).** Fast, portable, CPU-friendly; the backend is
   pluggable so an LLM/VLM backend can be added on-demand later.
4. **Pretrained OCR, deferred layout/RAG/agents.** The core stays a tight, finishable
   spine; breadth returns as advancements.
5. **Reproducibility first.** A reviewer can reconstruct and run the system from a clean
   checkout.
