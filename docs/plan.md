# Build Roadmap

The phased build plan for DocIntel. See [`proposal.md`](proposal.md) for the design,
[`research.md`](research.md) for the engineering environment, and [`pipeline.md`](pipeline.md)
for the reference architecture.

The roadmap is **core-first**: Phases 0–5 build the MLOps spine (an end-to-end pipeline
with a measurable business metric, served and monitored). Only once the core is green do
the **Advancements** (A1–A5) begin. Each phase delivers a self-contained, demonstrable
milestone.

**Legend:** 💻 local (CPU) · ☁️ Colab (GPU, batch) · 🔬 research / spike · 📦 deliverable

---

# CORE — the MLOps spine

## Phase 0 — Foundations & Environment

**Goal:** A clean repository that runs `docker compose up`, exposes a stub API, and has
storage and experiment tracking wired in.

### Research 🔬
- [ ] Confirm CORD + SROIE access, license, and format (HF `datasets` vs raw download).
- [ ] Decide dependency tooling (uv) for Python 3.12.
- [ ] Finalize repository layout.

### Tasks
- [ ] Project structure: `src/docintel/{pipeline,kie,api,validation,storage,scripts}`, `tests/`, `notebooks/`, `infra/`, `data/`.
- [ ] Dependency management via `pyproject.toml`; pin Python 3.12.
- [ ] `Dockerfile` (CPU base) + `docker-compose.yml`: `api`, `mlflow`, `minio`.
- [ ] FastAPI skeleton with `GET /health`.
- [ ] Configuration via Pydantic `BaseSettings`; structured JSON logging.
- [ ] Tooling: ruff + mypy + pre-commit; `pytest` running.
- [ ] Data download script (CORD/SROIE) → `data/raw/`; DVC init + tracking.

### Done when 📦
- [ ] `docker compose up` starts the service stack; `/health` returns 200.
- [ ] Lint, type-check, and tests pass locally and in a basic CI workflow.

---

## Phase 1 — OCR Baseline + `/extract`

**Goal:** The thinnest end-to-end slice — image in, raw OCR text + boxes out.

### Research 🔬
- [ ] PaddleOCR vs docTR on CPU: accuracy on receipts, latency/page, ONNX support, footprint.
- [ ] Image input contract (upload vs base64; size limits).

### Tasks
- [ ] `OCREngine` interface + chosen pretrained implementation.
- [ ] OpenCV preprocessing (deskew, denoise, resize) behind a toggle.
- [ ] `POST /extract` → `{text, boxes, confidence}` (typed response model).
- [ ] Unit tests on sample images; latency logged.

### Done when 📦
- [ ] `/extract` returns OCR results for a sample receipt; engine swappable via config.

---

## Phase 2 — KIE Fine-tune (LayoutLMv3) + MLflow

**Goal:** The flagship build-time step — fine-tune the extraction model on Colab and make
the run reproducible and tracked.

### Research 🔬
- [ ] LayoutLMv3 fine-tuning recipe for CORD (token classification).
- [ ] Label schema mapping (CORD fields → output JSON schema).

### Tasks
- [ ] ☁️ Colab notebook: dataset prep, fine-tune LayoutLMv3.
- [ ] ☁️ Log params, metrics, and artifacts to MLflow; register the chosen model.
- [ ] ☁️ Push the model artifact to MinIO.
- [ ] Evaluate: KIE F1 per field on the CORD test split, logged to MLflow.

### Done when 📦
- [ ] A fine-tuned LayoutLMv3 is registered in MLflow with F1 recorded and reproducible
      from the notebook.

---

## Phase 3 — Optimization: ONNX + INT8 + Benchmark

**Goal:** The inference-speedup story — export, quantize, and quantify the trade-off.

### Research 🔬
- [ ] ONNX export path for LayoutLMv3; INT8 quantization options in ONNX Runtime.
- [ ] Metric definitions: F1, CER/WER, p50/p95 latency, throughput, model size.

### Tasks
- [ ] ☁️/💻 Export the registered model to ONNX; INT8 quantize.
- [ ] Benchmark harness: fixed sample set, warm-up, repeated runs.
- [ ] Compare fp32 vs INT8 (F1 / latency / throughput / size); log to MLflow.
- [ ] `docs/benchmark.md` with tables + plots.

### Done when 📦
- [ ] A reproducible benchmark report justifies the served ONNX INT8 artifact.

---

## Phase 4 — Serving + Validation + Structured Schema + Persistence

**Goal:** A trustworthy, persisted, retrievable end-to-end response on CPU.

### Research 🔬
- [ ] Canonical output schema (fields, types, currency/date normalization).
- [ ] Validation rules worth enforcing (line items reconcile to total; date/currency sanity).

### Tasks
- [ ] 💻 `KIEBackend` interface + `LayoutLMv3OnnxBackend` (pulls registered model from MLflow).
- [ ] 💻 Wire KIE into `/extract`: image → preprocess → OCR → KIE → validation → fields.
- [ ] Pydantic `Document` schema + field normalizers.
- [ ] Rule engine: hard failures vs soft warnings, with confidence flags.
- [ ] Persist metadata (SQLite) + image (MinIO), linked by document id.
- [ ] `GET /documents/{id}` retrieval.

### Done when 📦
- [ ] `/extract` returns schema-validated structured fields with explicit validation flags,
      pulled from the MLflow-registered ONNX INT8 model; documents retrievable by id.

---

## Phase 5 — Monitoring, Observability & CI/CD

**Goal:** Production hygiene — metrics, logs, dashboards, automated checks. Completes the
core MLOps story.

### Research 🔬
- [ ] Prometheus metrics to expose (request count, latency histograms, error rate, KIE confidence).
- [ ] Grafana dashboard layout; Loki log pipeline.

### Tasks
- [ ] Instrument FastAPI with the Prometheus client; `/metrics` endpoint.
- [ ] Grafana dashboards + Loki log shipping in compose.
- [ ] GitHub Actions: lint, test, build + push image.

### Done when 📦
- [ ] Dashboards show live request/latency/error + model metrics; CI green on PRs.
- [ ] **Core complete:** `docker compose up` runs the full pipeline — fine-tuned model
      pulled from the registry, served on CPU, monitored — reproducible from a clean checkout.

---

# ADVANCEMENTS — after the core is green

Begin only once Phase 5's "core complete" criterion is met. Each advancement is additive
and pluggable; they may be done in any order based on appetite.

## A1 — Layout Detection

**Goal:** Add document-region structure ahead of OCR/KIE.

- [ ] 🔬 DocLayout-YOLO weights/license; ONNX export + CPU latency; does layout improve KIE?
- [ ] `LayoutDetector` interface + DocLayout-YOLO (ONNX Runtime).
- [ ] Integrate: preprocess → layout → region-scoped OCR → KIE.
- [ ] Visual debug output (boxes overlaid).
- [ ] 📦 `/extract` returns labeled regions; layout runs as ONNX on CPU within budget.

## A2 — GraphRAG over Extracted Data + `/ask`

**Goal:** Answer natural-language questions grounded in extracted documents.

- [ ] 🔬 Embedding model on CPU (quality vs latency); graph/chunking strategy for structured docs; RAGAS eval set.
- [ ] Build a graph/vector index over extracted JSON (with doc/field metadata).
- [ ] Retriever + prompt assembly; `POST /ask` → grounded answer + citations.
- [ ] RAGAS evaluation (faithfulness, answer relevancy).
- [ ] 📦 `/ask` answers over indexed documents with citations; RAGAS scores recorded.

## A3 — Agent Orchestration (LangGraph) + `/agent`

**Goal:** A multi-step agent chaining the capabilities as tools.

- [ ] 🔬 LangGraph state/graph design: extract → validate → retrieve → answer; failure handling, retries.
- [ ] Tools wrapping pipeline + RAG; LangGraph graph with conditional routing.
- [ ] Langfuse tracing on every node; expose via `POST /agent`.
- [ ] 📦 Agent handles compound tasks ("extract this doc, then answer X") in one call, fully traced.

## A4 — Kubernetes (kind) & Packaging

**Goal:** Demonstrate orchestration competence with production-style manifests.

- [ ] 🔬 `kind` cluster setup; CPU resource limits.
- [ ] K8s manifests: Deployment, Service, Ingress, ConfigMap/Secret, HPA.
- [ ] Deploy to `kind`; verify scaling behavior.
- [ ] 📦 App runs on `kind` from manifests in `infra/k8s/`; README documents architecture.

## A5 — On-Demand LLM KIE Backend

**Goal:** Showcase LLM fine-tuning + on-demand GPU serving without a persistent service.

- [ ] 🔬 QLoRA recipe for Qwen2.5-3B on instruction-style KIE; on-demand serving (Colab + ngrok).
- [ ] ☁️ QLoRA fine-tune Qwen2.5-3B; log to MLflow.
- [ ] Deploy as an on-demand endpoint behind the `KIEBackend` interface.
- [ ] Extend the benchmark: LLM vs encoder KIE (accuracy / latency / cost).
- [ ] 📦 `/extract?backend=llm` works on-demand; trade-offs documented.

---

## Cross-Cutting Standards (maintain throughout)

- [ ] Every component behind an interface and independently testable.
- [ ] No hardcoded constants — configuration via settings.
- [ ] Type hints + docstrings on public functions.
- [ ] Tests added with each phase; CI stays green.
- [ ] An MLflow run logged for every training / benchmark.
- [ ] README and architecture docs kept current.
- [ ] Each phase ends with a working `docker compose up`.

---

## First Step

Begin with **Phase 0**: scaffold the repository, `docker compose` stack, `/health`, and
dataset download. Every later phase builds on that slice; the core (Phases 0–5) ships
before any advancement begins.
