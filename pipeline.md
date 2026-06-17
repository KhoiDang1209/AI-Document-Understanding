# AI Document Understanding System — End-to-End Architecture

> **Goal:** Turn a document image (invoice / receipt) into **structured, validated JSON** — production-ready, scalable, and reliable.

This document summarizes the reference architecture in `End-to-End-Pipeline.png`.

---

## High-Level Flow

```
Document Image (SROIE invoice/receipt)
  → 1. Layout Analysis     (DocLayout-YOLO)
  → 2. Text Detection      (MixNet)
  → 3. Text Recognition    (PARSeq)
  → 4. Key Information Extraction (KIE backend: LayoutLMv3 | Qwen2.5-3B | Vintern-1B-3.5)
  → Structured Data (JSON)
  → API Serving (FastAPI) → Users & Integrations
```

Two flow types in the diagram:
- **Data Flow** (solid): runtime inference path.
- **Control / Deploy Flow** (dashed): training, registry, and deployment wiring.

---

## A. Training & Fine-Tuning

Offline stage that produces the trained models consumed by the inference pipeline.

| Step | Component | Purpose |
|------|-----------|---------|
| 1. Datasets | **SROIE Dataset** (invoices/receipts) | Training/eval data |
| 2. Layout Model Training | **DocLayout-YOLO** | Detect document regions |
| 3. Text Detection Training | **MixNet** | Locate text boxes |
| 4. OCR Recognition Training | **PARSeq** | Read text from boxes |
| 5. KIE Training / Fine-tuning | **LayoutLMv3 / Qwen / Vintern** | Extract key fields |
| Experiment Tracking | **MLflow** | Track runs, register models |

All trained models are pushed to the **MLflow Model Registry**.

---

## Inference Pipeline (Core)

### 1. DocLayout-YOLO — Layout Analysis
Classifies regions: **Text, Title, Table, Key Value, Other**.
*Output:* regions + bounding boxes.

### 2. MixNet — Text Detection
Finds individual text regions.
*Output:* text boxes + confidence.

### 3. PARSeq — Text Recognition
Reads text inside each detected box.
*Output:* text + coordinates + confidence (e.g. `Invoice No: INV-001234`, `Date`, `Vendor`, `Total`).

### 4. KIE — Key Information Extraction (pluggable backend)
Choose **one** backend:

| Option | Model | Notes |
|--------|-------|-------|
| A | **LayoutLMv3** | Document-AI model, served as **ONNX INT8** |
| B | **Qwen2.5-3B** | Instruction-based KIE, served via **vLLM** |
| C | **Vintern-1B-3.5** | Lightweight VLM, served via **vLLM** |

**Output:** `Structured Data (JSON)` — fields like `company`, `date`, `address`, `total`, line items, each with values/confidence.

---

## B. Model Optimization
1. **Export to ONNX**
2. **INT8 Quantization**
3. **Benchmark & Evaluation** — F1 Score, Latency, Throughput, Resource Usage.

---

## Model Registry & Storage
- **MLflow Model Registry** — versioned models.
- **Model Storage** — MinIO / GCS / S3 (object store backing the registry).

---

## C. API & Serving Layer
**FastAPI** REST API:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/ocr` | OCR inference |
| POST | `/kie` | KIE inference |
| GET | `/health` | Health check |

Characteristics: **High Performance, Async Support, Auto Docs (Swagger UI), Validation.**

---

## D. Deployment & Infrastructure
- **Docker Compose (recommended):** FastAPI Container + vLLM Server + Redis (optional).
- **Kubernetes (bonus):** Deployment → Service → Ingress → Auto Scaling.

---

## E. Monitoring & Observability
- **Prometheus** — metrics
- **Grafana** — dashboards
- **Loki** — logs
- **Alertmanager** — alerting

**Monitored signals:** Request Count / Throughput, Latency / Error Rate, GPU / CPU Usage, OCR Accuracy / KIE F1 Score.

---

## F. Users & Integrations
Consumers of the API: **Web App, Mobile App, Third-party System, Data Pipeline / Automation.**

---

## Component Summary by Layer

| Layer | Tech |
|-------|------|
| Data & Training | SROIE, DocLayout-YOLO, MixNet, PARSeq, LayoutLMv3/Qwen/Vintern |
| Pipeline (OCR/KIE) | Layout → Detection → Recognition → KIE |
| Optimization | ONNX, INT8 quantization, benchmarking |
| Registry & Storage | MLflow, MinIO/GCS/S3 |
| Serving | FastAPI, vLLM, Swagger |
| Infrastructure | Docker Compose, Kubernetes, Redis |
| Monitoring | Prometheus, Grafana, Loki, Alertmanager |
| Users | Web/Mobile apps, third-party systems, automation pipelines |
