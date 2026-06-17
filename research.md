# Feasibility Research — Document AI as a Personal AI-Engineer Portfolio Project

**Context:** Personal project to demonstrate end-to-end AI Engineering skills for job applications.
**Hardware budget:** 1 laptop + Google **Colab Pro** subscription. Goal is **$0 extra spend** beyond Colab Pro.

> **Assumption to confirm:** This doc assumes the laptop has **no (or only a small) NVIDIA GPU** and ~16 GB RAM — i.e. CPU-class local inference. Where a *local discrete GPU* changes the verdict, it's called out. Please confirm laptop specs (RAM, GPU/VRAM, OS) so the plan can be tightened.

---

## 1. TL;DR Verdict

**Yes — the project is viable**, but **not as a single 24/7 GPU-served system**. The realistic shape is:

- **Train / fine-tune / benchmark** → on **Colab Pro** (ephemeral GPU sessions).
- **Optimize** (ONNX + INT8) so models run **on the laptop CPU**.
- **Serve** the optimized pipeline via **FastAPI in Docker on the laptop** (and a free public demo on **Hugging Face Spaces**).
- **RAG + agents + MLOps + monitoring** → all fully doable **free**, mostly on CPU.

The one thing that is genuinely **over the horizon for free**: a **persistent, 24/7, GPU-backed LLM endpoint** (e.g. vLLM serving Qwen around the clock). Solve it by serving LLM features *on-demand* or with a quantized small model on CPU.

---

## 2. Hardware Reality Check

### Colab Pro (what it actually gives you)
- **GPU types:** T4 (16 GB) most common; sometimes L4 (24 GB), V100, or A100 (40 GB) when available. **No guaranteed type.**
- **Compute units:** ~100 units/month included. Burn rate varies wildly by GPU:
  - T4 ≈ low burn → roughly **tens of hours/month**.
  - A100 ≈ high burn → roughly **a handful of hours/month**.
- **High-RAM** runtime (~32–51 GB system RAM) available.
- **Background execution** + longer sessions, but **sessions still disconnect** and are **not guaranteed to last 24h**.
- **No persistent disk** — must mount Google Drive or re-download each session.

**Implication:** Colab Pro is a **batch compute environment** (train, quantize, benchmark, generate). It is **not** a hosting platform. Don't design anything that assumes a Colab GPU is always up.

### Laptop (assumed CPU-class)
- Great for: dev, Docker, FastAPI, vector DBs, ONNX/quantized inference of small models, RAG embeddings, orchestration code, local k8s (kind/minikube, CPU).
- Bad for: training, running unquantized 3B+ LLMs, high-throughput GPU serving.

### Free public hosting options (for a clickable demo link on your CV)
| Platform | Free tier | Good for |
|----------|-----------|----------|
| **Hugging Face Spaces** | Free **CPU** (2 vCPU/16 GB), Gradio/Docker | Public demo of the CPU pipeline + RAG |
| **Render / Railway / Fly.io** | Small free/credit tiers, CPU | FastAPI demo |
| **Modal / Replicate** | Some free credits, **serverless GPU** | On-demand LLM/VLM inference (not 24/7) |
| **Streamlit Community Cloud** | Free CPU | Quick UI |

> Free **persistent GPU** hosting essentially does not exist. Use **serverless GPU** (Modal/Replicate free credits) for occasional LLM calls, or quantized CPU models.

---

## 3. Component-by-Component Feasibility

Legend: ✅ free on laptop · 🟡 needs Colab Pro (batch) · 🟠 free but limited/on-demand · 🔴 over the horizon for free

| Stage | Model / Tech | Train | Inference | Verdict |
|-------|--------------|-------|-----------|---------|
| Layout | **DocLayout-YOLO** (small) | T4 fine-tune 🟡 | CPU/ONNX ✅ | ✅ Fully viable |
| Text detection | **MixNet** / DBNet | T4 🟡 | CPU ✅ | ✅ |
| Text recognition | **PARSeq** | T4 🟡 | CPU (slower) ✅ | ✅ |
| OCR (shortcut) | **PaddleOCR / docTR** pretrained | — | CPU ✅ | ✅ Use as baseline to skip training |
| KIE (encoder) | **LayoutLMv3** (~125–368M) | **QLoRA/full on T4** 🟡 | **ONNX INT8 on CPU** ✅ | ✅ Best fit |
| KIE (LLM) | **Qwen2.5-3B-Instruct** | **QLoRA on T4** 🟡 | 4-bit ~4 GB GPU 🟠 / CPU slow | 🟠 On-demand only |
| KIE (VLM) | **Vintern-1B / Qwen2-VL-2B** | LoRA on T4 🟡 | small GPU 🟠 | 🟠 |
| Optimization | **ONNX export, INT8 (static/dynamic), ONNXRuntime** | — | CPU ✅ | ✅ Strong portfolio piece |
| Optimization+ | **TensorRT, GPTQ/AWQ, distillation** | Colab 🟡 | GPU 🟠 | 🟡 Doable as experiments |
| Serving | **FastAPI + Uvicorn** | — | CPU ✅ | ✅ |
| LLM serving | **vLLM (24/7 GPU)** | — | GPU 🔴 | 🔴 Not free persistent → use serverless/on-demand |
| RAG | **bge-small/e5 embeddings + FAISS/Chroma/Qdrant** | — | CPU ✅ | ✅ Fully viable |
| Agents/Orchestr. | **LangGraph / LlamaIndex + tool use** | — | CPU (calls LLM) ✅ | ✅ |
| Registry | **MLflow** (local or SQLite+filestore) | — | CPU ✅ | ✅ |
| Storage | **MinIO** (local S3) | — | CPU ✅ | ✅ |
| Containerization | **Docker / docker-compose** | — | CPU ✅ | ✅ |
| Orchestration | **Kubernetes** | — | **kind/minikube CPU** ✅ / GPU cluster 🔴 | 🟡 Learn locally, CPU-only |
| Monitoring | **Prometheus + Grafana + Loki** | — | CPU ✅ | ✅ |
| LLM observability | **Langfuse / Phoenix** | — | CPU ✅ | ✅ |
| CI/CD | **GitHub Actions** (free tier) | — | cloud ✅ | ✅ |

---

## 4. Datasets (all free / public)

| Dataset | Task | Notes |
|---------|------|-------|
| **SROIE** (ICDAR 2019) | Receipt OCR + KIE (4 fields) | Small, classic, in the reference diagram |
| **CORD** | Receipt KIE (30+ fields) | Best for richer KIE; HF-hosted |
| **FUNSD** | Form understanding / KIE | 199 forms, entity+relation |
| **XFUND** | Multilingual forms | FUNSD's multilingual sibling |
| **DocVQA** | Document QA | Great for the **RAG / VQA** angle |
| **DocLayNet** / **PubLayNet** | Layout detection | Large, for layout model training |
| **RVL-CDIP** | Doc classification (16 classes) | Optional classification stage |
| **WildReceipt / Kleister** | Receipts / long docs KIE | Extra variety |

**Recommendation:** Start with **CORD** (clean, rich KIE labels, small enough for Colab) + **SROIE** (matches the diagram). Add **DocVQA** when you build the RAG layer.

---

## 5. What's Over the Horizon (and the cheap workaround)

| Ambition | Why it's blocked | Pragmatic substitute |
|----------|------------------|----------------------|
| 24/7 vLLM GPU endpoint | No free persistent GPU | Serverless GPU (Modal free credits) **or** 4-bit small model on-demand **or** CPU INT8 encoder KIE |
| Training 7B+ from scratch / full fine-tune | VRAM + compute units | **QLoRA on ≤3B**, or just fine-tune LayoutLMv3 |
| Real GPU-autoscaling K8s cluster | Costs money | **kind/minikube CPU** to demonstrate manifests, HPA, ingress |
| High-throughput load/latency benchmarking at scale | Single GPU, ephemeral | Benchmark CPU INT8 locally + one Colab GPU run; report curves, not scale |
| TensorRT / multi-GPU optimization | Hardware | ONNXRuntime + dynamic/static INT8 (covers the *skill* convincingly) |

None of these block the project — they just reshape *how* you demonstrate the skill.

---

## 6. AI-Engineer Skill Coverage Map

This is the point of the project — showing breadth. Mapping each AI-Engineer competency to a concrete, feasible deliverable:

| AI-Engineer competency | Covered by | Feasible? |
|------------------------|-----------|-----------|
| Data engineering / preprocessing | Doc → image → OCR dataset pipeline (CORD/SROIE) | ✅ |
| Model fine-tuning | LayoutLMv3 + QLoRA on Qwen2.5-3B (Colab) | ✅ |
| Model optimization | ONNX export + INT8 quantization + benchmark | ✅ (flagship) |
| Inference serving | FastAPI + batching + async | ✅ |
| **RAG** | Index extracted fields + DocVQA; bge embeddings + Qdrant; RAGAS eval | ✅ |
| **LLM orchestration / agents** | LangGraph agent: "extract → validate → answer" tool flow | ✅ |
| Evaluation | KIE F1, OCR CER/WER, latency/throughput, RAGAS faithfulness | ✅ |
| MLOps / experiment tracking | MLflow runs + model registry | ✅ |
| Data/model versioning | DVC + Git | ✅ |
| Containerization | Docker / docker-compose | ✅ |
| Orchestration | K8s manifests on kind (CPU) | 🟡 (CPU demo) |
| Monitoring / observability | Prometheus + Grafana + Loki; Langfuse for LLM | ✅ |
| CI/CD | GitHub Actions (lint, test, build, push image) | ✅ |
| Cost/latency trade-off analysis | Written benchmark report (encoder vs LLM KIE) | ✅ |

**Conclusion:** Every core AI-Engineer competency is coverable **for free** with this hardware. RAG and agents — which the original diagram lacks — are the easiest wins and worth adding explicitly.

---

## 7. Recommended Architecture Adjustments

1. **Split "build-time" vs "run-time" environments explicitly.** Colab = train/quantize/benchmark; laptop/Docker = serve. Make this a documented design decision (it *is* real MLOps).
2. **Default KIE backend = LayoutLMv3 (ONNX INT8 on CPU).** Cheap, fast, always-on. Keep Qwen/VLM as an **optional on-demand backend** behind the same `/kie` interface — demonstrates the pluggable-backend design without needing a persistent GPU.
3. **Add a RAG + agent layer on top of extracted JSON** ("ask questions about your documents"). This converts a pure pipeline into an *application* and covers RAG/agent skills.
4. **Public demo on HF Spaces (CPU)**; serverless GPU (Modal) only for the optional LLM path.
5. **Keep everything reproducible** (Docker, DVC, MLflow) so a reviewer can `docker compose up` and see it work.

---

## 8. Suggested Phased Plan (each phase = a shippable milestone)

| Phase | Deliverable | Where |
|-------|-------------|-------|
| 0 | Repo, env, Docker skeleton, MLflow, data download (CORD/SROIE) | Laptop |
| 1 | OCR baseline (PaddleOCR/docTR) + FastAPI `/ocr` | Laptop |
| 2 | Layout (DocLayout-YOLO) + recognition pipeline | Colab train → laptop serve |
| 3 | KIE with LayoutLMv3 fine-tune → ONNX INT8 → `/kie` | Colab + laptop |
| 4 | Optimization + benchmark report (accuracy/latency/size) | Both |
| 5 | RAG over extracted data + DocVQA, RAGAS eval | Laptop |
| 6 | LangGraph agent (extract→validate→answer) | Laptop |
| 7 | Monitoring (Prometheus/Grafana/Langfuse) + CI/CD | Laptop + GitHub |
| 8 | K8s manifests (kind) + HF Spaces public demo | Laptop + cloud |
| 9 (opt.) | QLoRA Qwen2.5-3B as on-demand LLM backend | Colab + Modal |

---

## 9. Open Questions to Confirm
1. **Laptop specs** — RAM, discrete GPU + VRAM, OS? (Changes local-GPU verdicts.)
2. **Primary CV target** — broad AI-Engineer generalist, or lean toward **LLM/RAG** vs **CV/Document-AI**? (Decides where to invest depth.)
3. **Demo expectation** — is a public clickable link required, or is a local `docker compose up` + README/video enough?
