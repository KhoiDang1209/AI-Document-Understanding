# Project Overview — What We Built

A single narrative of the whole project: the two products in this repo, how they
fit together, the engineering practices behind them, and the results — with links
to the detailed per-phase reports. For a hands-on demo, see
[`RUNBOOK.md`](RUNBOOK.md).

---

## 1. What this is

An **AI-engineering portfolio project** built to demonstrate the full lifecycle —
data → fine-tuning → optimization → serving → MLOps → observability → RAG → agents —
rather than chasing a single accuracy number. It contains **two products** on one
shared, typed, tested, reproducible codebase:

1. **Document AI pipeline (Phases 0–5)** — turns a receipt image into validated,
   structured, queryable JSON, served on CPU and fully observable.
2. **Contract Intelligence Platform (C1–C4)** — turns contract PDFs into extracted
   clauses, then answers questions over them with vector RAG, a knowledge graph,
   and an orchestrating agent.

Compute model: **build-time work (training, fine-tuning, ONNX export, INT8) runs on
Colab GPU; run-time (API, inference, observability) runs locally on CPU.** Models
are trained on Colab, registered in MLflow, and pulled locally as optimized ONNX.

---

## 2. Architecture at a glance

```
Document pipeline (Phases 0–5)          Contract Intelligence (C1–C4)
  Image → Preprocess → OCR (docTR)        PDF → C1 Extract (DeBERTa-v3 QA, ONNX-INT8)
        → KIE (LayoutLMv3, ONNX-INT8)          ├─→ C2 Vector RAG (hybrid BM25+dense, rerank)
        → Decode (BIO → fields)                ├─→ C3 GraphRAG (Neo4j date/renewal templates)
        → Validation (Pydantic + rules)        └─→ C4 Agent (LangGraph: route→retrieve→generate→critique)
        → Persist (SQLite + MinIO)
```

- Full reference architecture: [`pipeline.md`](pipeline.md)
- Contract flow + degradation matrix: [`architecture.md`](architecture.md)

---

## 3. Phase-by-phase (Document AI)

| Phase | Title | Outcome | Report |
|---|---|---|---|
| 0 | Foundations & environment | Repo, config, CI, compose skeleton | [report](phases/phase0/report_phase0.md) |
| 1 | OCR baseline + `/extract` | docTR words+boxes; upload pipeline | [report](phases/phase1/report_phase1.md) |
| 2 | KIE fine-tune (LayoutLMv3) + MLflow | Fine-tuned on CORD; registered | [report](phases/phase2/report_phase2.md) |
| 3 | Optimization: ONNX + INT8 + benchmark | CPU-served INT8; accuracy/latency/size | [report](phases/phase3/report_phase3.md), [benchmark](benchmark.md) |
| 4 | Serving + validation + schema + persistence | `Document` schema, rule engine, SQLite+MinIO | [report](phases/phase4/report_phase4.md) |
| 5 | Monitoring & observability | Prometheus + Loki + Grafana, provisioned | [report](phases/phase5/report_phase5.md) |

Validation **annotates, never blocks**: a receipt failing reconciliation still
returns `200` with `validation.ok = false` and the specific issues.

---

## 4. Contract Intelligence (C1–C4)

| Stage | Endpoint | What it does | Report |
|---|---|---|---|
| **C1** Extract | `POST /contracts/extract` | PDF → 41 CUAD clause types via a fine-tuned DeBERTa-v3 extractive-QA model, served as dynamic-INT8 ONNX on CPU. Dual-path (digital text layer vs docTR OCR). Auto-indexes into RAG + graph. | [report](phases/c1/report_c1.md) |
| **C2** Vector RAG | `POST /ask` | Hybrid BM25 + dense retrieval, focused query rewriting, optional cross-encoder rerank, grounded generate-or-degrade. Embedder fine-tuned on CUAD. | [report](phases/c2/report_c2.md), [retrieval boost](phases/c2-retrieval-boost/report_c2_retrieval_boost.md), [embed fine-tune](phases/c2-embed-finetune/report_c2_embed_finetune.md) |
| **C3** GraphRAG | `POST /ask` (graph route) | Neo4j knowledge graph of contracts→dates→renewal clauses; two deterministic Cypher templates (no text-to-Cypher). A rule router picks graph vs vector. | [report](phases/c3/report_c3.md) |
| **C4** Agent | `POST /agent` | LangGraph state machine: route → retrieve → generate → critique, with one bounded retry that falls back graph→vector. Best-effort Langfuse tracing. | — |

**Graceful degradation is mandatory:** when the LLM is unset/unreachable, `/ask` and
`/agent` return `200` with deterministic retrieval output, `answer: null`, and
`status: "degraded"`. The LLM is the only GPU-bound, optional component.

---

## 5. Headline results (honest read)

**KIE optimization (Phase 3):** INT8 ONNX served on CPU — accuracy/latency/size
trade-off in [`benchmark.md`](benchmark.md).

**C2 retrieval — the load-bearing evidence** (40 CUAD contracts, seed 0, 1,253
queries; from the [embed fine-tune report](phases/c2-embed-finetune/report_c2_embed_finetune.md)):

| Stack | Recall@1 | Recall@5 | Recall@30 | MRR |
|---|---|---|---|---|
| Stock (full stack) | 0.206 | 0.494 | 0.816 | 0.373 |
| **Fine-tuned, no reranker (recommended)** | **0.316** | **0.745** | **0.949** | **0.528** |

Recall@5 **0.494 → 0.745 (+51% relative)**; success bar (≥ 0.65) met. Three
formerly-zero clause categories recovered to recall@5 0.55 / 1.00 / 1.00. A notable
finding: the *generic* reranker, helpful on the stock embedder, **harms** the
fine-tuned ordering (0.745 → 0.513) — so production runs with rerank off, and a
CUAD-tuned reranker is the obvious next lever.

**RAGAS answer quality** (5 contracts, 40 questions, Qwen2.5-7B judge): mostly
**within noise** at this sample size — the retrieval eval is the real evidence. The
one consistent signal is answered-only faithfulness improving with better-retrieved
context. Reported transparently in the fine-tune report rather than overclaimed.

> Provenance: every retrieval eval JSON now records the **embedder identity**
> (`embedding_model` / `embedding_local_path`), so a stock vs fine-tuned run is
> distinguishable from the file itself — not just its name. The C3 fake↔real graph
> store is guarded by a full-row parity test that runs against a live Neo4j in CI.

---

## 6. Engineering practices

- **Typed & strict:** full type hints; `mypy --strict` on `src`.
- **Functional style:** small, focused functions; pure/testable logic split from I/O
  and rendering (e.g. the UI's `contract_client` / `eval_report` helpers vs the thin
  Streamlit page).
- **Test-driven:** behavioral changes land red→green; the suite is the contract.
  267 tests at last run.
- **CI:** ruff (lint + format), `mypy src`, and pytest on every push/PR; a separate
  job stands up a live Neo4j for the GraphRAG parity test.
- **Reproducible:** config via `DOCINTEL_`-prefixed settings; datasets tracked with
  DVC; models registered in MLflow; seeds fixed in evals.
- **Observable:** Prometheus metrics (incl. custom KIE-confidence and validation
  counters), Loki logs, provisioned Grafana dashboards — the stack boots fully wired.

---

## 7. The demo

A Streamlit app (`src/docintel/ui/app.py`) with six views — Overview (architecture
+ live health), Extract, Ask, Agent, Graph (C3 evidence network), and Metrics
(committed eval JSON). See [`RUNBOOK.md`](RUNBOOK.md) to run and walk it.

---

## 8. Reproduce

```bash
cd docintel
uv sync --all-extras                       # full env (sync ALL extras for the suite)
uv run ruff check . && uv run mypy src
uv run pytest                              # slow tests deselected by default

# Retrieval eval (writes embedder identity into the JSON)
python -m docintel.scripts.eval_rag --sample 40 --seed 0 --top-ks 1,3,5,30 --no-rerank
```

Full setup, model paths, LLM/tracing, and the demo walkthrough are in
[`RUNBOOK.md`](RUNBOOK.md).

---

## 9. Repository map

```
docintel/
  src/docintel/
    pipeline/  preprocess → layout → OCR → KIE → validation
    kie/       key-information extraction (LayoutLMv3, ONNX-INT8)
    contracts/ C1 clause extraction (DeBERTa-v3 QA)
    rag/       C2 hybrid retrieval + generate-or-degrade
    graph/     C3 Neo4j store, templates, router
    agent/     C4 LangGraph orchestration
    ui/        Streamlit demo (client + eval_report helpers + app)
    api/       FastAPI app + routes
    scripts/   eval_rag / eval_ragas / eval_graph, data download
  tests/       unit & integration (pytest)
  monitoring/  Prometheus / Loki / Promtail / Grafana (provisioned)
  docker-compose.yml, Dockerfile, pyproject.toml
docs/          this overview, runbook, architecture, benchmark, per-phase reports
models/        local model bundles (git-ignored)
```

---

## 10. Related docs

| Document | Purpose |
|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | Stand up the stack and walk the demo |
| [`pipeline.md`](pipeline.md) | End-to-end reference architecture |
| [`architecture.md`](architecture.md) | Contract Intelligence flow & degradation matrix |
| [`benchmark.md`](benchmark.md) | ONNX / INT8 accuracy · latency · size |
| [`phases/README.md`](phases/README.md) | Per-phase status and completion reports |
| [`plan.md`](plan.md) · [`proposal.md`](proposal.md) · [`research.md`](research.md) | Roadmap, design decisions, feasibility |
