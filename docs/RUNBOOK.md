# Demo Runbook — Contract Intelligence Platform

A step-by-step guide to stand up the platform and walk the Streamlit demo end to
end. For the "what and why" of everything built, see
[`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).

All commands run from the **`docintel/`** directory unless noted.

---

## 1. Prerequisites

- Docker + Docker Compose (for the full stack), or Python 3.12 + [uv](https://docs.astral.sh/uv/) for local dev.
- The model bundles are too large for git and live (git-ignored) under root `models/`:
  - `models/cord-layoutlmv3-onnx-int8` — KIE model (receipt pipeline).
  - the CUAD contract extractor ONNX bundle — C1.
  - `models/rag-embed-cuad` — the CUAD-fine-tuned embedder — C2 (optional but recommended).
- The generative **LLM is optional**. Without it, `/ask` and `/agent` return
  citations-only (graceful degradation) — the demo still works.

---

## 2. Bring up the stack

```bash
cd docintel
cp .env.example .env
docker compose up -d --build      # first build is slow: CPU torch + baked docTR weights
```

This starts the API (`:8000`), MLflow (`:5000`), MinIO (`:9000`/`:9001`), Qdrant
(`:6333`), Neo4j (`:7474`/`:7687`), Langfuse (`:3000`), and the Prometheus/Loki/
Grafana observability stack. The compose file wires the Neo4j password to the API
automatically.

Smoke test:

```bash
curl http://localhost:8000/health
```

> **Local (no Docker) alternative:** `uv sync --all-extras` then
> `uvicorn docintel.api.main:app --reload`. You will need Qdrant and Neo4j
> reachable (or accept that `/ask` graph routes and vector search degrade).

---

## 3. Point the API at the local models

The API loads models from the MLflow registry by default. For laptop CPU serving,
bypass the registry with the local-path escape hatches (in `.env` or the shell):

```bash
DOCINTEL_KIE_ONNX_LOCAL_PATH=/models/cord-layoutlmv3-onnx-int8   # receipt KIE
DOCINTEL_CONTRACT_ONNX_LOCAL_PATH=/path/to/cuad-extractor-onnx-int8   # C1

# Recommended C2 config: the fine-tuned embedder with the generic reranker OFF
DOCINTEL_RAG_EMBEDDING_LOCAL_PATH=/path/to/models/rag-embed-cuad
DOCINTEL_RAG_RERANK_MODEL=                                       # blank = no rerank
```

> **Why rerank off:** on the CUAD-fine-tuned ordering the generic ms-marco reranker
> *subtracts* recall@5 (0.745 → 0.513). See the
> [C2 embedder fine-tune report](phases/c2-embed-finetune/report_c2_embed_finetune.md).
>
> **Re-index after switching embedders:** the fine-tuned model defines a new
> embedding space. Delete the `contract_chunks` Qdrant collection; it rebuilds on
> the next `/contracts/extract`.

---

## 4. Run the demo UI

```bash
streamlit run src/docintel/ui/app.py
```

The API base URL and request timeout come from `Settings`
(`DOCINTEL_UI_API_BASE_URL`, default `http://localhost:8000`).

---

## 5. Walk the demo (six views)

| View | What to do | What you see |
|---|---|---|
| **Overview** | (nothing) | Architecture graph, **live API health**, and the configured backends (Qdrant / Neo4j / LLM). Red banner if the API is down. |
| **Extract (C1)** | Upload a contract PDF → **Extract** | Structured clauses (type, text, confidence), derived fields, raw JSON. The extracted contract is auto-indexed into RAG + the graph, and its id is remembered for the next views. |
| **Ask (C2/C3)** | Type a question → **Ask** | A grounded answer (or **citations-only** if no LLM), with the cited chunks. Tick "Only the last extracted contract" to scope. |
| **Agent (C4)** | Type a compound task → **Run agent** | The LangGraph agent's answer plus status / retries / steps and (if configured) a Langfuse trace id. Degrades to citations-only without an LLM. |
| **Graph (C3)** | Ask e.g. *"Which contracts expire within 90 days?"* → **Query graph** | The Neo4j-routed evidence rendered as a **contract → fact network** (Graphviz) plus the citation table. |
| **Metrics** | (nothing) | Committed eval JSON rendered as tables: retrieval recall@k / MRR (with **embedder identity**), RAGAS faithfulness / answer-relevancy, and per-category recall. |

### Degradation is a feature

If the LLM endpoint is unset/unreachable, `/ask` and `/agent` return **HTTP 200**
with the deterministic retrieval/routing output, `answer: null`, and
`status: "degraded"`. The UI surfaces this as a warning and shows citations. Nothing
crashes.

---

## 6. Optional: enable generation and tracing

**LLM (for real answers):** self-hosted open model on Colab GPU via vLLM, reached
through a local relay so an ngrok URL change can't break a long run:

```bash
DOCINTEL_LLM_BASE_URL=http://127.0.0.1:8899
DOCINTEL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
DOCINTEL_LLM_API_KEY=...            # if the endpoint requires it
```

**Langfuse tracing (best-effort, never blocks a request):**

```bash
docker compose up -d langfuse
DOCINTEL_LANGFUSE_HOST=http://localhost:3000
DOCINTEL_LANGFUSE_PUBLIC_KEY=pk-...
DOCINTEL_LANGFUSE_SECRET_KEY=sk-...
```

---

## 7. Populate the Metrics view

The Metrics view scans `DOCINTEL_UI_EVAL_DIR` (default `.` — the `docintel/` dir when
run as above) for `eval_rag*.json` and `eval_ragas*.json`. Generate a run:

```bash
# Retrieval recall@k / MRR over CUAD (writes the embedder identity into the JSON)
python -m docintel.scripts.eval_rag --sample 40 --seed 0 --top-ks 1,3,5,30 \
  --no-rerank --out eval_rag_finetuned.json

# LLM-judged answer quality (needs the Colab judge; serialize workers)
python -m docintel.scripts.eval_ragas --out eval_ragas_new.json
```

Refresh the Metrics tab and the new run appears, tagged with its embedder.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/ask` graph route returns 503 | Neo4j unreachable or password mismatch. Compose wires it; for local dev set `DOCINTEL_NEO4J_PASSWORD` to match the container's `NEO4J_AUTH`. |
| Vector answers look worse after fine-tuning | Reranker still on. Set `DOCINTEL_RAG_RERANK_MODEL=` (blank) and re-index. |
| `test_config_*` fail locally | A local `docintel/.env` shadows the defaults those tests assert. Run the suite with `.env` renamed aside (this is expected). |
| Metrics view is empty | No `eval_*.json` in `DOCINTEL_UI_EVAL_DIR`. Run step 7, or point the var at your results dir. |
| Container exits at startup | Ensure the image was built with the runtime extras (it is, via the Dockerfile); rebuild with `docker compose up --build`. |
