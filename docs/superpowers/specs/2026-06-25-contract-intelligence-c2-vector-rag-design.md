# Contract Intelligence — C2 Design: Vector RAG + `/ask`

**Date:** 2026-06-25
**Status:** Approved (design); ready for implementation planning. **Depends on C1.**
**Initiative:** Contract Intelligence Platform. **Roadmap:** C1 → **C2 (this doc)** → C3 → C4.

## Goal

Answer natural-language questions grounded in the extracted contracts, with **citations**.
`POST /ask` retrieves relevant clause/text chunks from a vector index, assembles a grounded
prompt, generates an answer with the **Colab-hosted LLM (via ngrok)**, and returns the answer
plus its supporting citations. Quality is measured with **RAGAS** (LLM judge = the same
Colab LLM, so eval is free).

## Context

- **C1** produces persisted `ContractDocument`s: per contract, a list of `ExtractedClause`
  `{clause_type, answer_text, char_start, char_end, confidence}` + full ingested text.
- Reuse the spine: **MLflow** (embedding + RAGAS eval), **MinIO/SQLite** (already hold the
  records), **Prometheus/Grafana/Loki** (latency, LLM call metrics, RAGAS scores).
- **New infra:** Qdrant (vector store) added to Compose. The LLM is external (Colab/ngrok),
  treated as **optional/intermittent** (see [[llm-serving-colab-ngrok]]).

## Decisions (locked during brainstorming)

1. **What gets indexed:** two complementary chunk kinds, both with metadata
   `{contract_id, clause_type?, char_start, char_end}`:
   - **Clause chunks** — one per `ExtractedClause` (precise, citation-ready).
   - **Paragraph chunks** — sliding paragraphs over the full ingested text (coverage for
     questions outside the 41 clause types).
2. **Embeddings on CPU:** a small sentence-transformer (`bge-small-en-v1.5` preferred,
   `all-MiniLM-L6-v2` fallback), **ONNX-exported** for CPU — same optimize-then-serve pattern.
   Retrieval quality (recall@k) evaluated and logged to **MLflow**.
3. **Vector store = Qdrant** as a Compose service with a persisted volume. (FAISS in-process
   was considered; Qdrant chosen for the production/observability story and C3 reuse.)
4. **Generation = Colab LLM via ngrok**, OpenAI-compatible, configured by
   `DOCINTEL_LLM_BASE_URL`. **Graceful degradation:** if the endpoint is down, `/ask` returns
   the **ranked, cited extractive chunks** (no synthesized prose) with a `degraded` flag. The
   CPU path never hard-depends on the GPU.
5. **Citations are mandatory and structured:** every answer carries citations
   `{contract_id, clause_type?, char_start, char_end}`. The prompt instructs the LLM to ground
   strictly in retrieved context; ungrounded answers are surfaced by the faithfulness metric.
6. **RAGAS eval set from CUAD:** build a QA set from CUAD clause questions (answer spans =
   ground-truth context). Metrics: **faithfulness, answer relevancy, context precision/recall**.
   Judge = Colab LLM. Scores → MLflow, surfaced in Grafana.
7. **No graph, no agent** — those are C3/C4. `/ask` here is vector-only retrieval.

## Architecture / new modules

```
src/docintel/rag/
  chunk.py     (new)  ContractDocument → clause chunks + paragraph chunks (+ metadata)
  embed.py     (new)  CPU ONNX embedding model (lazy on app.state)
  index.py     (new)  Qdrant upsert / collection mgmt
  retrieve.py  (new)  query → top-k chunks (with metadata for citations)
  prompt.py    (new)  assemble grounded prompt from retrieved chunks
  generate.py  (new)  OpenAI-compatible client → Colab LLM (timeout, degrade-on-failure)

api/routes/ask.py (new)  POST /ask → retrieve → prompt → generate → answer + citations
eval/ragas_eval.py (new) RAGAS over the CUAD QA set; scores → MLflow

infra/  qdrant service added to docker-compose; DOCINTEL_LLM_BASE_URL, QDRANT_URL, top_k config
```

## Data flow

```
(index build, offline)  ContractDocument → chunk() → embed() → index() [Qdrant]
(query)  question → embed → retrieve top-k → prompt → generate (Colab LLM)
         → answer + citations   (or degraded: cited chunks only)
         → metrics (retrieval/LLM latency, degraded count) → Prometheus
```

## Testing (TDD)

- **Unit:** chunking (clause + paragraph, offsets preserved); prompt assembly; generate client
  **degrades** correctly when the endpoint is unreachable (mock); citation construction.
- **Integration:** index a fixture contract → `/ask` returns answer + valid citations pointing
  at real spans; degraded path returns cited chunks with the flag.
- **Eval (not in CI):** `ragas_eval.py` produces RAGAS scores + retrieval recall@k → MLflow.

## Metrics delivered (for the CV)

| Metric | Where |
|---|---|
| **RAGAS** faithfulness / answer-relevancy / context-precision/recall | `ragas_eval.py` → MLflow |
| Retrieval **recall@k / MRR** | embedding eval → MLflow |
| `/ask` end-to-end **latency**, **degraded-rate** | Prometheus → Grafana |

## Out of scope (later)

- Knowledge graph + multi-hop traversal — **C3**.
- Agent orchestration — **C4**.

## Risks & mitigations

- **LLM intermittent:** designed-for via the degraded path (Decision 4).
- **Embedding quality on legal text:** measured (recall@k); swap model if weak.
- **Eval-set construction:** start from CUAD's native question/answer spans; keep it small,
  documented, versioned with the repo.
