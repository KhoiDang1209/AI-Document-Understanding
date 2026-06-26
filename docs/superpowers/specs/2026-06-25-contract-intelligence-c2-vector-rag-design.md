# Contract Intelligence — C2 Design: Vector RAG + `/ask`

**Date:** 2026-06-25 · **Revised:** 2026-06-26 (framework + embedding-runtime decisions)
**Status:** Approved (design); ready for implementation planning. **Depends on C1.**
**Initiative:** Contract Intelligence Platform. **Roadmap:** C1 → **C2 (this doc)** → C3 → C4.

## Goal

Answer natural-language questions grounded in the extracted contracts, with **citations**.
`POST /ask` retrieves relevant clause/text chunks from a vector index, assembles a grounded
prompt, generates an answer with the configured **OpenAI-compatible LLM** (Colab-hosted via
ngrok now; OpenAI API once the leftover Colab credits expire — see
[[llm-serving-colab-ngrok]]), and returns the answer plus its supporting citations. When no LLM
is reachable, `/ask` **degrades** to returning the ranked, cited chunks alone. Quality is
measured with **RAGAS** (LLM judge = the same configured LLM).

## Context

- **C1** produces persisted `ContractDocument`s: per contract, a list of `ExtractedClause`
  `{clause_type, answer_text, char_start, char_end, confidence}` + the full ingested text.
- Indexing is folded into the existing `POST /contracts/extract`: when a contract is extracted,
  its text + clauses are chunked, embedded, and upserted to Qdrant in the same flow
  (**best-effort** — a down Qdrant must never fail extraction).
- Reuse the spine: **MLflow** (retrieval + RAGAS eval), **MinIO/SQLite** (already hold the
  records), **Prometheus/Grafana/Loki** (latency, LLM call metrics, RAGAS scores).
- **New infra:** Qdrant (vector store) added to Compose. The LLM is external and treated as
  **optional/intermittent**.

## Decisions (locked during brainstorming)

1. **What gets indexed — at extract time, two complementary chunk kinds**, both with payload
   `{contract_id, chunk_index, chunk_kind, clause_type?, char_start, char_end}`:
   - **Clause chunks** — one per `ExtractedClause` (precise, citation-ready, carries `clause_type`).
   - **Paragraph chunks** — sliding char windows over the full ingested text (coverage for
     questions outside the 41 clause types; `clause_type=None`).
2. **Embeddings = `BAAI/bge-small-en-v1.5` (384-dim) via fastembed** — quantized ONNX on CPU,
   **no torch** in the serve path; matches the repo's ONNX-on-CPU serving theme. Wrapped behind
   the LangChain `Embeddings` interface. Retrieval quality (recall@k / MRR) evaluated → MLflow.
3. **Vector store = Qdrant** as a Compose service with a persisted volume, accessed via
   **`langchain-qdrant`**. (FAISS in-process was considered; Qdrant chosen for the
   production/observability story and C3 reuse.) Tests use Qdrant `location=":memory:"`.
4. **Framework = hybrid (hand-rolled primitives + LangChain orchestration).** Chunking and the
   fastembed embedder are hand-rolled and CPU-light; the vector store (`langchain-qdrant`),
   generation (`ChatOpenAI`), and the retrieve→prompt→generate **LCEL chain** use LangChain.
   Rationale: JD signal (LangChain/LangGraph) **and** continuity — `langchain-core` + `ChatOpenAI`
   plumbing carries straight into the **C4 LangGraph agent** — while staying lean (no torch).
5. **Generation = OpenAI-compatible `ChatOpenAI`**, configured by `DOCINTEL_LLM_BASE_URL` /
   `DOCINTEL_LLM_API_KEY` / `DOCINTEL_LLM_MODEL`. Switching Colab/ngrok → OpenAI changes only
   settings, no code. **Graceful degradation:** LLM "off" when `llm_base_url` is unset
   (`build_llm` returns `None`), or a configured call fails → `/ask` returns the ranked, cited
   chunks with `answer=null` and `generation_skipped=true`. The CPU path never hard-depends on
   any LLM being up, so the whole retrieval pipeline is testable on the laptop with no endpoint.
6. **Query scope = a single `POST /ask` with an optional `contract_id` filter.** Present → a
   Qdrant payload filter scopes retrieval to that contract; absent → corpus-wide. Each returned
   citation reports which contract (and clause type, if a clause chunk) it came from.
7. **Citations are mandatory and structured:** every answer carries citations
   `{contract_id, chunk_kind, clause_type?, char_start, char_end}`. The prompt instructs the LLM
   to ground strictly in retrieved context; ungrounded answers are surfaced by the faithfulness
   metric.
8. **RAGAS eval set from CUAD:** build a small QA set from CUAD clause questions (answer spans =
   ground-truth context). Metrics: **faithfulness, answer relevancy, context precision/recall**.
   Judge = configured LLM. Scores → MLflow, surfaced in Grafana. Harness built now; **run later**
   when an LLM is available (not in CI). Lives behind the optional `eval` extra.
9. **Idempotent upsert:** point ids are deterministic
   `uuid5(NAMESPACE, f"{contract_id}:{chunk_index}")`, so re-extracting a contract overwrites its
   chunks rather than duplicating them.
10. **No graph, no agent** — those are C3/C4. `/ask` here is vector-only retrieval.

## Architecture / new modules

```
src/docintel/rag/
  chunk.py   (new)  pure build_chunks(text, clauses, size, overlap) -> list[TextChunk]
                    (clause chunks + paragraph chunks, with offsets, kind, clause_type)
  embed.py   (new)  FastEmbedEmbeddings(LangChain Embeddings) over bge-small-en-v1.5; build_embedder
  store.py   (new)  Qdrant via langchain-qdrant: ensure collection, upsert (deterministic ids),
                    search(query, top_k, contract_id?) -> list[RetrievedChunk]
  index.py   (new)  index_contract(contract_id, text, clauses, store, settings) -> int
  llm.py     (new)  build_llm(settings) -> ChatOpenAI | None; grounded prompt; format_context
  answer.py  (new)  answer_question(question, store, llm, settings, contract_id=None) -> AskResponse
  schema.py  (new)  RetrievedChunk, AskRequest, AskResponse (pydantic)
  eval.py    (new)  RAGAS over the CUAD QA set; scores -> MLflow (run later; eval extra)

api/routes/ask.py (new)  POST /ask -> retrieve -> generate-or-degrade -> answer + citations
api/routes/contracts.py (modified)  after save_contract: best-effort index_contract(...)
api/main.py (modified)  register ask router; lifespan inits app.state.rag_store / rag_llm = None

infra/  qdrant service added to docker-compose; new DOCINTEL_ settings (below)
```

Each module is small and single-purpose; the embedder, vector store, and LLM are reached through
LangChain interfaces (or `None` for the LLM), so each unit is independently stubbable and
CPU-testable.

### Response schema

```
RetrievedChunk: { contract_id, chunk_index, chunk_kind: "clause"|"paragraph",
                  clause_type: str|None, text, score, char_start, char_end }
AskRequest:     { question: str, contract_id: str|None = None, top_k: int|None = None }
AskResponse:    { question, answer: str|None, generation_skipped: bool,
                  contract_id: str|None, citations: list[RetrievedChunk] }
```

## Data flow

```
(index, at extract time)
  POST /contracts/extract → ingest_pdf → text → C1 clause extract → persist
       → [best-effort] index_contract: build_chunks(text, clauses) → fastembed → Qdrant upsert
  → returns ContractDocument  (response shape unchanged)

(query)
  POST /ask {question, contract_id?, top_k?}
       → embed query → Qdrant search (filter by contract_id if given) → top-k chunks
       → if LLM reachable:  LCEL chain  prompt(context) | ChatOpenAI | StrOutputParser  → answer
       → else:              degrade (answer=null, generation_skipped=true)
       → AskResponse{ answer|null, generation_skipped, citations[] }
       → metrics (retrieval/LLM latency, degraded count) → Prometheus
```

## Configuration (additions to `Settings`, `DOCINTEL_` prefix — no hardcoded constants)

| Setting | Default |
|---|---|
| `rag_embedding_model` | `BAAI/bge-small-en-v1.5` |
| `rag_embedding_dim` | `384` |
| `rag_chunk_size` (chars, paragraph chunks) | `1200` |
| `rag_chunk_overlap` (chars) | `200` |
| `qdrant_url` | `http://qdrant:6333` |
| `qdrant_collection` | `contract_chunks` |
| `rag_top_k` | `5` |
| `llm_base_url` | `None` (LLM off → degrade) |
| `llm_api_key` | `None` |
| `llm_model` | `Qwen/Qwen2.5-7B-Instruct` |
| `llm_timeout_s` | `60.0` |

## Infrastructure & dependencies

- **docker-compose:** add a `qdrant` service (`qdrant/qdrant`, ports `6333`/`6334`,
  volume `qdrant-data:/qdrant/storage`); the `api` service gets
  `DOCINTEL_QDRANT_URL: http://qdrant:6333` and `depends_on: qdrant`.
- **pyproject extras:**
  - `rag = ["fastembed", "langchain-core", "langchain-openai", "langchain-qdrant", "qdrant-client"]`
  - `eval = ["ragas"]` (separate; RAGAS run only)
  - mypy `ignore_missing_imports` for `fastembed.*`, `qdrant_client.*`, `langchain.*`,
    `langchain_core.*`, `langchain_openai.*`, `langchain_qdrant.*`, `ragas.*`.

## Error handling

| Condition | Behavior |
|---|---|
| Empty/invalid `/ask` body | 422 (pydantic validation) |
| Qdrant unreachable on `/ask` | 503 |
| LLM unconfigured or call fails | Degrade: 200, `answer=null`, `generation_skipped=true`, citations populated |
| Indexing fails during extract | Best-effort: log and continue; extraction returns 200 |
| No chunks match | 200 with empty `citations` and degraded/empty answer |

## Testing (TDD, all CPU, no live LLM)

- **Unit:** chunking (clause + paragraph, offsets/kind/clause_type preserved, short text);
  store with Qdrant `location=":memory:"` (upsert + `contract_id` filter + deterministic-id
  idempotency); `answer` with stubbed store + stubbed LLM (generate path **and** degrade path);
  grounded-prompt + citation construction.
- **Integration:** `/ask` via FastAPI `dependency_overrides` — generate path returns answer +
  valid citations pointing at real spans; degraded path returns cited chunks with
  `generation_skipped=true`; `contract_id` filter passed through. Extract-still-succeeds-when-
  index-fails (best-effort).
- **Slow (deselected by default):** real bge-small embedding test (downloads model).
- **Eval (not in CI):** `eval.py` produces RAGAS scores + retrieval recall@k → MLflow.

## Metrics delivered (for the CV)

| Metric | Where |
|---|---|
| **RAGAS** faithfulness / answer-relevancy / context-precision/recall | `eval.py` → MLflow |
| Retrieval **recall@k / MRR** | embedding eval → MLflow |
| `/ask` end-to-end **latency**, **degraded-rate** | Prometheus → Grafana |

## Out of scope (later)

- Knowledge graph + multi-hop traversal — **C3**.
- Agent orchestration (LangGraph) — **C4** (this design's `langchain-core` + `ChatOpenAI`
  plumbing carries straight into it).

## Risks & mitigations

- **LLM intermittent / credits expire:** designed-for via the degraded path (Decision 5) and the
  settings-only Colab→OpenAI switch.
- **Embedding quality on legal text:** measured (recall@k); swap model if weak.
- **LangChain version churn:** pin the lean packages; keep LangChain at the orchestration seam
  only, primitives hand-rolled.
- **Eval-set construction:** start from CUAD's native question/answer spans; keep it small,
  documented, versioned with the repo.
