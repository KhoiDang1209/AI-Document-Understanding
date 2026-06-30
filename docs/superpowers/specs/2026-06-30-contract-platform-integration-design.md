# Contract Platform Integration (C1–C4) — Design

**Date:** 2026-06-30
**Status:** Approved (pending spec review)

## Goal

Tie the four Contract Intelligence stages into one presentable platform. The
**service layer is already wired**: `/contracts/extract` (C1) auto-indexes into
the RAG store (C2) and builds the graph (C3) at extract time, `/ask` routes
between graph and vector retrieval, and `/agent` (C4) orchestrates the C1–C3
tools via LangGraph. The remaining gaps are the human-facing surfaces and a
regression lock:

1. A contract-focused Streamlit UI that exercises C1 → C4 end to end.
2. An in-process end-to-end test proving the wiring.
3. An HTTP demo script for a portfolio walkthrough.
4. Architecture documentation tying the stages together.

## Non-Goals

- No new backends, auth, or config beyond what the demo needs.
- No changes to the `/extract` receipt-KIE **backend** route or the KIE
  pipeline — only the *UI* drops receipts.
- No changes to C1–C4 business logic. This work is integration + presentation.
- No committed sample PDF binary.

## Existing contracts (verified, do not change)

- `POST /contracts/extract` — multipart `file` (`application/pdf`) →
  `ContractDocument` `{id, source, clauses[{clause_type, answer_text,
  char_start, char_end, confidence}], derived{type: [texts]}, page_count,
  created_at}`. Best-effort indexing/graph-build; extraction still succeeds if
  the stores are down.
- `GET /contracts/{id}` → `ContractDocument` or 404.
- `POST /ask` — `{question (min_len 1), contract_id?, top_k?}` → `AskResponse`
  `{question, answer|null, generation_skipped, contract_id, citations[]}`.
- `POST /agent` — `{task (min_len 1), contract_id?}` → `AgentResponse`
  `{task, answer|null, status: "ok"|"degraded", contract_id, trace_id|null,
  retries, citations[], steps[]}`.
- `RetrievedChunk` `{contract_id, chunk_index, chunk_kind, clause_type|null,
  text, score, char_start, char_end}`.

Degradation is already handled server-side: missing LLM → `generation_skipped`
/ `status: "degraded"` with citations-only; missing graph store → router falls
back to vector. The UI and demo only **render** that status; they add no
degradation logic of their own.

---

## 1. Contract-only Streamlit UI

The current single receipt page is replaced by a single **contract** page. No
multipage routing (only one domain now).

### Files

- **Remove** `src/docintel/ui/client.py` and `tests/test_ui_client.py`
  (receipt-specific, become dead code).
- **Rewrite** `src/docintel/ui/app.py` — thin Streamlit rendering only. One page,
  three tabs: **Extract**, **Ask**, **Agent**.
- **Add** `src/docintel/ui/contract_client.py` — pure, testable HTTP + formatting
  helpers (mirrors the old `client.py` split so `app.py` stays presentation-only).
- **Add** `tests/test_ui_contract_client.py` — mirrors the existing
  `test_ui_client.py` mocked-`httpx.MockTransport` pattern.

### `contract_client.py` surface

A single shared `ContractApiError(Exception)` carrying a user-facing message,
plus:

- `extract_contract(base_url, timeout_s, filename, data) -> dict` — POST PDF to
  `/contracts/extract`.
- `ask_question(base_url, timeout_s, question, contract_id) -> dict` — POST
  `/ask`.
- `run_agent(base_url, timeout_s, task, contract_id) -> dict` — POST `/agent`.
- Formatting helpers: `clause_rows(document) -> list[dict]` (Type / Text /
  Confidence), `citation_rows(response) -> list[dict]` (Contract / Clause /
  Score / Text).

Each HTTP helper raises `ContractApiError` on transport failure (timeout,
connection refused) and on non-2xx (extracting `detail`), exactly like the old
`extract_receipt`.

### Page behavior (`app.py`)

- Reads `ui_api_base_url` and `ui_request_timeout_s` from `Settings`.
- **Extract tab:** PDF uploader → `extract_contract` → render source badge,
  `clause_rows` table, derived fields, raw-JSON expander. Store the returned
  `id` in `st.session_state["contract_id"]`.
- **Ask tab:** question text input + an "only this contract" toggle (uses the
  session `contract_id` when on) → `ask_question` → render answer (or a
  "generation skipped — showing citations" notice when `generation_skipped`),
  `citation_rows` table.
- **Agent tab:** task text input + same scope toggle → `run_agent` → render
  status badge (`ok`/`degraded`), answer, `steps` list, `retries`, citations,
  `trace_id` when present.
- `ContractApiError` is caught per action and shown via `st.error`.

### Testing

`test_ui_contract_client.py` covers, per HTTP helper: success, non-2xx with
`detail`, timeout, connection error; and the row-formatting helpers including
missing/`None` fields. No Streamlit rendering is unit-tested (matches current
approach).

---

## 2. End-to-end integration test

`tests/test_e2e_contract_pipeline.py` — one `TestClient`, all heavy deps faked,
no `slow` marker. Critically, the RAG and graph stores are **single shared
instances** bound to *both* the optional (extract-time) and non-optional
(ask/agent) getters, so data written during extract is visible to ask/agent.

### Fakes / overrides

- `get_settings` → `Settings(sqlite_path=tmp_path/...)`.
- `get_ocr_engine` → no-op; `get_s3_client` → `_FakeS3` (reuse from
  `tests/test_documents.py`).
- `get_contract_extractor` → stub returning clauses that include a `Parties`
  clause and an expiration-style clause, so the graph build produces a contract
  node and vector chunks exist.
- `ingest_pdf` monkeypatched → fixed `IngestedDoc(text=..., page_count=1,
  source="digital")` (avoids building a real PDF).
- One in-memory Qdrant store (`QdrantClient(":memory:")` via
  `build_vector_store`/`ensure_collection`) bound to **both**
  `get_rag_store_optional` and `get_rag_store`.
- One `InMemoryGraphStore` bound to **both** `get_graph_store_optional` and
  `get_graph_store`.
- `get_rag_llm` → `FakeListChatModel(responses=[...])`.

### Flow / assertions

1. `POST /contracts/extract` → 200; capture `contract_id`; assert clauses and
   `derived` present.
2. Assert the **shared** RAG store now returns chunks for that `contract_id`
   (via `docintel.rag.store.search`) **and** the shared graph store has the
   contract — proving extract fed C2 and C3.
3. `POST /ask` scoped to `contract_id` → 200; grounded answer from the fake LLM;
   citations reference that contract.
4. `POST /agent` with a task → 200; `status == "ok"`; non-empty `steps`;
   citations reference that contract.

This is the regression lock: it fails if the extract→index/graph wiring or the
ask/agent retrieval path breaks.

---

## 3. HTTP demo script

`src/docintel/scripts/demo_pipeline.py`, exposed as console entry
`docintel-demo` in `[project.scripts]`. Drives the **live** API over HTTP
(`httpx`), printing each stage's result.

### CLI

- `--base-url` (default `Settings().ui_api_base_url`).
- `--pdf PATH` (optional). When omitted, synthesize a tiny multi-clause sample
  contract PDF **in memory** with `pymupdf` (already a `contracts` dep) — no
  committed binary.
- `--timeout` (default `Settings().ui_request_timeout_s`).
- `--contract-id` is not an input; it is captured from the extract response and
  reused for ask/agent.

### Behavior

1. Extract → print `id`, `source`, clause count, `derived` keys.
2. `POST /ask` "What is the governing law?" scoped to the captured id → print
   router-independent answer, `generation_skipped`, citation count.
3. `POST /agent` "Summarize the parties and governing law." scoped to the id →
   print `status`, `steps`, answer, citation count, `trace_id`.
4. On connection failure, print a clear "could not reach the API at <url> — is
   the stack running?" message and exit non-zero.

`main()` returns an int exit code; `console_scripts` wraps it. The PDF-synthesis
helper is a small pure function (`build_sample_pdf() -> bytes`).

### Testing

`tests/test_demo_pipeline.py` — light, with `httpx.MockTransport` (same pattern
as the UI client test): assert the script POSTs to `/contracts/extract`,
`/ask`, `/agent` in order, threads the captured `contract_id`, and surfaces a
clear error/non-zero exit on a connection error. `build_sample_pdf()` asserted
to return non-empty `application/pdf` bytes (skip if `pymupdf` unavailable).

---

## 4. Documentation

### `docs/architecture.md` (new)

- **Components:** C1 extract, C2 RAG, C3 GraphRAG, C4 agent — one paragraph
  each, pointing at the owning module and route.
- **Data/request flow:** the extract → auto-index + graph-build fan-out;
  `/ask` routing (graph vs vector) with the shared degrade tail; `/agent`
  route → retrieve → generate → critique with one bounded retry.
- **Degradation matrix:** what each surface returns when the LLM / graph /
  vector store is unavailable.
- **Running the full stack:** Compose services, required env (e.g. the ONNX
  extractor path, optional LLM/Langfuse), `uvicorn` command, `streamlit run`
  command, and `docintel-demo`.
- A **Mermaid** flow diagram of the four stages and their stores.

### `README.md`

Add a "Contract Intelligence (C1–C4)" section: the endpoint table
(`/contracts/extract`, `/contracts/{id}`, `/ask`, `/agent`), the UI run command,
the demo command, and a link to `docs/architecture.md`.

---

## Verification

- `ruff check src tests` clean.
- `mypy src` strict clean.
- `pytest -q` green (new e2e + UI client + demo tests included; no new `slow`
  tests).
- Manual smoke (optional, requires stack): `streamlit run
  src/docintel/ui/app.py` and `docintel-demo` against a running API.

## Risks / open points

- **Streamlit import in tests:** only `contract_client.py` (pure, no Streamlit
  import) is unit-tested; `app.py` imports Streamlit but is not imported by the
  test suite, matching the current arrangement.
- **`pymupdf` text→PDF:** the synthesized sample must contain enough recognizable
  clause text for the stub-free live demo to extract something meaningful; this
  depends on the real ONNX extractor at demo time (not on CI).
