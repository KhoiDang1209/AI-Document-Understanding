# Contract Intelligence C4 — LangGraph Agent + `/agent` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangGraph agent that chains the existing C1–C3 capabilities as tools to answer compound contract questions in one `POST /agent` call, with bounded retries, a graceful degraded path when the LLM is down, and self-hosted Langfuse tracing.

**Architecture:** A new `agent/` package wraps C1 extract, C2 vector retrieve, C3 graph query, and the shared generate-or-degrade tail as thin functional **tools** (no logic duplication). A LangGraph state machine routes a task (`route → retrieve | graph_query → generate → critique`, with a bounded retry edge), reusing C3's rule-based `route` and C2's `generate_or_degrade`. Every node is traced to self-hosted Langfuse; tracing failure never breaks the request. `/agent` is a new route; `/ask` stays the single-shot path.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, **LangGraph** (state graph), **Langfuse** (self-hosted tracing, behind a guard), pytest, ruff + mypy (strict). Reuses C2/C3 `langchain-core` plumbing, `rag.store.search`, `rag.answer.generate_or_degrade`, `graph.router.route`, `graph.query.run_graph_query`, `contracts.extractor`.

## Global Constraints

- **Project root for all paths/commands is `docintel/`.** Source under `docintel/src/docintel/`, tests under `docintel/tests/`. Run pytest/ruff/mypy from `docintel/`. Use the project venv: `./.venv/Scripts/python.exe` (Windows) — invoke pytest as `python -m pytest`.
- Python `>=3.12`; **full type hints** on every def; **functional style preferred over classes** (the LangGraph state is a `TypedDict`; tools and nodes are plain functions).
- **No hardcoded constants in business logic** — tunables live in `Settings` (`DOCINTEL_` prefix). Reference text (prompts, label names) may be module-level constants, matching `rag/llm.py`/`graph/templates.py`.
- **C4 adds no new model or capability** — it only orchestrates C1–C3. Tools are thin adapters over existing functions; do not re-implement retrieval, generation, or extraction logic.
- **TDD:** failing test → run-it-fails → minimal impl → run-it-passes → commit, per step.
- Lint/type gate before each commit: `python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy src` (from `docintel/`).
- New third-party imports lacking stubs get a `mypy` override (`ignore_missing_imports`).
- **Graceful degradation is mandatory:** if the LLM endpoint (`llm_base_url`) is unset/down, `/agent` returns the deterministic tool outputs gathered so far with `status="degraded"` and HTTP 200 — never a hard failure. Tracing (Langfuse) is best-effort: a tracing error is logged and swallowed.
- Reuse, do not duplicate: citations **are** `docintel.rag.schema.RetrievedChunk`; the generate-or-degrade step **is** `docintel.rag.answer.generate_or_degrade`; routing **is** `docintel.graph.router.route`.
- Commit messages: `feat(agent): …` / `chore(agent): …`, ending with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

**New (`docintel/src/docintel/agent/`):**
- `__init__.py` — empty package marker.
- `schema.py` — `AgentRequest`, `AgentResponse`, `AgentState` (TypedDict).
- `tools.py` — `extract_tool`, `vector_retrieve_tool`, `graph_query_tool`, `generate_tool` (thin adapters).
- `trace.py` — `build_tracer(settings) -> Any | None`, `trace_id_of(tracer) -> str | None` (Langfuse, guarded).
- `graph.py` — `build_agent_graph(deps)`, `run_agent(task, contract_id, deps) -> AgentResponse`; the LangGraph nodes + conditional retry edge.
- `eval.py` — `evaluate_agent(cases, deps) -> dict[str, float]`, `log_to_mlflow(...)` (run later → MLflow).

**Modified:**
- `docintel/src/docintel/config.py` — add `langfuse_*` + `agent_enabled` + `agent_max_retries`.
- `docintel/src/docintel/api/metrics.py` — add `agent_run_total`, `agent_steps`, `agent_retries`.
- `docintel/src/docintel/api/main.py` — `app.state.agent_*` init + include `agent.router`.
- `docintel/src/docintel/api/routes/agent.py` (new) — `POST /agent`.
- `docintel/pyproject.toml` — `agent` extra + `langgraph.*`/`langfuse.*` mypy overrides.
- `docintel/docker-compose.yml` — `langfuse` + `langfuse-postgres` services; api `LANGFUSE_*` env.
- `docintel/.env.example` — `DOCINTEL_LANGFUSE_*` keys.
- `docintel/README.md` — `/agent` subsection.

**New tests:** `tests/test_config_agent.py`, `test_agent_schema.py`, `test_agent_tools.py`, `test_agent_trace.py`, `test_agent_graph.py`, `test_agent_metrics.py`, `test_agent_routes.py`, `test_agent_eval.py`.

---

## Task 1: Config, `agent` extra & Langfuse Compose services

**Files:**
- Modify: `docintel/src/docintel/config.py` (after the C3 block, ~line 109)
- Modify: `docintel/pyproject.toml` (add `agent` extra; extend mypy override list)
- Modify: `docintel/docker-compose.yml` (add `langfuse` + `langfuse-postgres`; api env)
- Modify: `docintel/.env.example`
- Test: `tests/test_config_agent.py`

**Interfaces:**
- Produces: `Settings` fields `langfuse_host: str`, `langfuse_public_key: str | None`, `langfuse_secret_key: str | None`, `agent_enabled: bool`, `agent_max_retries: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_agent.py
from __future__ import annotations

from docintel.config import Settings


def test_agent_settings_defaults() -> None:
    s = Settings()
    assert s.langfuse_host == "http://langfuse:3000"
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.agent_enabled is True
    assert s.agent_max_retries == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && python -m pytest tests/test_config_agent.py -v`
Expected: FAIL with `AttributeError` / missing field.

- [ ] **Step 3: Add the settings**

In `config.py`, after the `graph_default_within_days` line, add:

```python
    # Agent / LangGraph + Langfuse (C4)
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    agent_enabled: bool = True
    agent_max_retries: int = 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && python -m pytest tests/test_config_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Add the `agent` extra + mypy overrides**

In `pyproject.toml`, after the `eval = [...]` extra add:

```toml
agent = [
    "langgraph>=0.2,<0.3",
    "langfuse>=2.0,<3",
]
```

In the `[[tool.mypy.overrides]]` `module = [...]` list, append `"langgraph.*"` and `"langfuse.*"`.

- [ ] **Step 6: Add the Langfuse Compose services**

In `docker-compose.yml`, under `services.api.environment` add:

```yaml
      DOCINTEL_LANGFUSE_HOST: http://langfuse:3000
```

After the `neo4j:` service block add:

```yaml
  langfuse-postgres:
    image: postgres:16
    container_name: docintel-langfuse-postgres
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - langfuse-postgres-data:/var/lib/postgresql/data

  langfuse:
    image: langfuse/langfuse:2
    container_name: docintel-langfuse
    depends_on:
      - langfuse-postgres
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-postgres:5432/langfuse
      NEXTAUTH_SECRET: local-dev-secret
      SALT: local-dev-salt
      NEXTAUTH_URL: http://localhost:3000
    ports:
      - "3000:3000"
```

Under top-level `volumes:` add `  langfuse-postgres-data:`.

- [ ] **Step 7: Add `.env.example` keys**

Append to `docintel/.env.example`:

```bash
DOCINTEL_LANGFUSE_HOST=http://localhost:3000
DOCINTEL_LANGFUSE_PUBLIC_KEY=
DOCINTEL_LANGFUSE_SECRET_KEY=
```

- [ ] **Step 8: Verify compose is valid YAML**

Run: `cd docintel && ./.venv/Scripts/python.exe -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"`
Expected: no output, exit 0.

- [ ] **Step 9: Sync the agent extra**

Run: `cd docintel && uv sync --all-extras`
Expected: installs `langgraph` and `langfuse`.

- [ ] **Step 10: Commit**

```bash
git add docintel/src/docintel/config.py docintel/pyproject.toml docintel/docker-compose.yml docintel/.env.example tests/test_config_agent.py
git commit -m "feat(agent): add agent settings, extra, and Langfuse Compose services

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Agent schema (request, response, state)

**Files:**
- Create: `docintel/src/docintel/agent/__init__.py` (empty)
- Create: `docintel/src/docintel/agent/schema.py`
- Test: `tests/test_agent_schema.py`

**Interfaces:**
- Consumes: `RetrievedChunk` from `docintel.rag.schema`.
- Produces:
  - `AgentRequest(task: str [min_length=1], contract_id: str | None = None)`.
  - `AgentResponse(task: str, answer: str | None, status: Literal["ok", "degraded"], citations: list[RetrievedChunk], steps: list[str], retries: int, contract_id: str | None, trace_id: str | None)`.
  - `AgentState` — a `TypedDict` with keys: `task: str`, `contract_id: str | None`, `route_target: str`, `citations: list[RetrievedChunk]`, `answer: str | None`, `generation_skipped: bool`, `retries: int`, `fallback: bool`, `do_retry: bool`, `steps: Annotated[list[str], operator.add]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_schema.py
from __future__ import annotations

from docintel.agent.schema import AgentRequest, AgentResponse


def test_agent_request_requires_task() -> None:
    req = AgentRequest(task="When does contract X expire?")
    assert req.contract_id is None


def test_agent_response_defaults() -> None:
    resp = AgentResponse(
        task="t", answer=None, status="degraded", contract_id=None, trace_id=None, retries=0
    )
    assert resp.citations == [] and resp.steps == [] and resp.status == "degraded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && python -m pytest tests/test_agent_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.agent.schema`.

- [ ] **Step 3: Create the package + schema**

Create empty `docintel/src/docintel/agent/__init__.py`. Create `schema.py`:

```python
"""Pydantic request/response models and the LangGraph state for the C4 agent."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from docintel.rag.schema import RetrievedChunk


class AgentRequest(BaseModel):
    """A compound natural-language task, optionally scoped to one contract."""

    task: str = Field(min_length=1)
    contract_id: str | None = None


class AgentResponse(BaseModel):
    """The agent's grounded answer (or null when degraded) plus citations and trace id."""

    task: str
    answer: str | None
    status: Literal["ok", "degraded"]
    contract_id: str | None
    trace_id: str | None
    retries: int
    citations: list[RetrievedChunk] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph nodes."""

    task: str
    contract_id: str | None
    route_target: str
    citations: list[RetrievedChunk]
    answer: str | None
    generation_skipped: bool
    retries: int
    fallback: bool
    do_retry: bool
    steps: Annotated[list[str], operator.add]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && python -m pytest tests/test_agent_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docintel/src/docintel/agent/__init__.py docintel/src/docintel/agent/schema.py tests/test_agent_schema.py
git commit -m "feat(agent): add agent request/response/state schema

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Tool adapters over C1–C3

**Files:**
- Create: `docintel/src/docintel/agent/tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `ContractExtractor` (`docintel.contracts.extractor`), `search` (`docintel.rag.store`), `route` + `run_graph_query` (`docintel.graph`), `generate_or_degrade` (`docintel.rag.answer`), `build_derived` + `ContractDocument` (`docintel.contracts.schema`), `Settings`.
- Produces (all thin, side-effect-free except the stores they are handed):
  - `extract_tool(text: str, extractor: ContractExtractor, source: str = "digital") -> ContractDocument` — `extractor.extract(text)` → `ContractDocument` (uses `build_derived`).
  - `vector_retrieve_tool(question: str, store: Any, settings: Settings, contract_id: str | None = None) -> list[RetrievedChunk]` — wraps `search`.
  - `graph_query_tool(question: str, graph_store: Any | None, settings: Settings) -> list[RetrievedChunk]` — `route(question)`; if `target == "graph"` and store present → `run_graph_query`, else `[]`.
  - `generate_tool(question: str, citations: list[RetrievedChunk], llm: Any | None, contract_id: str | None) -> AskResponse` — wraps `generate_or_degrade`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_tools.py
from __future__ import annotations

from datetime import date

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docintel.agent.tools import (
    extract_tool,
    generate_tool,
    graph_query_tool,
    vector_retrieve_tool,
)
from docintel.config import Settings
from docintel.contracts.schema import ExtractedClause
from docintel.graph.schema import ExpirationFact, GraphContract
from docintel.graph.store import InMemoryGraphStore
from docintel.rag.schema import RetrievedChunk


class _FakeExtractor:
    def extract(self, text: str) -> list[ExtractedClause]:
        return [
            ExtractedClause(
                clause_type="Governing Law",
                answer_text="New York",
                char_start=0,
                char_end=8,
                confidence=0.9,
            )
        ]


def test_extract_tool_builds_document() -> None:
    doc = extract_tool("some contract text", _FakeExtractor())
    assert doc.clauses[0].clause_type == "Governing Law"
    assert doc.derived["Governing Law"] == ["New York"]


def test_graph_query_tool_routes_and_queries() -> None:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-01-01", answer_text="expires 2999-01-01", char_start=0, char_end=18
            ),
        )
    )
    chunks = graph_query_tool("which contracts expire within 400000 days?", store, Settings())
    assert [c.contract_id for c in chunks] == ["a"]


def test_graph_query_tool_returns_empty_for_non_graph_question() -> None:
    chunks = graph_query_tool("what is the governing law?", InMemoryGraphStore(), Settings())
    assert chunks == []


def test_generate_tool_degrades_without_llm() -> None:
    cite = RetrievedChunk(
        contract_id="a",
        chunk_index=0,
        chunk_kind="graph",
        clause_type="Governing Law",
        text="New York",
        score=1.0,
        char_start=0,
        char_end=8,
    )
    resp = generate_tool("q", [cite], None, "a")
    assert resp.generation_skipped is True and resp.citations == [cite]
    ok = generate_tool("q", [cite], FakeListChatModel(responses=["A."]), "a")
    assert ok.answer == "A."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && python -m pytest tests/test_agent_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the tools**

```python
"""Thin tool adapters that expose C1–C3 capabilities to the agent.

Each tool is a side-effect-free wrapper over an existing function — no retrieval,
generation, or extraction logic is duplicated here. The agent graph composes them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from docintel.contracts.extractor import ContractExtractor
from docintel.contracts.schema import ContractDocument, build_derived
from docintel.graph.query import run_graph_query
from docintel.graph.router import route
from docintel.rag.answer import generate_or_degrade
from docintel.rag.schema import AskResponse, RetrievedChunk
from docintel.rag.store import search
from docintel.config import Settings


def extract_tool(
    text: str, extractor: ContractExtractor, source: str = "digital"
) -> ContractDocument:
    """Extract clauses from raw contract text and assemble a ContractDocument."""
    clauses = extractor.extract(text)
    return ContractDocument(
        id=uuid4().hex,
        source="digital" if source == "digital" else "ocr",
        clauses=clauses,
        derived=build_derived(clauses),
        page_count=1,
        created_at=datetime.now(UTC).isoformat(),
    )


def vector_retrieve_tool(
    question: str, store: Any, settings: Settings, contract_id: str | None = None
) -> list[RetrievedChunk]:
    """Top-k vector retrieval over indexed contracts (optionally scoped to one)."""
    return search(store, question, settings.rag_top_k, contract_id)


def graph_query_tool(
    question: str, graph_store: Any | None, settings: Settings
) -> list[RetrievedChunk]:
    """Route the question; return cited graph facts, or [] when not a graph question."""
    decision = route(question)
    if decision.target != "graph" or graph_store is None:
        return []
    return run_graph_query(graph_store, decision, settings)


def generate_tool(
    question: str, citations: list[RetrievedChunk], llm: Any | None, contract_id: str | None
) -> AskResponse:
    """Generate a grounded answer from citations, or degrade to citations-only."""
    return generate_or_degrade(question, citations, llm, contract_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && python -m pytest tests/test_agent_tools.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint/type/commit**

```bash
cd docintel && python -m ruff check src/docintel/agent tests/test_agent_tools.py && python -m mypy src/docintel/agent/tools.py
git add docintel/src/docintel/agent/tools.py tests/test_agent_tools.py
git commit -m "feat(agent): add thin tool adapters over C1-C3

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Langfuse tracer (guarded, best-effort)

**Files:**
- Create: `docintel/src/docintel/agent/trace.py`
- Test: `tests/test_agent_trace.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces:
  - `build_tracer(settings: Settings) -> Any | None` — returns a Langfuse `CallbackHandler` when both `langfuse_public_key` and `langfuse_secret_key` are set, else `None`. Any construction error is logged and returns `None` (tracing must never break the agent).
  - `trace_id_of(tracer: Any | None) -> str | None` — best-effort read of the last trace id (`getattr` chain), `None` on absence or error.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_trace.py
from __future__ import annotations

from docintel.agent.trace import build_tracer, trace_id_of
from docintel.config import Settings


def test_build_tracer_disabled_without_keys() -> None:
    assert build_tracer(Settings()) is None


def test_trace_id_of_handles_none_and_missing() -> None:
    assert trace_id_of(None) is None

    class _NoId:
        pass

    assert trace_id_of(_NoId()) is None


def test_trace_id_of_reads_attribute() -> None:
    class _WithId:
        last_trace_id = "abc123"

    assert trace_id_of(_WithId()) == "abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && python -m pytest tests/test_agent_trace.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the tracer**

```python
"""Self-hosted Langfuse tracing for the agent — strictly best-effort.

A tracing failure (no keys, unreachable Langfuse, library error) must never break
/agent: build_tracer returns None and the graph runs untraced. The callback handler,
when present, is passed to LangChain via config={"callbacks": [tracer]}.
"""

from __future__ import annotations

import logging
from typing import Any

from docintel.config import Settings

logger = logging.getLogger("docintel.agent.trace")


def build_tracer(settings: Settings) -> Any | None:
    """Return a Langfuse CallbackHandler when configured, else None (never raises)."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        logger.warning("agent.trace.unavailable", exc_info=True)
        return None


def trace_id_of(tracer: Any | None) -> str | None:
    """Best-effort read of the handler's last trace id; None on absence or error."""
    if tracer is None:
        return None
    try:
        getter = getattr(tracer, "get_trace_id", None)
        if callable(getter):
            return getter() or None
        value = getattr(tracer, "last_trace_id", None)
        return str(value) if value else None
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && python -m pytest tests/test_agent_trace.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint/type/commit**

```bash
cd docintel && python -m ruff check src/docintel/agent/trace.py tests/test_agent_trace.py && python -m mypy src/docintel/agent/trace.py
git add docintel/src/docintel/agent/trace.py tests/test_agent_trace.py
git commit -m "feat(agent): add guarded Langfuse tracer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: LangGraph state machine + `run_agent`

**Files:**
- Create: `docintel/src/docintel/agent/graph.py`
- Test: `tests/test_agent_graph.py`

**Interfaces:**
- Consumes: `AgentState`, `AgentResponse` (`agent.schema`); the four tools (`agent.tools`); `Settings`.
- Produces:
  - `AgentDeps` — a frozen dataclass bundling injected dependencies: `settings: Settings`, `rag_store: Any`, `graph_store: Any | None`, `llm: Any | None`, `tracer: Any | None = None`.
  - `build_agent_graph(deps: AgentDeps) -> Any` — a compiled LangGraph. Nodes: `route_node`, `retrieve_node`, `generate_node`, `critique_node`. Entry `route_node`; conditional edge from `critique_node` back to `retrieve_node` (retry) or to `END`.
  - `run_agent(task: str, contract_id: str | None, deps: AgentDeps) -> AgentResponse`.
- **Routing/retry rules (deterministic, testable without an LLM):**
  - `route_node`: `route_target = route(task).target` ("graph"/"vector"); records a step.
  - `retrieve_node`: if `route_target == "graph"` and not `fallback` → `graph_query_tool`; else `vector_retrieve_tool` (with `contract_id`). On `fallback`, always use vector. Sets `citations`.
  - `generate_node`: `generate_tool` → sets `answer`, `generation_skipped`.
  - `critique_node`: if `citations` empty **and** `retries < settings.agent_max_retries` → set `fallback=True`, `retries += 1`, route back to `retrieve_node`; else END.
  - `run_agent` maps the final state to `AgentResponse`: `status = "ok"` iff `answer is not None and not generation_skipped`, else `"degraded"`; `trace_id = trace_id_of(deps.tracer)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_graph.py
from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient

from docintel.agent.graph import AgentDeps, run_agent
from docintel.config import Settings
from docintel.graph.schema import ExpirationFact, GraphContract
from docintel.graph.store import InMemoryGraphStore
from docintel.rag.embed import build_embedder
from docintel.rag.store import build_vector_store, ensure_collection, upsert_chunks
from docintel.rag.chunk import build_chunks


def _vector_store(settings: Settings) -> object:
    client = QdrantClient(":memory:")
    ensure_collection(client, settings.qdrant_collection, settings.rag_embedding_dim)
    store = build_vector_store(settings, build_embedder(settings), client=client)
    chunks = build_chunks("The governing law is the State of New York.", [], 1200, 200)
    upsert_chunks(store, "a", chunks)
    return store


def _graph_store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-01-01", answer_text="expires 2999-01-01", char_start=0, char_end=18
            ),
        )
    )
    return store


def test_vector_task_generates_answer() -> None:
    settings = Settings()
    deps = AgentDeps(
        settings=settings,
        rag_store=_vector_store(settings),
        graph_store=_graph_store(),
        llm=FakeListChatModel(responses=["Governing law is New York."]),
    )
    resp = run_agent("What is the governing law?", None, deps)
    assert resp.status == "ok"
    assert resp.answer == "Governing law is New York."
    assert resp.citations  # retrieved at least one chunk


def test_graph_task_degrades_without_llm() -> None:
    settings = Settings()
    deps = AgentDeps(
        settings=settings,
        rag_store=_vector_store(settings),
        graph_store=_graph_store(),
        llm=None,
    )
    resp = run_agent("which contracts expire within 400000 days?", None, deps)
    assert resp.status == "degraded"
    assert resp.answer is None
    assert [c.contract_id for c in resp.citations] == ["a"]


def test_retry_caps_and_marks_degraded_when_nothing_found() -> None:
    settings = Settings()  # agent_max_retries=1
    empty_vector = _vector_store(settings)
    deps = AgentDeps(
        settings=settings, rag_store=empty_vector, graph_store=InMemoryGraphStore(), llm=None
    )
    # A graph question whose store is empty -> graph yields [], fallback to vector also weak.
    resp = run_agent("which contracts expire within 1 day?", None, deps)
    assert resp.retries <= settings.agent_max_retries
    assert resp.status == "degraded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && python -m pytest tests/test_agent_graph.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the graph**

```python
"""LangGraph state machine that orchestrates the C1–C3 tools.

The graph is deliberately explicit (not a free-form ReAct loop): route → retrieve →
generate → critique, with one bounded retry edge. This keeps every step inspectable and
traceable, and lets the whole flow run deterministically (and testably) without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from docintel.agent.schema import AgentResponse, AgentState
from docintel.agent.tools import generate_tool, graph_query_tool, vector_retrieve_tool
from docintel.agent.trace import trace_id_of
from docintel.config import Settings


@dataclass(frozen=True)
class AgentDeps:
    """Dependencies injected into the graph (stores, llm, tracer)."""

    settings: Settings
    rag_store: Any
    graph_store: Any | None
    llm: Any | None
    tracer: Any | None = None


def _route_node(state: AgentState, deps: AgentDeps) -> AgentState:
    from docintel.graph.router import route

    target = route(state["task"]).target
    return {"route_target": target, "steps": [f"route:{target}"]}


def _retrieve_node(state: AgentState, deps: AgentDeps) -> AgentState:
    use_graph = state.get("route_target") == "graph" and not state.get("fallback", False)
    if use_graph:
        citations = graph_query_tool(state["task"], deps.graph_store, deps.settings)
        step = "retrieve:graph"
    else:
        citations = vector_retrieve_tool(
            state["task"], deps.rag_store, deps.settings, state.get("contract_id")
        )
        step = "retrieve:vector"
    return {"citations": citations, "steps": [f"{step}:{len(citations)}"]}


def _generate_node(state: AgentState, deps: AgentDeps) -> AgentState:
    # The tracer is propagated to this node's LLM call via the graph-level invoke config
    # (run_agent passes callbacks once); generate_tool reuses C2's generate_or_degrade.
    resp = generate_tool(
        state["task"], state.get("citations", []), deps.llm, state.get("contract_id")
    )
    return {
        "answer": resp.answer,
        "generation_skipped": resp.generation_skipped,
        "steps": ["generate"],
    }


def _critique_node(state: AgentState, deps: AgentDeps) -> AgentState:
    # Decide retry once, per iteration. do_retry is the single source of truth for the edge,
    # so the loop cannot continue after the cap is reached (fallback alone would never clear).
    no_citations = not state.get("citations")
    can_retry = state.get("retries", 0) < deps.settings.agent_max_retries
    if no_citations and can_retry:
        return {
            "do_retry": True,
            "fallback": True,
            "retries": state.get("retries", 0) + 1,
            "steps": ["critique:retry"],
        }
    return {"do_retry": False, "steps": ["critique:finish"]}


def _should_retry(state: AgentState, deps: AgentDeps) -> str:
    return "retrieve" if state.get("do_retry") else "end"


def build_agent_graph(deps: AgentDeps) -> Any:
    """Compile the route → retrieve → generate → critique graph with a bounded retry."""

    def bind(fn: Any) -> Any:
        return lambda state: fn(state, deps)

    graph = StateGraph(AgentState)
    graph.add_node("route", bind(_route_node))
    graph.add_node("retrieve", bind(_retrieve_node))
    graph.add_node("generate", bind(_generate_node))
    graph.add_node("critique", bind(_critique_node))
    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique", lambda state: _should_retry(state, deps), {"retrieve": "retrieve", "end": END}
    )
    return graph.compile()


def run_agent(task: str, contract_id: str | None, deps: AgentDeps) -> AgentResponse:
    """Run the compiled graph for one task and map the final state to AgentResponse."""
    compiled = build_agent_graph(deps)
    initial: AgentState = {"task": task, "contract_id": contract_id, "retries": 0, "steps": []}
    # Pass the tracer once at graph level; LangChain callback propagation reaches the node LLM call.
    config = {"callbacks": [deps.tracer]} if deps.tracer is not None else None
    final: AgentState = compiled.invoke(initial, config=config)
    answer = final.get("answer")
    skipped = final.get("generation_skipped", True)
    status = "ok" if (answer is not None and not skipped) else "degraded"
    return AgentResponse(
        task=task,
        answer=answer,
        status=status,
        contract_id=contract_id,
        trace_id=trace_id_of(deps.tracer),
        retries=final.get("retries", 0),
        citations=final.get("citations", []),
        steps=final.get("steps", []),
    )
```

> Note on the retry edge: `_critique_node` computes `do_retry` fresh each iteration (true only when there are no citations and `retries < agent_max_retries`), sets `fallback`, and increments `retries`; `_should_retry` routes back to `retrieve` iff `do_retry` is set, else to `END`. Because `do_retry` is recomputed every critique pass and goes false once the cap is hit, the graph makes at most `agent_max_retries` extra retrieve passes — it cannot loop unbounded (the earlier `fallback`-based edge could).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && python -m pytest tests/test_agent_graph.py -v`
Expected: PASS (3 tests). (First run downloads the bge-small fastembed model.)

- [ ] **Step 5: Lint/type/commit**

```bash
cd docintel && python -m ruff check src/docintel/agent/graph.py tests/test_agent_graph.py && python -m mypy src/docintel/agent/graph.py
git add docintel/src/docintel/agent/graph.py tests/test_agent_graph.py
git commit -m "feat(agent): add LangGraph state machine and run_agent

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Agent metrics

**Files:**
- Modify: `docintel/src/docintel/api/metrics.py`
- Test: `tests/test_agent_metrics.py`

**Interfaces:**
- Produces: `Metrics.agent_run_total: Counter` (label `status`), `Metrics.agent_retries: Counter`, `Metrics.agent_steps: Histogram`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_metrics.py
from __future__ import annotations

from prometheus_client import CollectorRegistry

from docintel.api.metrics import build_metrics


def test_agent_metrics_present() -> None:
    m = build_metrics(CollectorRegistry())
    m.agent_run_total.labels(status="ok").inc()
    m.agent_retries.inc()
    m.agent_steps.observe(4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && python -m pytest tests/test_agent_metrics.py -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Add the metrics**

In `metrics.py`, add three fields to the `Metrics` dataclass (after `router_decision_total`):

```python
    agent_run_total: Counter
    agent_retries: Counter
    agent_steps: Histogram
```

In `build_metrics(...)`, add to the constructor call (after `router_decision_total=...`):

```python
        agent_run_total=Counter(
            "docintel_agent_runs",
            "/agent runs, labelled by terminal status.",
            labelnames=("status",),
            registry=registry,
        ),
        agent_retries=Counter(
            "docintel_agent_retries",
            "Total retry passes taken across /agent runs.",
            registry=registry,
        ),
        agent_steps=Histogram(
            "docintel_agent_steps",
            "Number of graph steps executed per /agent run.",
            buckets=(1, 2, 3, 4, 5, 6, 8, 10),
            registry=registry,
        ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && python -m pytest tests/test_agent_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docintel/src/docintel/api/metrics.py tests/test_agent_metrics.py
git commit -m "feat(agent): add agent run/retry/step metrics

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: `POST /agent` route + app wiring

**Files:**
- Create: `docintel/src/docintel/api/routes/agent.py`
- Modify: `docintel/src/docintel/api/main.py` (init `app.state.agent_tracer = None`; `include_router(agent.router)`)
- Test: `tests/test_agent_routes.py`

**Interfaces:**
- Consumes: `AgentRequest`/`AgentResponse` (`agent.schema`), `AgentDeps`/`run_agent` (`agent.graph`), `build_tracer` (`agent.trace`), the existing `/ask` deps `get_rag_store` / `get_graph_store` / `get_rag_llm` (`api.routes.ask`), `get_contract_extractor` (`api.routes.contracts`) — **not used unless a future text-extract path needs it; the route reuses stores+llm**, `get_metrics`, `get_settings`.
- Produces: `get_agent_tracer(request, settings) -> Any | None` (cached on `app.state.agent_tracer`); `POST /agent` → `AgentResponse`. Records `agent_run_total{status}`, `agent_retries`, `agent_steps`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_routes.py
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient

from docintel.api.main import create_app
from docintel.api.routes.ask import get_graph_store, get_rag_llm, get_rag_store
from docintel.config import Settings, get_settings
from docintel.graph.schema import ExpirationFact, GraphContract
from docintel.graph.store import InMemoryGraphStore
from docintel.rag.chunk import build_chunks
from docintel.rag.embed import build_embedder
from docintel.rag.store import build_vector_store, ensure_collection, upsert_chunks


def _vector_store(settings: Settings) -> Any:
    client = QdrantClient(":memory:")
    ensure_collection(client, settings.qdrant_collection, settings.rag_embedding_dim)
    store = build_vector_store(settings, build_embedder(settings), client=client)
    upsert_chunks(store, "a", build_chunks("Governing law is New York.", [], 1200, 200))
    return store


def _graph_store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-01-01", answer_text="expires 2999-01-01", char_start=0, char_end=18
            ),
        )
    )
    return store


def _client(llm: Any) -> TestClient:
    app = create_app()
    settings = Settings()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rag_store] = lambda: _vector_store(settings)
    app.dependency_overrides[get_graph_store] = lambda: _graph_store()
    app.dependency_overrides[get_rag_llm] = lambda: llm
    return TestClient(app)


def test_agent_route_generates_answer() -> None:
    with _client(FakeListChatModel(responses=["New York."])) as client:
        resp = client.post("/agent", json={"task": "What is the governing law?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["answer"] == "New York."
    assert body["steps"] and body["trace_id"] is None


def test_agent_route_degrades_without_llm() -> None:
    with _client(None) as client:
        resp = client.post(
            "/agent", json={"task": "which contracts expire within 400000 days?"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded" and body["answer"] is None
    assert [c["contract_id"] for c in body["citations"]] == ["a"]


def test_agent_route_validates_empty_task() -> None:
    with _client(None) as client:
        resp = client.post("/agent", json={"task": ""})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && python -m pytest tests/test_agent_routes.py -v`
Expected: FAIL (route not mounted → 404, or import error).

- [ ] **Step 3: Implement the route**

Create `api/routes/agent.py`:

```python
"""The /agent endpoint: run the LangGraph agent over the C1–C3 tools."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from docintel.agent.graph import AgentDeps, run_agent
from docintel.agent.schema import AgentRequest, AgentResponse
from docintel.agent.trace import build_tracer
from docintel.api.metrics import Metrics
from docintel.api.routes.ask import get_graph_store, get_rag_llm, get_rag_store
from docintel.api.routes.extract import get_metrics
from docintel.config import Settings, get_settings

router = APIRouter(tags=["agent"])


def get_agent_tracer(
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Any | None:
    """Build the Langfuse tracer once and cache it (None when not configured)."""
    tracer = getattr(request.app.state, "agent_tracer", None)
    if tracer is None:
        tracer = build_tracer(settings)
        request.app.state.agent_tracer = tracer
    return tracer


@router.post("/agent", response_model=AgentResponse, summary="Run the contract agent on a task")
def agent(
    req: AgentRequest,
    settings: Settings = Depends(get_settings),  # noqa: B008
    rag_store: Any = Depends(get_rag_store),  # noqa: B008
    graph_store: Any | None = Depends(get_graph_store),  # noqa: B008
    llm: Any | None = Depends(get_rag_llm),  # noqa: B008
    tracer: Any | None = Depends(get_agent_tracer),  # noqa: B008
    metrics: Metrics = Depends(get_metrics),  # noqa: B008
) -> AgentResponse:
    """Run the agent graph for a compound task and return the grounded (or degraded) result."""
    deps = AgentDeps(
        settings=settings, rag_store=rag_store, graph_store=graph_store, llm=llm, tracer=tracer
    )
    response = run_agent(req.task, req.contract_id, deps)
    metrics.agent_run_total.labels(status=response.status).inc()
    if response.retries:
        metrics.agent_retries.inc(response.retries)
    metrics.agent_steps.observe(len(response.steps))
    return response
```

- [ ] **Step 4: Wire the route into the app**

In `api/main.py`:
1. Add `agent` to the routes import: `from docintel.api.routes import agent, ask, contracts, documents, extract, health`.
2. In `lifespan`, after `app.state.graph_store = None` add: `app.state.agent_tracer = None`.
3. In `create_app`, after `app.include_router(ask.router)` add: `app.include_router(agent.router)`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd docintel && python -m pytest tests/test_agent_routes.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Lint/type/commit**

```bash
cd docintel && python -m ruff check src/docintel/api tests/test_agent_routes.py && python -m mypy src/docintel/api/routes/agent.py src/docintel/api/main.py
git add docintel/src/docintel/api/routes/agent.py docintel/src/docintel/api/main.py tests/test_agent_routes.py
git commit -m "feat(agent): add POST /agent route and app wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Compound-task eval harness

**Files:**
- Create: `docintel/src/docintel/agent/eval.py`
- Test: `tests/test_agent_eval.py`

**Interfaces:**
- Consumes: `AgentDeps`, `run_agent` (`agent.graph`).
- Produces:
  - `EvalCase = tuple[str, str | None, set[str]]` — `(task, contract_id, expected_citation_contract_ids)`.
  - `evaluate_agent(cases: list[EvalCase], deps: AgentDeps) -> dict[str, float]` — runs each task; a case **succeeds** when the set of citation `contract_id`s equals the expected set (citation-grounding success, LLM-independent). Returns `{"success_rate", "n"}`.
  - `log_to_mlflow(metrics: dict[str, float], experiment: str = "agent-eval") -> None` — lazy MLflow import (run later, not in CI).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_eval.py
from __future__ import annotations

from docintel.agent.eval import evaluate_agent
from docintel.agent.graph import AgentDeps
from docintel.config import Settings
from docintel.graph.schema import ExpirationFact, GraphContract
from docintel.graph.store import InMemoryGraphStore
from qdrant_client import QdrantClient

from docintel.rag.chunk import build_chunks
from docintel.rag.embed import build_embedder
from docintel.rag.store import build_vector_store, ensure_collection, upsert_chunks


def _deps() -> AgentDeps:
    settings = Settings()
    client = QdrantClient(":memory:")
    ensure_collection(client, settings.qdrant_collection, settings.rag_embedding_dim)
    store = build_vector_store(settings, build_embedder(settings), client=client)
    upsert_chunks(store, "a", build_chunks("Governing law New York.", [], 1200, 200))
    graph = InMemoryGraphStore()
    graph.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-01-01", answer_text="expires 2999-01-01", char_start=0, char_end=18
            ),
        )
    )
    return AgentDeps(settings=settings, rag_store=store, graph_store=graph, llm=None)


def test_evaluate_agent_scores_grounding() -> None:
    cases = [("which contracts expire within 400000 days?", None, {"a"})]
    metrics = evaluate_agent(cases, _deps())
    assert metrics["success_rate"] == 1.0 and metrics["n"] == 1.0


def test_evaluate_agent_empty_cases() -> None:
    assert evaluate_agent([], _deps()) == {"success_rate": 0.0, "n": 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && python -m pytest tests/test_agent_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the eval**

```python
"""Compound-task eval for the agent: citation-grounding success rate (run later → MLflow).

Each case is (task, contract_id, expected_citation_contract_ids). Success is exact match of
the cited contract id set, so the metric is LLM-independent and runs on the laptop. The
MLflow tail is optional and lazy, matching graph/eval.py.
"""

from __future__ import annotations

from docintel.agent.graph import AgentDeps, run_agent

EvalCase = tuple[str, str | None, set[str]]


def evaluate_agent(cases: list[EvalCase], deps: AgentDeps) -> dict[str, float]:
    """Return {'success_rate', 'n'} over the cases (exact cited-contract-id-set match)."""
    if not cases:
        return {"success_rate": 0.0, "n": 0.0}
    correct = 0
    for task, contract_id, expected in cases:
        response = run_agent(task, contract_id, deps)
        got = {c.contract_id for c in response.citations}
        if got == expected:
            correct += 1
    return {"success_rate": correct / len(cases), "n": float(len(cases))}


def log_to_mlflow(metrics: dict[str, float], experiment: str = "agent-eval") -> None:
    """Log eval metrics to MLflow (lazy import; called only when running the eval, not in CI)."""
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && python -m pytest tests/test_agent_eval.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint/type/commit**

```bash
cd docintel && python -m ruff check src/docintel/agent/eval.py tests/test_agent_eval.py && python -m mypy src/docintel/agent/eval.py
git add docintel/src/docintel/agent/eval.py tests/test_agent_eval.py
git commit -m "feat(agent): add compound-task grounding eval harness

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: README docs + full-suite gate

**Files:**
- Modify: `docintel/README.md` (add a `/agent` subsection under the contract-intelligence section)

- [ ] **Step 1: Document `/agent`**

Add a subsection describing: the LangGraph route → retrieve/graph → generate → critique flow; the bounded retry; the degraded path (LLM down → cited tool outputs + `status="degraded"`); self-hosted Langfuse tracing (`DOCINTEL_LANGFUSE_*`, `docker compose up langfuse`); and an example `curl -X POST /agent -d '{"task": "..."}'`. Mirror the existing `/ask` subsection's style.

- [ ] **Step 2: Run the full gate**

Run:
```bash
cd docintel && python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy src && python -m pytest
```
Expected: ruff/format/mypy clean; all tests pass (new agent tests green, prior suite unchanged).

- [ ] **Step 3: Commit**

```bash
git add docintel/README.md
git commit -m "docs(agent): document /agent endpoint and degraded path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes, scope decisions & deferred items

- **Scope: `/agent` reasons over already-indexed contracts** (`task` + optional `contract_id`); PDF ingestion stays on `POST /contracts/extract`. `extract_tool` is provided (and tested) for a text-extract step, but wiring a multipart upload through the graph is out of C4 scope — C4's goal is *orchestration*, and indexing already happens at extract time. If a "extract-then-answer in one call" demo is wanted later, add an `ingest_extract` entry node that calls `extract_tool` + `index_contract` + `build_contract` before `route`.
- **Eval location:** placed at `agent/eval.py` (matching `graph/eval.py`/`rag/eval.py`), not the spec's `eval/agent_eval.py`, to follow the established per-package pattern.
- **Eval metric:** citation-grounding success rate (LLM-independent) so it runs on the laptop and in a fresh session. The richer "correct, cited answer" judgement needs the Colab LLM + Langfuse traces — run that pass when the endpoint is up (same deferral as C2 RAGAS).
- **Langfuse `trace_id` API** differs across versions; `trace_id_of` reads it best-effort and returns `None` on any mismatch, so a Langfuse upgrade can't break `/agent`. `langfuse` is pinned `>=2,<3`.
- **Retry bound:** `agent_max_retries` (default 1) caps the single fallback pass; `_should_retry` cannot loop unbounded.

## Self-review (done)

- **Spec coverage:** LangGraph state machine (Task 5) ✓; tools wrap C1–C3 (Task 3) ✓; tool-calling LLM via existing `build_llm`/`get_rag_llm` reuse (Task 7) ✓; graceful degradation (Tasks 3/5/7 tests) ✓; bounded retries + failure path (Task 5) ✓; Langfuse on nodes, trace_id returned (Tasks 4/5/7) ✓; `/agent` new route, `/ask` unchanged (Task 7) ✓; compound-task success rate → MLflow (Task 8) ✓; Prometheus steps/retries/status + Grafana-ready (Task 6) ✓; Langfuse + Postgres in Compose, `LANGFUSE_*` config (Task 1) ✓. **Out of scope** honored: no new model; PDF-through-agent deferred (noted).
- **Placeholder scan:** none — every code/diff step is concrete.
- **Type consistency:** `AgentDeps`, `AgentState`, `AgentResponse`, tool signatures, and the `route_target`/`citations`/`retries`/`fallback` state keys are used identically across Tasks 2/3/5/7/8.
