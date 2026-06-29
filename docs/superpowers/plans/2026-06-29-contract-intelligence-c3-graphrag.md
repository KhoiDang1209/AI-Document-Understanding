# Contract Intelligence C3 — GraphRAG (Neo4j) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, date-centric knowledge graph over the C1 extractions and route `POST /ask` between vector retrieval (C2) and two multi-hop Cypher templates (graph), reusing C2's LLM for prose and degrading gracefully when it is down.

**Architecture:** A `graph/` package builds `Contract`/`Date`/`Renewal-ClauseType` nodes from each `ContractDocument` (best-effort, at extract time, beside the existing Qdrant indexing). A pure rule-based router classifies a question to a Cypher template (`expiring_within`, `auto_renewing_expiring_within`) or to the vector path. Graph traversal returns cited facts that reuse C2's `RetrievedChunk` citation shape, then C2's `generate_answer` produces prose (or degrades). All graph logic is exercised through a `GraphStore` Protocol with an in-memory fake; a single deselected test checks real-Neo4j Cypher parity.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, `neo4j` Python driver (behind a Protocol), pytest, ruff + mypy (strict). Reuses C2's `langchain-core`/`ChatOpenAI` LLM plumbing.

## Global Constraints

- **Project root for all paths/commands is `docintel/`.** Source lives under `docintel/src/docintel/`, tests under `docintel/tests/`. Run pytest/ruff/mypy from `docintel/`.
- Python `>=3.12`; **full type hints** on every def; **functional style preferred over classes** (classes only where a Protocol/stateful store needs them).
- **No hardcoded constants in business logic** — tunables live in `Settings` (`DOCINTEL_` prefix). Reference data (clause-type names, Cypher strings, fixed namespaces) may be module-level constants, matching `questions.py`/`store.py`.
- **TDD:** failing test → run-it-fails → minimal impl → run-it-passes → commit, per step.
- Lint/type gate before each commit: `ruff check src tests && ruff format --check src tests && mypy src` (run from `docintel/`).
- New third-party imports that lack stubs get a `mypy` override (`ignore_missing_imports`).
- Reuse, do not duplicate: the graph citation type **is** C2's `docintel.rag.schema.RetrievedChunk`; the generate-or-degrade tail is shared with C2's `answer_question`.
- Commit messages: `feat(graph): …` / `chore(graph): …`, ending with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

**New (`docintel/src/docintel/graph/`):**
- `__init__.py` — empty package marker.
- `schema.py` — `ExpirationFact`, `RenewalFact`, `GraphContract`, `RouteDecision` (pydantic).
- `normalize.py` — `parse_iso_date(text)`, `build_graph_contract(doc)`.
- `store.py` — `GraphStore` Protocol, `InMemoryGraphStore` (fake, runs both templates in Python), `build_graph_store(settings)`.
- `templates.py` — `TEMPLATES` Cypher registry + `get_template(name)`; `Neo4jGraphStore`.
- `router.py` — `route(question) -> RouteDecision` (pure, rule-based).
- `query.py` — `run_graph_query(store, decision, settings, reference_date=None) -> list[RetrievedChunk]`.
- `build.py` — `build_contract(doc, store) -> bool`.

**Modified:**
- `docintel/src/docintel/config.py` — add `neo4j_*` + `graph_enabled` + `graph_default_within_days`.
- `docintel/src/docintel/rag/answer.py` — extract shared `generate_or_degrade(...)`.
- `docintel/src/docintel/api/metrics.py` — add `graph_query_latency`, `router_decision_total`.
- `docintel/src/docintel/api/routes/ask.py` — graph store deps + router wiring + metrics.
- `docintel/src/docintel/api/routes/contracts.py` — best-effort `build_contract` at extract time.
- `docintel/src/docintel/api/main.py` — `app.state.graph_store = None`.
- `docintel/pyproject.toml` — `graph` extra + `neo4j` mypy override.
- `docintel/docker-compose.yml` — `neo4j` service + api env/depends_on + volume.

**New eval:** `docintel/src/docintel/graph/eval.py` — `graphrag_eval` (MLflow; run later).

**New tests:** `tests/test_graph_schema.py`, `test_graph_normalize.py`, `test_graph_store.py`, `test_graph_templates.py`, `test_graph_router.py`, `test_graph_query.py`, `test_graph_build.py`, `test_graph_ask_routes.py`, `test_graph_neo4j_parity.py` (deselected), `test_graph_eval.py`.

---

## Task 1: Config, dependency extra & Compose service

**Files:**
- Modify: `docintel/src/docintel/config.py` (after the C2 RAG block, ~line 101)
- Modify: `docintel/pyproject.toml` (add `graph` extra; extend mypy override list)
- Modify: `docintel/docker-compose.yml` (add `neo4j` service, api env + depends_on, volume)
- Test: `tests/test_config_graph.py`

**Interfaces:**
- Produces: `Settings` fields `neo4j_uri: str`, `neo4j_user: str`, `neo4j_password: str`, `neo4j_database: str`, `graph_enabled: bool`, `graph_default_within_days: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_graph.py
from __future__ import annotations

from docintel.config import Settings


def test_graph_settings_defaults() -> None:
    s = Settings()
    assert s.neo4j_uri == "bolt://neo4j:7687"
    assert s.neo4j_user == "neo4j"
    assert s.neo4j_database == "neo4j"
    assert s.graph_enabled is True
    assert s.graph_default_within_days == 90
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_config_graph.py -v`
Expected: FAIL with `AttributeError` / missing field.

- [ ] **Step 3: Add the settings**

In `config.py`, after the `llm_timeout_s` line (end of the C2 block), add:

```python
    # GraphRAG / Neo4j (C3)
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"
    neo4j_database: str = "neo4j"
    graph_enabled: bool = True
    graph_default_within_days: int = 90
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && pytest tests/test_config_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Add the `graph` extra + mypy override**

In `pyproject.toml`, after the `rag = [...]` extra add:

```toml
graph = [
    "neo4j>=5.0",
]
```

In the `[[tool.mypy.overrides]]` `module = [...]` list, append `"neo4j.*"` to the existing array.

- [ ] **Step 6: Add the Neo4j Compose service**

In `docker-compose.yml`, under `services.api.environment` add:

```yaml
      DOCINTEL_NEO4J_URI: bolt://neo4j:7687
```

Under `services.api.depends_on` add `- neo4j`. After the `qdrant:` service block add:

```yaml
  neo4j:
    image: neo4j:5.23
    container_name: docintel-neo4j
    environment:
      NEO4J_AUTH: neo4j/neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j-data:/data
```

Under top-level `volumes:` add `  neo4j-data:`.

- [ ] **Step 7: Verify compose is valid YAML**

Run: `cd docintel && python -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"`
Expected: no output, exit 0.

- [ ] **Step 8: Commit**

```bash
git add docintel/src/docintel/config.py docintel/pyproject.toml docintel/docker-compose.yml tests/test_config_graph.py
git commit -m "feat(graph): add Neo4j settings, graph extra, and Compose service

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Graph schema models

**Files:**
- Create: `docintel/src/docintel/graph/__init__.py` (empty)
- Create: `docintel/src/docintel/graph/schema.py`
- Test: `tests/test_graph_schema.py`

**Interfaces:**
- Produces:
  - `ExpirationFact(iso_date: str, answer_text: str, char_start: int, char_end: int)`
  - `RenewalFact(answer_text: str, char_start: int, char_end: int)`
  - `GraphContract(contract_id: str, expiration: ExpirationFact | None = None, renewal: RenewalFact | None = None)`
  - `RouteDecision(target: Literal["graph", "vector"], template: str | None = None, within_days: int | None = None)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_schema.py
from __future__ import annotations

from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact, RouteDecision


def test_graph_contract_holds_optional_facts() -> None:
    gc = GraphContract(
        contract_id="c1",
        expiration=ExpirationFact(
            iso_date="2025-12-31", answer_text="expires on December 31, 2025", char_start=10, char_end=40
        ),
        renewal=RenewalFact(answer_text="auto-renews annually", char_start=50, char_end=70),
    )
    assert gc.expiration is not None and gc.expiration.iso_date == "2025-12-31"
    assert gc.renewal is not None and gc.renewal.char_start == 50


def test_graph_contract_defaults_none() -> None:
    gc = GraphContract(contract_id="c2")
    assert gc.expiration is None and gc.renewal is None


def test_route_decision_minimal() -> None:
    d = RouteDecision(target="vector")
    assert d.template is None and d.within_days is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_graph_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.graph.schema`.

- [ ] **Step 3: Create the package + schema**

Create empty `docintel/src/docintel/graph/__init__.py`. Create `schema.py`:

```python
"""Pydantic models for the C3 graph layer: normalized facts and the router decision."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ExpirationFact(BaseModel):
    """A contract's expiration date, normalized to ISO, with its citation span."""

    iso_date: str
    answer_text: str
    char_start: int
    char_end: int


class RenewalFact(BaseModel):
    """Presence of a renewal/auto-renewal clause, with its citation span."""

    answer_text: str
    char_start: int
    char_end: int


class GraphContract(BaseModel):
    """The minimal, normalized projection of a ContractDocument upserted into the graph."""

    contract_id: str
    expiration: ExpirationFact | None = None
    renewal: RenewalFact | None = None


class RouteDecision(BaseModel):
    """Where /ask should send a question, plus the chosen template and parameters."""

    target: Literal["graph", "vector"]
    template: str | None = None
    within_days: int | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && pytest tests/test_graph_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docintel/src/docintel/graph/__init__.py docintel/src/docintel/graph/schema.py tests/test_graph_schema.py
git commit -m "feat(graph): add graph schema models

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Normalization — ContractDocument → GraphContract

**Files:**
- Create: `docintel/src/docintel/graph/normalize.py`
- Test: `tests/test_graph_normalize.py`

**Interfaces:**
- Consumes: `ContractDocument`, `ExtractedClause` from `docintel.contracts.schema`; `GraphContract`, `ExpirationFact`, `RenewalFact` from `docintel.graph.schema`.
- Produces:
  - `EXPIRATION_CLAUSE = "Expiration Date"`, `RENEWAL_CLAUSE = "Renewal Term"` (module constants, exact CUAD category names from `questions.py`).
  - `parse_iso_date(text: str) -> str | None` — first date found, as `YYYY-MM-DD`, else `None`.
  - `build_graph_contract(doc: ContractDocument) -> GraphContract`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_normalize.py
from __future__ import annotations

from docintel.contracts.schema import ContractDocument, ExtractedClause
from docintel.graph.normalize import build_graph_contract, parse_iso_date


def test_parse_iso_date_long_form() -> None:
    assert parse_iso_date("This Agreement shall expire on December 31, 2025.") == "2025-12-31"


def test_parse_iso_date_numeric_and_iso() -> None:
    assert parse_iso_date("term ends 01/05/2026") == "2026-01-05"
    assert parse_iso_date("until 2027-03-09 inclusive") == "2027-03-09"


def test_parse_iso_date_unparseable_returns_none() -> None:
    assert parse_iso_date("expires at the end of the term") is None


def _doc(clauses: list[ExtractedClause]) -> ContractDocument:
    return ContractDocument(
        id="c1", source="digital", clauses=clauses, derived={}, page_count=1, created_at="t"
    )


def test_build_graph_contract_extracts_expiration_and_renewal() -> None:
    doc = _doc(
        [
            ExtractedClause(
                clause_type="Expiration Date",
                answer_text="expire on December 31, 2025",
                char_start=10,
                char_end=37,
                confidence=0.9,
            ),
            ExtractedClause(
                clause_type="Renewal Term",
                answer_text="renews for successive one-year terms",
                char_start=50,
                char_end=86,
                confidence=0.8,
            ),
        ]
    )
    gc = build_graph_contract(doc)
    assert gc.contract_id == "c1"
    assert gc.expiration is not None and gc.expiration.iso_date == "2025-12-31"
    assert gc.renewal is not None and gc.renewal.char_start == 50


def test_build_graph_contract_skips_unparseable_expiration() -> None:
    doc = _doc(
        [
            ExtractedClause(
                clause_type="Expiration Date",
                answer_text="end of the term",
                char_start=0,
                char_end=15,
                confidence=0.9,
            )
        ]
    )
    gc = build_graph_contract(doc)
    assert gc.expiration is None and gc.renewal is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_graph_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement normalization**

```python
"""Rule-based normalization: a ContractDocument's clauses → a minimal GraphContract.

Date parsing is intentionally rule-based (a small set of explicit formats); unparseable
expiration text is skipped, never blocking the build. Party/governing-law normalization is
out of scope for C3.
"""

from __future__ import annotations

import re
from datetime import datetime

from docintel.contracts.schema import ContractDocument, ExtractedClause
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact

# Exact CUAD category names (reference data; see contracts/questions.py).
EXPIRATION_CLAUSE = "Expiration Date"
RENEWAL_CLAUSE = "Renewal Term"

# (compiled regex, strptime format) pairs tried in order. ISO is handled first verbatim.
_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b((?:January|February|March|April|May|June|July|August|September|October"
            r"|November|December)\s+\d{1,2},\s+\d{4})\b"
        ),
        "%B %d, %Y",
    ),
    (re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"), "%m/%d/%Y"),
)


def parse_iso_date(text: str) -> str | None:
    """Return the first date in ``text`` as ``YYYY-MM-DD``, or None if none parse."""
    iso = _ISO.search(text)
    if iso is not None:
        try:
            return datetime.strptime(iso.group(1), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    for pattern, fmt in _DATE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        try:
            return datetime.strptime(match.group(1), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _first(clauses: list[ExtractedClause], clause_type: str) -> ExtractedClause | None:
    return next((c for c in clauses if c.clause_type == clause_type), None)


def build_graph_contract(doc: ContractDocument) -> GraphContract:
    """Project a ContractDocument onto the minimal graph node set (expiration + renewal)."""
    expiration: ExpirationFact | None = None
    exp_clause = _first(doc.clauses, EXPIRATION_CLAUSE)
    if exp_clause is not None:
        iso = parse_iso_date(exp_clause.answer_text)
        if iso is not None:
            expiration = ExpirationFact(
                iso_date=iso,
                answer_text=exp_clause.answer_text,
                char_start=exp_clause.char_start,
                char_end=exp_clause.char_end,
            )

    renewal: RenewalFact | None = None
    ren_clause = _first(doc.clauses, RENEWAL_CLAUSE)
    if ren_clause is not None:
        renewal = RenewalFact(
            answer_text=ren_clause.answer_text,
            char_start=ren_clause.char_start,
            char_end=ren_clause.char_end,
        )

    return GraphContract(contract_id=doc.id, expiration=expiration, renewal=renewal)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && pytest tests/test_graph_normalize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docintel/src/docintel/graph/normalize.py tests/test_graph_normalize.py
git commit -m "feat(graph): normalize ContractDocument to GraphContract

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: GraphStore Protocol + in-memory fake + builder

**Files:**
- Create: `docintel/src/docintel/graph/store.py`
- Test: `tests/test_graph_store.py`

**Interfaces:**
- Consumes: `GraphContract` from `docintel.graph.schema`; `Settings`.
- Produces:
  - `GraphStore` Protocol: `upsert_contract(self, gc: GraphContract) -> None`; `run_template(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]`.
  - `InMemoryGraphStore` implementing both templates in Python. Row keys: `contract_id`, `iso_date`, `exp_answer`, `exp_start`, `exp_end`, and (auto-renew only) `ren_answer`, `ren_start`, `ren_end`.
  - Template names: `"expiring_within"`, `"auto_renewing_expiring_within"`. Params used: `lower` (ISO), `upper` (ISO).
  - `build_graph_store(settings: Settings) -> GraphStore | None` — `None` when `graph_enabled` is False (real Neo4j impl arrives in Task 5; here it raises `NotImplementedError` if enabled — replaced in Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_store.py
from __future__ import annotations

from docintel.config import Settings
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact
from docintel.graph.store import InMemoryGraphStore, build_graph_store


def _gc(cid: str, iso: str, renews: bool) -> GraphContract:
    return GraphContract(
        contract_id=cid,
        expiration=ExpirationFact(iso_date=iso, answer_text=f"exp {iso}", char_start=0, char_end=5),
        renewal=RenewalFact(answer_text="renews", char_start=10, char_end=16) if renews else None,
    )


def test_expiring_within_filters_by_date_window() -> None:
    store = InMemoryGraphStore()
    store.upsert_contract(_gc("a", "2025-02-01", renews=False))
    store.upsert_contract(_gc("b", "2025-12-31", renews=True))
    rows = store.run_template("expiring_within", {"lower": "2025-01-01", "upper": "2025-06-30"})
    assert [r["contract_id"] for r in rows] == ["a"]
    assert rows[0]["exp_answer"] == "exp 2025-02-01"


def test_auto_renewing_requires_renewal_clause() -> None:
    store = InMemoryGraphStore()
    store.upsert_contract(_gc("a", "2025-02-01", renews=False))
    store.upsert_contract(_gc("b", "2025-03-01", renews=True))
    rows = store.run_template(
        "auto_renewing_expiring_within", {"lower": "2025-01-01", "upper": "2025-06-30"}
    )
    assert [r["contract_id"] for r in rows] == ["b"]
    assert rows[0]["ren_answer"] == "renews"


def test_upsert_is_idempotent_per_contract() -> None:
    store = InMemoryGraphStore()
    store.upsert_contract(_gc("a", "2025-02-01", renews=False))
    store.upsert_contract(_gc("a", "2025-02-02", renews=False))
    rows = store.run_template("expiring_within", {"lower": "2025-01-01", "upper": "2025-12-31"})
    assert len(rows) == 1 and rows[0]["iso_date"] == "2025-02-02"


def test_build_graph_store_disabled_returns_none() -> None:
    assert build_graph_store(Settings(graph_enabled=False)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_graph_store.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the Protocol, fake, and builder**

```python
"""GraphStore abstraction: a Protocol plus an in-memory fake for CPU-only tests.

The fake re-implements the two date templates in Python so unit tests never need a live
Neo4j. The real Neo4jGraphStore (Task 5) runs the equivalent Cypher; a deselected parity
test guards against drift. build_graph_store returns None when the graph path is disabled.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from docintel.config import Settings
from docintel.graph.schema import GraphContract

_AUTO_RENEW = "auto_renewing_expiring_within"
_EXPIRING = "expiring_within"


@runtime_checkable
class GraphStore(Protocol):
    """Minimal store interface: upsert one contract's subgraph; run a named template."""

    def upsert_contract(self, gc: GraphContract) -> None: ...

    def run_template(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


class InMemoryGraphStore:
    """Dict-backed fake; runs the two date templates in Python. Upsert is keyed by id."""

    def __init__(self) -> None:
        self._data: dict[str, GraphContract] = {}

    def upsert_contract(self, gc: GraphContract) -> None:
        self._data[gc.contract_id] = gc

    def run_template(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        lower, upper = params["lower"], params["upper"]
        rows: list[dict[str, Any]] = []
        for gc in self._data.values():
            if gc.expiration is None or not (lower <= gc.expiration.iso_date <= upper):
                continue
            if name == _AUTO_RENEW and gc.renewal is None:
                continue
            row: dict[str, Any] = {
                "contract_id": gc.contract_id,
                "iso_date": gc.expiration.iso_date,
                "exp_answer": gc.expiration.answer_text,
                "exp_start": gc.expiration.char_start,
                "exp_end": gc.expiration.char_end,
            }
            if name == _AUTO_RENEW and gc.renewal is not None:
                row["ren_answer"] = gc.renewal.answer_text
                row["ren_start"] = gc.renewal.char_start
                row["ren_end"] = gc.renewal.char_end
            rows.append(row)
        return rows


def build_graph_store(settings: Settings) -> GraphStore | None:
    """Return a GraphStore, or None when the graph path is disabled."""
    if not settings.graph_enabled:
        return None
    from docintel.graph.templates import Neo4jGraphStore

    return Neo4jGraphStore(settings)
```

> Note: the `build_graph_store` import of `Neo4jGraphStore` is satisfied in Task 5. To keep Task 4 green in isolation, the `test_build_graph_store_disabled_returns_none` test only exercises the disabled branch (no import). Do not add an enabled-path unit test here.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && pytest tests/test_graph_store.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add docintel/src/docintel/graph/store.py tests/test_graph_store.py
git commit -m "feat(graph): add GraphStore protocol, in-memory fake, and builder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Cypher templates + Neo4jGraphStore

**Files:**
- Create: `docintel/src/docintel/graph/templates.py`
- Test: `tests/test_graph_templates.py`

**Interfaces:**
- Consumes: `Settings`, `GraphContract`, `get_template`.
- Produces:
  - `TEMPLATES: dict[str, str]` and `get_template(name: str) -> str`.
  - `Neo4jGraphStore` implementing `GraphStore` (lazy `neo4j` import; `upsert_contract` MERGEs nodes/edges; `run_template` runs `get_template(name)` with `params`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_templates.py
from __future__ import annotations

import pytest

from docintel.graph.templates import TEMPLATES, get_template


def test_templates_registry_has_both_patterns() -> None:
    assert set(TEMPLATES) == {"expiring_within", "auto_renewing_expiring_within"}


def test_expiring_template_filters_on_date_bounds() -> None:
    cypher = get_template("expiring_within")
    assert "EXPIRES_ON" in cypher and "$lower" in cypher and "$upper" in cypher
    assert "contract_id" in cypher


def test_auto_renew_template_requires_has_clause() -> None:
    cypher = get_template("auto_renewing_expiring_within")
    assert "HAS_CLAUSE" in cypher and "ren_answer" in cypher


def test_get_template_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_template("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_graph_templates.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement templates + Neo4j store**

```python
"""Parameterized Cypher templates and the real Neo4j-backed GraphStore.

Only two deterministic templates are exposed (no Cypher hallucination, no text-to-Cypher).
Citations travel on the EXPIRES_ON / HAS_CLAUSE relationships so answers stay grounded.
"""

from __future__ import annotations

from typing import Any

from docintel.config import Settings
from docintel.graph.normalize import RENEWAL_CLAUSE
from docintel.graph.schema import GraphContract

TEMPLATES: dict[str, str] = {
    "expiring_within": (
        "MATCH (c:Contract)-[r:EXPIRES_ON]->(d:Date) "
        "WHERE d.iso >= $lower AND d.iso <= $upper "
        "RETURN c.id AS contract_id, d.iso AS iso_date, "
        "r.answer_text AS exp_answer, r.char_start AS exp_start, r.char_end AS exp_end "
        "ORDER BY d.iso"
    ),
    "auto_renewing_expiring_within": (
        "MATCH (c:Contract)-[r:EXPIRES_ON]->(d:Date) "
        "MATCH (c)-[hr:HAS_CLAUSE]->(:ClauseType {name: $renewal}) "
        "WHERE d.iso >= $lower AND d.iso <= $upper "
        "RETURN c.id AS contract_id, d.iso AS iso_date, "
        "r.answer_text AS exp_answer, r.char_start AS exp_start, r.char_end AS exp_end, "
        "hr.answer_text AS ren_answer, hr.char_start AS ren_start, hr.char_end AS ren_end "
        "ORDER BY d.iso"
    ),
}

_UPSERT = (
    "MERGE (c:Contract {id: $contract_id}) "
    "WITH c "
    "OPTIONAL MATCH (c)-[old:EXPIRES_ON|HAS_CLAUSE]->() DELETE old "
    "WITH c "
    "FOREACH (_ IN CASE WHEN $iso IS NULL THEN [] ELSE [1] END | "
    "  MERGE (d:Date {iso: $iso}) "
    "  MERGE (c)-[r:EXPIRES_ON]->(d) "
    "  SET r.answer_text = $exp_answer, r.char_start = $exp_start, r.char_end = $exp_end) "
    "FOREACH (_ IN CASE WHEN $ren_answer IS NULL THEN [] ELSE [1] END | "
    "  MERGE (ct:ClauseType {name: $renewal}) "
    "  MERGE (c)-[hr:HAS_CLAUSE]->(ct) "
    "  SET hr.answer_text = $ren_answer, hr.char_start = $ren_start, hr.char_end = $ren_end)"
)


def get_template(name: str) -> str:
    """Return the Cypher for a template name (raises KeyError if unknown)."""
    return TEMPLATES[name]


class Neo4jGraphStore:
    """Real GraphStore backed by the official neo4j driver (lazy-imported)."""

    def __init__(self, settings: Settings) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self._database = settings.neo4j_database

    def upsert_contract(self, gc: GraphContract) -> None:
        params: dict[str, Any] = {
            "contract_id": gc.contract_id,
            "renewal": RENEWAL_CLAUSE,
            "iso": gc.expiration.iso_date if gc.expiration else None,
            "exp_answer": gc.expiration.answer_text if gc.expiration else None,
            "exp_start": gc.expiration.char_start if gc.expiration else None,
            "exp_end": gc.expiration.char_end if gc.expiration else None,
            "ren_answer": gc.renewal.answer_text if gc.renewal else None,
            "ren_start": gc.renewal.char_start if gc.renewal else None,
            "ren_end": gc.renewal.char_end if gc.renewal else None,
        }
        with self._driver.session(database=self._database) as session:
            session.run(_UPSERT, **params)

    def run_template(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        cypher = get_template(name)
        merged = {"renewal": RENEWAL_CLAUSE, **params}
        with self._driver.session(database=self._database) as session:
            return [record.data() for record in session.run(cypher, **merged)]

    def close(self) -> None:
        self._driver.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && pytest tests/test_graph_templates.py -v`
Expected: PASS (4 tests). (`Neo4jGraphStore` is import-only here; the lazy `neo4j` import is not triggered.)

- [ ] **Step 5: Type-check (lazy import must not break mypy)**

Run: `cd docintel && mypy src/docintel/graph/templates.py src/docintel/graph/store.py`
Expected: Success (the `neo4j.*` override from Task 1 silences the missing stub).

- [ ] **Step 6: Commit**

```bash
git add docintel/src/docintel/graph/templates.py tests/test_graph_templates.py
git commit -m "feat(graph): add Cypher templates and Neo4j-backed store

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Rule-based router

**Files:**
- Create: `docintel/src/docintel/graph/router.py`
- Test: `tests/test_graph_router.py`

**Interfaces:**
- Consumes: `RouteDecision` from `docintel.graph.schema`.
- Produces: `route(question: str) -> RouteDecision` — pure, no LLM, no settings. Extracts `within_days` from `"… N day(s)"`; leaves it `None` when absent (query fills the default).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_router.py
from __future__ import annotations

from docintel.graph.router import route


def test_expiry_question_routes_to_graph() -> None:
    d = route("Which contracts expire within 90 days?")
    assert d.target == "graph"
    assert d.template == "expiring_within"
    assert d.within_days == 90


def test_auto_renew_plus_expiry_routes_to_auto_template() -> None:
    d = route("Which auto-renewing contracts expire within 30 days?")
    assert d.template == "auto_renewing_expiring_within"
    assert d.within_days == 30


def test_renewal_wording_detected() -> None:
    assert route("list contracts that renew and expire in 14 days").template == (
        "auto_renewing_expiring_within"
    )


def test_no_day_count_leaves_within_none() -> None:
    d = route("which contracts are expiring soon?")
    assert d.target == "graph" and d.within_days is None


def test_non_graph_question_routes_to_vector() -> None:
    d = route("What is the governing law of this agreement?")
    assert d.target == "vector" and d.template is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_graph_router.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the router**

```python
"""Pure rule-based router: classify a question to a graph template or the vector path.

No LLM and no settings are consulted, so routing is deterministic and works in the degraded
(LLM-down) path. Parameter extraction is regex-only.
"""

from __future__ import annotations

import re

from docintel.graph.schema import RouteDecision

_EXPIRY = re.compile(r"\bexpir", re.IGNORECASE)
_RENEWAL = re.compile(r"\b(?:auto[\s-]?renew|renew)", re.IGNORECASE)
_WITHIN_DAYS = re.compile(r"(\d+)\s*day", re.IGNORECASE)


def route(question: str) -> RouteDecision:
    """Map a question to {graph template | vector}, extracting an N-day window if present."""
    if not _EXPIRY.search(question):
        return RouteDecision(target="vector")
    days_match = _WITHIN_DAYS.search(question)
    within_days = int(days_match.group(1)) if days_match else None
    template = (
        "auto_renewing_expiring_within"
        if _RENEWAL.search(question)
        else "expiring_within"
    )
    return RouteDecision(target="graph", template=template, within_days=within_days)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && pytest tests/test_graph_router.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add docintel/src/docintel/graph/router.py tests/test_graph_router.py
git commit -m "feat(graph): add rule-based question router

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Graph query — template rows → cited RetrievedChunks

**Files:**
- Create: `docintel/src/docintel/graph/query.py`
- Test: `tests/test_graph_query.py`

**Interfaces:**
- Consumes: `GraphStore`, `RouteDecision`, `Settings`, `RetrievedChunk` (`docintel.rag.schema`).
- Produces: `run_graph_query(store: GraphStore, decision: RouteDecision, settings: Settings, reference_date: date | None = None) -> list[RetrievedChunk]`. Uses `decision.within_days or settings.graph_default_within_days`; window is `[reference_date, reference_date + within]`. Emits one `RetrievedChunk` for the expiration clause and, for the auto-renew template, an extra one for the renewal clause. `chunk_kind="graph"`, `score=1.0`, `chunk_index=0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_query.py
from __future__ import annotations

from datetime import date

from docintel.config import Settings
from docintel.graph.query import run_graph_query
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact, RouteDecision
from docintel.graph.store import InMemoryGraphStore


def _store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2026-02-01", answer_text="expires 2026-02-01", char_start=0, char_end=10
            ),
            renewal=RenewalFact(answer_text="auto-renews", char_start=20, char_end=31),
        )
    )
    return store


def test_expiring_within_emits_expiration_citation() -> None:
    decision = RouteDecision(target="graph", template="expiring_within", within_days=60)
    chunks = run_graph_query(_store(), decision, Settings(), reference_date=date(2026, 1, 15))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.contract_id == "a" and c.chunk_kind == "graph"
    assert c.clause_type == "Expiration Date" and c.char_start == 0 and c.char_end == 10


def test_auto_renew_emits_two_citations() -> None:
    decision = RouteDecision(
        target="graph", template="auto_renewing_expiring_within", within_days=60
    )
    chunks = run_graph_query(_store(), decision, Settings(), reference_date=date(2026, 1, 15))
    kinds = {c.clause_type for c in chunks}
    assert kinds == {"Expiration Date", "Renewal Term"}


def test_default_window_used_when_within_none() -> None:
    decision = RouteDecision(target="graph", template="expiring_within", within_days=None)
    # graph_default_within_days=90 -> 2026-01-15 .. 2026-04-15 includes 2026-02-01
    chunks = run_graph_query(_store(), decision, Settings(), reference_date=date(2026, 1, 15))
    assert len(chunks) == 1


def test_out_of_window_returns_empty() -> None:
    decision = RouteDecision(target="graph", template="expiring_within", within_days=5)
    chunks = run_graph_query(_store(), decision, Settings(), reference_date=date(2026, 1, 15))
    assert chunks == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_graph_query.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the query**

```python
"""Run a routed template against a GraphStore and return cited RetrievedChunks.

Date bounds are computed in Python (reference_date .. reference_date + N days) and passed as
ISO strings, keeping the Cypher a pure string comparison. Citations reuse C2's RetrievedChunk
shape so /ask returns one consistent citation contract across vector and graph answers.
"""

from __future__ import annotations

from datetime import date, timedelta

from docintel.config import Settings
from docintel.graph.normalize import EXPIRATION_CLAUSE, RENEWAL_CLAUSE
from docintel.graph.schema import RouteDecision
from docintel.graph.store import GraphStore
from docintel.rag.schema import RetrievedChunk


def run_graph_query(
    store: GraphStore,
    decision: RouteDecision,
    settings: Settings,
    reference_date: date | None = None,
) -> list[RetrievedChunk]:
    """Execute the routed template and map result rows to cited RetrievedChunks."""
    assert decision.template is not None  # router guarantees this for target == "graph"
    start = reference_date or date.today()
    within = decision.within_days or settings.graph_default_within_days
    params = {"lower": start.isoformat(), "upper": (start + timedelta(days=within)).isoformat()}
    rows = store.run_template(decision.template, params)

    chunks: list[RetrievedChunk] = []
    for row in rows:
        chunks.append(
            RetrievedChunk(
                contract_id=row["contract_id"],
                chunk_index=0,
                chunk_kind="graph",
                clause_type=EXPIRATION_CLAUSE,
                text=row["exp_answer"],
                score=1.0,
                char_start=row["exp_start"],
                char_end=row["exp_end"],
            )
        )
        if "ren_answer" in row:
            chunks.append(
                RetrievedChunk(
                    contract_id=row["contract_id"],
                    chunk_index=0,
                    chunk_kind="graph",
                    clause_type=RENEWAL_CLAUSE,
                    text=row["ren_answer"],
                    score=1.0,
                    char_start=row["ren_start"],
                    char_end=row["ren_end"],
                )
            )
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && pytest tests/test_graph_query.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add docintel/src/docintel/graph/query.py tests/test_graph_query.py
git commit -m "feat(graph): run routed template and emit cited chunks

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Build at extract time (build.py + route + lifespan wiring)

**Files:**
- Create: `docintel/src/docintel/graph/build.py`
- Modify: `docintel/src/docintel/api/main.py` (lifespan: `app.state.graph_store = None`)
- Modify: `docintel/src/docintel/api/routes/contracts.py` (best-effort graph build after `index_contract`)
- Test: `tests/test_graph_build.py`

**Interfaces:**
- Consumes: `ContractDocument`, `GraphStore`, `build_graph_contract`.
- Produces: `build_contract(doc: ContractDocument, store: GraphStore) -> bool` — upserts the contract's subgraph; returns whether any fact was present.
- The contracts route gains a `get_graph_store_optional` dependency (added in Task 9's ask.py; for this task, import it from `docintel.api.routes.ask`). If Task 8 runs before Task 9, add the dependency stub described below first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_build.py
from __future__ import annotations

from docintel.contracts.schema import ContractDocument, ExtractedClause
from docintel.graph.build import build_contract
from docintel.graph.store import InMemoryGraphStore


def _doc() -> ContractDocument:
    return ContractDocument(
        id="c1",
        source="digital",
        clauses=[
            ExtractedClause(
                clause_type="Expiration Date",
                answer_text="expires on December 31, 2025",
                char_start=0,
                char_end=28,
                confidence=0.9,
            )
        ],
        derived={},
        page_count=1,
        created_at="t",
    )


def test_build_contract_upserts_and_reports_facts() -> None:
    store = InMemoryGraphStore()
    assert build_contract(_doc(), store) is True
    rows = store.run_template("expiring_within", {"lower": "2025-01-01", "upper": "2025-12-31"})
    assert [r["contract_id"] for r in rows] == ["c1"]


def test_build_contract_with_no_facts_returns_false() -> None:
    empty = ContractDocument(
        id="c2", source="digital", clauses=[], derived={}, page_count=1, created_at="t"
    )
    store = InMemoryGraphStore()
    assert build_contract(empty, store) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_graph_build.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement build.py**

```python
"""Upsert one ContractDocument's subgraph into a GraphStore (called best-effort at extract)."""

from __future__ import annotations

from docintel.contracts.schema import ContractDocument
from docintel.graph.normalize import build_graph_contract
from docintel.graph.store import GraphStore


def build_contract(doc: ContractDocument, store: GraphStore) -> bool:
    """Normalize and upsert the contract; return True if any fact (expiration/renewal) existed."""
    gc = build_graph_contract(doc)
    store.upsert_contract(gc)
    return gc.expiration is not None or gc.renewal is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd docintel && pytest tests/test_graph_build.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire `graph_store` into the lifespan**

In `api/main.py` lifespan, after `app.state.rag_llm = None` add:

```python
    app.state.graph_store = None
```

- [ ] **Step 6: Add the best-effort build call in the contracts route**

In `api/routes/contracts.py`:
1. Add imports near the existing graph-free imports:

```python
from docintel.api.routes.ask import get_graph_store_optional
from docintel.graph.build import build_contract
```

2. Add a parameter to `extract_contract`'s signature (after `rag_store`):

```python
    graph_store: Any = Depends(get_graph_store_optional),  # noqa: B008
```

3. After the existing `if rag_store is not None:` indexing block, add:

```python
    if graph_store is not None:
        try:
            build_contract(doc, graph_store)
        except Exception:
            logger.warning(
                "contracts.extract.graph_build_failed",
                extra={"contract_id": doc.id},
                exc_info=True,
            )
```

> `get_graph_store_optional` is defined in Task 9. If executing Task 8 before Task 9, first add the dependency block from Task 9 Step 3 to `ask.py`, then return here.

- [ ] **Step 7: Run the contracts route tests**

Run: `cd docintel && pytest tests/test_contracts_routes.py tests/test_graph_build.py -v`
Expected: PASS (existing route tests still green; build wired without breaking extract).

- [ ] **Step 8: Commit**

```bash
git add docintel/src/docintel/graph/build.py docintel/src/docintel/api/main.py docintel/src/docintel/api/routes/contracts.py tests/test_graph_build.py
git commit -m "feat(graph): build contract subgraph best-effort at extract time

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Route /ask through the router (+ shared degrade tail + metrics)

**Files:**
- Modify: `docintel/src/docintel/rag/answer.py` (extract `generate_or_degrade`)
- Modify: `docintel/src/docintel/api/metrics.py` (add graph metrics)
- Modify: `docintel/src/docintel/api/routes/ask.py` (graph deps + router wiring + metrics)
- Test: `tests/test_rag_answer.py` (extend), `tests/test_graph_ask_routes.py` (new)

**Interfaces:**
- Consumes: `route`, `run_graph_query`, `build_graph_store`, `RetrievedChunk`, `AskResponse`, `Metrics`.
- Produces:
  - `docintel.rag.answer.generate_or_degrade(question, citations, llm, contract_id) -> AskResponse` (reused by vector + graph paths).
  - `docintel.api.routes.ask.get_graph_store(request, settings) -> GraphStore | None` and `get_graph_store_optional(...) -> GraphStore | None` (cached on `app.state.graph_store`).
  - `Metrics.graph_query_latency: Histogram`, `Metrics.router_decision_total: Counter` (label `target`).

- [ ] **Step 1: Write the failing test for the shared tail**

Append to `tests/test_rag_answer.py`:

```python
def test_generate_or_degrade_is_shared() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from docintel.rag.answer import generate_or_degrade
    from docintel.rag.schema import RetrievedChunk

    cite = RetrievedChunk(
        contract_id="c1",
        chunk_index=0,
        chunk_kind="graph",
        clause_type="Expiration Date",
        text="expires 2026-01-01",
        score=1.0,
        char_start=0,
        char_end=18,
    )
    ok = generate_or_degrade("q", [cite], FakeListChatModel(responses=["A."]), "c1")
    assert ok.answer == "A." and ok.generation_skipped is False and ok.citations == [cite]
    degraded = generate_or_degrade("q", [cite], None, "c1")
    assert degraded.answer is None and degraded.generation_skipped is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_rag_answer.py::test_generate_or_degrade_is_shared -v`
Expected: FAIL with `ImportError: cannot import name 'generate_or_degrade'`.

- [ ] **Step 3: Refactor answer.py to extract the shared tail**

Replace the body of `answer_question` so it delegates, and add `generate_or_degrade`:

```python
def generate_or_degrade(
    question: str,
    citations: list[RetrievedChunk],
    llm: Any | None,
    contract_id: str | None,
) -> AskResponse:
    """Generate a grounded answer from citations, or degrade to citations-only."""
    if llm is None:
        return AskResponse(
            question=question,
            answer=None,
            generation_skipped=True,
            contract_id=contract_id,
            citations=citations,
        )
    try:
        answer = generate_answer(llm, question, format_context(citations))
    except Exception:
        return AskResponse(
            question=question,
            answer=None,
            generation_skipped=True,
            contract_id=contract_id,
            citations=citations,
        )
    return AskResponse(
        question=question,
        answer=answer,
        generation_skipped=False,
        contract_id=contract_id,
        citations=citations,
    )


def answer_question(
    question: str,
    store: Any,
    llm: Any | None,
    settings: Settings,
    contract_id: str | None = None,
    top_k: int | None = None,
) -> AskResponse:
    """Retrieve top-k chunks, then generate a grounded answer or degrade to citations."""
    citations = search(store, question, top_k or settings.rag_top_k, contract_id)
    return generate_or_degrade(question, citations, llm, contract_id)
```

Add the import at the top of `answer.py`:

```python
from docintel.rag.schema import AskResponse, RetrievedChunk
```

- [ ] **Step 4: Run answer tests (old + new) to verify they pass**

Run: `cd docintel && pytest tests/test_rag_answer.py -v`
Expected: PASS (existing 3 + new 1).

- [ ] **Step 5: Add the graph metrics**

In `api/metrics.py`, import `Histogram` is already present. Add two fields to `Metrics`:

```python
    graph_query_latency: Histogram
    router_decision_total: Counter
```

In `build_metrics(...)`, add to the constructor call:

```python
        graph_query_latency=Histogram(
            "docintel_graph_query_latency_seconds",
            "Latency of graph template queries on /ask.",
            registry=registry,
        ),
        router_decision_total=Counter(
            "docintel_router_decisions",
            "/ask routing decisions, labelled by target.",
            labelnames=("target",),
            registry=registry,
        ),
```

- [ ] **Step 6: Write the failing /ask graph route test**

```python
# tests/test_graph_ask_routes.py
from __future__ import annotations

from datetime import date
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docintel.api.main import create_app
from docintel.api.routes.ask import get_graph_store, get_rag_llm
from docintel.config import Settings, get_settings
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact
from docintel.graph.store import InMemoryGraphStore


def _graph_store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    # Far-future date so the default window keeps the fixture valid over time.
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-02-01", answer_text="expires 2999-02-01", char_start=0, char_end=18
            ),
            renewal=RenewalFact(answer_text="auto-renews", char_start=20, char_end=31),
        )
    )
    return store


def test_graph_route_degrades_without_llm() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings()
    app.dependency_overrides[get_graph_store] = lambda: _graph_store()
    app.dependency_overrides[get_rag_llm] = lambda: None
    with TestClient(app) as client:
        resp = client.post(
            "/ask", json={"question": "which auto-renewing contracts expire within 400000 days?"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["generation_skipped"] is True
    kinds = {c["clause_type"] for c in body["citations"]}
    assert kinds == {"Expiration Date", "Renewal Term"}


def test_graph_route_generates_with_llm() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings()
    app.dependency_overrides[get_graph_store] = lambda: _graph_store()
    app.dependency_overrides[get_rag_llm] = lambda: FakeListChatModel(responses=["One contract."])
    with TestClient(app) as client:
        resp = client.post(
            "/ask", json={"question": "which contracts expire within 400000 days?"}
        )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "One contract."
```

- [ ] **Step 7: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_graph_ask_routes.py -v`
Expected: FAIL (`get_graph_store` missing, or vector path taken).

- [ ] **Step 8: Wire the router into ask.py**

In `api/routes/ask.py`:

1. Add imports:

```python
import time

from docintel.api.metrics import Metrics
from docintel.api.routes.extract import get_metrics
from docintel.graph.query import run_graph_query
from docintel.graph.router import route
from docintel.graph.store import build_graph_store
from docintel.rag.answer import generate_or_degrade
```

2. Add the graph-store dependencies (mirroring the rag-store ones):

```python
def ensure_graph_store(app: Any, settings: Settings) -> Any:
    """Build the graph store once and cache it on app.state (None when disabled)."""
    store = getattr(app.state, "graph_store", None)
    if store is None:
        store = build_graph_store(settings)
        app.state.graph_store = store
    return store


def get_graph_store(request: Request, settings: Settings = Depends(get_settings)) -> Any | None:  # noqa: B008
    """Graph store dependency for /ask routing; None when graph is disabled."""
    return ensure_graph_store(request.app, settings)


def get_graph_store_optional(
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Any | None:
    """Best-effort graph store for extract-time build; returns None instead of raising."""
    try:
        return ensure_graph_store(request.app, settings)
    except Exception:
        logger.warning("graph.store.unavailable", exc_info=True)
        return None
```

3. Replace the `ask(...)` handler so it routes first. The vector path stays unchanged; the graph path runs only when a graph store is present:

```python
@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question grounded in indexed contracts",
)
def ask(
    req: AskRequest,
    settings: Settings = Depends(get_settings),  # noqa: B008
    store: Any = Depends(get_rag_store),  # noqa: B008
    graph_store: Any | None = Depends(get_graph_store),  # noqa: B008
    llm: Any | None = Depends(get_rag_llm),  # noqa: B008
    metrics: Metrics = Depends(get_metrics),  # noqa: B008
) -> AskResponse:
    """Route to graph or vector retrieval, then generate a grounded answer or degrade."""
    decision = route(req.question)
    target = decision.target if (decision.target == "graph" and graph_store is not None) else "vector"
    metrics.router_decision_total.labels(target=target).inc()
    if target == "graph":
        start = time.perf_counter()
        try:
            citations = run_graph_query(graph_store, decision, settings)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Graph store unavailable."
            ) from exc
        metrics.graph_query_latency.observe(time.perf_counter() - start)
        return generate_or_degrade(req.question, citations, llm, req.contract_id)
    try:
        return answer_question(req.question, store, llm, settings, req.contract_id, req.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store unavailable."
        ) from exc
```

> Note: `get_rag_store` still runs for graph questions (it does no network at build time, matching C2). This keeps the dependency wiring unchanged and the vector path always available as the fallback.

- [ ] **Step 9: Run the new + existing ask tests**

Run: `cd docintel && pytest tests/test_graph_ask_routes.py tests/test_rag_routes.py -v`
Expected: PASS (graph routes green; the three C2 vector-route tests still green — vector questions like "governing law?" route to vector).

- [ ] **Step 10: Full gate**

Run: `cd docintel && ruff check src tests && ruff format --check src tests && mypy src && pytest`
Expected: all green.

- [ ] **Step 11: Commit**

```bash
git add docintel/src/docintel/rag/answer.py docintel/src/docintel/api/metrics.py docintel/src/docintel/api/routes/ask.py tests/test_rag_answer.py tests/test_graph_ask_routes.py
git commit -m "feat(graph): route /ask between graph and vector with shared degrade tail

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Deselected real-Neo4j parity test + GraphRAG eval harness

**Files:**
- Create: `tests/test_graph_neo4j_parity.py` (marked `slow`, deselected by default)
- Create: `docintel/src/docintel/graph/eval.py`
- Test: `tests/test_graph_eval.py`

**Interfaces:**
- Consumes: `Neo4jGraphStore`, `InMemoryGraphStore`, `route`, `run_graph_query`.
- Produces: `docintel.graph.eval.evaluate_multihop(store, cases) -> dict[str, float]` returning `{"multihop_accuracy": float, "n": int}`; `log_to_mlflow(metrics)` (thin, run-later). `EvalCase = (question, expected_contract_ids)`.

- [ ] **Step 1: Write the parity test (deselected)**

```python
# tests/test_graph_neo4j_parity.py
from __future__ import annotations

import os

import pytest

from docintel.config import Settings
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact
from docintel.graph.store import InMemoryGraphStore

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not os.getenv("DOCINTEL_NEO4J_URI"), reason="no live Neo4j")
def test_cypher_matches_fake() -> None:
    from docintel.graph.templates import Neo4jGraphStore

    gc = GraphContract(
        contract_id="p1",
        expiration=ExpirationFact(
            iso_date="2026-02-01", answer_text="exp", char_start=0, char_end=3
        ),
        renewal=RenewalFact(answer_text="ren", char_start=4, char_end=7),
    )
    real = Neo4jGraphStore(Settings())
    fake = InMemoryGraphStore()
    real.upsert_contract(gc)
    fake.upsert_contract(gc)
    params = {"lower": "2026-01-01", "upper": "2026-12-31"}
    real_rows = real.run_template("auto_renewing_expiring_within", params)
    fake_rows = fake.run_template("auto_renewing_expiring_within", params)
    real.close()
    assert {r["contract_id"] for r in real_rows} == {r["contract_id"] for r in fake_rows}
```

- [ ] **Step 2: Confirm it is deselected by default**

Run: `cd docintel && pytest tests/test_graph_neo4j_parity.py -v`
Expected: `1 deselected` (the default `-m 'not slow'` filters it).

- [ ] **Step 3: Write the failing eval test**

```python
# tests/test_graph_eval.py
from __future__ import annotations

from datetime import date

from docintel.config import Settings
from docintel.graph.eval import evaluate_multihop
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact
from docintel.graph.store import InMemoryGraphStore


def _store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2026-02-01", answer_text="e", char_start=0, char_end=1
            ),
            renewal=RenewalFact(answer_text="r", char_start=2, char_end=3),
        )
    )
    store.upsert_contract(
        GraphContract(
            contract_id="b",
            expiration=ExpirationFact(
                iso_date="2026-02-01", answer_text="e", char_start=0, char_end=1
            ),
        )
    )
    return store


def test_evaluate_multihop_scores_expected_contracts() -> None:
    cases = [
        ("which contracts expire within 60 days?", {"a", "b"}),
        ("which auto-renewing contracts expire within 60 days?", {"a"}),
    ]
    metrics = evaluate_multihop(
        _store(), cases, Settings(), reference_date=date(2026, 1, 15)
    )
    assert metrics["multihop_accuracy"] == 1.0
    assert metrics["n"] == 2
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd docintel && pytest tests/test_graph_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 5: Implement eval.py**

```python
"""GraphRAG evaluation: multi-hop accuracy over a constructed case set (run later → MLflow).

Each case is (question, expected_contract_ids). We route + query the graph and compare the
returned contract id set to the expected set. MLflow logging is a thin, optional tail so the
metric function stays unit-testable on the laptop with no tracking server.
"""

from __future__ import annotations

from datetime import date

from docintel.config import Settings
from docintel.graph.query import run_graph_query
from docintel.graph.router import route
from docintel.graph.store import GraphStore

EvalCase = tuple[str, set[str]]


def evaluate_multihop(
    store: GraphStore,
    cases: list[EvalCase],
    settings: Settings,
    reference_date: date | None = None,
) -> dict[str, float]:
    """Return {'multihop_accuracy', 'n'} over the cases (exact contract-id-set match)."""
    if not cases:
        return {"multihop_accuracy": 0.0, "n": 0}
    correct = 0
    for question, expected in cases:
        decision = route(question)
        if decision.target != "graph":
            continue
        chunks = run_graph_query(store, decision, settings, reference_date)
        got = {c.contract_id for c in chunks}
        if got == expected:
            correct += 1
    return {"multihop_accuracy": correct / len(cases), "n": float(len(cases))}


def log_to_mlflow(metrics: dict[str, float], experiment: str = "graphrag-eval") -> None:
    """Log eval metrics to MLflow (import lazy; called only when running the eval, not in CI)."""
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd docintel && pytest tests/test_graph_eval.py -v`
Expected: PASS.

- [ ] **Step 7: Full gate**

Run: `cd docintel && ruff check src tests && ruff format --check src tests && mypy src && pytest`
Expected: all green; parity test reported as deselected.

- [ ] **Step 8: Commit**

```bash
git add tests/test_graph_neo4j_parity.py docintel/src/docintel/graph/eval.py tests/test_graph_eval.py
git commit -m "feat(graph): add Neo4j parity test and multi-hop eval harness

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (each C3 spec decision → task):
- Deterministic graph from extractions → Tasks 3, 4, 5, 8. ✔
- Trimmed schema (Contract + expiration Date + Renewal ClauseType; edges EXPIRES_ON/HAS_CLAUSE with citations) → Tasks 2, 3, 5 (`_UPSERT`), 7. ✔
- Date-only normalization, flag/skip unparseable → Task 3. ✔
- Build folded into `/contracts/extract`, best-effort, idempotent → Tasks 4 (idempotent fake), 5 (`_UPSERT` deletes old edges), 8 (route wiring). ✔
- Two Cypher templates → Task 5. ✔
- Rule-based router, no LLM, returns {graph|vector} → Task 6. ✔
- Hybrid answer assembly, same citation contract, degrade → Tasks 7 (RetrievedChunk reuse), 9 (`generate_or_degrade`). ✔
- `/ask` gains router, one endpoint → Task 9. ✔
- `neo4j` driver behind GraphStore Protocol → Tasks 4, 5. ✔
- `graph_enabled` flag → Tasks 1, 4. ✔
- Config additions + Compose Neo4j service → Task 1. ✔
- Eval (multi-hop accuracy → MLflow) → Task 10. ✔
- Testing: protocol + fake unit tests, deselected real-Neo4j parity → all tasks + Task 10. ✔
- Metrics: graph query latency + router decision mix → Task 9. ✔

**Note on graph-vs-vector lift metric:** the spec lists a "graph-augmented vs. vector-only" comparison. Task 10 delivers `multihop_accuracy` over graph routing; the vector-only comparison is a follow-on within `eval.py` (same case set scored through the C2 vector path) and is left as an explicit extension to run when an LLM/index is available — it is not unit-testable on the laptop and would otherwise add an untestable step. Flag this to the user if full parity with the spec's metric table is required in-phase.

**Placeholder scan:** no TBD/TODO; every code step contains complete code. ✔

**Type consistency:** `GraphStore.run_template` row keys (`contract_id`, `iso_date`, `exp_answer`, `exp_start`, `exp_end`, `ren_answer`, `ren_start`, `ren_end`) are produced identically by `InMemoryGraphStore` (Task 4), the Cypher `RETURN ... AS` aliases (Task 5), and consumed by `run_graph_query` (Task 7). `RouteDecision` fields (`target`, `template`, `within_days`) consistent across Tasks 2, 6, 7. `generate_or_degrade` signature consistent between Tasks 9 definition and 7/9 callers. ✔
