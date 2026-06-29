# Contract Intelligence — C3 Design: GraphRAG (Neo4j)

**Date:** 2026-06-25 · **Revised:** 2026-06-29 (scope trimmed to date-centric templates;
build folded into extract; rule-based router; GraphStore protocol + fake testing)
**Status:** Approved (design); ready for implementation planning. **Depends on C1, C2.**
**Initiative:** Contract Intelligence Platform. **Roadmap:** C1 → C2 → **C3 (this doc)** → C4.

## Goal

Answer **multi-hop** questions that flat vector RAG handles poorly — concretely, the
date-driven patterns *"which contracts expire within N days?"* and *"which auto-renewing
contracts expire within N days?"*. Build a **knowledge graph from the C1 extractions** (no LLM
re-reading the corpus), traverse it with parameterized Cypher, and feed the selected, cited
facts to the LLM for prose. `/ask` gains a rule-based router between vector retrieval (C2) and
graph retrieval (C3).

## Context

- **C1** gives persisted `ContractDocument`s: per contract, a list of `ExtractedClause`
  `{clause_type, answer_text, char_start, char_end, confidence}` + a `derived` view.
- **C2** gives vector retrieval, the `ChatOpenAI` LLM client (optional/intermittent, degrades),
  and `POST /ask`. Indexing is folded **best-effort** into `POST /contracts/extract`.
- Reuse the spine: MLflow (GraphRAG eval), Prometheus/Grafana/Loki, the `Protocol`-backend
  pattern (`KIEBackend`, `ContractExtractor`).
- **New infra:** Neo4j added to Compose (persisted volume).
- **C1's fine-tuned model is not finished** (still in training). C3 is developed and tested
  entirely on **fixtures**, so it does not block on the model.

## Decisions (locked during brainstorming)

1. **Deterministic graph from extractions** (structured KG, not LLM graph extraction).

2. **Trimmed, date-centric schema (strict YAGNI).** Only what the shipped templates need:
   - **Nodes:** `Contract`, `Date` (expiration), `ClauseType` (the renewal/auto-renewal signal).
   - **Edges:** `EXPIRES_ON` (`Contract`→`Date`, carries the **expiration clause citation**),
     `HAS_CLAUSE` (`Contract`→`Renewal` `ClauseType`, carries the **renewal clause citation**).
   - **Deferred** (re-addable when a template needs them): `Party`, `GoverningLaw` nodes;
     party / governing-law normalization; the shared-governing-law and party templates.

3. **Light normalization, date-only.** `normalize.py` parses the relevant expiration clause
   text → ISO date; unparseable dates are **flagged and skipped, never block** the build.
   Rule-based and documented. (Party/governing-law normalization is out of C3.)

4. **Build folded into `POST /contracts/extract`, best-effort.** After extract + persist, the
   contract's subgraph is upserted to Neo4j alongside the existing Qdrant `index_contract`
   call. A down/unreachable Neo4j logs a warning and **never fails extraction** (mirrors the
   C2 indexing contract). Upsert is **idempotent per `contract_id`** — re-extracting overwrites
   that contract's subgraph. (This supersedes the earlier "(graph build, offline)" wording.)

5. **Querying = two parameterized Cypher templates** (deterministic, no Cypher hallucination):
   - `expiring_within` — contracts whose `EXPIRES_ON` date is within `N` days of a reference
     date (default: today).
   - `auto_renewing_expiring_within` — contracts that **have a renewal clause** AND expire
     within `N` days.
   **Text-to-Cypher is dropped from C3** (was a gated stretch goal; rule-based routing removes
   the need).

6. **Rule-based router (no LLM in the routing path).** `router.py` is pure: keyword/regex rules
   classify a question to a template and extract params (`N` days, `expir*`, `renew*`/
   `auto-renew*`, optional reference date). Returns `{vector | graph}`. Because routing and
   traversal need no LLM, **graph retrieval works fully in the degraded path** and is
   deterministic + CPU-testable. **Hybrid routing is deferred** (not needed for two date
   templates; noted as future).

7. **Hybrid answer assembly, same citation contract as C2.** Graph traversal selects the
   candidate contracts/clauses → those **cited facts** (contract_id + clause char spans) are
   passed to C2's `generate` for the prose answer. **Degraded** (LLM unset or call fails):
   return the graph result set directly with `generation_skipped=true`.

8. **`/ask` gains the router; one endpoint, no new route.** The LLM is used **only** for the
   final prose — never for routing or parameter parsing.

9. **Driver = official `neo4j` Python driver** behind a thin `GraphStore` `Protocol`
   (`upsert_contract(...)`, `run_template(name, params) -> rows`). No `langchain-neo4j` —
   routing is deterministic. The protocol mirrors the repo's existing backend protocols.

10. **`graph_enabled` settings flag** keeps the graph path optional (like the LLM path), so the
    whole pipeline stays runnable on the laptop with no Neo4j up.

11. **Eval:** a constructed **multi-hop question set** over the two templates; report multi-hop
    answer accuracy and a **graph-augmented vs. vector-only** comparison → MLflow.

## Architecture / new modules

```
src/docintel/graph/
  normalize.py (new)  expiration clause text → ISO date (rule-based; flag unparseable)
  store.py     (new)  GraphStore Protocol + Neo4jGraphStore (official neo4j driver) + in-memory fake
  build.py     (new)  ContractDocument → normalize() → upsert nodes/edges (idempotent per contract)
  templates.py (new)  two parameterized Cypher templates (expiring_within, auto_renewing_expiring_within)
  query.py     (new)  run template via GraphStore → cited facts {contract_id, clause spans}
  router.py    (new)  pure rule-based: question → {vector | graph} (+ extracted params)
  schema.py    (new)  GraphFact / graph result models (pydantic)

api/routes/ask.py        (modified)  route → vector (C2) and/or graph (C3) → generate-or-degrade → answer + citations
api/routes/contracts.py  (modified)  after index_contract: best-effort graph build_contract(...)
api/main.py              (modified)  lifespan inits app.state.graph_store = None
eval/graphrag_eval.py    (new)       multi-hop accuracy; graph-vs-vector comparison → MLflow

infra/  neo4j service added to docker-compose; NEO4J_* settings
```

## Data flow

```
(build, at extract time)
  POST /contracts/extract → ingest → C1 extract → persist → [best-effort] index_contract (C2)
       → [best-effort] build_contract: normalize() → GraphStore upsert (Neo4j)
  → returns ContractDocument  (response shape unchanged)

(query)
  POST /ask {question, contract_id?, top_k?}
       → router (rule-based) → {graph | vector}
       → graph:  run Cypher template (Neo4j) → cited facts
                 → if LLM reachable: generate prose; else degrade (return result set)
       → vector: C2 path unchanged
       → AskResponse{ answer|null, generation_skipped, citations[] }
       → metrics (graph query latency, router decision mix) → Prometheus
```

## Configuration (additions to `Settings`, `DOCINTEL_` prefix — no hardcoded constants)

| Setting | Default |
|---|---|
| `neo4j_uri` | `bolt://neo4j:7687` |
| `neo4j_user` | `neo4j` |
| `neo4j_password` | `neo4j` (dev; overridden in env) |
| `neo4j_database` | `neo4j` |
| `graph_enabled` | `true` |

## Testing (TDD)

- **Unit (CPU, no Docker, no LLM):** via the **in-memory `GraphStore` fake** —
  `build()` maps a fixture `ContractDocument` → expected node/edge upserts; `normalize()`
  (ISO dates, unparseable flagged); each template returns expected rows on a seeded fake;
  `router` classification + param extraction on labeled examples; `query` assembles cited facts.
- **Integration:** `/ask` via FastAPI `dependency_overrides` with the fake store — a multi-hop
  question returns the correct contracts with citations; degraded path returns the graph result
  set with `generation_skipped=true`; extract-still-succeeds-when-graph-build-fails (best-effort).
- **Deselected (needs Docker):** one real-Neo4j test confirms the Cypher templates return the
  same rows as the fake (parity), guarding against fake drift.
- **Eval (not in CI):** `graphrag_eval.py` → multi-hop accuracy + graph-vs-vector → MLflow.
- Format/lint/type-check: ruff + mypy (strict) + pytest, per repo standard. mypy
  `ignore_missing_imports` for `neo4j.*`.

## Metrics delivered (for the CV)

| Metric | Where |
|---|---|
| **Multi-hop answer accuracy** | `graphrag_eval.py` → MLflow |
| **Graph-augmented vs. vector-only** retrieval lift | `graphrag_eval.py` → MLflow |
| Graph query **latency**, router decision mix | Prometheus → Grafana |

## Out of scope (later)

- Agent orchestration — **C4**.
- LLM-extracted entity graph / community summaries (the rejected GraphRAG flavor).
- Text-to-Cypher; hybrid (graph+vector) routing; `Party` / `GoverningLaw` nodes and their
  templates/normalization — added when a future template needs them.

## Risks & mitigations

- **Fake drift from real Cypher:** the in-memory fake re-implements the two templates in Python;
  a deselected real-Neo4j parity test guards it. Two simple templates keep this tractable.
- **Date parsing** from free-text expiration clauses: rule-based + flag unparseable; never block.
- **Neo4j down at extract:** best-effort build (Decision 4); extraction always returns 200.
- **C1 model unfinished:** C3 built/tested on fixtures; independent of the registered extractor.
