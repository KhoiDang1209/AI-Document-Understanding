# Contract Intelligence — C3 Design: GraphRAG (Neo4j)

**Date:** 2026-06-25
**Status:** Approved (design); ready for implementation planning. **Depends on C1, C2.**
**Initiative:** Contract Intelligence Platform. **Roadmap:** C1 → C2 → **C3 (this doc)** → C4.

## Goal

Answer **cross-contract** and **multi-hop** questions that flat vector RAG handles poorly —
e.g. "which auto-renewing contracts expire within 90 days?", "which contracts share a
governing law with contract X?". Build a **knowledge graph from the C1 extractions** (no LLM
re-reading the corpus), traverse it, and feed the selected, cited facts to the LLM. `/ask`
gains routing between vector retrieval (C2) and graph retrieval (C3).

## Context

- **C1** gives structured `ExtractedClause`s + a derived view (parties, effective/expiration/
  renewal dates, governing law). **C2** gives vector retrieval, the LLM client, and `/ask`.
- Reuse the spine: MLflow (GraphRAG eval), Prometheus/Grafana/Loki.
- **New infra:** Neo4j added to Compose (persisted volume).

## Decisions (locked during brainstorming)

1. **Graph built deterministically from extractions** (the chosen "structured KG" flavor — not
   Microsoft-style LLM graph extraction). Nodes: `Contract`, `Party`, `ClauseType`, `Date`,
   `GoverningLaw`. Edges: `HAS_CLAUSE`, `PARTY_TO`, `GOVERNED_BY`, `EFFECTIVE_ON`,
   `EXPIRES_ON`, `RENEWS_ON`. Each clause node keeps its citation `{contract_id, char span}`.
2. **Light normalization only** (needed for traversal to work): parse dates from the relevant
   clause types to ISO; normalize party / governing-law strings (casing, whitespace, simple
   alias merge). Kept minimal and rule-based; documented.
3. **Querying = parameterized Cypher templates** for the common multi-hop patterns
   (deterministic, no Cypher hallucination), with **LLM used only for intent + entity parsing**
   (map the NL question → a template + parameters). **Text-to-Cypher is a stretch goal**, gated
   behind the templates so metrics stay stable.
4. **Hybrid answer assembly:** graph traversal selects the candidate contracts/clauses → those
   cited facts are passed to the LLM (C2's `generate`) for the prose answer. Same
   citation contract as C2. Same **degraded** behavior when the LLM is down (return the
   graph result set directly).
5. **`/ask` gains a router:** classify the question as semantic (→ vector, C2) or
   structured/multi-hop (→ graph, C3); hybrid questions can use both. Router is simple and
   explainable (rules + optional LLM intent classification). One endpoint, not a new route.
6. **Eval:** a constructed **multi-hop question set** over the corpus; report multi-hop
   answer accuracy and a **graph-augmented vs. vector-only** retrieval comparison → MLflow.

## Architecture / new modules

```
src/docintel/graph/
  build.py     (new)  ContractDocument records → Neo4j nodes/edges (deterministic)
  normalize.py (new)  date parsing + party/governing-law normalization (rule-based)
  templates.py (new)  parameterized Cypher templates for multi-hop patterns
  query.py     (new)  run template (+ optional text-to-Cypher stretch); return cited facts
  router.py    (new)  question → {vector | graph | hybrid}

api/routes/ask.py (extended)  route → vector (C2) and/or graph (C3) → generate → answer + citations
eval/graphrag_eval.py (new)   multi-hop accuracy; graph-vs-vector comparison → MLflow

infra/  neo4j service added to docker-compose; NEO4J_URI/auth config
```

## Data flow

```
(graph build, offline)  ContractDocument(s) → normalize() → build() [Neo4j]
(query)  question → router → {Qdrant retrieve | Cypher template (Neo4j) | both}
         → cited facts → generate (Colab LLM) → answer + citations
         (degraded: return graph/vector result set directly)
```

## Testing (TDD)

- **Unit:** build() maps extractions → expected nodes/edges on a fixture; normalize() (dates,
  party aliases); each Cypher template returns expected rows on a seeded test graph; router
  classification on labeled examples.
- **Integration:** seed 2–3 fixture contracts → a multi-hop `/ask` returns the correct
  contracts with citations; degraded path returns the graph result set.
- **Eval (not in CI):** `graphrag_eval.py` → multi-hop accuracy + graph-vs-vector → MLflow.

## Metrics delivered (for the CV)

| Metric | Where |
|---|---|
| **Multi-hop answer accuracy** | `graphrag_eval.py` → MLflow |
| **Graph-augmented vs. vector-only** retrieval lift | `graphrag_eval.py` → MLflow |
| Graph query **latency**, router decision mix | Prometheus → Grafana |

## Out of scope (later)

- Agent orchestration — **C4**.
- LLM-extracted entity graph / community summaries (the rejected GraphRAG flavor).

## Risks & mitigations

- **Entity resolution** (party names vary across contracts): start with simple rule-based
  alias merge; measured via graph-eval, improve only if it limits accuracy.
- **Date parsing** from free-text clauses: rule-based + flag unparseable; never block.
- **Text-to-Cypher reliability:** kept a stretch goal behind deterministic templates.
