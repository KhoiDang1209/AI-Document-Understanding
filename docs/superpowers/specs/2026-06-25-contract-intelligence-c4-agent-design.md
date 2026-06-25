# Contract Intelligence — C4 Design: LangGraph Agent + `/agent`

**Date:** 2026-06-25
**Status:** Approved (design); ready for implementation planning. **Depends on C1, C2, C3.**
**Initiative:** Contract Intelligence Platform. **Roadmap:** C1 → C2 → C3 → **C4 (this doc)**.

## Goal

A multi-step agent that chains the platform's capabilities as **tools** to handle compound
tasks in one call — e.g. "extract this contract, then tell me when it auto-renews and whether
its governing law matches our other vendor contracts." `POST /agent` runs a LangGraph state
machine (extract → retrieve → graph-query → answer, with validation + retries), fully traced
in self-hosted **Langfuse**.

## Context

- **C1/C2/C3** expose callable functions: contract extraction, vector retrieval, graph query,
  grounded generation — all with citations. C4 wraps them as agent tools; it adds no new model.
- Reuse the spine: Prometheus/Grafana/Loki. **New infra:** self-hosted **Langfuse** (+ its
  Postgres) in Compose — **no API cost**.
- LLM = Colab/ngrok (the agent's reasoning + tool-calling model); **optional/intermittent**.

## Decisions (locked during brainstorming)

1. **Agent = LangGraph state machine**, not a free-form ReAct loop — explicit nodes and
   conditional edges so behavior is inspectable and the per-node traces are clean.
   Nodes: `ingest_extract` (if a new doc is attached) → `route` → `retrieve` (vector) /
   `graph_query` → `generate` → `critique/validate` → (retry or finish).
2. **Tools wrap C1–C3 functions** (extract, vector_retrieve, graph_query, generate) — thin
   adapters, no logic duplication. The agent composes existing capabilities.
3. **Tool-calling-capable open model** (e.g. Qwen2.5-Instruct) served on Colab/ngrok — chosen
   in the plan for reliable function-calling. **Graceful degradation:** if the LLM endpoint is
   down, `/agent` returns the deterministic tool outputs gathered so far + a `degraded` status;
   it never hard-fails on GPU absence.
4. **Bounded retries + explicit failure handling:** each node has a max-retry and a typed
   failure path; the graph cannot loop unbounded. Failures are traced, not swallowed.
5. **Langfuse tracing on every node** (self-hosted) — inputs/outputs/latency/tokens per step;
   the trace id is returned in the response for inspection.
6. **`/agent` is a new route**; `/ask` (C2/C3) stays the single-shot grounded-QA path. The
   agent is for compound, multi-step requests.
7. **Eval:** a small **compound-task set**; report task success rate (did it produce a correct,
   cited answer end-to-end) with traces as evidence → MLflow + Langfuse.

## Architecture / new modules

```
src/docintel/agent/
  tools.py  (new)  thin tool adapters over C1 extract, C2 retrieve/generate, C3 graph_query
  graph.py  (new)  LangGraph state graph: nodes, conditional routing, retries, failure paths
  trace.py  (new)  Langfuse callback wiring (self-hosted endpoint from config)

api/routes/agent.py (new)  POST /agent → run graph → answer + citations + trace_id (+ degraded)
eval/agent_eval.py  (new)  compound-task success rate → MLflow

infra/  langfuse (+ postgres) services added to docker-compose; LANGFUSE_* config
```

## Data flow

```
POST /agent (task [+ optional contract])
 → LangGraph: ingest_extract? → route → retrieve / graph_query → generate → critique
   (bounded retries; every node traced to Langfuse)
 → answer + citations + trace_id     (degraded: deterministic tool outputs + status)
 → metrics (steps, retries, success, latency) → Prometheus
```

## Testing (TDD)

- **Unit:** each tool adapter calls the underlying C1–C3 function correctly (mocked); the graph
  takes the expected path for representative inputs; retry/failure caps hold; degrade-on-LLM-down.
- **Integration:** a compound task over fixture contracts runs end-to-end → correct cited
  answer + a non-empty Langfuse trace; degraded path returns tool outputs + status.
- **Eval (not in CI):** `agent_eval.py` → compound-task success rate → MLflow.

## Metrics delivered (for the CV)

| Metric | Where |
|---|---|
| **Compound-task success rate** | `agent_eval.py` → MLflow |
| Per-node **traces** (inputs/outputs/latency/tokens) | Langfuse |
| Agent **steps / retries / latency**, degraded-rate | Prometheus → Grafana |

## Out of scope

- New models or capabilities — C4 only orchestrates C1–C3.
- Multi-user sessions / long-term agent memory.

## Risks & mitigations

- **Open-model tool-calling reliability:** pick a function-calling-capable model (Qwen2.5);
  keep the graph explicit (not free-form) so a weak step can't derail the whole run.
- **LLM intermittent:** designed-for via the degraded path (Decision 3).
- **Langfuse self-host complexity:** standard Docker Compose deployment; isolated from the
  serving path (tracing failure must not break `/agent`).
```
