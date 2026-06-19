# Phase 7 — Agent Orchestration (LangGraph)

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started**.

**Goal:** A multi-step agent chaining the capabilities as tools.

## Research 🔬
- [ ] LangGraph state/graph design: extract → validate → retrieve → answer.
- [ ] Tool boundaries, failure handling, retries.

## Tasks
- [ ] Tools wrapping pipeline + RAG.
- [ ] LangGraph graph with state + conditional routing.
- [ ] Langfuse tracing on every node.
- [ ] Expose via `POST /agent`.

## Done when 📦
- [ ] Agent handles compound tasks ("extract this doc, then answer X") in one call, fully traced.

## Report
On completion, add `report_phase7.md` to this folder.
