# Phase 6 — RAG over Extracted Data + `/ask`

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started**.

**Goal:** Answer natural-language questions grounded in extracted documents.

## Research 🔬
- [ ] Embedding model (`bge-small-en` vs `e5-small`) on CPU: quality vs latency.
- [ ] Chunking strategy for structured docs (per-field / per-doc / hybrid).
- [ ] RAGAS metrics + curated eval question set.
- [ ] Answer LLM: on-demand Qwen (Colab/serverless) vs compact CPU model; per-query routing.

## Tasks
- [ ] Embed extracted JSON → Qdrant (with doc/field metadata).
- [ ] Retriever + prompt assembly; `POST /ask` → grounded answer + citations.
- [ ] RAGAS evaluation (faithfulness, answer relevancy).

## Done when 📦
- [ ] `/ask` answers questions over indexed documents with citations; RAGAS scores recorded.

## Report
On completion, add `report_phase6.md` to this folder.
