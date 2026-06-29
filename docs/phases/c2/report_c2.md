# C2 Completion Report — Vector RAG + `/ask`

**Date:** 2026-06-29
**Status:** ✅ Code complete & merged. ✅ Headline **retrieval** metrics measured (this report). ⏳ RAGAS answer-quality deferred (needs the intermittent Colab LLM judge).
**Spec:** [`docs/superpowers/specs/2026-06-25-contract-intelligence-c2-vector-rag-design.md`](../../superpowers/specs/2026-06-25-contract-intelligence-c2-vector-rag-design.md)
**Plan:** [`docs/superpowers/plans/2026-06-26-contract-intelligence-c2.md`](../../superpowers/plans/2026-06-26-contract-intelligence-c2.md)

## What C2 delivers

A vector-RAG layer over the C1 `ContractDocument` output:

```
ContractDocument
 → chunk (clause chunks + sliding paragraph windows)
 → embed (bge-small-en-v1.5 via fastembed, ONNX, CPU, no torch)
 → Qdrant (deterministic point ids; indexed best-effort at extract time)
 → POST /ask: retrieve top-k → assemble prompt → generate via Colab/ngrok LLM
   (optional/intermittent → degrade to cited extractive chunks) → answer + citations
```

Modules: `rag/{chunk,embed,store,index,llm,answer,schema,eval}.py`, `api/routes/ask.py`, Qdrant Compose service. All behind the established DocIntel patterns (lazy `app.state` backends, graceful degrade, per-registry Prometheus metrics).

## Headline metrics — retrieval quality

**Method.** A 40-contract sample of CUAD (`theatticusproject/cuad-qa`, seed 0) is chunked into **paragraph chunks only** (clause chunks excluded so the metric reflects semantic passage retrieval, not trivial gold-span lookup) and indexed into an in-memory Qdrant with the production embed/index/search path. For each of the 1,253 gold-answered CUAD clause questions, retrieval is filtered to its own contract; a question is a *hit@k* if a paragraph chunk covering the gold answer span is in the top-k.

| Metric | Value |
|---|---|
| Contracts indexed | 40 |
| Paragraph chunks | 3,004 |
| Gold-answered queries | 1,253 |
| **Recall@1** | **0.141** |
| **Recall@3** | **0.277** |
| **Recall@5** | **0.355** |
| **MRR** | **0.238** |

**Per-category spread (recall@5):** strong where the clause has distinctive vocabulary — Insurance 0.86, Source Code Escrow 0.82, Renewal Term 0.71, Uncapped Liability 0.67, Termination For Convenience 0.66, Governing Law 0.60; weak where the clause is diffuse or the generic query is a poor signal — License Grant 0.10, Exclusivity 0.15, and several at 0.00 (Non-Compete, No-Solicit, Rofr/Rofo/Rofn).

**Honest reading.** These are a **zero-shot baseline**: bge-small embeddings, no reranker, and — critically — the query is the *generic CUAD category template* ("Highlight the parts … related to 'X' …"), a deliberately weak retrieval signal against a single buried clause in a long contract. The numbers measure the floor of the unaided retriever; the per-category analysis shows it already works well for lexically distinctive clauses. Obvious levers (not pursued here): a cross-encoder reranker, hybrid BM25+dense, or category-name queries.

Reproduce: `python -m docintel.scripts.eval_rag --sample 40 --seed 0`

## Deferred — RAGAS answer quality (needs Colab LLM)

Faithfulness / answer-relevancy require a live LLM judge (the Colab/ngrok endpoint, `DOCINTEL_LLM_BASE_URL`), which is optional/intermittent and not configured at report time. The `/ask` generation path itself is implemented and degrades cleanly when the endpoint is down. RAGAS runs when the endpoint is up → MLflow.

## Verification

- New eval runner `src/docintel/scripts/eval_rag.py` reuses `rag.eval.recall_at_k` / `mrr` and the production `store.search` path; pure helpers unit-tested.
- Gates: `ruff check` / `ruff format --check` / `mypy src` clean.
