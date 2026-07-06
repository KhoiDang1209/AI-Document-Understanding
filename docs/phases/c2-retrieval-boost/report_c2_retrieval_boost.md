# C2 Retrieval Boost Report — Hybrid Search + Reranking + Focused Queries

**Date:** 2026-07-06
**Status:** ✅ Code complete. ✅ Retrieval recall re-measured. ✅ RAGAS re-measured against the live Colab LLM (this report).
**Baseline:** [`docs/phases/c2/report_c2.md`](../c2/report_c2.md) (recall@5 0.355) and
[`docs/phases/eval-llm/report_eval_llm.md`](../eval-llm/report_eval_llm.md) (RAGAS faithfulness 0.365 / answer_relevancy 0.196).

## Why

The eval-llm report diagnosed the low RAGAS scores as **retrieval-bound**: recall@5 was
0.355, so two-thirds of questions never saw the gold passage and the model (correctly)
refused — and RAGAS `answer_relevancy` hard-zeros refusals. This phase attacks retrieval
coverage directly and makes the eval report the refusal behaviour honestly.

## What changed

| Lever | Where | What |
|---|---|---|
| Hybrid dense+BM25 search | `rag/store.py` | Qdrant collection gains a named BM25 sparse vector (`Qdrant/bm25` via fastembed, IDF modifier); `QdrantVectorStore` runs in `RetrievalMode.HYBRID` (RRF fusion). Benefits `/ask` **and** the C4 agent's vector tool automatically. |
| Focused query rewriting | `rag/query.py` | `focus_query()` strips the CUAD template boilerplate to `"<category>: <details>"`; natural questions pass through unchanged. Applied in `/ask` and both evals. |
| Cross-encoder reranking | `rag/rerank.py` | `Xenova/ms-marco-MiniLM-L-6-v2` (fastembed ONNX, CPU) rescores a top-30 candidate pool down to top-k. Wired into `/ask` via a cached, graceful-degrade dependency; disabled by blanking `DOCINTEL_RAG_RERANK_MODEL`. |
| Eval realism (RAGAS) | `scripts/eval_ragas.py` | Indexes **clause chunks from the real C1 extractor** (production parity; graceful fallback to paragraphs-only), uses the production retrieval stack, defaults to n=40 questions, writes a per-sample CSV, and splits metrics into answered vs refused with a `refusal_rate`. |
| Eval ablations | `scripts/eval_rag.py` | `--no-rerank` / `--raw-query` flags isolate each lever's contribution. |

New settings (env-overridable): `rag_sparse_model` (`Qdrant/bm25`),
`rag_rerank_model` (`Xenova/ms-marco-MiniLM-L-6-v2`), `rag_rerank_candidates` (30).

## Headline metrics — retrieval quality (40 contracts, seed 0, 1,253 queries)

| Stack | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| C2 baseline (dense only, raw CUAD query) | 0.141 | 0.277 | 0.355 | 0.238 |
| + hybrid BM25 (raw query) | 0.166 | 0.340 | 0.429 | 0.287 |
| + focused query (no rerank) | 0.178 | 0.376 | 0.468 | 0.312 |
| **+ cross-encoder rerank (full stack)** | **0.206** | **0.399** | **0.494** | **0.345** |

Every lever contributes, in the expected order: hybrid BM25 is the biggest single lift
(+0.074 recall@5), focused queries add +0.039, and the reranker adds +0.026 more (and the
largest recall@1/MRR gains, as a reranker should).

**Recall@5: 0.355 → 0.494 (+39% relative). MRR: 0.238 → 0.345 (+45%).**

Per-category movement (recall@5): formerly-zero categories now retrieve — No-Solicit Of
Employees 0.00 → 0.86, Covenant Not To Sue → 0.61, Non-Disparagement → 0.67; lexically
distinctive categories stay strong (Source Code Escrow 1.00, Renewal Term 0.97, Governing
Law 0.93, Insurance 0.84). Still at zero: Non-Compete, Price Restrictions,
Unlimited/All-You-Can-Eat-License — genuinely diffuse clauses that likely need
category-specific query expansion or a stronger (GPU) reranker.

Reproduce: `python -m docintel.scripts.eval_rag --sample 40 --seed 0`
(ablations: `--no-rerank`, `--raw-query`).

## RAGAS answer quality — re-run with the new stack

5 contracts, **40 questions** (vs 8 before), seed 0, clause chunks from the real C1
extractor indexed alongside paragraphs, judge = the live Colab Qwen2.5-7B endpoint,
serialized (`max_workers=1`). Per-sample scores in the run's `eval_ragas_samples.csv`.

| Metric | Before (n=8) | After (n=40) |
|---|---|---|
| Faithfulness | 0.365 | **0.661** (+81%) |
| Answer relevancy (all) | 0.196 | **0.487** (+148%) |
| Answer relevancy (answered-only, n=25) | — | **0.713** |
| Faithfulness (answered-only) | — | 0.662 |
| Refusal rate | — | 0.375 |

The answered/refused split is the honest framing: a grounded refusal on a retrieval miss
is correct behaviour that `answer_relevancy` scores as 0; blending it into one number
hides whether generation or retrieval is at fault. Read together: when retrieval
surfaces the right context (62.5% of questions), answers are relevant (0.71) and
reasonably faithful; the remaining 37.5% are refusals that track the retrieval misses —
consistent with recall@5 ≈ 0.49 plus the clause-chunk index covering additional
questions. The next lift still comes from retrieval coverage, not the generator.

Reproduce: `python -m docintel.scripts.eval_ragas --contracts 5 --questions 40 --seed 0
--samples-csv samples.csv` (needs `DOCINTEL_LLM_BASE_URL`).

## Operational notes

- **Existing Qdrant collections must be re-indexed** (the hybrid collection adds a sparse
  vector; `ensure_collection` only creates, never migrates). Delete the collection (or the
  compose volume) and re-run `/extract` to re-index.
- First `/ask` after deploy downloads two small ONNX models to the fastembed cache
  (BM25 weights, ~1 MB; MiniLM reranker, ~90 MB). Reranker load failure degrades
  gracefully to hybrid-only search.
- CPU cost: reranking adds ~2–3 s per query on the laptop (30 candidates × 1,200-char
  chunks). Fine for interactive `/ask`; it is the dominant cost in bulk evals
  (the 1,253-query eval took ~65 min).
- **C1 clause extraction dominates `eval_ragas` wall-time on CPU** (~2.5 h for 5 fp32
  DeBERTa contracts vs ~30 min for generation + serialized judging). For faster
  iterations, cache extractions or run the eval on a machine with the INT8/GPU bundle.
- The C4 agent's vector tool gets hybrid search automatically but not reranking
  (would need `AgentDeps` plumbing) — noted as follow-up.
- Pre-existing on master, not touched: mypy error in `contracts/extractor.py`
  (transformers stub drift); `ruff format` drift in 5 UI/demo files.

## Verification

- TDD throughout: new tests `test_rag_query.py`, `test_rag_rerank.py`; extended
  `test_rag_store.py` (hybrid keyword-match-first, fake sparse embedder — no downloads),
  `test_rag_answer.py` (retrieval_query override, rerank reorder/truncate),
  `test_rag_routes.py` (reranker dependency caching + graceful degrade),
  `test_eval_scripts.py` (refusal detection).
- Gates: `ruff check` clean; `ruff format --check` clean on all touched files;
  `mypy src` clean except the pre-existing extractor error; **full pytest suite green**.
- Live smoke: `eval_rag --sample 2` end-to-end through the production store/rerank path.
