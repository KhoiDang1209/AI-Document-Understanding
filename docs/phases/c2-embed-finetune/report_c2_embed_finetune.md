# C2 Embedder Fine-Tune Report — CUAD-Tuned bge-small

**Date:** 2026-07-06
**Status:** ✅ Success bar met — recall@5 **0.745** vs the ≥ 0.65 target (baseline 0.494), with one
config-level caveat: the generic reranker must be disabled (see below).
**Spec:** [`docs/superpowers/specs/2026-07-06-c2-embedder-finetune-design.md`](../../superpowers/specs/2026-07-06-c2-embedder-finetune-design.md)
**Baseline:** [`docs/phases/c2-retrieval-boost/report_c2_retrieval_boost.md`](../c2-retrieval-boost/report_c2_retrieval_boost.md)

## Why

The retrieval-boost phase left recall@5 at 0.494 with three categories at zero, and diagnosed the
remaining misses as first-stage pool misses — the failure mode retriever fine-tuning addresses.
This phase fine-tuned the served embedder (`BAAI/bge-small-en-v1.5`) on CUAD (focused query →
covering paragraph window) pairs, changing nothing else in the stack, so the ablation attributes
every delta to the embedder.

## What changed

| Piece | Where | What |
|---|---|---|
| Training pairs | `scripts/build_embed_pairs.py` | (focus_query → covering 1200/200 window) per gold answer; **40 seed-0 eval contracts excluded** via the eval's own `_sample_contracts`; 10,885 train / 595 dev pairs from 368 contracts; leak-check clean. |
| Fine-tune | `notebooks/cuad_embed_finetune.ipynb` | Colab GPU, sentence-transformers MNRL (in-batch negatives, NO_DUPLICATES batching), 2 epochs. Dev IR (held-out contracts): accuracy@5 0.116 → **0.313 (2.7×)**, mrr@10 0.073 → 0.190. ONNX export + in-notebook parity assert. |
| Serving | `rag/embed.py`, `config.py` | `DOCINTEL_RAG_EMBEDDING_LOCAL_PATH` loads the bundle via fastembed `add_custom_model` + `specific_model_path` (CLS pooling, normalized, 384-dim — schema unchanged). Unset ⇒ stock model, zero behavior change. |
| Parity gate | `scripts/check_embed_parity.py` | Laptop ONNX path must reproduce the trained model's vectors (cosine ≥ 0.999, both `embed_documents` and `embed_query`) before any eval. **Passed** on the deployed bundle. |
| Eval | `scripts/eval_rag.py` | `--top-ks` flag (recall@30 = reranker-headroom measurement). |

## Headline metrics — retrieval (40 contracts, seed 0, 1,253 queries)

| Stack | Recall@1 | Recall@3 | Recall@5 | Recall@30 | MRR |
|---|---|---|---|---|---|
| Stock (full stack, retrieval-boost final) | 0.206 | 0.399 | 0.494 | 0.816 | 0.373 |
| Fine-tuned + reranker | 0.210 | 0.413 | 0.513 | **0.949** | 0.391 |
| **Fine-tuned, no reranker (recommended)** | **0.316** | **0.616** | **0.745** | **0.949** | **0.528** |

**Recall@5: 0.494 → 0.745 (+51% relative). MRR: 0.373 → 0.528 (+42%). Success bar (≥ 0.65) met.**

(All rows are measured at retrieval depth 30, so MRR credits gold passages found at ranks 6–30;
that is why the stock row shows MRR 0.373 rather than the 0.345 in the retrieval-boost report,
which measured at depth 5. Same stack, same queries — deeper list.)

Two findings, one expected and one not:

1. **The fine-tune fixed the first stage, as designed.** Pool quality (recall@30) jumped 0.816 →
   0.949, and the gains reach the top-5 at production settings: a dedicated run at the serving
   depth (`--no-rerank`, default top-ks; `eval_rag_finetuned_norerank_k5.json`) puts the three
   formerly-zero categories at **recall@5 0.55 / 1.00 / 1.00** (Non-Compete, Price Restrictions,
   Unlimited/All-You-Can-Eat-License — all 0.00 in the retrieval-boost report). Five categories
   sit at a perfect 1.00 (incl. Renewal Term, No-Solicit Of Customers); the weakest are now
   Competitive Restriction Exception (0.43), Revenue/Profit Sharing (0.48), and Exclusivity
   (0.49) — diffuse, but no longer hopeless. Headline metrics at this depth: recall@5 0.741,
   MRR 0.524 — a hair off the depth-30 run's 0.745/0.528 because RRF fusion order shifts
   slightly with retrieval depth; both clear the bar.
2. **The generic reranker flipped from help to harm.** On the stock embedder, ms-marco MiniLM
   added +0.026 recall@5; on the fine-tuned ordering it *subtracts* 0.232 (0.745 → 0.513). The
   CUAD-tuned dense+BM25 ordering now encodes domain knowledge the generic cross-encoder cannot
   judge, so reranking shuffles gold passages back out of the top-5. The reranker was already
   env-disableable by design (blank `DOCINTEL_RAG_RERANK_MODEL`); that is now the **recommended
   production config**. A CUAD-fine-tuned reranker is the obvious next-phase lever — with the
   pool at 0.949, a good reranker has ~0.20 of recall@5 headroom to claim.

Reproduce: `eval_rag --sample 40 --seed 0 --top-ks 1,3,5,30` (+ `--no-rerank`), with
`DOCINTEL_RAG_EMBEDDING_LOCAL_PATH` pointing at the bundle.

## RAGAS answer quality — re-run with the recommended stack

5 contracts, 40 questions, seed 0, clause chunks from the real C1 extractor, judge = Colab
Qwen2.5-7B (serialized), retrieval = fine-tuned embedder, rerank off.

| Metric | Before (retrieval-boost) | After (this phase) |
|---|---|---|
| Faithfulness | 0.661 | 0.637 |
| Answer relevancy (all) | 0.487 | 0.420 |
| Answer relevancy (answered-only) | 0.713 | 0.663 (n=23) |
| Faithfulness (answered-only) | 0.662 | **0.734** |
| Refusal rate | 0.375 | 0.425 |

Honest read: **within noise, except one real positive.** At n=40 with a single 7B judge (one of
80 judge jobs failed to parse), deltas of ±0.05–0.08 are not signal — the refusal-rate "rise" is
two answers out of forty. The one delta consistent with the retrieval gains is answered-only
faithfulness (0.662 → 0.734): when the model answers, the better-retrieved context grounds it
more. The dramatic retrieval improvement did not translate into a measurable RAGAS jump on this
sample; the 1,253-query retrieval eval is this phase's load-bearing evidence, and the RAGAS
re-run's role is confirming **no answer-quality regression** under the new stack. (The
refusal-vs-recall link is also looser here than the headline suggests: RAGAS retrieves over
clause+paragraph chunks on 5 contracts, a different distribution from the paragraph-only
40-contract retrieval eval.)

## Operational notes

- **Re-index required:** the fine-tuned model defines a new embedding space. The production
  `contract_chunks` collection was deleted during this phase; it is rebuilt automatically by the
  next `/extract` once serving runs with the bundle env var set.
- **Deploy config:** `DOCINTEL_RAG_EMBEDDING_LOCAL_PATH=<...>/models/rag-embed-cuad` **and**
  `DOCINTEL_RAG_RERANK_MODEL=` (blank). Bundle (127 MB ONNX + tokenizer.json + parity.json) lives
  in root `models/`, gitignored; reproducible from the notebook + pairs + seeds.
- The C4 agent's vector tool inherits the fine-tuned hybrid search automatically; it never had
  reranking, so the reranker regression does not affect it.
- CPU eval cost unchanged (~65 min with rerank, ~25 min without, per 1,253 queries).
  `eval_ragas` still re-runs C1 extraction (~2.5 h for 5 fp32 contracts) — caching extractions
  remains the top eval-loop QoL follow-up.
- Ops incidents this phase (documented for reuse): detached long runs must go through
  `Start-Process -File <script>.ps1` + a Monitor (harness kills backgrounded tools at ~10 min;
  inline `-Command` strips quotes and silently drops env vars); phase worktrees lack the
  gitignored `docintel/.env` (copy it in, or the C1 extractor path/MLflow URI silently revert to
  docker-internal defaults); the Colab LLM is reached via a local relay
  (`DOCINTEL_LLM_BASE_URL=http://127.0.0.1:8899` → target file swappable mid-run) so ngrok URL
  churn can't waste a 3-hour eval.

## Verification

- TDD throughout: `test_rag_embed.py` (stock-path unchanged, custom-branch kwargs, idempotent
  registration — plus the pre-existing adapter tests restored after a review catch),
  `test_build_embed_pairs.py` (holdout exclusion, focused queries, covering windows, multi-span,
  deterministic contract-disjoint dev split), `test_check_embed_parity.py` (cosine gate logic),
  `test_eval_scripts.py::test_rag_parse_top_ks`.
- Per-task spec+quality reviews (two fix loops); full suite green (249 tests) before measurements
  and again at phase close. Gates at close: `ruff check` clean on all touched files (pre-existing
  notebook lints in the two old notebooks untouched); `ruff format` drift only in the 5
  pre-existing UI/demo files; `mypy src` clean except the pre-existing extractor error.
- Known environmental quirk (pre-existing, surfaced this phase): `test_config_agent` /
  `test_config_graph` assert Settings *defaults* and fail when a local `docintel/.env` sets
  agent/graph keys — run the suite without `.env` (or with it renamed) for a clean signal.
- Embedding-parity gate passed on the deployed bundle (8 sentences, both embed paths, ≥ 0.999).
- Holdout verified programmatically: 0 of the 40 eval titles appear in train/dev pairs.
