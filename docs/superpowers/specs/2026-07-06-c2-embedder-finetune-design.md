# Contract Intelligence — C2 Embedder Fine-Tune Design: CUAD-Tuned bge-small

**Date:** 2026-07-06
**Status:** Approved (design); ready for implementation planning. **Depends on C2 + c2-retrieval-boost.**
**Initiative:** Contract Intelligence Platform. **Baseline:**
[`docs/phases/c2-retrieval-boost/report_c2_retrieval_boost.md`](../../phases/c2-retrieval-boost/report_c2_retrieval_boost.md)
(recall@5 **0.494**, MRR 0.345, RAGAS faithfulness 0.661 / answer_relevancy 0.487, refusal rate 0.375).

## Goal

Lift first-stage retrieval by fine-tuning the dense embedder (`BAAI/bge-small-en-v1.5`)
on CUAD question→passage pairs. **Success bar: recall@5 ≥ 0.65** on the standard eval
(`eval_rag --sample 40 --seed 0`), up from 0.494 (+0.15 absolute). Single lever — BM25,
RRF fusion, focused queries, and the cross-encoder reranker stay exactly as shipped in
c2-retrieval-boost, so the ablation attributes any gain to the embedder alone.

## Why this lever

The retrieval-boost report showed the remaining misses are **first-stage pool misses**:
categories like Non-Compete and Price Restrictions are semantically diffuse, the gold
clause never enters the top-30 candidate pool, and no reranker can recover it. That is
the failure mode retriever fine-tuning addresses. Refusals (37.5%) track retrieval
misses, so a recall lift flows directly into the RAGAS numbers.

## Decisions (locked during brainstorming)

1. **Fine-tune the bi-encoder, not the reranker.** The reranker is capped by
   first-stage recall@30; the bi-encoder *is* the first stage. (Reranker fine-tune and
   category query expansion remain documented follow-ups if this lands short.)
2. **Base = the served model** (`bge-small-en-v1.5`, 384-dim). Embedding dim, Qdrant
   schema, and CPU latency are unchanged; only the weights improve.
3. **Train on the production task, not an easier proxy:** queries are
   `focus_query(question)` (what `/ask` actually embeds); positives are the **1200/200
   paragraph windows covering the gold span**, built with the production `build_chunks`
   — the same text distribution the model retrieves over at serve time.
4. **Strict contract-level holdout:** the 40 seed-0 eval titles (via the eval's own
   `_sample_contracts(dataset, 40, 0)`) are excluded from training. The RAGAS eval's 5
   contracts are the first 5 of the same seed-0 shuffle, so one exclusion list covers
   both evals. ~470 remaining contracts form the training pool; 20 of them are held back
   as a dev set for during-training IR validation.
5. **Loss = MultipleNegativesRankingLoss** (in-batch negatives), sentence-transformers,
   one Colab GPU session. No hard-negative mining in v1 — it is the standard follow-up,
   not a prerequisite.
6. **Serve via fastembed's custom-model path** (`TextEmbedding.add_custom_model`,
   available in the installed fastembed 0.8.0): a new optional setting
   `rag_embedding_local_path` (env `DOCINTEL_RAG_EMBEDDING_LOCAL_PATH`, same pattern as
   `DOCINTEL_KIE_ONNX_LOCAL_PATH`) points at a local ONNX bundle in root `models/`
   (gitignored). Unset ⇒ stock model, zero behavior change.
7. **Colab = train, laptop CPU = serve** (project-wide constraint). Export to ONNX in
   the same Colab session; the bundle ships `tokenizer.json` for the `tokenizers`
   runtime (the C1 lesson: never depend on cross-version `tokenizer_config` loading).

## Architecture / new & touched modules

```
src/docintel/scripts/build_embed_pairs.py  (new)   CUAD → train/dev JSONL pairs; pure
                                                   pair-builder function + CLI; reuses
                                                   _sample_contracts + build_chunks
src/docintel/rag/embed.py                  (touch) build_embedder(): if
                                                   rag_embedding_local_path is set,
                                                   register the bundle via
                                                   add_custom_model (CLS pooling,
                                                   normalized, bge query prefix) and
                                                   load it; else stock path unchanged
src/docintel/config.py                     (touch) rag_embedding_local_path: str = ""
src/docintel/scripts/eval_rag.py           (touch) expose top_ks (recall@30) on the CLI
notebooks/cuad_embed_finetune.ipynb        (new)   Colab: load JSONL → MNRL fine-tune →
                                                   dev IR eval → ONNX export → zip bundle
models/rag-embed-cuad/                     (local) ONNX bundle: model.onnx,
                                                   tokenizer.json, config (gitignored)
```

`_sample_contracts` moves (or is imported) so `eval_rag` and `build_embed_pairs` share
one definition — the holdout must be computed by the same code the eval uses.

## Data flow

```
CUAD (theatticusproject/cuad-qa, train split)
 → build_embed_pairs: exclude 40 seed-0 eval titles
   → per answered question: (focus_query(q), covering 1200-char window(s))  [~10–15k pairs]
   → train.jsonl + dev.jsonl (20 dev contracts)
 → Colab notebook: bge-small + MNRL → dev InformationRetrievalEvaluator
   → ONNX export (optimum) + tokenizer.json → bundle zip → Drive → laptop models/
 → laptop: DOCINTEL_RAG_EMBEDDING_LOCAL_PATH=models/rag-embed-cuad
   → build_embedder loads custom model → hybrid store (unchanged) → /ask, evals
```

## Embedding-parity guard

Before trusting eval numbers, verify the ONNX/fastembed serving path reproduces the
trained model: embed a fixed sentence set in Colab with the fine-tuned
sentence-transformers model and on the laptop through `build_embedder`; cosine
similarity per sentence must be ≥ 0.999. This catches pooling (bge = CLS, not mean),
normalization, and query-prefix mismatches — the three silent killers of exported
bi-encoders. Training must embed queries exactly as the serving path does (whether
fastembed applies the bge query instruction in `query_embed` is verified at
implementation time, and the parity guard catches any mismatch).

## Eval protocol

1. **Pre-measure recall@30** of the current stack (one `eval_rag` run with `top_ks`
   extended) — establishes reranker headroom and lets the report attribute gains to
   "better pool" vs "better ranking".
2. **Headline table** (all on `--sample 40 --seed 0`): stock vs fine-tuned embedder ×
   {full stack, `--no-rerank`} → recall@1/3/5/30, MRR, per-category recall@5 (watch the
   three zero categories).
3. **RAGAS re-run** with the fine-tuned stack (5 contracts / 40 questions / seed 0,
   Colab judge, `max_workers=1`) — expected: refusal rate drops with recall. Runs last;
   C1 extraction dominates its wall-time (~3 h CPU), so it is the closing measurement,
   not an iteration loop.
4. **Success:** recall@5 ≥ 0.65 closes the phase. If short, the report states honestly
   what was reached and which follow-up (hard negatives, query expansion) is next.

## Testing (TDD)

- **Unit:** pair-builder — holdout titles never appear in output; queries are focused;
  positives are exactly the covering windows; multi-span questions yield multiple pairs;
  JSONL round-trips. `build_embedder` — local-path setting selects the custom-model
  branch (fake/registered model, no downloads); unset preserves stock behavior; query
  prefix applied in `embed_query` only.
- **Integration:** existing rag/store/answer/routes suites must stay green with the
  setting unset (default path untouched).
- **Live (not CI):** parity guard above, then the eval protocol.

## Operational notes

- **Re-index required after switching embedders:** same collection schema, but the dense
  vectors live in a new space — delete the Qdrant collection (or compose volume) and
  re-run `/extract`, exactly as the retrieval-boost deploy required.
- Bundle stays out of git (root `models/`, same as the KIE/C1 bundles); the notebook and
  pair JSONLs are reproducible from CUAD + seed.
- Colab pip gotcha applies if the notebook installs `docintel` from git: bump the
  package version or install from the wheel, since Colab silently reuses stale builds.

## Metrics delivered (for the CV)

| Metric | Where |
|---|---|
| recall@1/3/5/30 + MRR, stock vs fine-tuned × rerank ablation | `eval_rag` → phase report |
| Per-category recall@5 movement (incl. former zero categories) | `eval_rag` → phase report |
| RAGAS faithfulness / answer_relevancy / refusal rate, before→after | `eval_ragas` → phase report |
| Dev IR metrics during training | notebook / MLflow |

## Out of scope

- Hard-negative mining, reranker fine-tuning, category query expansion (follow-ups).
- Changing chunking, BM25, fusion, reranker, prompts, or the C4 agent path.
- Faithfulness/grounding prompt work (separate phase if pursued).

## Risks & mitigations

- **Fine-tune underperforms (< 0.65):** dev IR eval in-notebook catches a broken run
  before the expensive laptop eval; the phase still ships the training loop + honest
  report, with hard negatives as the scoped next step.
- **Export/serving drift:** the embedding-parity guard is a hard gate before any eval.
- **Overfitting to CUAD template phrasing:** queries are focused (template boilerplate
  already stripped), and the dev set is held-out contracts, not held-out questions.
- **Tokenizer/version drift Colab↔laptop:** bundle `tokenizer.json`; fastembed tokenizes
  via the `tokenizers` runtime, not `transformers` (the C1 incident cannot recur).
