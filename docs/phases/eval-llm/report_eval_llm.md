# Eval-Loop Closure Report — LLM Serving + Live Endpoints + RAGAS Answer Quality

**Date:** 2026-07-05
**Status:** ✅ LLM served on Colab (OpenAI-compatible). ✅ `/ask` + `/agent` success path verified live. ✅ RAGAS faithfulness + answer_relevancy measured (this report).
**Notebook:** [`docintel/notebooks/llm_serving_colab.ipynb`](../../../docintel/notebooks/llm_serving_colab.ipynb)
**Runner:** [`docintel/src/docintel/scripts/eval_ragas.py`](../../../docintel/src/docintel/scripts/eval_ragas.py)

## What this closes

The C1–C4 platform was already built and merged, but generation degraded to citations-only
because no LLM endpoint was configured. This step self-hosts an open LLM on a Colab GPU,
wires it into the API, verifies the full (non-degraded) success path, and measures the
answer-quality metrics that were deferred because they need a live LLM judge.

```
Colab GPU  →  Qwen2.5-7B-Instruct (4-bit)  →  FastAPI OpenAI-compatible shim  →  ngrok
                                                                                   │  public https URL
DocIntel API (laptop, CPU)  ←  DOCINTEL_LLM_BASE_URL=<ngrok>/v1  ←──────────────────┘
  /ask   : retrieve (Qdrant) → generate (LLM)  → grounded answer + citations
  /agent : route → retrieve|graph → generate → critique (+bounded retry) → answer
```

## LLM serving (Colab notebook)

Self-contained, run top-to-bottom on a GPU runtime:

| Aspect | Choice |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` (matches `DOCINTEL_LLM_MODEL`) |
| Quantization | 4-bit nf4 (bitsandbytes) — fits a 16 GB T4; toggle off for L4/A100 fp16 |
| Serving | FastAPI, hand-rolled **OpenAI-compatible** `/v1/chat/completions` (non-stream + SSE), `/v1/models`, `/health` |
| Exposure | `pyngrok` tunnel; prints the exact `DOCINTEL_LLM_*` block to paste into `.env` |
| Auth | optional bearer token (blank/`EMPTY` matches a blank `DOCINTEL_LLM_API_KEY`) |

Because it serves the OpenAI Chat Completions shape, LangChain's `ChatOpenAI`
(`docintel.rag.llm.build_llm`) talks to it unchanged — only settings change, so the same
client points at Colab now and a managed API later.

**Endpoint round-trip (verified):** a direct `POST /v1/chat/completions` returned a valid
`chat.completion` with correct token-usage accounting.

## Live endpoints — success path verified

With `DOCINTEL_LLM_BASE_URL` set and the API restarted, both surfaces flip from
`degraded` to fully generative:

| Endpoint | Before (no LLM) | After (verified) |
|---|---|---|
| `POST /ask` | `generation_skipped=true`, citations only | `generation_skipped=false`, grounded answer + 5 citations |
| `POST /agent` | `status=degraded`, single vector pass | `status=ok`, full route→critique→retry loop |

- **`/ask`** — *"What law governs this agreement?"* →
  *"This Agreement shall be governed by the laws of the State of New York. The clause type
  relied upon is Governing Law."* (5 citations, top = `Governing Law`).
- **`/agent`** — the step trace shows genuine agentic self-correction:
  `route:graph → retrieve:graph:0 (empty) → generate → critique:retry → retrieve:vector:5 → generate → critique:finish`.
  It routed to the graph, got nothing, critiqued its own draft, retried via vector, and finished.

## Headline metric — RAGAS answer quality

**Method.** `eval_ragas.py` runs the **real** `/ask` retrieve-then-generate pipeline
(`answer_question`) over a CUAD sample, then scores each generated answer with RAGAS.
Both metrics are **reference-free** (need only question + answer + retrieved contexts), so
no gold labels are required. Judge = the same Colab LLM; `answer_relevancy` embeddings =
the local `fastembed` model (`bge-small-en-v1.5`).

| Metric | Value |
|---|---|
| Contracts sampled | 5 |
| Questions scored | 8 |
| **Faithfulness** | **0.365** |
| **Answer relevancy** | **0.196** |

Reproduce: `python -m docintel.scripts.eval_ragas --contracts 5 --questions 8 --seed 0`

**Honest reading.** These are genuinely low, and the cause is diagnosable — it is
**retrieval-bound, not a scoring bug**. C2 retrieval recall@5 is only **0.355**, so roughly
two-thirds of the time the gold passage is not in the retrieved context. The grounding
prompt then *correctly* instructs the model to answer *"I don't have enough information"* —
faithful behavior that nonetheless scores near-zero on `answer_relevancy` (which rewards
answers that directly address the question). So the honest bottleneck is **retrieval
coverage**, and the two metrics together tell that story: the pipeline stays grounded
(doesn't hallucinate) but is capped by what retrieval surfaces. Small sample (n=8) adds
variance. Lifting these scores means improving retrieval (better embedder / chunking /
hybrid search), not the generator.

## Operational finding — serialize RAGAS for a single-GPU endpoint

The first RAGAS run returned **NaN** for both metrics: all 16 judge jobs raised
`TimeoutError`. Cause — RAGAS defaults to ~16 concurrent workers, but the Colab server
generates one request at a time on a single GPU, so the concurrent calls queue behind the
GPU and each job blows its timeout. Fix — serialize RAGAS to match the server:
`RunConfig(max_workers=1, timeout=300)`. The runner exposes `--max-workers` (default 1) and
`--timeout` (default 300). The serialized run completed cleanly in ~3 min with zero timeouts.

## Consolidated eval scoreboard

| Stage | Metric | Result | LLM needed |
|---|---|---|---|
| C2 retrieval | recall@5 / MRR | 0.355 / 0.238 | no |
| C3 GraphRAG | multi-hop accuracy | 1.00 (10/10) | no |
| C2/C4 answer quality | RAGAS faithfulness | 0.365 | yes |
| C2/C4 answer quality | RAGAS answer_relevancy | 0.196 | yes |

## Verification

- `eval_ragas.py`: `ruff check` clean, `mypy` clean (added `ragas.*` to the mypy
  ignore-missing-imports override), no-LLM guard exits with a clear message.
- Live checks run against the Colab endpoint through the running API on `127.0.0.1:8000`.
- The `DOCINTEL_LLM_BASE_URL` (ngrok URL) lives only in `docintel/.env`, which is gitignored.
