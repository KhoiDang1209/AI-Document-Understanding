# C1 Completion Report — Contract Ingestion & Clause Extraction

**Date:** 2026-06-25
**Branch:** `worktree-contract-intelligence-c1` → merged to `master`
**Spec:** [`docs/superpowers/specs/2026-06-25-contract-intelligence-c1-ingestion-extraction-design.md`](../../superpowers/specs/2026-06-25-contract-intelligence-c1-ingestion-extraction-design.md)
**Plan:** [`docs/superpowers/plans/2026-06-25-contract-intelligence-c1.md`](../../superpowers/plans/2026-06-25-contract-intelligence-c1.md)
**Status:** ✅ Code complete & verified. ⏳ Model bundle + headline metrics pending one Colab GPU run (see below).

## What C1 delivers

The first phase of the **Contract Intelligence Platform** — a contract-extraction pipeline that runs *alongside* the existing receipt pipeline (the receipt `/extract` path is untouched). End-to-end shape:

```
Contract PDF
 → ingest (dual-path: PyMuPDF embedded text │ docTR OCR)
 → extract (fine-tuned extractive-QA, ONNX-INT8, sliding window over 41 CUAD clause questions)
 → aggregate spans → ContractDocument (clauses + derived view)
 → persist (SQLite contracts table + MinIO PDF)
 → POST /contracts/extract  ·  GET /contracts/{id}
```

Every stage reuses the established DocIntel patterns (MLflow registry / local-path escape hatch, raw `onnxruntime.InferenceSession`, lazy `app.state` backends, per-registry Prometheus metrics, stdlib SQLite).

## What was built (13 tasks + 1 fix, all reviewed)

| Area | Modules |
|---|---|
| Config | `contract_*` settings in `config.py` |
| Schema | `contracts/schema.py` — `ContractDocument`, `ExtractedClause`, `build_derived` |
| Questions | `contracts/questions.py` — 41 CUAD categories + question template |
| Ingestion | `contracts/ingest.py` — dual-path (PyMuPDF / docTR), `contracts` extra (pymupdf) |
| Decode | `contracts/aggregate.py` — window logits → char-offset spans |
| Serving | `contracts/extractor.py` — `CuadQaOnnxExtractor` (input-name-driven ONNX feed, sliding window) |
| Persistence | `storage/contracts_db.py` — `contracts` table (save/get/upsert) |
| Metrics | `api/metrics.py` — `contract_clause_confidence`, `contract_clauses_total` + `record_contract_extraction` |
| API | `api/routes/contracts.py` + `main.py` wiring (415/413/404) |
| Build-time | `contracts/qa_config.py`, `contracts/train_qa.py`; `optimize/export.py::export_qa_to_onnx` |
| Eval | `contracts/eval.py` (F1/ANLS/AUPR), `contracts/ocr_cer.py` (CER); `scikit-learn` in `train` extra |
| Notebooks/docs | `notebooks/cuad_finetune.ipynb`, `notebooks/cuad_onnx_export.ipynb`, README subsection |

## Verification

- **Tests:** 134 passed, 1 deselected (slow). New: schema, questions, ingest (both paths), aggregation, extractor decode, persistence, metrics, routes, train helpers, export seam, eval metrics.
- **Gates:** `ruff check` clean · `ruff format --check` clean · `mypy src` clean (55 files, strict).
- **Process:** subagent-driven — fresh implementer + task reviewer per task, plus a final opus whole-branch review. One Important finding (ONNX `token_type_ids` feed) fixed and re-reviewed.

## ⏳ Not yet done — requires a Colab GPU run (first thing for hands-on metrics)

The **serving + eval code is complete**, but the model artifact does not exist yet. To produce the headline CV metrics:

1. Run `notebooks/cuad_finetune.ipynb` on Colab → fine-tune on CUAD → register `cuad-extractor` in MLflow.
2. Run `notebooks/cuad_onnx_export.ipynb` → export + INT8 quantize → register `cuad-extractor-onnx-int8`; compute **AUPR/F1/ANLS** (clean text) and **CER** (OCR sample); log to MLflow.
3. Pull the INT8 bundle locally (or set `DOCINTEL_CONTRACT_ONNX_LOCAL_PATH`) and smoke-test `POST /contracts/extract` with a real contract PDF.

Until step 1–2 run, `/contracts/extract` will error on model load (no registered bundle) — expected.

## Deferred Minor findings (tracked for later cleanup, none blocking)

- `test_contracts_extractor.py`: dead code in `_FakeEncoding` (`_offsets`/`num_windows`/`offsets()`).
- Routes: no 413 oversize test; `init_contracts_db` called per GET (mirrors existing `documents.py`).
- `aggregate.py`: `n_best` applied per-window then again across windows — possible mild recall loss on long contracts; consider widening the per-window candidate pool.
- `rasterize_pages`: no grayscale-pixmap guard (RGB default; real-world safe).
- Notebooks: finetune pip pins `@master` (manual swap on feature branches); README says `GET /contracts/{id}` while the path param is `{contract_id}` (no runtime impact).

## Next session — C2 (Vector RAG + `/ask`)

Spec already written: [`...-c2-vector-rag-design.md`](../../superpowers/specs/2026-06-25-contract-intelligence-c2-vector-rag-design.md). C2 consumes the `ContractDocument` output from C1:
- Chunk (clause + paragraph) → CPU ONNX embeddings → **Qdrant** (new Compose service).
- `POST /ask`: retrieve → assemble prompt → generate via the **Colab/ngrok LLM** (`DOCINTEL_LLM_BASE_URL`, optional/intermittent → degrade to cited extractive spans).
- **RAGAS** eval (judge = same Colab LLM) → MLflow.

Recommended C2 kickoff: `/writing-plans` on the C2 spec, then resume subagent-driven execution in a fresh `contract-intelligence-c2` worktree.
