# Contract Intelligence — C1 Design: Ingestion & Clause Extraction

**Date:** 2026-06-25
**Status:** Approved (design); ready for implementation planning.
**Initiative:** DocIntel evolves from a receipt extractor into a **Contract Intelligence Platform**.
**Roadmap:** C1 (this doc) → C2 Vector RAG → C3 GraphRAG → C4 LangGraph Agent.

## Goal

Turn a contract PDF (born-digital **or** scanned) into a **structured contract record** —
the 41 CUAD clause types extracted as cited text spans with per-field confidence —
served on CPU from an MLflow-registered ONNX-INT8 model, and persisted so every contract
is retrievable by id. C1 is the **foundation**: C2–C4 all consume its structured output.

This phase delivers the project's two headline metric families:
- **Extraction:** AUPR / F1 / ANLS on CUAD (comparable to the published baseline).
- **OCR:** CER on a scanned-contract sample.

## Context (what already exists, and what's new)

**Existing DocIntel spine (reused unchanged):**
- Fine-tune → ONNX export → INT8 → **MLflow** registry → `onnxruntime` CPU serving — the
  exact pattern proven for `cord-layoutlmv3-onnx-int8`. The contract extractor follows it.
- Phase 3 lesson: run ONNX via **raw `onnxruntime.InferenceSession`**, not the Optimum
  wrapper (it silently drops inputs).
- OCR engine (**docTR**) is lazy-loaded once onto `app.state` (`api/routes/extract.py`).
- Persistence: **SQLite** (metadata) + **MinIO** (source bytes). **Prometheus/Grafana/Loki**
  observability via `prometheus-fastapi-instrumentator` + a per-app custom registry.
- `KIEBackend`-style `Protocol` interface for pluggable model backends.

**The receipt pipeline (`/extract`, LayoutLMv3) stays as-is** — a proven component. C1 adds
a **parallel contract path**; it does not modify or break `/extract`.

**The dataset — CUAD (Contract Understanding Atticus Dataset):**
- 510 commercial contracts; 13,000+ expert annotations across **41 clause types**.
- Native format: **span-selection QA** (SQuAD 2.0-style — one question per clause type,
  answers may be empty). Ships as both PDFs and clean text. CC BY 4.0.
- Official baseline: a fine-tuned extractive-QA transformer (RoBERTa) reporting **AUPR**.

## Decisions (locked during brainstorming)

1. **Domain & dataset:** English contract intelligence on **CUAD**. Chosen because one
   dataset yields both extraction metrics *and* downstream RAG QA ground truth, and the
   extracted output feeds both the vector index (C2) and the knowledge graph (C3) — maximum
   cohesion ("one pipeline, two goals: OCR + RAG").

2. **Extractor = fine-tuned extractive-QA model**, not an LLM. A RoBERTa/DeBERTa-style
   encoder fine-tuned on CUAD's 41 clause types as SQuAD2 span selection (no-answer allowed).
   Keeps the "I fine-tune **and** optimize models" narrative and reuses the existing
   fine-tune→ONNX-INT8→serve spine. Base model picked in the plan by Colab budget
   (DeBERTa-v3-base preferred; RoBERTa-base fallback).

3. **Long-document handling:** contracts far exceed 512 tokens. Use the standard CUAD
   **sliding-window** (stride) over the document **per clause-question**; aggregate the
   highest-scoring spans across windows. This is the established CUAD inference pattern.

4. **Dual-path ingestion (auto-detected):**
   - **Born-digital** PDF (has a text layer) → extract text + char offsets via **PyMuPDF**.
   - **Scanned** PDF / image → render pages → **docTR** OCR (reuse existing engine).
   - Detection: if the embedded text layer is non-trivial, use it; else OCR.

5. **Metric fidelity (resolves the OCR-vs-clean-span tension):**
   - **Extraction metrics (AUPR/F1/ANLS)** are computed on the **clean CUAD text** so they
     are directly comparable to published baselines.
   - **CER** is reported **separately** on a sampled subset of contracts rendered to images
     and OCR'd (OCR output vs. clean text). This keeps both numbers honest and the OCR goal
     genuinely exercised, without OCR noise polluting the extraction benchmark.

6. **Serving = new contract path, no breaking change.** New route **`POST /contracts/extract`**
   returns a `ContractDocument`; **`GET /contracts/{id}`** retrieves it. The extractor sits
   behind a `ContractExtractor` `Protocol` (mirrors `KIEBackend`), lazy-loaded onto
   `app.state`. `/extract` (receipts) is untouched.

7. **Structured record schema (`ContractDocument`):** per contract — an id, source ref, and
   a list of `ExtractedClause { clause_type, answer_text, char_start, char_end, confidence }`
   (one or more spans per type; empty when absent). Plus a thin `derived` view surfacing the
   directly-available temporal/party clause types (e.g. effective/expiration/renewal dates,
   parties, governing law) for convenient downstream use. **No normalization/reasoning** in
   C1 — that is C2+. Persisted: metadata + clauses → SQLite; source PDF → MinIO.

8. **MLflow:** new experiment `cuad-extractor` logs AUPR/F1/ANLS (and CER on the OCR sample);
   the optimized artifact registers as **`cuad-extractor-onnx-int8`**. Serving pulls it from
   the registry (or `DOCINTEL_CONTRACT_ONNX_LOCAL_PATH`, mirroring the existing local-path
   escape hatch for large bundles).

9. **No LLM in C1.** The extractor is the fine-tuned transformer. The Colab/ngrok LLM and
   vector/graph stores arrive in C2/C3.

## Architecture / new modules

```
notebooks/
  cuad_finetune.ipynb          (Colab GPU)  fine-tune extractive-QA on CUAD → MLflow
  cuad_onnx_export.ipynb       (Colab GPU)  ONNX export + INT8 + register

src/docintel/contracts/
  ingest.py        (new)   dual-path: PyMuPDF text │ docTR OCR → reconstructed text + offsets
  questions.py     (new)   41 clause types → CUAD question templates (from dataset)
  extractor.py     (new)   ContractExtractor Protocol + DebertaQaOnnxExtractor
                           (raw onnxruntime; sliding-window per question)
  aggregate.py     (new)   per-question window logits → ranked spans → ExtractedClause list
  schema.py        (new)   Pydantic ContractDocument, ExtractedClause, derived view

api/routes/contracts.py (new)  POST /contracts/extract, GET /contracts/{id}

eval/
  cuad_eval.py     (new)   AUPR / F1 / ANLS on CUAD test split
  ocr_cer.py       (new)   CER on the rendered/OCR'd contract sample

storage/                (extended)  contract metadata + clauses tables; PDF object put/get
```

## Components

### `contracts/ingest.py`
- `ingest(pdf_bytes) -> IngestedDoc` where `IngestedDoc = { text, page_offsets, source }`.
- Born-digital: PyMuPDF `get_text` per page, concatenated with tracked char offsets.
- Scanned: render pages to images (PyMuPDF rasterize) → existing docTR engine → text.
- `source` records which path ran (for metrics + the CER sample selection).

### `contracts/questions.py`
- Static mapping: 41 clause types → the CUAD question string(s). No hardcoded constants in
  business logic — the mapping lives in a data file / module-level table loaded once.

### `contracts/extractor.py`
- `ContractExtractor` — `Protocol`: `extract(text) -> list[ExtractedClause]`.
- `DebertaQaOnnxExtractor` — pulls `cuad-extractor-onnx-int8` from MLflow (or local path),
  lazy on `app.state`. For each clause question: tokenize `(question, context)` with
  **sliding-window stride**, run **raw `onnxruntime.InferenceSession`**, collect start/end
  logits per window.

### `contracts/aggregate.py`
- Per question: convert window start/end logits → candidate spans, map back to **document
  char offsets**, rank by score, apply a no-answer threshold, keep top-k → `ExtractedClause`s.

### `api/routes/contracts.py`
- `POST /contracts/extract`: upload PDF → ingest → extract → `ContractDocument` → persist →
  return. Records Prometheus metrics (extraction confidence histogram, per-clause hit
  counter) reusing the existing custom registry pattern.
- `GET /contracts/{id}`: retrieve persisted record.

## Data flow

```
PDF bytes
 → ingest()           dual-path → text (+ offsets, source)
 → extractor.extract() per-question sliding-window QA (ONNX-INT8, CPU)
 → aggregate()         → list[ExtractedClause] (+ derived view)
 → ContractDocument    persist: SQLite (meta+clauses) + MinIO (PDF)
 → response (+ metrics → Prometheus, logs → Loki)
```

## Testing (TDD)

- **Unit:** path detection (digital vs scanned) on fixtures; question-template completeness
  (all 41 present); span aggregation (window logits → correct char offsets, no-answer
  threshold); schema validation/serialization.
- **Integration:** one small fixture contract end-to-end → `ContractDocument` with expected
  clause types present; `GET /contracts/{id}` round-trip.
- **Eval (not in CI):** `cuad_eval.py` reproduces AUPR/F1/ANLS on the CUAD test split;
  `ocr_cer.py` reports CER on the OCR sample. Numbers recorded to MLflow + the phase report.
- Format/lint/type-check: ruff + mypy (strict) + pytest, per repo standard.

## Metrics delivered (for the CV)

| Metric | Where measured | Comparable to |
|---|---|---|
| Clause extraction **AUPR / F1 / ANLS** | `cuad_eval.py`, CUAD test split | CUAD published baseline |
| OCR **CER** | `ocr_cer.py`, scanned sample | docTR on contracts |
| CPU **p50/p95 latency**, model **size** (fp32 vs INT8) | benchmark, MLflow | existing Phase-3 style |

## Out of scope (later phases)

- Vector index, embeddings, `/ask`, RAGAS — **C2**.
- Knowledge graph (Neo4j), GraphRAG traversal — **C3**.
- LangGraph agent, Langfuse, Colab/ngrok LLM — **C2 (LLM) / C4 (agent)**.
- Clause normalization, date parsing, cross-contract reasoning.

## Risks & mitigations

- **CPU latency:** 41 questions × sliding windows over a long contract can be slow on CPU.
  Mitigate with INT8, batching windows, and (if needed) gating which windows run. Measured,
  not assumed — latency is a tracked metric, not a blocker.
- **Colab fine-tune budget:** DeBERTa-v3-base on CUAD must fit Colab Pro session limits;
  RoBERTa-base is the fallback. Decided in the plan, not here.
- **OCR span alignment:** avoided by design — extraction metrics use clean text; OCR is
  measured only via CER (Decision 5).
- **Large model bundle:** kept off git like the LayoutLMv3 bundle; served from MLflow or
  `DOCINTEL_CONTRACT_ONNX_LOCAL_PATH`.
```
