# Contract Intelligence C1 — Ingestion & Clause Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a contract PDF (born-digital or scanned) into a persisted, structured `ContractDocument` — the 41 CUAD clause types as cited text spans — served on CPU from an ONNX-INT8 extractive-QA model, retrievable by id.

**Architecture:** A new `contracts/` package mirrors the existing `kie/` serving pattern: dual-path ingestion (PyMuPDF digital text │ docTR OCR) → a sliding-window ONNX QA extractor → span aggregation → `ContractDocument` → SQLite + MinIO. New routes `POST /contracts/extract` and `GET /contracts/{id}` sit alongside the untouched receipt `/extract`. Build-time fine-tuning + ONNX export reuse the proven `kie/train.py` and `optimize/export.py` patterns. Metrics (AUPR/F1/ANLS, CER) come from build-time eval modules.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, PyMuPDF (fitz), Transformers (tokenizer), ONNX Runtime, MLflow, MinIO, SQLite, Prometheus, pytest, ruff, mypy.

## Global Constraints

- **Python 3.12+**, full type hints on every def; `from __future__ import annotations` at top of each module.
- **Functional over classes**; keep functions small and focused.
- **No hardcoded constants** — every knob lives in `Settings` (env prefix `DOCINTEL_`). CUAD's 41 clause category names are reference *data* (a module-level table), not tunable constants.
- **Heavy libraries imported inside functions** (mlflow, transformers, onnxruntime, optimum, fitz, datasets, sklearn) so modules load cheaply on the laptop.
- **Minimal changes**: do not modify the receipt `/extract` path. Match existing style.
- **Lint/type/test** must pass: `uv run ruff check . && uv run ruff format --check .`, `uv run mypy src`, `uv run pytest`.
- **All commands run from `docintel/`.** Run `uv sync --all-extras` before starting (the whole suite needs all extras).
- C1 adds **no LLM, no vector store, no graph** — those are C2/C3/C4.

---

### Task 1: Contract settings

**Files:**
- Modify: `src/docintel/config.py` (add contract knobs to `Settings`)
- Test: `tests/test_config.py` (add cases)

**Interfaces:**
- Produces: new `Settings` fields used by every later task — `contract_model_name: str`, `contract_registered_model_name: str`, `contract_onnx_registered_model_name: str`, `contract_onnx_model_version: str`, `contract_onnx_local_path: str | None`, `contract_max_seq_length: int`, `contract_doc_stride: int`, `contract_n_best: int`, `contract_max_answer_length: int`, `contract_no_answer_threshold: float`, `contract_max_upload_mb: float`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
def test_contract_settings_defaults() -> None:
    from docintel.config import Settings

    s = Settings()
    assert s.contract_model_name == "microsoft/deberta-v3-base"
    assert s.contract_onnx_registered_model_name == "cuad-extractor-onnx-int8"
    assert s.contract_onnx_local_path is None
    assert s.contract_max_seq_length == 512
    assert s.contract_doc_stride == 128
    assert s.contract_n_best == 5
    assert s.contract_max_answer_length == 256
    assert s.contract_max_upload_mb == 25.0


def test_contract_settings_env_override(monkeypatch: object) -> None:
    from docintel.config import Settings

    monkeypatch.setenv("DOCINTEL_CONTRACT_ONNX_LOCAL_PATH", "/models/cuad")  # type: ignore[attr-defined]
    s = Settings()
    assert s.contract_onnx_local_path == "/models/cuad"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_contract_settings_defaults -v`
Expected: FAIL (`AttributeError: ... contract_model_name`).

- [ ] **Step 3: Add the fields**

```python
# in src/docintel/config.py, inside Settings, after the Phase-6 UI block
    # Contract Intelligence (C1)
    contract_model_name: str = "microsoft/deberta-v3-base"
    contract_registered_model_name: str = "cuad-extractor"
    contract_onnx_registered_model_name: str = "cuad-extractor-onnx-int8"
    contract_onnx_model_version: str = "1"
    contract_onnx_local_path: str | None = Field(
        default=None,
        description="Local contract ONNX bundle dir; if set, load it instead of the registry.",
    )
    contract_max_seq_length: int = 512
    contract_doc_stride: int = 128
    contract_n_best: int = 5
    contract_max_answer_length: int = 256
    contract_no_answer_threshold: float = 0.0
    contract_max_upload_mb: float = 25.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/docintel/config.py tests/test_config.py
git commit -m "feat(contracts): add contract-pipeline settings"
```

---

### Task 2: ContractDocument schema

**Files:**
- Create: `src/docintel/contracts/__init__.py` (empty)
- Create: `src/docintel/contracts/schema.py`
- Test: `tests/test_contracts_schema.py`

**Interfaces:**
- Produces:
  - `ExtractedClause(clause_type: str, answer_text: str, char_start: int, char_end: int, confidence: float)`
  - `ContractDocument(id: str, source: Literal["digital", "ocr"], clauses: list[ExtractedClause], derived: dict[str, list[str]], page_count: int, created_at: str)`
  - `build_derived(clauses: list[ExtractedClause]) -> dict[str, list[str]]` — groups answer_texts by clause_type.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts_schema.py
from __future__ import annotations

from docintel.contracts.schema import ContractDocument, ExtractedClause, build_derived


def _clause(t: str, text: str) -> ExtractedClause:
    return ExtractedClause(clause_type=t, answer_text=text, char_start=0, char_end=len(text), confidence=0.9)


def test_build_derived_groups_by_type() -> None:
    clauses = [_clause("Parties", "Acme"), _clause("Parties", "Globex"), _clause("Governing Law", "NY")]
    derived = build_derived(clauses)
    assert derived == {"Parties": ["Acme", "Globex"], "Governing Law": ["NY"]}


def test_contract_document_roundtrips() -> None:
    doc = ContractDocument(
        id="abc",
        source="digital",
        clauses=[_clause("Parties", "Acme")],
        derived={"Parties": ["Acme"]},
        page_count=3,
        created_at="2026-06-25T00:00:00+00:00",
    )
    again = ContractDocument.model_validate_json(doc.model_dump_json())
    assert again == doc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_schema.py -v`
Expected: FAIL (`ModuleNotFoundError: docintel.contracts.schema`).

- [ ] **Step 3: Create the package and schema**

```python
# src/docintel/contracts/__init__.py
```

```python
# src/docintel/contracts/schema.py
"""Structured contract record schema for the C1 extraction path."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field


class ExtractedClause(BaseModel):
    """One extracted clause span with its char offsets into the ingested text."""

    clause_type: str
    answer_text: str
    char_start: int
    char_end: int
    confidence: float


class ContractDocument(BaseModel):
    """Structured extraction result for one contract."""

    id: str
    source: Literal["digital", "ocr"]
    clauses: list[ExtractedClause] = Field(default_factory=list)
    derived: dict[str, list[str]] = Field(default_factory=dict)
    page_count: int
    created_at: str


def build_derived(clauses: list[ExtractedClause]) -> dict[str, list[str]]:
    """Group clause answer texts by clause type, preserving extraction order."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for clause in clauses:
        grouped[clause.clause_type].append(clause.answer_text)
    return dict(grouped)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/docintel/contracts/__init__.py src/docintel/contracts/schema.py tests/test_contracts_schema.py
git commit -m "feat(contracts): add ContractDocument schema"
```

---

### Task 3: CUAD clause categories + question templates

**Files:**
- Create: `src/docintel/contracts/questions.py`
- Test: `tests/test_contracts_questions.py`

**Interfaces:**
- Produces:
  - `CLAUSE_CATEGORIES: tuple[str, ...]` — the canonical 41 CUAD categories.
  - `question_for(category: str) -> str`
  - `all_questions() -> list[tuple[str, str]]` — `(category, question)` for all 41.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts_questions.py
from __future__ import annotations

from docintel.contracts.questions import CLAUSE_CATEGORIES, all_questions, question_for


def test_there_are_41_unique_categories() -> None:
    assert len(CLAUSE_CATEGORIES) == 41
    assert len(set(CLAUSE_CATEGORIES)) == 41


def test_every_question_mentions_its_category_and_is_nonempty() -> None:
    pairs = all_questions()
    assert len(pairs) == 41
    for category, question in pairs:
        assert category in question
        assert question == question_for(category)
        assert len(question) > len(category)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_questions.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Create the questions module**

```python
# src/docintel/contracts/questions.py
"""The 41 CUAD clause categories and their span-selection questions.

The category list is reference data from CUAD (Hendrycks et al., 2021). The
question follows CUAD's per-category template, so questions stay DRY.
"""

from __future__ import annotations

CLAUSE_CATEGORIES: tuple[str, ...] = (
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "Competitive Restriction Exception",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
)


def question_for(category: str) -> str:
    """Return the CUAD-style question for one clause category."""
    return (
        f'Highlight the parts (if any) of this contract related to "{category}" '
        "that should be reviewed by a lawyer."
    )


def all_questions() -> list[tuple[str, str]]:
    """Return ``(category, question)`` for all 41 clause categories."""
    return [(category, question_for(category)) for category in CLAUSE_CATEGORIES]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_questions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/docintel/contracts/questions.py tests/test_contracts_questions.py
git commit -m "feat(contracts): add CUAD clause categories + question templates"
```

---

### Task 4: Dual-path ingestion (PyMuPDF + docTR)

**Files:**
- Create: `src/docintel/contracts/ingest.py`
- Modify: `pyproject.toml` (add `contracts` extra + mypy override for `fitz`)
- Test: `tests/test_contracts_ingest.py`

**Interfaces:**
- Consumes: `OCREngine` from `docintel.pipeline.ocr`.
- Produces:
  - `IngestedDoc(text: str, page_count: int, source: Literal["digital", "ocr"])` (frozen dataclass)
  - `select_source(total_text_chars: int, min_chars: int) -> Literal["digital", "ocr"]`
  - `extract_digital_pages(data: bytes) -> list[str]`
  - `rasterize_pages(data: bytes) -> list[Image]`
  - `ingest_pdf(data: bytes, ocr_engine: OCREngine, settings: Settings) -> IngestedDoc`

- [ ] **Step 1: Add the `contracts` extra and mypy override**

```toml
# pyproject.toml — add a new optional-dependencies entry
contracts = [
    "pymupdf>=1.24",
]
```

```toml
# pyproject.toml — extend the existing mypy overrides "module" list with "fitz.*"
module = ["datasets.*", "huggingface_hub.*", "cv2.*", "doctr.*", "PIL.*", "seqeval.*", "mlflow.*", "optimum.*", "onnx.*", "onnxruntime.*", "matplotlib.*", "boto3.*", "botocore.*", "transformers.*", "fitz.*", "sklearn.*"]
```

Run: `uv sync --all-extras`
Expected: resolves, installs `pymupdf`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_contracts_ingest.py
from __future__ import annotations

from typing import Any

import fitz  # PyMuPDF
import numpy as np

from docintel.config import Settings
from docintel.contracts.ingest import (
    extract_digital_pages,
    ingest_pdf,
    select_source,
)
from docintel.pipeline.types import OCRResult


def _digital_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def test_select_source_threshold() -> None:
    assert select_source(total_text_chars=500, min_chars=20) == "digital"
    assert select_source(total_text_chars=3, min_chars=20) == "ocr"


def test_extract_digital_pages_reads_text() -> None:
    pages = extract_digital_pages(_digital_pdf("Hello Contract World"))
    assert any("Hello Contract World" in p for p in pages)


def test_ingest_digital_path() -> None:
    settings = Settings()
    doc = ingest_pdf(_digital_pdf("Governing Law: New York."), ocr_engine=_boom, settings=settings)
    assert doc.source == "digital"
    assert "Governing Law" in doc.text
    assert doc.page_count == 1


def _boom(image: Any) -> OCRResult:  # must NOT be called on the digital path
    raise AssertionError("OCR engine should not run on a born-digital PDF")


def test_ingest_ocr_path(monkeypatch: Any) -> None:
    settings = Settings()
    monkeypatch.setattr("docintel.contracts.ingest.extract_digital_pages", lambda data: [""])
    monkeypatch.setattr(
        "docintel.contracts.ingest.rasterize_pages",
        lambda data: [np.zeros((4, 4, 3), dtype=np.uint8)],
    )

    def _fake_ocr(image: Any) -> OCRResult:
        return OCRResult(text="SCANNED CLAUSE", words=[], confidence=0.0, image_width=4, image_height=4)

    doc = ingest_pdf(b"%PDF-fake", ocr_engine=_fake_ocr, settings=settings)
    assert doc.source == "ocr"
    assert "SCANNED CLAUSE" in doc.text
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_ingest.py -v`
Expected: FAIL (`ModuleNotFoundError: docintel.contracts.ingest`).

- [ ] **Step 4: Implement ingestion**

```python
# src/docintel/contracts/ingest.py
"""Dual-path contract ingestion: born-digital text (PyMuPDF) or scanned OCR (docTR).

A born-digital PDF carries an extractable text layer; a scanned PDF does not, so
its pages are rasterized and sent through the existing OCR engine. Heavy imports
(fitz) live inside functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from docintel.config import Settings
from docintel.pipeline.ocr import Image, OCREngine

_MIN_DIGITAL_CHARS = 32  # below this, treat the document as scanned


@dataclass(frozen=True)
class IngestedDoc:
    """Reconstructed contract text plus which ingestion path produced it."""

    text: str
    page_count: int
    source: Literal["digital", "ocr"]


def select_source(total_text_chars: int, min_chars: int) -> Literal["digital", "ocr"]:
    """Choose the digital path when the embedded text layer is non-trivial."""
    return "digital" if total_text_chars >= min_chars else "ocr"


def extract_digital_pages(data: bytes) -> list[str]:
    """Return per-page embedded text from a PDF (empty strings for image-only pages)."""
    import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        return [page.get_text("text") for page in doc]


def rasterize_pages(data: bytes) -> list[Image]:
    """Render each PDF page to a BGR image array for OCR."""
    import fitz

    images: list[Image] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap()
            arr: NDArray[np.uint8] = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            rgb = arr[:, :, :3]
            images.append(np.ascontiguousarray(rgb[:, :, ::-1]))  # RGB -> BGR
    return images


def ingest_pdf(data: bytes, ocr_engine: OCREngine, settings: Settings) -> IngestedDoc:
    """Reconstruct contract text via the digital text layer, or OCR if absent."""
    pages = extract_digital_pages(data)
    total_chars = sum(len(p.strip()) for p in pages)
    source = select_source(total_chars, _MIN_DIGITAL_CHARS)
    if source == "digital":
        return IngestedDoc(text="\n".join(pages), page_count=len(pages), source="digital")
    images = rasterize_pages(data)
    ocr_text = "\n".join(ocr_engine(image).text for image in images)
    return IngestedDoc(text=ocr_text, page_count=len(images), source="ocr")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_ingest.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/docintel/contracts/ingest.py tests/test_contracts_ingest.py
git commit -m "feat(contracts): add dual-path PDF ingestion (PyMuPDF + docTR)"
```

---

### Task 5: Span aggregation

**Files:**
- Create: `src/docintel/contracts/aggregate.py`
- Test: `tests/test_contracts_aggregate.py`

**Interfaces:**
- Consumes: `ExtractedClause` from `docintel.contracts.schema`.
- Produces:
  - `WindowSpan(start_char: int, end_char: int, score: float)` (frozen dataclass)
  - `best_spans_from_window(start_logits, end_logits, offset_mapping, n_best, max_answer_length) -> list[WindowSpan]`
  - `aggregate_clause(clause_type, text, windows, n_best, no_answer_threshold) -> list[ExtractedClause]` where `windows: list[WindowSpan]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts_aggregate.py
from __future__ import annotations

import numpy as np

from docintel.contracts.aggregate import (
    WindowSpan,
    aggregate_clause,
    best_spans_from_window,
)


def test_best_spans_picks_highest_scoring_valid_span() -> None:
    # 4 tokens; token 0 is the CLS/no-answer slot (offset (0,0)).
    start = np.array([0.1, 2.0, 0.0, 0.0])
    end = np.array([0.1, 0.0, 3.0, 0.0])
    offsets = [(0, 0), (0, 5), (6, 11), (12, 16)]
    spans = best_spans_from_window(start, end, offsets, n_best=3, max_answer_length=20)
    top = spans[0]
    assert (top.start_char, top.end_char) == (0, 11)
    assert top.score == 5.0


def test_aggregate_clause_applies_no_answer_threshold() -> None:
    text = "alpha beta gamma"
    windows = [WindowSpan(0, 5, score=4.0), WindowSpan(6, 10, score=-1.0)]
    clauses = aggregate_clause("Parties", text, windows, n_best=2, no_answer_threshold=0.0)
    assert len(clauses) == 1
    assert clauses[0].clause_type == "Parties"
    assert clauses[0].answer_text == "alpha"
    assert clauses[0].char_start == 0 and clauses[0].char_end == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_aggregate.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement aggregation**

```python
# src/docintel/contracts/aggregate.py
"""Turn per-window QA logits into char-offset clause spans (SQuAD-style decode)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from docintel.contracts.schema import ExtractedClause


@dataclass(frozen=True)
class WindowSpan:
    """A candidate answer span in document char offsets with its logit score."""

    start_char: int
    end_char: int
    score: float


def best_spans_from_window(
    start_logits: Any,
    end_logits: Any,
    offset_mapping: Sequence[tuple[int, int]],
    n_best: int,
    max_answer_length: int,
) -> list[WindowSpan]:
    """Return the top-``n_best`` (start, end) spans for one tokenized window.

    Tokens whose offset is ``(0, 0)`` (special/question tokens) are skipped as
    span endpoints. Scores are ``start_logit + end_logit``.
    """
    start = np.asarray(start_logits)
    end = np.asarray(end_logits)
    starts = np.argsort(start)[::-1][:n_best]
    ends = np.argsort(end)[::-1][:n_best]
    spans: list[WindowSpan] = []
    for s in starts:
        for e in ends:
            if e < s or (e - s + 1) > max_answer_length:
                continue
            s_off = offset_mapping[int(s)]
            e_off = offset_mapping[int(e)]
            if s_off == (0, 0) or e_off == (0, 0):
                continue
            spans.append(
                WindowSpan(
                    start_char=int(s_off[0]),
                    end_char=int(e_off[1]),
                    score=float(start[s] + end[e]),
                )
            )
    spans.sort(key=lambda span: span.score, reverse=True)
    return spans[:n_best]


def aggregate_clause(
    clause_type: str,
    text: str,
    windows: Sequence[WindowSpan],
    n_best: int,
    no_answer_threshold: float,
) -> list[ExtractedClause]:
    """Rank spans across windows; keep those above threshold, up to ``n_best``."""
    ranked = sorted(windows, key=lambda span: span.score, reverse=True)
    clauses: list[ExtractedClause] = []
    for span in ranked[:n_best]:
        if span.score < no_answer_threshold:
            continue
        clauses.append(
            ExtractedClause(
                clause_type=clause_type,
                answer_text=text[span.start_char : span.end_char],
                char_start=span.start_char,
                char_end=span.end_char,
                confidence=float(1.0 / (1.0 + np.exp(-span.score))),
            )
        )
    return clauses
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_aggregate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/docintel/contracts/aggregate.py tests/test_contracts_aggregate.py
git commit -m "feat(contracts): add QA span aggregation"
```

---

### Task 6: ONNX QA extractor backend

**Files:**
- Create: `src/docintel/contracts/extractor.py`
- Test: `tests/test_contracts_extractor.py`

**Interfaces:**
- Consumes: `Settings`; `all_questions()`; `best_spans_from_window`, `aggregate_clause`, `WindowSpan`; `ExtractedClause`; `download_registered_model` from `docintel.optimize.export`.
- Produces:
  - `ContractExtractor` Protocol: `extract(self, text: str) -> list[ExtractedClause]`
  - `resolve_contract_bundle(settings, download, tracking_uri=None) -> Path`
  - `CuadQaOnnxExtractor` with classmethod `load(settings, tracking_uri=None)` and `extract(text)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts_extractor.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from docintel.config import Settings
from docintel.contracts.extractor import CuadQaOnnxExtractor, resolve_contract_bundle


def test_resolve_contract_bundle_prefers_local_path() -> None:
    settings = Settings(contract_onnx_local_path="/models/cuad")
    out = resolve_contract_bundle(settings, download=_unused_download)
    assert out == Path("/models/cuad")


def _unused_download(name: str, version: str, dest: Path, uri: str | None) -> Path:
    raise AssertionError("download must not run when a local path is set")


class _FakeEncoding:
    """Mimics a transformers BatchEncoding with overflow windows + offsets."""

    def __init__(self) -> None:
        # one window, 3 tokens: [special, "alpha", "beta"]
        self._offsets = [[(0, 0), (0, 5), (6, 10)]]
        self.data = {"input_ids": np.array([[0, 1, 2]]), "attention_mask": np.array([[1, 1, 1]])}

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def num_windows(self) -> int:
        return len(self._offsets)

    def offsets(self, i: int) -> list[tuple[int, int]]:
        return self._offsets[i]


def test_extract_decodes_one_clause_per_strong_question(monkeypatch: Any) -> None:
    settings = Settings(contract_no_answer_threshold=0.0, contract_n_best=1)

    class _FakeSession:
        def run(self, _: Any, feeds: dict[str, Any]) -> list[Any]:
            # start favors token 1, end favors token 2 -> span "alpha beta"? offsets (0,10)
            start = np.array([[0.0, 5.0, 0.0]])
            end = np.array([[0.0, 0.0, 5.0]])
            return [start, end]

    # Only the first question yields a clause; force a single question for the test.
    monkeypatch.setattr(
        "docintel.contracts.extractor.all_questions", lambda: [("Parties", "Who are the parties?")]
    )
    extractor = CuadQaOnnxExtractor.__new__(CuadQaOnnxExtractor)
    extractor._session = _FakeSession()  # type: ignore[attr-defined]
    extractor._settings = settings  # type: ignore[attr-defined]
    extractor._encode = lambda question, text: _FakeEncoding()  # type: ignore[attr-defined]

    clauses = extractor.extract("alpha beta")
    assert len(clauses) == 1
    assert clauses[0].clause_type == "Parties"
    assert clauses[0].answer_text == "alpha beta"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_extractor.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the extractor**

```python
# src/docintel/contracts/extractor.py
"""Serve the ONNX-INT8 CUAD extractive-QA model on CPU.

For each of the 41 clause questions, the (question, contract) pair is tokenized
with a sliding window (doc stride). Each window runs through a raw
``onnxruntime.InferenceSession`` (start/end logits); spans are decoded to
document char offsets and aggregated per clause. Heavy imports live in functions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from docintel.config import Settings
from docintel.contracts.aggregate import WindowSpan, aggregate_clause, best_spans_from_window
from docintel.contracts.questions import all_questions
from docintel.contracts.schema import ExtractedClause


def resolve_contract_bundle(
    settings: Settings,
    download: Callable[[str, str, Path, str | None], Path],
    tracking_uri: str | None = None,
) -> Path:
    """Return the ONNX bundle dir: the local override if set, else MLflow download."""
    if settings.contract_onnx_local_path:
        return Path(settings.contract_onnx_local_path)
    dest = Path(settings.data_dir) / "models" / settings.contract_onnx_registered_model_name
    return download(
        settings.contract_onnx_registered_model_name,
        settings.contract_onnx_model_version,
        dest,
        tracking_uri or settings.mlflow_tracking_uri,
    )


class ContractExtractor(Protocol):
    """Maps contract text to extracted clause spans."""

    def extract(self, text: str) -> list[ExtractedClause]: ...


class CuadQaOnnxExtractor:
    """ONNX-INT8 extractive-QA model served via onnxruntime."""

    def __init__(self, session: Any, tokenizer: Any, settings: Settings) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._settings = settings

    @classmethod
    def load(cls, settings: Settings, tracking_uri: str | None = None) -> CuadQaOnnxExtractor:
        """Load the INT8 model (local override or MLflow) and build the backend."""
        import onnxruntime as ort
        from transformers import AutoTokenizer

        from docintel.optimize.export import download_registered_model

        bundle = resolve_contract_bundle(settings, download_registered_model, tracking_uri)
        onnx_path = next(Path(bundle).rglob("*quantized*.onnx"), None) or next(
            Path(bundle).rglob("*.onnx"), None
        )
        if onnx_path is None:
            raise FileNotFoundError(f"No .onnx file found under bundle: {bundle}")
        session = ort.InferenceSession(str(onnx_path))
        tokenizer = AutoTokenizer.from_pretrained(settings.contract_model_name)
        return cls(session, tokenizer, settings)

    def _encode(self, question: str, text: str) -> Any:
        """Tokenize (question, text) into overflowing windows with offset maps."""
        return self._tokenizer(
            question,
            text,
            truncation="only_second",
            max_length=self._settings.contract_max_seq_length,
            stride=self._settings.contract_doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="np",
        )

    def _run_question(self, question: str, clause_type: str, text: str) -> list[ExtractedClause]:
        encoding = self._encode(question, text)
        spans: list[WindowSpan] = []
        num_windows = encoding["input_ids"].shape[0]
        for i in range(num_windows):
            feeds = {
                "input_ids": encoding["input_ids"][i : i + 1].astype(np.int64),
                "attention_mask": encoding["attention_mask"][i : i + 1].astype(np.int64),
            }
            start_logits, end_logits = self._session.run(None, feeds)
            offsets = [tuple(o) for o in encoding["offset_mapping"][i]]
            spans.extend(
                best_spans_from_window(
                    start_logits[0],
                    end_logits[0],
                    offsets,
                    self._settings.contract_n_best,
                    self._settings.contract_max_answer_length,
                )
            )
        return aggregate_clause(
            clause_type,
            text,
            spans,
            self._settings.contract_n_best,
            self._settings.contract_no_answer_threshold,
        )

    def extract(self, text: str) -> list[ExtractedClause]:
        """Run all 41 clause questions over the contract text."""
        clauses: list[ExtractedClause] = []
        for clause_type, question in all_questions():
            clauses.extend(self._run_question(question, clause_type, text))
        return clauses
```

> Note: the test exercises decode logic via injected `_session`/`_encode`; the real `_encode`/token-grad path runs only with the model present (covered by the build-time eval in Task 12). The test monkeypatches `_encode` to return a fake encoding, so adjust `_run_question` to read windows via `encoding["input_ids"].shape[0]` — already done above; the fake encoding in the test exposes `input_ids`/`offset_mapping` through `__getitem__`. Update the test's `_FakeEncoding` to also expose `offset_mapping`:

```python
# adjust _FakeEncoding.__init__ in the test:
        self.data = {
            "input_ids": np.array([[0, 1, 2]]),
            "attention_mask": np.array([[1, 1, 1]]),
            "offset_mapping": np.array([[(0, 0), (0, 5), (6, 10)]]),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_extractor.py -v`
Expected: PASS.

- [ ] **Step 5: Run lint + types**

Run: `uv run ruff check src/docintel/contracts && uv run mypy src/docintel/contracts`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/docintel/contracts/extractor.py tests/test_contracts_extractor.py
git commit -m "feat(contracts): add ONNX-INT8 CUAD QA extractor backend"
```

---

### Task 7: Contract persistence (SQLite)

**Files:**
- Create: `src/docintel/storage/contracts_db.py`
- Test: `tests/test_contracts_db.py`

**Interfaces:**
- Consumes: `ContractDocument`.
- Produces:
  - `init_contracts_db(path: str) -> None`
  - `save_contract(path: str, doc: ContractDocument, pdf_key: str) -> None`
  - `get_contract(path: str, contract_id: str) -> tuple[ContractDocument, str] | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts_db.py
from __future__ import annotations

from typing import Any

from docintel.contracts.schema import ContractDocument, ExtractedClause
from docintel.storage.contracts_db import get_contract, init_contracts_db, save_contract


def _doc(cid: str) -> ContractDocument:
    return ContractDocument(
        id=cid,
        source="digital",
        clauses=[ExtractedClause(clause_type="Parties", answer_text="Acme", char_start=0, char_end=4, confidence=0.9)],
        derived={"Parties": ["Acme"]},
        page_count=1,
        created_at="2026-06-25T00:00:00+00:00",
    )


def test_save_and_get_roundtrip(tmp_path: Any) -> None:
    path = str(tmp_path / "c.db")
    init_contracts_db(path)
    save_contract(path, _doc("c1"), "c1.pdf")
    found = get_contract(path, "c1")
    assert found is not None
    doc, key = found
    assert doc == _doc("c1")
    assert key == "c1.pdf"


def test_get_missing_returns_none(tmp_path: Any) -> None:
    path = str(tmp_path / "c.db")
    init_contracts_db(path)
    assert get_contract(path, "nope") is None


def test_save_upserts(tmp_path: Any) -> None:
    path = str(tmp_path / "c.db")
    init_contracts_db(path)
    save_contract(path, _doc("c1"), "old.pdf")
    save_contract(path, _doc("c1"), "new.pdf")
    found = get_contract(path, "c1")
    assert found is not None and found[1] == "new.pdf"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_db.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement persistence** (mirrors `storage/db.py`)

```python
# src/docintel/storage/contracts_db.py
"""SQLite persistence for ContractDocument records (stdlib sqlite3, no ORM)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from docintel.contracts.schema import ContractDocument

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contracts (
    id TEXT PRIMARY KEY,
    contract_json TEXT NOT NULL,
    pdf_key TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


@contextmanager
def _connect(path: str) -> Iterator[sqlite3.Connection]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_contracts_db(path: str) -> None:
    """Create the contracts table if it does not exist."""
    with _connect(path) as conn:
        conn.execute(_SCHEMA)


def save_contract(path: str, doc: ContractDocument, pdf_key: str) -> None:
    """Upsert one contract record by id."""
    with _connect(path) as conn:
        conn.execute(
            "INSERT INTO contracts (id, contract_json, pdf_key, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "contract_json=excluded.contract_json, pdf_key=excluded.pdf_key",
            (doc.id, doc.model_dump_json(), pdf_key, doc.created_at),
        )


def get_contract(path: str, contract_id: str) -> tuple[ContractDocument, str] | None:
    """Return ``(ContractDocument, pdf_key)`` for an id, or None if absent."""
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT contract_json, pdf_key FROM contracts WHERE id = ?", (contract_id,)
        ).fetchone()
    if row is None:
        return None
    return ContractDocument.model_validate_json(row[0]), row[1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_db.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/docintel/storage/contracts_db.py tests/test_contracts_db.py
git commit -m "feat(contracts): add SQLite persistence for contract records"
```

---

### Task 8: Contract metrics

**Files:**
- Modify: `src/docintel/api/metrics.py`
- Test: `tests/test_metrics.py` (add cases)

**Interfaces:**
- Consumes: `ContractDocument`.
- Produces: extend `Metrics` with `contract_clause_confidence: Histogram` and `contract_clause_total: Counter`; add `record_contract_extraction(metrics: Metrics, doc: ContractDocument) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_metrics.py
def test_record_contract_extraction_counts_clauses() -> None:
    from prometheus_client import CollectorRegistry

    from docintel.api.metrics import build_metrics, record_contract_extraction
    from docintel.contracts.schema import ContractDocument, ExtractedClause

    registry = CollectorRegistry()
    metrics = build_metrics(registry)
    doc = ContractDocument(
        id="c1",
        source="ocr",
        clauses=[
            ExtractedClause(clause_type="Parties", answer_text="Acme", char_start=0, char_end=4, confidence=0.8),
        ],
        derived={"Parties": ["Acme"]},
        page_count=2,
        created_at="2026-06-25T00:00:00+00:00",
    )
    record_contract_extraction(metrics, doc)
    assert registry.get_sample_value("docintel_contract_clauses_total", {"source": "ocr"}) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py::test_record_contract_extraction_counts_clauses -v`
Expected: FAIL (`ImportError: record_contract_extraction`).

- [ ] **Step 3: Extend metrics**

```python
# in src/docintel/api/metrics.py
# add to the imports:
from docintel.contracts.schema import ContractDocument

# add two fields to the Metrics dataclass:
    contract_clause_confidence: Histogram
    contract_clause_total: Counter

# in build_metrics(...), add to the returned Metrics(...):
        contract_clause_confidence=Histogram(
            "docintel_contract_clause_confidence",
            "Per-clause extraction confidence observed on /contracts/extract.",
            buckets=_CONFIDENCE_BUCKETS,
            registry=registry,
        ),
        contract_clause_total=Counter(
            "docintel_contract_clauses",
            "Clauses extracted on /contracts/extract, labelled by ingestion source.",
            labelnames=("source",),
            registry=registry,
        ),

# add at module end:
def record_contract_extraction(metrics: Metrics, doc: ContractDocument) -> None:
    """Record one extracted contract: clause confidences + clause count by source."""
    for clause in doc.clauses:
        metrics.contract_clause_confidence.observe(clause.confidence)
    metrics.contract_clause_total.labels(source=doc.source).inc(len(doc.clauses))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/docintel/api/metrics.py tests/test_metrics.py
git commit -m "feat(contracts): record contract extraction metrics"
```

---

### Task 9: API routes + app wiring

**Files:**
- Create: `src/docintel/api/routes/contracts.py`
- Modify: `src/docintel/api/main.py` (register router + init `app.state.contract_extractor`)
- Test: `tests/test_contracts_routes.py`

**Interfaces:**
- Consumes: `ingest_pdf`, `CuadQaOnnxExtractor`/`ContractExtractor`, `ContractDocument`/`build_derived`, `record_contract_extraction`, `init_contracts_db`/`save_contract`/`get_contract`, `ensure_bucket`/`make_s3_client`/`put_image`, `get_ocr_engine`/`get_s3_client`/`get_metrics` (reused from `extract.py`).
- Produces: `get_contract_extractor(request) -> ContractExtractor`; `POST /contracts/extract -> ContractDocument`; `GET /contracts/{contract_id} -> ContractDocument`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts_routes.py
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docintel.api.main import create_app
from docintel.api.routes.contracts import get_contract_extractor
from docintel.api.routes.extract import get_ocr_engine, get_s3_client
from docintel.config import Settings, get_settings
from docintel.contracts.schema import ExtractedClause
from tests.test_documents import _FakeS3


class _StubExtractor:
    def extract(self, text: str) -> list[ExtractedClause]:
        return [ExtractedClause(clause_type="Parties", answer_text="Acme", char_start=0, char_end=4, confidence=0.9)]


@pytest.fixture
def client(tmp_path: Any) -> Iterator[TestClient]:
    app = create_app()
    settings = Settings(sqlite_path=str(tmp_path / "c.db"))
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_ocr_engine] = lambda: (lambda image: None)
    app.dependency_overrides[get_s3_client] = lambda: _FakeS3()
    app.dependency_overrides[get_contract_extractor] = lambda: _StubExtractor()
    # ingest is monkeypatched per-test to avoid building a real PDF
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_extract_rejects_non_pdf(client: TestClient) -> None:
    resp = client.post("/contracts/extract", files={"file": ("x.txt", b"hi", "text/plain")})
    assert resp.status_code == 415


def test_extract_and_retrieve(client: TestClient, monkeypatch: Any) -> None:
    from docintel.contracts.ingest import IngestedDoc

    monkeypatch.setattr(
        "docintel.api.routes.contracts.ingest_pdf",
        lambda data, ocr_engine, settings: IngestedDoc(text="Acme and Globex", page_count=1, source="digital"),
    )
    resp = client.post("/contracts/extract", files={"file": ("c.pdf", b"%PDF-1.4", "application/pdf")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "digital"
    assert body["derived"]["Parties"] == ["Acme"]
    cid = body["id"]

    got = client.get(f"/contracts/{cid}")
    assert got.status_code == 200
    assert got.json()["id"] == cid

    assert client.get("/contracts/missing").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_routes.py -v`
Expected: FAIL (`ModuleNotFoundError: docintel.api.routes.contracts`).

- [ ] **Step 3: Implement the routes** (mirrors `extract.py` + `documents.py`)

```python
# src/docintel/api/routes/contracts.py
"""The /contracts endpoints: PDF -> ingest -> QA extract -> persist -> ContractDocument."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from docintel.api.metrics import Metrics, record_contract_extraction
from docintel.api.routes.extract import get_metrics, get_ocr_engine, get_s3_client
from docintel.config import Settings, get_settings
from docintel.contracts.extractor import ContractExtractor, CuadQaOnnxExtractor
from docintel.contracts.ingest import ingest_pdf
from docintel.contracts.schema import ContractDocument, build_derived
from docintel.pipeline.ocr import OCREngine
from docintel.storage.contracts_db import get_contract, init_contracts_db, save_contract
from docintel.storage.objects import ensure_bucket, put_image

logger = logging.getLogger("docintel.api.contracts")
router = APIRouter(tags=["contracts"])

_PDF_TYPE = "application/pdf"


def get_contract_extractor(request: Request) -> ContractExtractor:
    """Return the process-wide contract extractor, loading it once on first use."""
    backend: ContractExtractor | None = getattr(request.app.state, "contract_extractor", None)
    if backend is None:
        backend = CuadQaOnnxExtractor.load(get_settings())
        request.app.state.contract_extractor = backend
    return backend


@router.post(
    "/contracts/extract",
    response_model=ContractDocument,
    summary="Extract structured clauses from a contract PDF",
)
async def extract_contract(
    file: UploadFile,
    settings: Settings = Depends(get_settings),  # noqa: B008
    engine: OCREngine = Depends(get_ocr_engine),  # noqa: B008
    extractor: ContractExtractor = Depends(get_contract_extractor),  # noqa: B008
    s3: Any = Depends(get_s3_client),  # noqa: B008
    metrics: Metrics = Depends(get_metrics),  # noqa: B008
) -> ContractDocument:
    """Ingest a PDF, extract clauses, persist, and return a ContractDocument."""
    if file.content_type != _PDF_TYPE:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type!r}; use application/pdf.",
        )
    data = await file.read()
    max_bytes = int(settings.contract_max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Upload exceeds the {settings.contract_max_upload_mb} MB limit.",
        )

    start = time.perf_counter()
    ingested = ingest_pdf(data, engine, settings)
    clauses = extractor.extract(ingested.text)
    doc = ContractDocument(
        id=uuid.uuid4().hex,
        source=ingested.source,
        clauses=clauses,
        derived=build_derived(clauses),
        page_count=ingested.page_count,
        created_at=datetime.now(UTC).isoformat(),
    )
    latency_ms = (time.perf_counter() - start) * 1000

    pdf_key = f"{doc.id}.pdf"
    ensure_bucket(s3, settings.minio_bucket)
    put_image(s3, settings.minio_bucket, pdf_key, data, _PDF_TYPE)
    init_contracts_db(settings.sqlite_path)
    save_contract(settings.sqlite_path, doc, pdf_key)

    logger.info(
        "contracts.extract.complete",
        extra={
            "contract_id": doc.id,
            "latency_ms": round(latency_ms, 2),
            "clauses": len(doc.clauses),
            "source": doc.source,
        },
    )
    record_contract_extraction(metrics, doc)
    return doc


@router.get(
    "/contracts/{contract_id}",
    response_model=ContractDocument,
    summary="Retrieve an extracted contract",
)
def read_contract(
    contract_id: str,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> ContractDocument:
    """Return a previously extracted contract by id."""
    init_contracts_db(settings.sqlite_path)
    found = get_contract(settings.sqlite_path, contract_id)
    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found.")
    return found[0]
```

- [ ] **Step 4: Register the router and init state**

```python
# src/docintel/api/main.py
# 1) extend the routes import:
from docintel.api.routes import contracts, documents, extract, health
# 2) in lifespan(), alongside the other app.state lines:
    app.state.contract_extractor = None
# 3) in create_app(), after app.include_router(documents.router):
    app.include_router(contracts.router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_routes.py tests/test_health.py -v`
Expected: PASS (health confirms app still builds).

- [ ] **Step 6: Commit**

```bash
git add src/docintel/api/routes/contracts.py src/docintel/api/main.py tests/test_contracts_routes.py
git commit -m "feat(contracts): add /contracts/extract and /contracts/{id} routes"
```

---

### Task 10: Build-time QA training module

**Files:**
- Create: `src/docintel/contracts/qa_config.py`
- Create: `src/docintel/contracts/train_qa.py`
- Test: `tests/test_contracts_train_qa.py`

**Interfaces:**
- Produces:
  - `QaTrainingConfig` (Pydantic) with: `model_name: str`, `num_train_epochs: float`, `learning_rate: float`, `train_batch_size: int`, `eval_batch_size: int`, `weight_decay: float`, `warmup_ratio: float`, `seed: int`, `max_seq_length: int`, `doc_stride: int`.
  - `build_qa_training_arguments(config, output_dir) -> Any`
  - `collect_repro_params(config, dataset_revision, git_sha) -> dict[str, str]`
  - `save_qa_bundle(model, tokenizer, metrics, bundle_dir) -> Path`
  - `run_qa_training(...)` (runs on Colab; not unit-tested).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts_train_qa.py
from __future__ import annotations

import json
from pathlib import Path

from docintel.contracts.qa_config import QaTrainingConfig
from docintel.contracts.train_qa import collect_repro_params, save_qa_bundle


def test_collect_repro_params_are_all_strings() -> None:
    cfg = QaTrainingConfig()
    params = collect_repro_params(cfg, dataset_revision="cuad-v1", git_sha="abc123")
    assert params["model_name"] == cfg.model_name
    assert params["dataset_revision"] == "cuad-v1"
    assert params["git_sha"] == "abc123"
    assert all(isinstance(v, str) for v in params.values())


class _FakeSavable:
    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "marker").write_text("ok", encoding="utf-8")


def test_save_qa_bundle_writes_artifacts(tmp_path: Path) -> None:
    bundle = save_qa_bundle(_FakeSavable(), _FakeSavable(), {"f1": 0.5}, tmp_path / "bundle")
    assert (bundle / "model" / "marker").exists()
    assert (bundle / "tokenizer" / "marker").exists()
    assert json.loads((bundle / "metrics.json").read_text(encoding="utf-8")) == {"f1": 0.5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_train_qa.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement config + training helpers** (mirrors `kie/config.py` + `kie/train.py`)

```python
# src/docintel/contracts/qa_config.py
"""Configuration for fine-tuning the CUAD extractive-QA model (build-time)."""

from __future__ import annotations

from pydantic import BaseModel


class QaTrainingConfig(BaseModel):
    """Hyperparameters for CUAD QA fine-tuning."""

    model_name: str = "microsoft/deberta-v3-base"
    num_train_epochs: float = 3.0
    learning_rate: float = 3e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    max_seq_length: int = 512
    doc_stride: int = 128
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
```

```python
# src/docintel/contracts/train_qa.py
"""CUAD extractive-QA fine-tuning helpers; run_qa_training executes on Colab GPU."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from docintel.contracts.qa_config import QaTrainingConfig


def build_qa_training_arguments(config: QaTrainingConfig, output_dir: str) -> Any:
    """Map QaTrainingConfig to Hugging Face TrainingArguments."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        seed=config.seed,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        logging_steps=50,
    )


def collect_repro_params(
    config: QaTrainingConfig, dataset_revision: str, git_sha: str
) -> dict[str, str]:
    """Flatten reproducibility-relevant values into MLflow string params."""
    return {
        "model_name": config.model_name,
        "num_train_epochs": str(config.num_train_epochs),
        "learning_rate": str(config.learning_rate),
        "train_batch_size": str(config.train_batch_size),
        "weight_decay": str(config.weight_decay),
        "warmup_ratio": str(config.warmup_ratio),
        "seed": str(config.seed),
        "max_seq_length": str(config.max_seq_length),
        "doc_stride": str(config.doc_stride),
        "dataset_revision": dataset_revision,
        "git_sha": git_sha,
    }


def save_qa_bundle(
    model: Any, tokenizer: Any, metrics: Mapping[str, float], bundle_dir: Path
) -> Path:
    """Write a self-contained model bundle for download + ONNX export."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(bundle_dir / "model"))
    tokenizer.save_pretrained(str(bundle_dir / "tokenizer"))
    (bundle_dir / "metrics.json").write_text(json.dumps(dict(metrics)), encoding="utf-8")
    return bundle_dir


def run_qa_training(
    config: QaTrainingConfig,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    bundle_dir: Path,
    dataset_revision: str,
    git_sha: str,
    output_dir: str = "outputs",
) -> Path:
    """Fine-tune the QA model, log to MLflow, and save a bundle. Runs on Colab."""
    import mlflow
    from transformers import AutoModelForQuestionAnswering, Trainer, set_seed

    set_seed(config.seed)
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_name)
    args = build_qa_training_arguments(config, output_dir)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    with mlflow.start_run():
        mlflow.log_params(collect_repro_params(config, dataset_revision, git_sha))
        trainer.train()
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        bundle = save_qa_bundle(model, tokenizer, eval_metrics, bundle_dir)
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
    return bundle
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_train_qa.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/docintel/contracts/qa_config.py src/docintel/contracts/train_qa.py tests/test_contracts_train_qa.py
git commit -m "feat(contracts): add CUAD QA fine-tuning module"
```

---

### Task 11: ONNX export for QA

**Files:**
- Modify: `src/docintel/optimize/export.py` (add `export_qa_to_onnx`)
- Test: `tests/test_optimize_export.py` (add a seam test)

**Interfaces:**
- Produces: `export_qa_to_onnx(model_dir: Path, out_dir: Path) -> Path` (uses `ORTModelForQuestionAnswering`). Quantization reuses the existing `optimize/quantize.py::quantize_dynamic_int8`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_optimize_export.py
def test_export_qa_to_onnx_is_importable() -> None:
    from docintel.optimize.export import export_qa_to_onnx

    assert callable(export_qa_to_onnx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_optimize_export.py::test_export_qa_to_onnx_is_importable -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Add the QA export function**

```python
# append to src/docintel/optimize/export.py
def export_qa_to_onnx(model_dir: Path, out_dir: Path) -> Path:
    """Export a question-answering model to ONNX (fp32) via Optimum."""
    from optimum.onnxruntime import ORTModelForQuestionAnswering

    model = ORTModelForQuestionAnswering.from_pretrained(str(model_dir), export=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return out_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimize_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/docintel/optimize/export.py tests/test_optimize_export.py
git commit -m "feat(contracts): add ONNX export for the QA model"
```

---

### Task 12: Evaluation — CUAD metrics + OCR CER

**Files:**
- Create: `src/docintel/contracts/eval.py`
- Create: `src/docintel/contracts/ocr_cer.py`
- Modify: `pyproject.toml` (add `scikit-learn` to the `train` extra)
- Test: `tests/test_contracts_eval.py`

**Interfaces:**
- Produces:
  - `normalize_answer(text: str) -> str`
  - `token_f1(prediction: str, ground_truth: str) -> float`
  - `anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float`
  - `average_precision(scores: list[float], labels: list[int]) -> float` (wraps sklearn)
  - `cer(reference: str, hypothesis: str) -> float` (in `ocr_cer.py`)

- [ ] **Step 1: Add scikit-learn to the `train` extra**

```toml
# pyproject.toml — extend the existing train extra list with:
    "scikit-learn>=1.4",
```

Run: `uv sync --all-extras`
Expected: resolves, installs `scikit-learn`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_contracts_eval.py
from __future__ import annotations

from docintel.contracts.eval import anls, average_precision, normalize_answer, token_f1
from docintel.contracts.ocr_cer import cer


def test_normalize_answer_strips_articles_punctuation_case() -> None:
    assert normalize_answer("The  Agreement.") == "agreement"


def test_token_f1_exact_and_partial() -> None:
    assert token_f1("new york law", "new york law") == 1.0
    assert 0.0 < token_f1("new york", "new york law") < 1.0
    assert token_f1("", "anything") == 0.0


def test_anls_identical_is_one_and_far_is_zero() -> None:
    assert anls("acme corp", "acme corp") == 1.0
    assert anls("acme", "zzzzzzzz") == 0.0


def test_average_precision_perfect_ranking() -> None:
    ap = average_precision(scores=[0.9, 0.8, 0.1], labels=[1, 1, 0])
    assert ap == 1.0


def test_cer_basic() -> None:
    assert cer("contract", "contract") == 0.0
    assert cer("contract", "contracX") == 1 / 8
    assert cer("", "") == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_eval.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement the metrics**

```python
# src/docintel/contracts/eval.py
"""CUAD extraction metrics: normalized token-F1, ANLS, and AUPR.

Pure functions over strings/score lists so they are CPU-testable and reusable by
the build-time eval notebook. AUPR wraps scikit-learn.
"""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles, and collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1 between two normalized answer strings."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    for tok in pred_tokens:
        if tok in gold_tokens:
            common[tok] = min(pred_tokens.count(tok), gold_tokens.count(tok))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity for a single (pred, gold) pair."""
    pred = prediction.strip().lower()
    gold = ground_truth.strip().lower()
    if not pred and not gold:
        return 1.0
    longest = max(len(pred), len(gold))
    if longest == 0:
        return 1.0
    similarity = 1.0 - _levenshtein(pred, gold) / longest
    return similarity if similarity >= threshold else 0.0


def average_precision(scores: list[float], labels: list[int]) -> float:
    """Area under the precision-recall curve (AUPR) via scikit-learn."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))
```

```python
# src/docintel/contracts/ocr_cer.py
"""Character Error Rate for the OCR ingestion path."""

from __future__ import annotations

from docintel.contracts.eval import _levenshtein


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit distance / len(reference); 0.0 when both empty."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / len(reference)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_eval.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/docintel/contracts/eval.py src/docintel/contracts/ocr_cer.py tests/test_contracts_eval.py
git commit -m "feat(contracts): add CUAD eval metrics (F1/ANLS/AUPR) + OCR CER"
```

---

### Task 13: Colab notebooks + docs

**Files:**
- Create: `notebooks/cuad_finetune.ipynb`
- Create: `notebooks/cuad_onnx_export.ipynb`
- Modify: `README.md` (add a "Contract Intelligence (C1)" subsection)

**Interfaces:** none (orchestration only; all logic lives in the tested modules above).

- [ ] **Step 1: Create `notebooks/cuad_finetune.ipynb`**

A minimal notebook (markdown + code cells) that, on Colab GPU:
1. `!pip install -e ".[train,kie]"` and `mlflow.set_tracking_uri(...)`.
2. `from datasets import load_dataset; ds = load_dataset("theatticusproject/cuad-qa")`.
3. Tokenize with `AutoTokenizer.from_pretrained(QaTrainingConfig().model_name)` using `doc_stride`/`max_seq_length` (standard SQuAD prep).
4. `from docintel.contracts.train_qa import run_qa_training` → run; bundle saved + logged to MLflow.
5. Register the bundle as `cuad-extractor` (same `mlflow.register_model` step as the KIE notebook).

- [ ] **Step 2: Create `notebooks/cuad_onnx_export.ipynb`**

1. Download the registered `cuad-extractor` bundle (`download_registered_model`).
2. `from docintel.optimize.export import export_qa_to_onnx` → fp32 ONNX.
3. `from docintel.optimize.quantize import quantize_dynamic_int8` → INT8.
4. Run `cuad_eval` (F1/ANLS/AUPR) on clean text + `ocr_cer` on a rendered/OCR'd sample; log to MLflow.
5. Register the INT8 bundle as `cuad-extractor-onnx-int8`.

- [ ] **Step 3: Document the contract path in README**

Add a short subsection under the pipeline docs describing `POST /contracts/extract`, the dual-path ingest, the 41 clause types, and `DOCINTEL_CONTRACT_ONNX_LOCAL_PATH` (mirroring the existing KIE-model note).

- [ ] **Step 4: Full verification**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest`
Expected: all clean / PASS (slow tests deselected by default).

- [ ] **Step 5: Commit**

```bash
git add notebooks/cuad_finetune.ipynb notebooks/cuad_onnx_export.ipynb README.md
git commit -m "docs(contracts): add CUAD fine-tune + ONNX export notebooks and README"
```

---

## Self-Review

**Spec coverage check** (against the C1 design):
- Dual-path ingestion → Task 4. ✅
- Fine-tuned extractive-QA on 41 CUAD clauses → Tasks 3, 6, 10. ✅
- Sliding-window long-doc handling → Task 6 (`_encode` stride) + Task 5 (aggregation). ✅
- ONNX-INT8 serve, MLflow registry / local-path escape hatch → Tasks 6, 11, 13. ✅
- `ContractDocument` (clauses + derived) → Task 2. ✅
- New routes, no breaking change to `/extract` → Task 9. ✅
- Persistence (SQLite + MinIO PDF) → Tasks 7, 9. ✅
- Metrics: AUPR/F1/ANLS + CER → Task 12; serving metrics → Task 8. ✅
- MLflow experiment + registered names → config (Task 1) + notebooks (Task 13). ✅
- No LLM/vector/graph in C1 → respected. ✅

**Placeholder scan:** none — every code/test step contains complete content. The notebooks (Task 13) are orchestration over already-tested modules; cell intents are explicit.

**Type consistency:** `ExtractedClause`/`ContractDocument` fields are used identically across Tasks 2, 5, 6, 7, 8, 9. `WindowSpan` produced in Task 5 is consumed in Task 6. `resolve_contract_bundle`/`CuadQaOnnxExtractor.load` signatures match Task 9's dependency. `save_qa_bundle` writes `tokenizer/` which Task 6's `load()` reads via `settings.contract_model_name` (tokenizer reload from base name — consistent).
