# Contract Intelligence C2 — Vector RAG + `/ask` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make extracted contracts queryable in natural language — index each contract's text + clauses into Qdrant at extract time, and serve a `POST /ask` endpoint that retrieves cited chunks and (when an LLM is reachable) generates a grounded answer, degrading to cited chunks otherwise.

**Architecture:** A new `src/docintel/rag/` package mirrors the existing `contracts/` serving style: hand-rolled pure chunking + a fastembed (ONNX, no torch) embedder behind LangChain's `Embeddings` interface, Qdrant via `langchain-qdrant`, generation via `ChatOpenAI` (ngrok now / OpenAI later), and a retrieve→generate-or-degrade orchestrator. Indexing is folded best-effort into `POST /contracts/extract`; `/ask` is a new route. Heavy libraries are imported inside functions so modules load cheaply on the laptop.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, fastembed, langchain-core, langchain-openai, langchain-qdrant, qdrant-client, Qdrant, pytest, ruff, mypy.

## Global Constraints

- **Python 3.12+**, full type hints on every `def`; `from __future__ import annotations` at the top of each module.
- **Functional over classes**; keep functions small and focused. (The one class is `FastEmbedEmbeddings`, required to satisfy LangChain's `Embeddings` interface — mirrors the existing `CuadQaOnnxExtractor` class exception.)
- **No hardcoded constants** — every knob lives in `Settings` (env prefix `DOCINTEL_`). The grounding prompt text and the point-id UUID namespace are reference *data* (module-level), not tunable constants.
- **Heavy libraries imported inside functions** (fastembed, qdrant_client, langchain_qdrant, langchain_openai) so modules load cheaply on the laptop. `langchain_core` (prompts, output parsers, `Embeddings`) is light and may be imported at module top.
- **Vector store params via `Any`** in signatures (avoid importing the heavy/untyped `QdrantVectorStore` at module top), matching the repo's use of `Any` for the ONNX session/tokenizer in `contracts/extractor.py`.
- **Minimal changes**: do not modify the receipt `/extract` path or the C1 clause-extraction logic; the `ContractDocument` response shape is unchanged. Match existing style.
- **Lint/type/test** must pass: `uv run ruff check . && uv run ruff format --check .`, `uv run mypy src`, `uv run pytest`.
- **All commands run from `docintel/`.** Run `uv sync --all-extras` before starting (the whole suite needs all extras).
- **Graceful degrade is mandatory:** when `llm_base_url` is unset or a configured LLM call fails, `/ask` returns 200 with `answer=null`, `generation_skipped=true`, and populated `citations`. The CPU path never hard-depends on any LLM being up.

---

### Task 1: RAG settings + dependencies

**Files:**
- Modify: `src/docintel/config.py` (add RAG knobs to `Settings`, after `contract_max_upload_mb`)
- Modify: `pyproject.toml` (add `rag` extra; extend mypy `ignore_missing_imports` module list)
- Test: `tests/test_config.py` (append two tests)

**Interfaces:**
- Produces: new `Settings` fields used by every later task — `rag_embedding_model: str`, `rag_embedding_dim: int`, `rag_chunk_size: int`, `rag_chunk_overlap: int`, `qdrant_url: str`, `qdrant_collection: str`, `rag_top_k: int`, `llm_base_url: str | None`, `llm_api_key: str | None`, `llm_model: str`, `llm_timeout_s: float`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
def test_rag_settings_defaults() -> None:
    s = Settings()
    assert s.rag_embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.rag_embedding_dim == 384
    assert s.rag_chunk_size == 1200
    assert s.rag_chunk_overlap == 200
    assert s.qdrant_url == "http://qdrant:6333"
    assert s.qdrant_collection == "contract_chunks"
    assert s.rag_top_k == 5
    assert s.llm_base_url is None
    assert s.llm_api_key is None
    assert s.llm_model == "Qwen/Qwen2.5-7B-Instruct"
    assert s.llm_timeout_s == 60.0


def test_rag_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCINTEL_LLM_BASE_URL", "http://ngrok/v1")
    monkeypatch.setenv("DOCINTEL_QDRANT_URL", "http://localhost:6333")
    s = Settings()
    assert s.llm_base_url == "http://ngrok/v1"
    assert s.qdrant_url == "http://localhost:6333"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_rag_settings_defaults -v`
Expected: FAIL with `AttributeError` / assertion on missing `rag_embedding_model`.

- [ ] **Step 3: Add the settings**

```python
# in src/docintel/config.py, immediately after `contract_max_upload_mb: float = 25.0`
    # RAG / Vector retrieval (C2)
    rag_embedding_model: str = "BAAI/bge-small-en-v1.5"
    rag_embedding_dim: int = 384
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "contract_chunks"
    rag_top_k: int = 5
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_timeout_s: float = 60.0
```

- [ ] **Step 4: Add the `rag` extra and mypy overrides to `pyproject.toml`**

```toml
# add to [project.optional-dependencies], after the `contracts` extra
rag = [
    "fastembed>=0.3",
    "langchain-core>=0.3",
    "langchain-openai>=0.2",
    "langchain-qdrant>=0.2",
    "qdrant-client>=1.11",
]
```

```toml
# replace the existing [[tool.mypy.overrides]] module line, appending the new modules:
module = ["datasets.*", "huggingface_hub.*", "cv2.*", "doctr.*", "PIL.*", "seqeval.*", "mlflow.*", "optimum.*", "onnx.*", "onnxruntime.*", "matplotlib.*", "boto3.*", "botocore.*", "transformers.*", "fitz.*", "sklearn.*", "torch.*", "fastembed.*", "qdrant_client.*", "langchain.*", "langchain_core.*", "langchain_openai.*", "langchain_qdrant.*"]
```

- [ ] **Step 5: Install and verify**

Run: `uv sync --all-extras && uv run pytest tests/test_config.py -q`
Expected: PASS (all config tests).

- [ ] **Step 6: Commit**

```bash
git add src/docintel/config.py pyproject.toml uv.lock tests/test_config.py
git commit -m "feat(rag): add C2 RAG settings and rag dependency extra"
```

---

### Task 2: Chunking (`rag/chunk.py`)

**Files:**
- Create: `src/docintel/rag/chunk.py`
- Test: `tests/test_rag_chunk.py`

**Interfaces:**
- Consumes: `docintel.contracts.schema.ExtractedClause` (`clause_type: str`, `answer_text: str`, `char_start: int`, `char_end: int`, `confidence: float`).
- Produces: `TextChunk` (frozen dataclass: `text: str`, `char_start: int`, `char_end: int`, `chunk_index: int`, `chunk_kind: Literal["clause", "paragraph"]`, `clause_type: str | None`); `build_chunks(text: str, clauses: list[ExtractedClause], size: int, overlap: int) -> list[TextChunk]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_chunk.py
from __future__ import annotations

from docintel.contracts.schema import ExtractedClause
from docintel.rag.chunk import TextChunk, build_chunks


def _clause() -> ExtractedClause:
    return ExtractedClause(
        clause_type="Governing Law", answer_text="New York", char_start=0, char_end=8, confidence=0.9
    )


def test_clause_chunks_come_first_and_carry_type() -> None:
    chunks = build_chunks("New York law applies here.", [_clause()], size=10, overlap=2)
    assert chunks[0] == TextChunk(
        text="New York", char_start=0, char_end=8, chunk_index=0,
        chunk_kind="clause", clause_type="Governing Law",
    )


def test_paragraph_chunks_cover_text_with_unique_indices() -> None:
    chunks = build_chunks("abcdefghij", [], size=4, overlap=1)
    para = [c for c in chunks if c.chunk_kind == "paragraph"]
    assert [c.text for c in para] == ["abcd", "defg", "ghij", "j"]
    assert [c.chunk_index for c in para] == [0, 1, 2, 3]
    assert para[0].clause_type is None


def test_indices_are_sequential_across_both_kinds() -> None:
    chunks = build_chunks("abcdefghij", [_clause()], size=4, overlap=1)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_kind == "clause"
    assert chunks[1].chunk_kind == "paragraph"


def test_blank_clause_is_skipped() -> None:
    blank = ExtractedClause(
        clause_type="X", answer_text="   ", char_start=0, char_end=3, confidence=0.1
    )
    chunks = build_chunks("abcd", [blank], size=4, overlap=1)
    assert all(c.chunk_kind == "paragraph" for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_chunk.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.rag.chunk`.

- [ ] **Step 3: Write the implementation**

```python
# src/docintel/rag/chunk.py
"""Pure chunking of a contract into clause chunks + sliding paragraph chunks.

Clause chunks (one per ExtractedClause) are precise and citation-ready; paragraph
chunks are overlapping char windows over the full ingested text, giving coverage
for questions outside the 41 clause types. No model or I/O — fully CPU-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from docintel.contracts.schema import ExtractedClause


@dataclass(frozen=True)
class TextChunk:
    """One indexable chunk with its char offsets into the ingested text."""

    text: str
    char_start: int
    char_end: int
    chunk_index: int
    chunk_kind: Literal["clause", "paragraph"]
    clause_type: str | None


def _paragraph_chunks(text: str, size: int, overlap: int, start_index: int) -> list[TextChunk]:
    """Overlapping char windows over ``text``, indexed from ``start_index``."""
    step = max(size - overlap, 1)
    chunks: list[TextChunk] = []
    index = start_index
    position = 0
    length = len(text)
    while position < length:
        window = text[position : position + size]
        if window.strip():
            chunks.append(
                TextChunk(
                    text=window,
                    char_start=position,
                    char_end=min(position + size, length),
                    chunk_index=index,
                    chunk_kind="paragraph",
                    clause_type=None,
                )
            )
            index += 1
        if position + size >= length:
            break
        position += step
    return chunks


def build_chunks(
    text: str, clauses: list[ExtractedClause], size: int, overlap: int
) -> list[TextChunk]:
    """Build clause chunks (first) then paragraph chunks, with unique sequential indices."""
    chunks: list[TextChunk] = []
    index = 0
    for clause in clauses:
        if not clause.answer_text.strip():
            continue
        chunks.append(
            TextChunk(
                text=clause.answer_text,
                char_start=clause.char_start,
                char_end=clause.char_end,
                chunk_index=index,
                chunk_kind="clause",
                clause_type=clause.clause_type,
            )
        )
        index += 1
    chunks.extend(_paragraph_chunks(text, size, overlap, index))
    return chunks
```

Also create the empty package marker note: `src/docintel/rag/__init__.py` already exists (a docstring stub) — leave it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_chunk.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/docintel/rag/chunk.py tests/test_rag_chunk.py
git commit -m "feat(rag): pure clause + paragraph chunking"
```

---

### Task 3: Response schema (`rag/schema.py`)

**Files:**
- Create: `src/docintel/rag/schema.py`
- Test: `tests/test_rag_schema.py`

**Interfaces:**
- Produces: `RetrievedChunk` (`contract_id: str`, `chunk_index: int`, `chunk_kind: str`, `clause_type: str | None`, `text: str`, `score: float`, `char_start: int`, `char_end: int`); `AskRequest` (`question: str`, `contract_id: str | None = None`, `top_k: int | None = None`); `AskResponse` (`question: str`, `answer: str | None`, `generation_skipped: bool`, `contract_id: str | None`, `citations: list[RetrievedChunk]`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_schema.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from docintel.rag.schema import AskRequest, AskResponse, RetrievedChunk


def test_ask_request_defaults() -> None:
    req = AskRequest(question="What is the governing law?")
    assert req.contract_id is None
    assert req.top_k is None


def test_ask_request_requires_question() -> None:
    with pytest.raises(ValidationError):
        AskRequest()  # type: ignore[call-arg]


def test_ask_response_roundtrip() -> None:
    chunk = RetrievedChunk(
        contract_id="c1", chunk_index=0, chunk_kind="clause", clause_type="Governing Law",
        text="New York", score=0.42, char_start=0, char_end=8,
    )
    resp = AskResponse(
        question="q", answer=None, generation_skipped=True, contract_id=None, citations=[chunk]
    )
    assert resp.model_dump()["citations"][0]["clause_type"] == "Governing Law"
    assert resp.answer is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.rag.schema`.

- [ ] **Step 3: Write the implementation**

```python
# src/docintel/rag/schema.py
"""Pydantic models for the /ask request/response and retrieved citations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One retrieved chunk used as a grounded citation."""

    contract_id: str
    chunk_index: int
    chunk_kind: str
    clause_type: str | None
    text: str
    score: float
    char_start: int
    char_end: int


class AskRequest(BaseModel):
    """A natural-language question, optionally scoped to one contract."""

    question: str
    contract_id: str | None = None
    top_k: int | None = None


class AskResponse(BaseModel):
    """The grounded answer (or null when degraded) plus its citations."""

    question: str
    answer: str | None
    generation_skipped: bool
    contract_id: str | None
    citations: list[RetrievedChunk] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_schema.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/docintel/rag/schema.py tests/test_rag_schema.py
git commit -m "feat(rag): /ask request/response and citation schema"
```

---

### Task 4: Embedder (`rag/embed.py`)

**Files:**
- Create: `src/docintel/rag/embed.py`
- Test: `tests/test_rag_embed.py`

**Interfaces:**
- Consumes: `docintel.config.Settings` (`rag_embedding_model`).
- Produces: `FastEmbedEmbeddings(model: Any)` implementing LangChain `Embeddings` (`embed_documents(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`); `build_embedder(settings: Settings) -> FastEmbedEmbeddings`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_embed.py
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from docintel.rag.embed import FastEmbedEmbeddings


class _FakeModel:
    def embed(self, texts: list[str]) -> list[Any]:
        return [np.array([float(len(t))] * 3) for t in texts]

    def query_embed(self, texts: list[str]) -> list[Any]:
        return [np.array([1.0, 2.0, 3.0]) for _ in texts]


def test_embed_documents_returns_lists() -> None:
    embedder = FastEmbedEmbeddings(_FakeModel())
    assert embedder.embed_documents(["ab", "cde"]) == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]


def test_embed_query_returns_single_vector() -> None:
    embedder = FastEmbedEmbeddings(_FakeModel())
    assert embedder.embed_query("x") == [1.0, 2.0, 3.0]


@pytest.mark.slow
def test_build_embedder_real_model_has_384_dims() -> None:
    from docintel.config import Settings
    from docintel.rag.embed import build_embedder

    embedder = build_embedder(Settings())
    vector = embedder.embed_query("termination for convenience")
    assert len(vector) == 384
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_embed.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: docintel.rag.embed`.

- [ ] **Step 3: Write the implementation**

```python
# src/docintel/rag/embed.py
"""bge-small-en-v1.5 embeddings via fastembed (ONNX, CPU, no torch).

Wrapped behind LangChain's ``Embeddings`` interface so the Qdrant store can use it.
The fastembed model is imported lazily so this module stays cheap to import.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from docintel.config import Settings


class FastEmbedEmbeddings(Embeddings):
    """Adapt a fastembed text-embedding model to the LangChain Embeddings API."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model.query_embed([text])))
        return list(vector.tolist())


def build_embedder(settings: Settings) -> FastEmbedEmbeddings:
    """Load the configured fastembed model and wrap it."""
    from fastembed import TextEmbedding

    return FastEmbedEmbeddings(TextEmbedding(model_name=settings.rag_embedding_model))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_embed.py -v -m "not slow"`
Expected: PASS (2 tests; the slow real-model test is deselected).

- [ ] **Step 5: Commit**

```bash
git add src/docintel/rag/embed.py tests/test_rag_embed.py
git commit -m "feat(rag): fastembed bge-small embedder behind LangChain Embeddings"
```

---

### Task 5: Vector store (`rag/store.py`)

**Files:**
- Create: `src/docintel/rag/store.py`
- Test: `tests/test_rag_store.py`

**Interfaces:**
- Consumes: `docintel.config.Settings` (`qdrant_url`, `qdrant_collection`, `rag_embedding_dim`); `docintel.rag.chunk.TextChunk`; `langchain_core.embeddings.Embeddings`.
- Produces: `chunk_point_id(contract_id: str, chunk_index: int) -> str`; `ensure_collection(client: Any, collection: str, dim: int) -> None`; `build_vector_store(settings: Settings, embedder: Embeddings, client: Any | None = None) -> Any` (a `QdrantVectorStore`); `upsert_chunks(store: Any, contract_id: str, chunks: list[TextChunk]) -> int`; `search(store: Any, query: str, top_k: int, contract_id: str | None = None) -> list[RetrievedChunk]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_store.py
from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from docintel.config import Settings
from docintel.rag.chunk import TextChunk
from docintel.rag.store import (
    build_vector_store,
    chunk_point_id,
    ensure_collection,
    search,
    upsert_chunks,
)

_DIM = 8


@pytest.fixture
def store() -> object:
    settings = Settings(rag_embedding_dim=_DIM, qdrant_collection="contracts_test")
    client = QdrantClient(location=":memory:")
    vector_store = build_vector_store(settings, DeterministicFakeEmbedding(size=_DIM), client=client)
    ensure_collection(client, settings.qdrant_collection, _DIM)
    return vector_store


def _chunks() -> list[TextChunk]:
    return [
        TextChunk("New York law", 0, 12, 0, "clause", "Governing Law"),
        TextChunk("some paragraph body", 0, 19, 1, "paragraph", None),
    ]


def test_point_id_is_deterministic() -> None:
    assert chunk_point_id("c1", 0) == chunk_point_id("c1", 0)
    assert chunk_point_id("c1", 0) != chunk_point_id("c1", 1)


def test_upsert_then_search_filters_by_contract(store: object) -> None:
    assert upsert_chunks(store, "c1", _chunks()) == 2
    hits = search(store, "law", top_k=5, contract_id="c1")
    assert {h.contract_id for h in hits} == {"c1"}
    assert {h.chunk_kind for h in hits} == {"clause", "paragraph"}
    assert search(store, "law", top_k=5, contract_id="c2") == []


def test_reupsert_is_idempotent(store: object) -> None:
    upsert_chunks(store, "c1", _chunks())
    upsert_chunks(store, "c1", _chunks())
    assert store.client.count("contracts_test").count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_store.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.rag.store`.

- [ ] **Step 3: Write the implementation**

```python
# src/docintel/rag/store.py
"""Qdrant vector store wiring via langchain-qdrant.

Deterministic point ids make re-indexing a contract idempotent. ``build_vector_store``
does no network I/O (so /ask can construct it without connecting); ``ensure_collection``
and ``search``/``upsert_chunks`` are the network calls. Heavy imports live in functions.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.embeddings import Embeddings

from docintel.config import Settings
from docintel.rag.chunk import TextChunk
from docintel.rag.schema import RetrievedChunk

# Fixed namespace for deterministic chunk point ids (reference data, not a tunable knob).
_POINT_NAMESPACE = uuid.UUID("6f1a2b3c-0000-4000-8000-000000000000")


def chunk_point_id(contract_id: str, chunk_index: int) -> str:
    """Stable UUID for a (contract, chunk) so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{contract_id}:{chunk_index}"))


def ensure_collection(client: Any, collection: str, dim: int) -> None:
    """Create the cosine collection if it does not yet exist."""
    from qdrant_client.http.models import Distance, VectorParams

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def build_vector_store(settings: Settings, embedder: Embeddings, client: Any | None = None) -> Any:
    """Construct a QdrantVectorStore (no network). Pass ``client`` for tests (``:memory:``)."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    if client is None:
        client = QdrantClient(url=settings.qdrant_url)
    return QdrantVectorStore(
        client=client, collection_name=settings.qdrant_collection, embedding=embedder
    )


def upsert_chunks(store: Any, contract_id: str, chunks: list[TextChunk]) -> int:
    """Embed and upsert chunks with deterministic ids; returns the number upserted."""
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    metadatas = [
        {
            "contract_id": contract_id,
            "chunk_index": c.chunk_index,
            "chunk_kind": c.chunk_kind,
            "clause_type": c.clause_type,
            "char_start": c.char_start,
            "char_end": c.char_end,
        }
        for c in chunks
    ]
    ids = [chunk_point_id(contract_id, c.chunk_index) for c in chunks]
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(chunks)


def search(
    store: Any, query: str, top_k: int, contract_id: str | None = None
) -> list[RetrievedChunk]:
    """Top-k similarity search, optionally filtered to one contract."""
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    query_filter = None
    if contract_id is not None:
        query_filter = Filter(
            must=[FieldCondition(key="metadata.contract_id", match=MatchValue(value=contract_id))]
        )
    results = store.similarity_search_with_score(query, k=top_k, filter=query_filter)
    chunks: list[RetrievedChunk] = []
    for document, score in results:
        meta = document.metadata
        chunks.append(
            RetrievedChunk(
                contract_id=meta["contract_id"],
                chunk_index=meta["chunk_index"],
                chunk_kind=meta["chunk_kind"],
                clause_type=meta.get("clause_type"),
                text=document.page_content,
                score=float(score),
                char_start=meta["char_start"],
                char_end=meta["char_end"],
            )
        )
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/docintel/rag/store.py tests/test_rag_store.py
git commit -m "feat(rag): Qdrant store — idempotent upsert and filtered search"
```

---

### Task 6: Index orchestration (`rag/index.py`)

**Files:**
- Create: `src/docintel/rag/index.py`
- Test: `tests/test_rag_index.py`

**Interfaces:**
- Consumes: `Settings` (`rag_chunk_size`, `rag_chunk_overlap`, `qdrant_collection`, `rag_embedding_dim`); `build_chunks`; `ensure_collection`, `upsert_chunks`; `ExtractedClause`.
- Produces: `index_contract(contract_id: str, text: str, clauses: list[ExtractedClause], store: Any, settings: Settings) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_index.py
from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from docintel.config import Settings
from docintel.contracts.schema import ExtractedClause
from docintel.rag.index import index_contract
from docintel.rag.store import build_vector_store, search

_DIM = 8


def test_index_contract_indexes_clause_and_paragraph_chunks() -> None:
    settings = Settings(
        rag_embedding_dim=_DIM, qdrant_collection="t", rag_chunk_size=8, rag_chunk_overlap=2
    )
    store = build_vector_store(
        settings, DeterministicFakeEmbedding(size=_DIM), client=QdrantClient(location=":memory:")
    )
    clauses = [
        ExtractedClause(
            clause_type="Governing Law", answer_text="New York",
            char_start=0, char_end=8, confidence=0.9,
        )
    ]
    count = index_contract("c1", "New York law applies here.", clauses, store, settings)
    assert count >= 2  # one clause chunk + paragraph chunks
    hits = search(store, "law", top_k=10, contract_id="c1")
    assert any(h.chunk_kind == "clause" for h in hits)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_index.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.rag.index`.

- [ ] **Step 3: Write the implementation**

```python
# src/docintel/rag/index.py
"""Index one contract's text + clauses into Qdrant (called at extract time)."""

from __future__ import annotations

from typing import Any

from docintel.config import Settings
from docintel.contracts.schema import ExtractedClause
from docintel.rag.chunk import build_chunks
from docintel.rag.store import ensure_collection, upsert_chunks


def index_contract(
    contract_id: str,
    text: str,
    clauses: list[ExtractedClause],
    store: Any,
    settings: Settings,
) -> int:
    """Chunk, embed, and upsert one contract; returns the number of chunks indexed."""
    ensure_collection(store.client, settings.qdrant_collection, settings.rag_embedding_dim)
    chunks = build_chunks(text, clauses, settings.rag_chunk_size, settings.rag_chunk_overlap)
    return upsert_chunks(store, contract_id, chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_index.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/docintel/rag/index.py tests/test_rag_index.py
git commit -m "feat(rag): index_contract orchestration (chunk -> embed -> upsert)"
```

---

### Task 7: LLM client + prompt (`rag/llm.py`)

**Files:**
- Create: `src/docintel/rag/llm.py`
- Test: `tests/test_rag_llm.py`

**Interfaces:**
- Consumes: `Settings` (`llm_base_url`, `llm_api_key`, `llm_model`, `llm_timeout_s`); `RetrievedChunk`.
- Produces: `build_llm(settings: Settings) -> Any | None` (a `ChatOpenAI` or `None`); `build_prompt() -> Any` (a `ChatPromptTemplate`); `format_context(chunks: list[RetrievedChunk]) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_llm.py
from __future__ import annotations

from docintel.config import Settings
from docintel.rag.llm import build_llm, format_context
from docintel.rag.schema import RetrievedChunk


def test_build_llm_is_none_when_unconfigured() -> None:
    assert build_llm(Settings(llm_base_url=None)) is None


def test_build_llm_when_configured() -> None:
    llm = build_llm(Settings(llm_base_url="http://ngrok/v1", llm_api_key="k", llm_model="m"))
    assert llm is not None
    assert llm.model_name == "m"


def test_format_context_numbers_and_labels_chunks() -> None:
    chunks = [
        RetrievedChunk(
            contract_id="c1", chunk_index=0, chunk_kind="clause", clause_type="Governing Law",
            text="New York", score=0.9, char_start=0, char_end=8,
        ),
        RetrievedChunk(
            contract_id="c1", chunk_index=1, chunk_kind="paragraph", clause_type=None,
            text="misc text", score=0.5, char_start=10, char_end=19,
        ),
    ]
    out = format_context(chunks)
    assert "[1] (Governing Law, contract c1): New York" in out
    assert "[2] (Excerpt, contract c1): misc text" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.rag.llm`.

- [ ] **Step 3: Write the implementation**

```python
# src/docintel/rag/llm.py
"""LLM client (OpenAI-compatible) + grounded prompt for /ask.

``build_llm`` returns ``None`` when no endpoint is configured, which drives the
graceful-degrade path. The same ``ChatOpenAI`` points at the Colab/ngrok server now
and the OpenAI API later — only settings change. ``langchain_openai`` is imported lazily.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from docintel.config import Settings
from docintel.rag.schema import RetrievedChunk

# Grounding instruction (reference text, not a tunable knob).
RAG_SYSTEM_PROMPT = (
    "You are a contract analysis assistant. Answer the question using ONLY the provided "
    "context. Cite the clause types you relied on. If the context does not contain the "
    "answer, say you do not have enough information."
)


def build_llm(settings: Settings) -> Any | None:
    """Return an OpenAI-compatible chat model, or None when no endpoint is configured."""
    if not settings.llm_base_url:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "EMPTY",
        model=settings.llm_model,
        timeout=settings.llm_timeout_s,
        temperature=0,
    )


def build_prompt() -> ChatPromptTemplate:
    """Grounded chat prompt with {question} and {context} slots."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, labelled context block."""
    lines = []
    for position, chunk in enumerate(chunks, start=1):
        label = chunk.clause_type or "Excerpt"
        lines.append(f"[{position}] ({label}, contract {chunk.contract_id}): {chunk.text}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_llm.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/docintel/rag/llm.py tests/test_rag_llm.py
git commit -m "feat(rag): OpenAI-compatible LLM client and grounded prompt"
```

---

### Task 8: Answer orchestration (`rag/answer.py`)

**Files:**
- Create: `src/docintel/rag/answer.py`
- Test: `tests/test_rag_answer.py`

**Interfaces:**
- Consumes: `Settings` (`rag_top_k`); `search`; `build_prompt`, `format_context`; `AskResponse`, `RetrievedChunk`.
- Produces: `generate_answer(llm: Any, question: str, context: str) -> str`; `answer_question(question: str, store: Any, llm: Any | None, settings: Settings, contract_id: str | None = None, top_k: int | None = None) -> AskResponse`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_answer.py
from __future__ import annotations

from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient

from docintel.config import Settings
from docintel.contracts.schema import ExtractedClause
from docintel.rag.answer import answer_question
from docintel.rag.index import index_contract
from docintel.rag.store import build_vector_store

_DIM = 8


def _seeded_store(settings: Settings) -> Any:
    store = build_vector_store(
        settings, DeterministicFakeEmbedding(size=_DIM), client=QdrantClient(location=":memory:")
    )
    clauses = [
        ExtractedClause(
            clause_type="Governing Law", answer_text="New York",
            char_start=0, char_end=8, confidence=0.9,
        )
    ]
    index_contract("c1", "New York law applies here.", clauses, store, settings)
    return store


def _settings() -> Settings:
    return Settings(
        rag_embedding_dim=_DIM, qdrant_collection="t", rag_chunk_size=8,
        rag_chunk_overlap=2, rag_top_k=5,
    )


def test_generate_path_returns_answer_and_citations() -> None:
    settings = _settings()
    store = _seeded_store(settings)
    llm = FakeListChatModel(responses=["Governed by New York law."])
    resp = answer_question("governing law?", store, llm, settings, contract_id="c1")
    assert resp.answer == "Governed by New York law."
    assert resp.generation_skipped is False
    assert len(resp.citations) >= 1


def test_degrade_when_no_llm() -> None:
    settings = _settings()
    store = _seeded_store(settings)
    resp = answer_question("governing law?", store, None, settings, contract_id="c1")
    assert resp.answer is None
    assert resp.generation_skipped is True
    assert len(resp.citations) >= 1


def test_degrade_when_llm_errors() -> None:
    settings = _settings()
    store = _seeded_store(settings)

    def _boom(_: Any) -> Any:
        raise RuntimeError("llm down")

    resp = answer_question("governing law?", store, RunnableLambda(_boom), settings, contract_id="c1")
    assert resp.answer is None
    assert resp.generation_skipped is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_answer.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.rag.answer`.

- [ ] **Step 3: Write the implementation**

```python
# src/docintel/rag/answer.py
"""Retrieve-then-(generate-or-degrade) orchestration for /ask."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser

from docintel.config import Settings
from docintel.rag.llm import build_prompt, format_context
from docintel.rag.schema import AskResponse
from docintel.rag.store import search


def generate_answer(llm: Any, question: str, context: str) -> str:
    """Run the LCEL chain prompt | llm | parser and return the answer text."""
    chain = build_prompt() | llm | StrOutputParser()
    return str(chain.invoke({"question": question, "context": context}))


def answer_question(
    question: str,
    store: Any,
    llm: Any | None,
    settings: Settings,
    contract_id: str | None = None,
    top_k: int | None = None,
) -> AskResponse:
    """Retrieve top-k chunks, then generate a grounded answer or degrade to citations."""
    citations = search(store, question, top_k or settings.rag_top_k, contract_id)
    if llm is None:
        return AskResponse(
            question=question, answer=None, generation_skipped=True,
            contract_id=contract_id, citations=citations,
        )
    try:
        answer = generate_answer(llm, question, format_context(citations))
    except Exception:
        return AskResponse(
            question=question, answer=None, generation_skipped=True,
            contract_id=contract_id, citations=citations,
        )
    return AskResponse(
        question=question, answer=answer, generation_skipped=False,
        contract_id=contract_id, citations=citations,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_answer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/docintel/rag/answer.py tests/test_rag_answer.py
git commit -m "feat(rag): answer orchestration with graceful degrade"
```

---

### Task 9: `/ask` route + app wiring

**Files:**
- Create: `src/docintel/api/routes/ask.py`
- Modify: `src/docintel/api/main.py` (register router; init `app.state.rag_store` / `rag_llm`)
- Test: `tests/test_rag_routes.py`

**Interfaces:**
- Consumes: `answer_question`; `build_embedder`, `build_vector_store`, `build_llm`; `AskRequest`, `AskResponse`; `get_settings`.
- Produces: `ensure_rag_store(app: Any, settings: Settings) -> Any`; `get_rag_store(request: Request, settings: Settings) -> Any` (raises on store-build failure); `get_rag_store_optional(request: Request, settings: Settings) -> Any | None` (never raises); `get_rag_llm(request: Request, settings: Settings) -> Any | None`; `POST /ask`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_routes.py
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient

from docintel.api.main import create_app
from docintel.api.routes.ask import get_rag_llm, get_rag_store
from docintel.config import Settings, get_settings
from docintel.contracts.schema import ExtractedClause
from docintel.rag.index import index_contract
from docintel.rag.store import build_vector_store

_DIM = 8


def _settings() -> Settings:
    return Settings(rag_embedding_dim=_DIM, qdrant_collection="t", rag_chunk_size=8, rag_chunk_overlap=2)


def _seeded_store(settings: Settings) -> Any:
    store = build_vector_store(
        settings, DeterministicFakeEmbedding(size=_DIM), client=QdrantClient(location=":memory:")
    )
    index_contract(
        "c1", "New York law applies here.",
        [ExtractedClause(clause_type="Governing Law", answer_text="New York", char_start=0, char_end=8, confidence=0.9)],
        store, settings,
    )
    return store


def test_ask_degrades_without_llm() -> None:
    settings = _settings()
    store = _seeded_store(settings)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rag_store] = lambda: store
    app.dependency_overrides[get_rag_llm] = lambda: None
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "governing law?", "contract_id": "c1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] is None
    assert body["generation_skipped"] is True
    assert len(body["citations"]) >= 1


def test_ask_generates_with_llm() -> None:
    settings = _settings()
    store = _seeded_store(settings)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rag_store] = lambda: store
    app.dependency_overrides[get_rag_llm] = lambda: FakeListChatModel(responses=["NY law."])
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "governing law?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "NY law."


def test_ask_returns_503_when_store_unavailable() -> None:
    class _BoomStore:
        def similarity_search_with_score(self, *a: Any, **k: Any) -> Any:
            raise ConnectionError("qdrant down")

    settings = _settings()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rag_store] = lambda: _BoomStore()
    app.dependency_overrides[get_rag_llm] = lambda: None
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "x"})
    assert resp.status_code == 503
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_routes.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.api.routes.ask`.

- [ ] **Step 3: Write the route**

```python
# src/docintel/api/routes/ask.py
"""The /ask endpoint: retrieve cited chunks and generate-or-degrade an answer."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from docintel.config import Settings, get_settings
from docintel.rag.answer import answer_question
from docintel.rag.embed import build_embedder
from docintel.rag.llm import build_llm
from docintel.rag.schema import AskRequest, AskResponse
from docintel.rag.store import build_vector_store

logger = logging.getLogger("docintel.api.ask")
router = APIRouter(tags=["rag"])


def ensure_rag_store(app: Any, settings: Settings) -> Any:
    """Build the vector store once and cache it on app.state (no network at build time)."""
    store = getattr(app.state, "rag_store", None)
    if store is None:
        store = build_vector_store(settings, build_embedder(settings))
        app.state.rag_store = store
    return store


def get_rag_store(request: Request, settings: Settings = Depends(get_settings)) -> Any:  # noqa: B008
    """Vector store dependency for /ask (propagates build failures)."""
    return ensure_rag_store(request.app, settings)


def get_rag_store_optional(
    request: Request, settings: Settings = Depends(get_settings)  # noqa: B008
) -> Any | None:
    """Best-effort vector store for indexing; returns None instead of raising."""
    try:
        return ensure_rag_store(request.app, settings)
    except Exception:
        logger.warning("rag.store.unavailable", exc_info=True)
        return None


def get_rag_llm(request: Request, settings: Settings = Depends(get_settings)) -> Any | None:  # noqa: B008
    """LLM dependency: cached when configured, else None (drives graceful degrade)."""
    llm = getattr(request.app.state, "rag_llm", None)
    if llm is None:
        llm = build_llm(settings)
        request.app.state.rag_llm = llm
    return llm


@router.post("/ask", response_model=AskResponse, summary="Ask a question grounded in indexed contracts")
def ask(
    req: AskRequest,
    settings: Settings = Depends(get_settings),  # noqa: B008
    store: Any = Depends(get_rag_store),  # noqa: B008
    llm: Any | None = Depends(get_rag_llm),  # noqa: B008
) -> AskResponse:
    """Retrieve cited chunks and answer; degrade to citations when no LLM is reachable."""
    try:
        return answer_question(req.question, store, llm, settings, req.contract_id, req.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store unavailable."
        ) from exc
```

- [ ] **Step 4: Wire the router and app.state in `main.py`**

```python
# in src/docintel/api/main.py: add ask to the routes import
from docintel.api.routes import ask, contracts, documents, extract, health
```

```python
# in lifespan(), alongside the other app.state initialisers:
    app.state.rag_store = None
    app.state.rag_llm = None
```

```python
# in create_app(), after app.include_router(contracts.router):
    app.include_router(ask.router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_routes.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/docintel/api/routes/ask.py src/docintel/api/main.py tests/test_rag_routes.py
git commit -m "feat(rag): POST /ask route with degrade and 503-on-store-failure"
```

---

### Task 10: Extract-time best-effort indexing

**Files:**
- Modify: `src/docintel/api/routes/contracts.py` (index after persist; add `rag_store` optional dependency + log field)
- Modify: `tests/test_contracts_routes.py` (override `get_rag_store_optional` in the fixture; add a best-effort test)

**Interfaces:**
- Consumes: `get_rag_store_optional` and `ensure_rag_store` from `ask.py`; `index_contract`.
- Produces: no new public interface; `POST /contracts/extract` now also indexes (best-effort).

- [ ] **Step 1: Write the failing test (and update the fixture)**

```python
# in tests/test_contracts_routes.py — add to imports:
from docintel.api.routes.ask import get_rag_store_optional

# in the `client` fixture, add this override before `with TestClient(app) as c:`
    app.dependency_overrides[get_rag_store_optional] = lambda: None
```

```python
# append a new test to tests/test_contracts_routes.py
def test_extract_succeeds_when_indexing_fails(client: TestClient, monkeypatch: Any) -> None:
    from docintel.contracts.ingest import IngestedDoc

    monkeypatch.setattr(
        "docintel.api.routes.contracts.ingest_pdf",
        lambda data, ocr_engine, settings: IngestedDoc(
            text="Acme and Globex", page_count=1, source="digital"
        ),
    )

    def _boom(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("qdrant down")

    monkeypatch.setattr("docintel.api.routes.contracts.index_contract", _boom)

    resp = client.post(
        "/contracts/extract", files={"file": ("c.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert resp.status_code == 200  # indexing failure must not break extraction
```

Note: this test passes a non-None store via the fixture override? The fixture overrides
`get_rag_store_optional` to `None`, which would skip indexing entirely. For *this* test we
need a non-None store so `index_contract` is reached and raises. Override it locally:

```python
# at the top of test_extract_succeeds_when_indexing_fails, before the post:
    client.app.dependency_overrides[get_rag_store_optional] = lambda: object()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_routes.py::test_extract_succeeds_when_indexing_fails -v`
Expected: FAIL (currently `index_contract` is not imported/called in `contracts.py`, so the monkeypatch target does not exist → `AttributeError`).

- [ ] **Step 3: Add best-effort indexing to `contracts.py`**

```python
# add imports near the existing contracts.py imports
from typing import Any  # already imported; keep single import

from docintel.api.routes.ask import get_rag_store_optional
from docintel.rag.index import index_contract
```

```python
# add a parameter to extract_contract's signature (after `metrics`):
    rag_store: Any = Depends(get_rag_store_optional),  # noqa: B008
```

```python
# after the existing `save_contract(settings.sqlite_path, doc, pdf_key)` line:
    chunks_indexed = 0
    if rag_store is not None:
        try:
            chunks_indexed = index_contract(
                doc.id, ingested.text, doc.clauses, rag_store, settings
            )
        except Exception:
            logger.warning("contracts.extract.index_failed", extra={"contract_id": doc.id}, exc_info=True)
```

```python
# add chunks_indexed to the existing completion-log `extra` dict:
            "chunks_indexed": chunks_indexed,
```

- [ ] **Step 4: Run the contracts route tests**

Run: `uv run pytest tests/test_contracts_routes.py -v`
Expected: PASS (existing tests + the new best-effort test).

- [ ] **Step 5: Commit**

```bash
git add src/docintel/api/routes/contracts.py tests/test_contracts_routes.py
git commit -m "feat(rag): index contracts at extract time (best-effort)"
```

---

### Task 11: Qdrant Compose service + env example

**Files:**
- Modify: `docker-compose.yml` (add `qdrant` service; add env + depends_on to `api`)
- Modify: `tests/test_compose.py` (add `qdrant` to `EXPECTED_SERVICES`; fix the stale comment)
- Modify: `.env.example` (add the new DOCINTEL_ keys)

**Interfaces:**
- Consumes: nothing new.
- Produces: a runnable Qdrant service for non-test serving.

- [ ] **Step 1: Update the compose guard test**

```python
# in tests/test_compose.py — replace the comment + EXPECTED_SERVICES set
# Core MLOps spine, the Phase 5 observability stack, and the C2 Qdrant vector store.
# The Streamlit UI runs locally (not containerised).
EXPECTED_SERVICES = {
    "api",
    "mlflow",
    "minio",
    "prometheus",
    "loki",
    "promtail",
    "grafana",
    "qdrant",
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_compose.py -v`
Expected: FAIL — `qdrant` expected but not in `docker-compose.yml`.

- [ ] **Step 3: Add the qdrant service and api wiring to `docker-compose.yml`**

```yaml
# add under `services:` (e.g. after the minio service)
  qdrant:
    image: qdrant/qdrant:v1.11.0
    container_name: docintel-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant-data:/qdrant/storage
```

```yaml
# in the `api` service `environment:` block, add:
      DOCINTEL_QDRANT_URL: http://qdrant:6333
```

```yaml
# in the `api` service `depends_on:` list, add:
      - qdrant
```

```yaml
# in the top-level `volumes:` block, add:
  qdrant-data:
```

- [ ] **Step 4: Add the new keys to `.env.example`**

```bash
# append to .env.example (keys must match Settings field names, DOCINTEL_ prefix)
DOCINTEL_QDRANT_URL=http://localhost:6333
DOCINTEL_QDRANT_COLLECTION=contract_chunks
DOCINTEL_RAG_TOP_K=5
DOCINTEL_LLM_BASE_URL=
DOCINTEL_LLM_API_KEY=
DOCINTEL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

- [ ] **Step 5: Run the guard tests**

Run: `uv run pytest tests/test_compose.py tests/test_env_example.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml tests/test_compose.py .env.example
git commit -m "feat(rag): add Qdrant compose service and env example keys"
```

---

### Task 12: Retrieval evaluation harness (`rag/eval.py`)

**Files:**
- Create: `src/docintel/rag/eval.py`
- Test: `tests/test_rag_eval.py`

**Scope note:** This task delivers the *retrieval* metrics (recall@k, MRR) that are pure and CPU-testable now. RAGAS faithfulness/answer-relevancy require a live LLM judge and are run later from the C2 eval notebook (with the Colab/ngrok LLM), exactly as C1's training runs on Colab — out of this automated plan's scope. The `eval` extra is added here for that notebook.

**Files (additional):**
- Modify: `pyproject.toml` (add the `eval` extra)

**Interfaces:**
- Produces: `recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float`; `mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_eval.py
from __future__ import annotations

from docintel.rag.eval import mrr, recall_at_k


def test_recall_at_k_counts_relevant_in_top_k() -> None:
    assert recall_at_k(["a", "b", "c", "d"], {"b", "z"}, k=2) == 0.5  # b found, z not
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0
    assert recall_at_k([], {"a"}, k=3) == 0.0


def test_mrr_uses_first_relevant_rank() -> None:
    assert mrr(["a", "b", "c"], {"b"}) == 0.5  # rank 2
    assert mrr(["a", "b"], {"a"}) == 1.0
    assert mrr(["a", "b"], {"z"}) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: docintel.rag.eval`.

- [ ] **Step 3: Write the implementation**

```python
# src/docintel/rag/eval.py
"""Pure retrieval-quality metrics for the C2 vector index (logged to MLflow by the
eval notebook). RAGAS answer-quality metrics need a live LLM judge and run separately.
"""

from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant ids found within the top-k retrieved ids."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant id (0.0 if none retrieved)."""
    for rank, identifier in enumerate(retrieved_ids, start=1):
        if identifier in relevant_ids:
            return 1.0 / rank
    return 0.0
```

- [ ] **Step 4: Add the `eval` extra to `pyproject.toml`**

```toml
# add to [project.optional-dependencies], after the `rag` extra
eval = [
    "ragas>=0.2",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_eval.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/docintel/rag/eval.py pyproject.toml tests/test_rag_eval.py
git commit -m "feat(rag): retrieval recall@k and MRR metrics; add eval extra"
```

---

### Final verification

- [ ] **Step 1: Full lint, type, and test sweep**

Run:
```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```
Expected: ruff clean; mypy `Success`; pytest all pass (slow tests deselected by default).

- [ ] **Step 2: Note any mypy `no-untyped-call` follow-ups**

If mypy reports `no-untyped-call` on a LangChain/qdrant call (as happened with `get_last_checkpoint` in C1), append a narrow `# type: ignore[no-untyped-call]` to that exact call line and re-run `uv run mypy src`.

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-06-25-contract-intelligence-c2-vector-rag-design.md`):
- Index at extract time, reuse C1 ingest → Task 10 (+ Task 6). ✅
- Best-effort indexing → Task 10. ✅
- Single `/ask`, optional `contract_id` filter → Tasks 5 (filter), 8, 9. ✅
- Clause + paragraph chunks → Task 2. ✅
- bge-small-en-v1.5 via fastembed (no torch) → Task 4. ✅
- Qdrant via langchain-qdrant; `:memory:` for tests → Task 5. ✅
- LangChain hybrid (LCEL chain, ChatOpenAI) → Tasks 7, 8. ✅
- Graceful degrade (answer=null, generation_skipped) → Tasks 8, 9. ✅
- Clause-typed structured citations → Tasks 3 (`clause_type`), 5, 7. ✅
- Settings (no hardcoded constants) → Task 1. ✅
- Deterministic idempotent ids → Task 5. ✅
- Error handling (422/503/best-effort) → Tasks 9, 10. ✅
- docker-compose qdrant + env → Task 11. ✅
- Retrieval recall@k/MRR → Task 12. RAGAS faithfulness/etc. explicitly deferred to the eval notebook (spec: "run later, not in CI"). ✅ (noted as out-of-plan automated scope)
- Testing matrix (chunk/store/answer/route/best-effort/slow) → Tasks 2,5,8,9,10,4. ✅

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every test shows assertions.

**Type consistency:** `TextChunk` fields, `RetrievedChunk` fields, `index_contract(contract_id, text, clauses, store, settings)`, `answer_question(question, store, llm, settings, contract_id, top_k)`, `search(store, query, top_k, contract_id)`, and `build_vector_store(settings, embedder, client)` signatures match across the tasks that produce and consume them. `ensure_rag_store`/`get_rag_store`/`get_rag_store_optional`/`get_rag_llm` names match between Tasks 9 and 10.

**One known integration detail:** `langchain-qdrant` stores chunk metadata under the payload key `metadata`, so the `contract_id` filter uses key `"metadata.contract_id"` (Task 5). If a future `langchain-qdrant` version changes the payload key, update that one string.
