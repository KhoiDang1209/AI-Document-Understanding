# C2 Embedder Fine-Tune (CUAD-Tuned bge-small) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift retrieval recall@5 from 0.494 to ≥ 0.65 by fine-tuning `BAAI/bge-small-en-v1.5` on CUAD (focused query → covering paragraph window) pairs, served through fastembed's custom-model path.

**Architecture:** A laptop-side pair-builder script exports train/dev JSONL from CUAD (excluding the 40 seed-0 eval contracts); a Colab notebook fine-tunes with MultipleNegativesRankingLoss and exports an ONNX bundle (+ reference vectors); a new optional setting loads the bundle via `TextEmbedding.add_custom_model` + `specific_model_path`; a parity script gates the bundle before evals. BM25, RRF fusion, focused queries, and the reranker are untouched.

**Tech Stack:** Python 3.12, uv, fastembed 0.8 (ONNX CPU), sentence-transformers + optimum (Colab only), datasets, Qdrant in-memory evals, pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-c2-embedder-finetune-design.md`

## Global Constraints

- Run everything through `uv run` from `docintel/`; full pytest requires `uv sync --all-extras` first (uv sync *replaces* the env — never sync a subset of extras before testing).
- Gates per task: `uv run ruff check`, `uv run ruff format <touched files>`, `uv run mypy src` (one pre-existing error in `contracts/extractor.py` is accepted), `uv run pytest` for the touched test files, full suite before the final task's commit.
- Do NOT touch: `rag/store.py`, `rag/rerank.py`, `rag/query.py`, `rag/answer.py`, prompts, chunk sizes (`rag_chunk_size=1200`, `rag_chunk_overlap=200`), or the C4 agent path.
- Holdout is sacred: the 40 titles from `_sample_contracts(dataset, 40, seed=0)` (defined in `src/docintel/scripts/eval_rag.py`) must never appear in training or dev data.
- The ONNX bundle lives in the repo-root `models/` directory (gitignored); it is never committed.
- Serving default must be unchanged: with `rag_embedding_local_path` unset, behavior is byte-for-byte the stock path.
- fastembed's `query_embed` applies **no prefix** for bge models (verified on fastembed 0.8.0), so training uses raw focused queries — no instruction prefix anywhere.
- Style: full type hints, functional components, small focused functions, no hardcoded constants (settings/env), match existing module docstring style.

## File Structure

```
docintel/
  src/docintel/config.py                       (modify)  + rag_embedding_local_path
  src/docintel/rag/embed.py                    (modify)  custom-model branch in build_embedder
  src/docintel/scripts/build_embed_pairs.py    (create)  CUAD → train/dev JSONL pairs
  src/docintel/scripts/check_embed_parity.py   (create)  bundle parity gate (cosine ≥ 0.999)
  src/docintel/scripts/eval_rag.py             (modify)  --top-ks CLI flag (recall@30)
  notebooks/cuad_embed_finetune.ipynb          (create)  Colab: MNRL fine-tune → ONNX bundle
  tests/test_rag_embed.py                      (create)  build_embedder branch tests
  tests/test_build_embed_pairs.py              (create)  pair-builder tests
  tests/test_check_embed_parity.py             (create)  parity check() tests
  tests/test_eval_scripts.py                   (modify)  _parse_top_ks test
  .env.example                                 (modify)  + DOCINTEL_RAG_EMBEDDING_LOCAL_PATH
docs/phases/c2-embed-finetune/report_c2_embed_finetune.md  (create, final task)
```

---

### Task 1: Config setting + custom-model serving branch in `rag/embed.py`

**Files:**
- Modify: `docintel/src/docintel/config.py` (RAG block, after `rag_embedding_dim`, line ~92)
- Modify: `docintel/src/docintel/rag/embed.py`
- Modify: `docintel/.env.example` (RAG section)
- Test: `docintel/tests/test_rag_embed.py` (create)

**Interfaces:**
- Consumes: `Settings` (pydantic-settings, env prefix `DOCINTEL_`), existing `FastEmbedEmbeddings`.
- Produces: `Settings.rag_embedding_local_path: str | None`; `build_embedder(settings)` returns a `FastEmbedEmbeddings` backed by the local bundle when the setting is set. Later tasks (parity script, evals) rely on exactly this switch.

- [ ] **Step 1: Write the failing tests**

Create `docintel/tests/test_rag_embed.py`:

```python
"""Unit tests for build_embedder's stock vs local-bundle branches (no model downloads)."""

from __future__ import annotations

from typing import Any

import pytest

import docintel.rag.embed as embed_module
from docintel.config import Settings
from docintel.rag.embed import build_embedder


class _FakeTextEmbedding:
    """Records constructor and add_custom_model calls; embeds nothing."""

    events: list[tuple[str, Any]] = []

    @classmethod
    def add_custom_model(cls, **kwargs: Any) -> None:
        cls.events.append(("register", kwargs))

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        type(self).events.append(("init", (model_name, kwargs)))


@pytest.fixture()
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTextEmbedding]:
    _FakeTextEmbedding.events = []
    monkeypatch.setattr("fastembed.TextEmbedding", _FakeTextEmbedding)
    monkeypatch.setattr(embed_module, "_registered_custom_models", set())
    return _FakeTextEmbedding


def test_build_embedder_stock_path_unchanged(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    build_embedder(Settings())
    assert fake_fastembed.events == [
        ("init", ("BAAI/bge-small-en-v1.5", {}))
    ]  # no registration, no extra kwargs


def test_build_embedder_local_bundle(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    settings = Settings(rag_embedding_local_path="models/rag-embed-cuad")
    build_embedder(settings)
    registers = [e for e in fake_fastembed.events if e[0] == "register"]
    inits = [e for e in fake_fastembed.events if e[0] == "init"]
    assert len(registers) == 1
    assert registers[0][1]["model_file"] == "model.onnx"
    assert registers[0][1]["dim"] == settings.rag_embedding_dim
    assert inits[0][1][0] == embed_module._CUSTOM_MODEL_NAME
    assert inits[0][1][1]["specific_model_path"] == "models/rag-embed-cuad"


def test_build_embedder_registers_once(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    settings = Settings(rag_embedding_local_path="models/rag-embed-cuad")
    build_embedder(settings)
    build_embedder(settings)
    registers = [e for e in fake_fastembed.events if e[0] == "register"]
    assert len(registers) == 1  # idempotent registration
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd docintel && uv run pytest tests/test_rag_embed.py -v`
Expected: FAIL with `AttributeError: module 'docintel.rag.embed' has no attribute '_registered_custom_models'` (and/or `Settings` rejecting `rag_embedding_local_path`).

- [ ] **Step 3: Add the setting**

In `docintel/src/docintel/config.py`, after `rag_embedding_dim: int = 384` (line ~92), add (mirrors `contract_onnx_local_path`, line 79):

```python
    rag_embedding_local_path: str | None = Field(
        default=None,
        description="Local fine-tuned embedding ONNX bundle dir; if set, overrides rag_embedding_model.",
    )
```

In `docintel/.env.example`, add to the RAG section:

```
DOCINTEL_RAG_EMBEDDING_LOCAL_PATH=
```

(`tests/test_env_example.py` validates the key maps to a real Settings field.)

- [ ] **Step 4: Implement the custom-model branch**

Replace `build_embedder` in `docintel/src/docintel/rag/embed.py` (keep `FastEmbedEmbeddings` untouched) and add the registration helper:

```python
_CUSTOM_MODEL_NAME = "docintel/bge-small-cuad"
_registered_custom_models: set[str] = set()


def _register_custom_model(name: str, dim: int) -> None:
    """Idempotently register the local fine-tuned bundle layout with fastembed."""
    if name in _registered_custom_models:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    TextEmbedding.add_custom_model(
        model=name,
        pooling=PoolingType.CLS,
        normalization=True,
        sources=ModelSource(hf=name),  # never downloaded: a local path is always passed
        dim=dim,
        model_file="model.onnx",
    )
    _registered_custom_models.add(name)


def build_embedder(settings: Settings) -> FastEmbedEmbeddings:
    """Load the configured fastembed model (local fine-tuned bundle if set) and wrap it."""
    from fastembed import TextEmbedding

    if settings.rag_embedding_local_path:
        _register_custom_model(_CUSTOM_MODEL_NAME, settings.rag_embedding_dim)
        return FastEmbedEmbeddings(
            TextEmbedding(
                model_name=_CUSTOM_MODEL_NAME,
                specific_model_path=settings.rag_embedding_local_path,
            )
        )
    return FastEmbedEmbeddings(TextEmbedding(model_name=settings.rag_embedding_model))
```

Note: bge pooling is CLS with L2 normalization — this must match the notebook's export (Task 4) and is cross-checked by the parity gate (Task 3).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd docintel && uv run pytest tests/test_rag_embed.py tests/test_config.py tests/test_env_example.py -v`
Expected: all PASS.

- [ ] **Step 6: Gates**

Run: `cd docintel && uv run ruff check src/docintel/config.py src/docintel/rag/embed.py tests/test_rag_embed.py && uv run ruff format src/docintel/config.py src/docintel/rag/embed.py tests/test_rag_embed.py && uv run mypy src`
Expected: clean (except the pre-existing `contracts/extractor.py` error).

- [ ] **Step 7: Commit**

```bash
git add docintel/src/docintel/config.py docintel/src/docintel/rag/embed.py docintel/.env.example docintel/tests/test_rag_embed.py
git commit -m "feat(rag): serve local fine-tuned embedding bundle via fastembed custom model"
```

---

### Task 2: Pair-builder script `build_embed_pairs.py`

**Files:**
- Create: `docintel/src/docintel/scripts/build_embed_pairs.py`
- Test: `docintel/tests/test_build_embed_pairs.py`

**Interfaces:**
- Consumes: `focus_query` (`rag/query.py`), `build_chunks` (`rag/chunk.py`), `_sample_contracts` and `_covering_chunk_indices` (`scripts/eval_rag.py`) — reusing the eval's own holdout/covering code is the point; do not reimplement them.
- Produces: `build_pairs(dataset, exclude_titles, size, overlap) -> list[dict[str, str]]` (keys `query`, `positive`, `title`) and `split_dev(pairs, dev_contracts, seed) -> tuple[list[dict], list[dict]]`. CLI writes `train.jsonl`, `dev.jsonl`, `meta.json` to `--out-dir`. The notebook (Task 4) consumes the JSONL files: one JSON object per line with keys `query`/`positive`/`title`.

- [ ] **Step 1: Write the failing tests**

Create `docintel/tests/test_build_embed_pairs.py`:

```python
"""Unit tests for the CUAD embedding-pair builder (fake dataset, no downloads)."""

from __future__ import annotations

from docintel.scripts.build_embed_pairs import build_pairs, split_dev

_CONTEXT = "".join(f"sentence {i:03d}. " for i in range(40))  # 560 chars


def _row(title: str, category: str, starts: list[int]) -> dict[str, object]:
    question = (
        f'Highlight the parts (if any) of this contract related to "{category}" '
        f"that should be reviewed by a lawyer. Details: some details about {category}"
    )
    return {
        "title": title,
        "context": _CONTEXT,
        "question": question,
        "answers": {"text": ["x"] * len(starts), "answer_start": starts},
    }


def _dataset() -> list[dict[str, object]]:
    return [
        _row("A", "Governing Law", [10]),
        _row("A", "Insurance", []),  # unanswered -> no pair
        _row("B", "Non-Compete", [10, 400]),  # two spans -> two windows
        _row("HELD-OUT", "Governing Law", [10]),
    ]


def test_build_pairs_excludes_holdout_and_unanswered() -> None:
    pairs = build_pairs(_dataset(), {"HELD-OUT"}, size=100, overlap=20)
    assert {p["title"] for p in pairs} == {"A", "B"}


def test_build_pairs_query_is_focused() -> None:
    pairs = build_pairs(_dataset(), set(), size=100, overlap=20)
    queries = {p["query"] for p in pairs if p["title"] == "A"}
    assert queries == {"Governing Law: some details about Governing Law"}


def test_build_pairs_positive_is_covering_window() -> None:
    pairs = build_pairs(_dataset(), set(), size=100, overlap=20)
    for pair in pairs:
        assert pair["positive"] in _CONTEXT  # a real window of the contract text
        assert len(pair["positive"]) <= 100


def test_build_pairs_multi_span_yields_multiple_windows() -> None:
    pairs = build_pairs(_dataset(), set(), size=100, overlap=20)
    b_pairs = [p for p in pairs if p["title"] == "B"]
    assert len(b_pairs) >= 2
    assert len({p["positive"] for p in b_pairs}) >= 2  # distinct windows for distant spans


def test_split_dev_is_contract_disjoint_and_deterministic() -> None:
    pairs = build_pairs(_dataset(), set(), size=100, overlap=20)
    train, dev = split_dev(pairs, dev_contracts=1, seed=0)
    train_titles = {p["title"] for p in train}
    dev_titles = {p["title"] for p in dev}
    assert dev_titles and not (train_titles & dev_titles)
    train2, dev2 = split_dev(pairs, dev_contracts=1, seed=0)
    assert (train, dev) == (train2, dev2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd docintel && uv run pytest tests/test_build_embed_pairs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'docintel.scripts.build_embed_pairs'`.

- [ ] **Step 3: Implement the script**

Create `docintel/src/docintel/scripts/build_embed_pairs.py`:

```python
"""Build CUAD (focused query -> covering paragraph window) pairs for embedder fine-tuning.

Positives are the production 1200/200 paragraph windows covering each gold span (built
with ``rag.chunk.build_chunks``), so the model trains on the exact text distribution it
retrieves over at serve time. The 40 seed-0 eval contracts (the same
``_sample_contracts`` call ``eval_rag`` uses) are excluded; a further ``--dev-contracts``
slice of the training pool becomes the dev set for during-training IR validation.

Reproduce::

    python -m docintel.scripts.build_embed_pairs --out-dir data/processed/embed_pairs
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from docintel.config import get_settings
from docintel.rag.chunk import build_chunks
from docintel.rag.query import focus_query
from docintel.scripts.eval_rag import _covering_chunk_indices, _sample_contracts


def build_pairs(
    dataset: Any, exclude_titles: set[str], size: int, overlap: int
) -> list[dict[str, str]]:
    """(focused query, covering paragraph window) per gold answer, minus held-out titles."""
    contexts: dict[str, str] = {}
    questions: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for ex in dataset:
        title = ex["title"]
        if title in exclude_titles:
            continue
        contexts.setdefault(title, ex["context"])
        starts = ex["answers"]["answer_start"]
        if starts:
            questions[title].append((ex["question"], list(starts)))

    pairs: list[dict[str, str]] = []
    for title, items in questions.items():
        chunks = build_chunks(contexts[title], [], size, overlap)
        by_index = {chunk.chunk_index: chunk for chunk in chunks}
        for question, starts in items:
            covering: set[int] = set()
            for start in starts:
                covering |= _covering_chunk_indices(chunks, start)
            query = focus_query(question)
            for index in sorted(covering):
                pairs.append(
                    {"query": query, "positive": by_index[index].text, "title": title}
                )
    return pairs


def split_dev(
    pairs: list[dict[str, str]], dev_contracts: int, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Deterministic contract-disjoint train/dev split of the built pairs."""
    titles = sorted({pair["title"] for pair in pairs})
    random.Random(seed).shuffle(titles)
    dev_titles = set(titles[:dev_contracts])
    train = [pair for pair in pairs if pair["title"] not in dev_titles]
    dev = [pair for pair in pairs if pair["title"] in dev_titles]
    return train, dev


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="CUAD embedding fine-tune pair builder.")
    parser.add_argument("--out-dir", type=str, default="data/processed/embed_pairs")
    parser.add_argument("--holdout-sample", type=int, default=40)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--dev-contracts", type=int, default=20)
    parser.add_argument("--dev-seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    settings = get_settings()
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    holdout = set(_sample_contracts(dataset, args.holdout_sample, args.holdout_seed))
    pairs = build_pairs(dataset, holdout, settings.rag_chunk_size, settings.rag_chunk_overlap)
    train, dev = split_dev(pairs, args.dev_contracts, args.dev_seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "dev.jsonl", dev)
    meta = {
        "train_pairs": len(train),
        "dev_pairs": len(dev),
        "train_contracts": len({p["title"] for p in train}),
        "dev_contracts": len({p["title"] for p in dev}),
        "holdout_titles": sorted(holdout),
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "holdout_titles"}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd docintel && uv run pytest tests/test_build_embed_pairs.py tests/test_eval_scripts.py -v`
Expected: all PASS (eval_scripts still green — we only *import* from `eval_rag`).

- [ ] **Step 5: Gates**

Run: `cd docintel && uv run ruff check src/docintel/scripts/build_embed_pairs.py tests/test_build_embed_pairs.py && uv run ruff format src/docintel/scripts/build_embed_pairs.py tests/test_build_embed_pairs.py && uv run mypy src`
Expected: clean (modulo the pre-existing extractor error).

- [ ] **Step 6: Commit**

```bash
git add docintel/src/docintel/scripts/build_embed_pairs.py docintel/tests/test_build_embed_pairs.py
git commit -m "feat(rag): CUAD embedding-pair builder with eval-contract holdout"
```

---

### Task 3: Parity gate `check_embed_parity.py`

**Files:**
- Create: `docintel/src/docintel/scripts/check_embed_parity.py`
- Test: `docintel/tests/test_check_embed_parity.py`

**Interfaces:**
- Consumes: `build_embedder` + `Settings(rag_embedding_local_path=...)` from Task 1; `parity.json` written by the notebook (Task 4) into the bundle: `{"sentences": [str], "vectors": [[float]]}` (vectors from the fine-tuned sentence-transformers model, normalized).
- Produces: `check(reference, produced, threshold) -> list[tuple[str, float]]` (failures) and a CLI: `python -m docintel.scripts.check_embed_parity --bundle models/rag-embed-cuad` exiting non-zero on any failure. Task 5 runs this as a hard gate.

- [ ] **Step 1: Write the failing tests**

Create `docintel/tests/test_check_embed_parity.py`:

```python
"""Unit tests for the embedding parity gate's pure comparison logic."""

from __future__ import annotations

from docintel.scripts.check_embed_parity import check

_REFERENCE = {
    "sentences": ["alpha", "beta"],
    "vectors": [[1.0, 0.0], [0.6, 0.8]],
}


def test_check_passes_on_identical_vectors() -> None:
    assert check(_REFERENCE, [[1.0, 0.0], [0.6, 0.8]], threshold=0.999) == []


def test_check_flags_diverging_vector_with_its_sentence() -> None:
    failures = check(_REFERENCE, [[1.0, 0.0], [0.8, 0.6]], threshold=0.999)
    assert [sentence for sentence, _ in failures] == ["beta"]
    assert all(cos < 0.999 for _, cos in failures)


def test_check_is_scale_invariant() -> None:
    # cosine ignores magnitude: a scaled copy still passes
    assert check(_REFERENCE, [[2.0, 0.0], [1.2, 1.6]], threshold=0.999) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd docintel && uv run pytest tests/test_check_embed_parity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'docintel.scripts.check_embed_parity'`.

- [ ] **Step 3: Implement the script**

Create `docintel/src/docintel/scripts/check_embed_parity.py`:

```python
"""Gate: the local ONNX bundle must reproduce the trained model's embeddings.

The Colab notebook writes ``parity.json`` (fixed sentences + their vectors from the
fine-tuned sentence-transformers model) into the bundle. This script embeds the same
sentences through the production ``build_embedder`` path — both the document and the
query call — and fails if any cosine similarity drops below the threshold. Catches
pooling (bge = CLS), normalization, and tokenizer drift in the export.

Reproduce::

    python -m docintel.scripts.check_embed_parity --bundle models/rag-embed-cuad
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_DEFAULT_THRESHOLD = 0.999


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def check(
    reference: dict[str, list], produced: list[list[float]], threshold: float
) -> list[tuple[str, float]]:
    """Return (sentence, cosine) for every produced vector below the threshold."""
    failures: list[tuple[str, float]] = []
    for sentence, expected, got in zip(
        reference["sentences"], reference["vectors"], produced, strict=True
    ):
        cosine = _cosine(expected, got)
        if cosine < threshold:
            failures.append((sentence, cosine))
    return failures


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Embedding bundle parity gate.")
    parser.add_argument("--bundle", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD)
    args = parser.parse_args()

    from docintel.config import Settings
    from docintel.rag.embed import build_embedder

    reference = json.loads((Path(args.bundle) / "parity.json").read_text(encoding="utf-8"))
    embedder = build_embedder(Settings(rag_embedding_local_path=args.bundle))
    sentences = reference["sentences"]

    doc_failures = check(reference, embedder.embed_documents(sentences), args.threshold)
    query_failures = check(
        reference, [embedder.embed_query(sentence) for sentence in sentences], args.threshold
    )
    for label, failures in (("documents", doc_failures), ("query", query_failures)):
        for sentence, cosine in failures:
            print(f"FAIL [{label}] cosine={cosine:.6f}  {sentence[:80]!r}")
    if doc_failures or query_failures:
        sys.exit(1)
    print(f"parity OK: {len(sentences)} sentences, both paths, threshold {args.threshold}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd docintel && uv run pytest tests/test_check_embed_parity.py -v`
Expected: all PASS.

- [ ] **Step 5: Gates**

Run: `cd docintel && uv run ruff check src/docintel/scripts/check_embed_parity.py tests/test_check_embed_parity.py && uv run ruff format src/docintel/scripts/check_embed_parity.py tests/test_check_embed_parity.py && uv run mypy src`
Expected: clean (modulo the pre-existing extractor error).

- [ ] **Step 6: Commit**

```bash
git add docintel/src/docintel/scripts/check_embed_parity.py docintel/tests/test_check_embed_parity.py
git commit -m "feat(rag): embedding bundle parity gate (cosine vs trained model)"
```

---

### Task 4: `eval_rag --top-ks` + Colab fine-tune notebook

**Files:**
- Modify: `docintel/src/docintel/scripts/eval_rag.py` (CLI block, lines 142–158)
- Modify: `docintel/tests/test_eval_scripts.py` (append one test)
- Create: `docintel/notebooks/cuad_embed_finetune.ipynb`

**Interfaces:**
- Consumes: `eval_rag.run(sample, seed, top_ks=...)` already accepts `top_ks` — only the CLI needs the flag. Notebook consumes `train.jsonl`/`dev.jsonl` (Task 2 format).
- Produces: `_parse_top_ks(raw: str) -> tuple[int, ...]` in `eval_rag`; a bundle zip `rag-embed-cuad.zip` containing `model.onnx`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `config.json`, `parity.json` — exactly the layout Task 1's `model_file="model.onnx"` and Task 3's gate expect.

- [ ] **Step 1: Write the failing test**

Append to `docintel/tests/test_eval_scripts.py`:

```python
def test_rag_parse_top_ks() -> None:
    assert eval_rag._parse_top_ks("1,3,5,30") == (1, 3, 5, 30)
    assert eval_rag._parse_top_ks("5") == (5,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docintel && uv run pytest tests/test_eval_scripts.py::test_rag_parse_top_ks -v`
Expected: FAIL with `AttributeError: ... has no attribute '_parse_top_ks'`.

- [ ] **Step 3: Implement the flag**

In `docintel/src/docintel/scripts/eval_rag.py`, add below `_covering_chunk_indices` (line ~45):

```python
def _parse_top_ks(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(","))
```

In `main()`, add the argument and pass it through:

```python
    parser.add_argument(
        "--top-ks", type=str, default="1,3,5", help="Comma-separated recall@k cutoffs."
    )
```

```python
    metrics = run(
        args.sample,
        args.seed,
        top_ks=_parse_top_ks(args.top_ks),
        rerank=not args.no_rerank,
        focused=not args.raw_query,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd docintel && uv run pytest tests/test_eval_scripts.py -v`
Expected: all PASS.

- [ ] **Step 5: Create the notebook**

Create `docintel/notebooks/cuad_embed_finetune.ipynb` (follow the structure of the existing `cuad_finetune.ipynb`: markdown header cell, then code cells). Cells:

**Cell 1 (markdown):**

```markdown
# CUAD Embedder Fine-Tune — bge-small-en-v1.5 (C2 retrieval)

Fine-tunes `BAAI/bge-small-en-v1.5` on (focused query → covering paragraph window)
pairs built by `docintel.scripts.build_embed_pairs` (eval contracts held out), then
exports an ONNX bundle for fastembed CPU serving.

**Inputs (upload to Drive):** `train.jsonl`, `dev.jsonl` from
`data/processed/embed_pairs/`. **Output:** `rag-embed-cuad.zip` →
laptop `models/rag-embed-cuad/`. Runtime: GPU (T4 is fine), ~20–40 min.
```

**Cell 2 (setup):**

```python
%pip -q install -U sentence-transformers datasets "optimum[onnxruntime]" onnx

from google.colab import drive

drive.mount("/content/drive")
PAIRS_DIR = "/content/drive/MyDrive/docintel/embed_pairs"  # adjust to where you uploaded
OUT_DIR = "/content/drive/MyDrive/docintel"
```

**Cell 3 (load pairs):**

```python
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


train_pairs = read_jsonl(f"{PAIRS_DIR}/train.jsonl")
dev_pairs = read_jsonl(f"{PAIRS_DIR}/dev.jsonl")
print(len(train_pairs), "train pairs,", len(dev_pairs), "dev pairs")
```

**Cell 4 (dev IR evaluator):**

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

corpus, corpus_ids = {}, {}
for pair in dev_pairs:
    corpus_ids.setdefault(pair["positive"], f"d{len(corpus_ids)}")
    corpus[corpus_ids[pair["positive"]]] = pair["positive"]

queries, relevant = {}, {}
for i, pair in enumerate(dev_pairs):
    qid = f"q{i}"
    queries[qid] = pair["query"]
    relevant.setdefault(qid, set()).add(corpus_ids[pair["positive"]])

dev_evaluator = InformationRetrievalEvaluator(
    queries, corpus, relevant, name="cuad-dev", accuracy_at_k=[1, 3, 5], map_at_k=[5]
)
```

**Cell 5 (train — MNRL, no-duplicates batching):**

```python
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("baseline dev:", dev_evaluator(model))

train_ds = Dataset.from_list(
    [{"anchor": p["query"], "positive": p["positive"]} for p in train_pairs]
)
args = SentenceTransformerTrainingArguments(
    output_dir="/content/ckpt",
    num_train_epochs=2,
    per_device_train_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MNRL: avoid duplicate positives in-batch
    logging_steps=50,
    eval_strategy="no",
    save_strategy="no",
    report_to=[],
)
trainer = SentenceTransformerTrainer(
    model=model, args=args, train_dataset=train_ds, loss=MultipleNegativesRankingLoss(model)
)
trainer.train()
print("fine-tuned dev:", dev_evaluator(model))
```

**Cell 6 (save + ONNX export):**

```python
ST_DIR, BUNDLE = "/content/bge-small-cuad", "/content/rag-embed-cuad"
model.save_pretrained(ST_DIR)

from optimum.onnxruntime import ORTModelForFeatureExtraction

ort_model = ORTModelForFeatureExtraction.from_pretrained(ST_DIR, export=True)
ort_model.save_pretrained(BUNDLE)  # writes model.onnx + config.json

import shutil

for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    shutil.copy(f"{ST_DIR}/{name}", f"{BUNDLE}/{name}")
```

**Cell 7 (parity vectors — the laptop gate's reference):**

```python
import json

PARITY_SENTENCES = [
    "Governing Law: which state or country's law governs the agreement",
    "Non-Compete: restrictions on competing with the counterparty",
    "Insurance: requirement for one party to maintain insurance coverage",
    "This Agreement shall be governed by the laws of the State of New York.",
    "Licensee shall not solicit employees of Licensor during the term.",
    "Either party may terminate this Agreement upon thirty (30) days notice.",
    "All disputes shall be resolved by binding arbitration in London.",
    "The term of this Agreement is five (5) years from the Effective Date.",
]
vectors = model.encode(PARITY_SENTENCES, normalize_embeddings=True).tolist()
with open(f"{BUNDLE}/parity.json", "w", encoding="utf-8") as handle:
    json.dump({"sentences": PARITY_SENTENCES, "vectors": vectors}, handle)
```

**Cell 8 (sanity-check ONNX in-Colab, then zip to Drive):**

```python
# In-Colab parity: ONNX (CLS pooling + L2 norm) vs the sentence-transformers model.
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BUNDLE)
encoded = tokenizer(PARITY_SENTENCES, padding=True, truncation=True, return_tensors="pt")
onnx_out = ort_model(**encoded).last_hidden_state[:, 0]  # CLS
onnx_out = onnx_out / np.linalg.norm(onnx_out.numpy(), axis=1, keepdims=True)
st_out = np.asarray(vectors)
cosines = (onnx_out.numpy() * st_out).sum(axis=1)
print("in-Colab parity cosines:", cosines.round(6))
assert cosines.min() >= 0.999, "ONNX export drifted from the trained model"

shutil.make_archive(f"{OUT_DIR}/rag-embed-cuad", "zip", BUNDLE)
print("bundle saved to Drive:", f"{OUT_DIR}/rag-embed-cuad.zip")
```

- [ ] **Step 6: Gates + commit**

Run: `cd docintel && uv run ruff check src/docintel/scripts/eval_rag.py tests/test_eval_scripts.py && uv run ruff format src/docintel/scripts/eval_rag.py tests/test_eval_scripts.py && uv run mypy src && uv run pytest tests/test_eval_scripts.py -v`
Expected: clean, all PASS.

```bash
git add docintel/src/docintel/scripts/eval_rag.py docintel/tests/test_eval_scripts.py docintel/notebooks/cuad_embed_finetune.ipynb
git commit -m "feat(eval): --top-ks flag + Colab embedder fine-tune notebook"
```

---

### Task 5: Baseline measurements + pair export (laptop, before training)

No code changes — measurement and data generation. Results feed the final report's ablation table.

**Files:**
- Create (untracked outputs): `docintel/eval_rag_stock_k30.json`, `docintel/data/processed/embed_pairs/{train,dev}.jsonl` + `meta.json`

- [ ] **Step 1: Full test suite green before measuring**

Run: `cd docintel && uv sync --all-extras && uv run pytest`
Expected: all PASS.

- [ ] **Step 2: Build the pairs**

Run: `cd docintel && uv run python -m docintel.scripts.build_embed_pairs --out-dir data/processed/embed_pairs`
Expected: meta printed with roughly 10–15k train pairs, ~450 train contracts, 20 dev contracts. Sanity-check `meta.json`: `holdout_titles` has 40 entries.

- [ ] **Step 3: Stock recall@30 pre-measure (reranker-headroom baseline)**

Run: `cd docintel && uv run python -m docintel.scripts.eval_rag --sample 40 --seed 0 --top-ks 1,3,5,30 --out eval_rag_stock_k30.json`
Expected: recall@1/3/5 ≈ 0.206/0.399/0.494 (matches the retrieval-boost report); recall@30 is the new number — record it. Takes ~65 min on the laptop CPU.

- [ ] **Step 4: Verify holdout discipline (one-liner)**

Run (from `docintel/`):

```bash
uv run python -c "
import json
from datasets import load_dataset
from docintel.scripts.eval_rag import _sample_contracts
meta = json.load(open('data/processed/embed_pairs/meta.json', encoding='utf-8'))
ds = load_dataset('theatticusproject/cuad-qa', split='train', trust_remote_code=True)
eval_titles = set(_sample_contracts(ds, 40, 0))
train_titles = {json.loads(l)['title'] for l in open('data/processed/embed_pairs/train.jsonl', encoding='utf-8')}
dev_titles = {json.loads(l)['title'] for l in open('data/processed/embed_pairs/dev.jsonl', encoding='utf-8')}
assert not (eval_titles & (train_titles | dev_titles)), 'HOLDOUT LEAK'
assert set(meta['holdout_titles']) == eval_titles
print('holdout clean:', len(eval_titles), 'eval titles excluded from', len(train_titles | dev_titles), 'training titles')
"
```

Expected: `holdout clean: 40 eval titles excluded from ~450 training titles`.

- [ ] **Step 5: Hand off to Colab**

Upload `train.jsonl` and `dev.jsonl` to Drive (`MyDrive/docintel/embed_pairs/`). No commit (all outputs are gitignored or transient; keep `eval_rag_stock_k30.json` for the report).

---

### Task 6: Colab training run + bundle deployment + parity gate

Manual/interactive task (Colab GPU session) — the notebook from Task 4 does the work.

- [ ] **Step 1: Run the notebook on Colab (GPU runtime)**

Execute all cells of `notebooks/cuad_embed_finetune.ipynb`. Record: baseline vs fine-tuned dev IR metrics (expect accuracy@5 to rise materially; if fine-tuned ≤ baseline, stop — do not deploy; investigate pairs/loss before burning laptop eval hours). The in-notebook ONNX parity assert (Cell 8) must pass.

- [ ] **Step 2: Deploy the bundle to the laptop**

Download `rag-embed-cuad.zip` from Drive, extract to the repo-root `models/` directory:

```powershell
Expand-Archive rag-embed-cuad.zip -DestinationPath "D:\AI Document Understanding\models\rag-embed-cuad"
```

Expected contents: `model.onnx`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `parity.json`.

- [ ] **Step 3: Run the parity gate (hard gate — nothing proceeds if it fails)**

Run: `cd docintel && uv run python -m docintel.scripts.check_embed_parity --bundle "D:/AI Document Understanding/models/rag-embed-cuad"`
Expected: `parity OK: 8 sentences, both paths, threshold 0.999`. If it fails: the export drifted (pooling/normalization/tokenizer) — fix the notebook export, re-run; do not lower the threshold.

---

### Task 7: Final evals + phase report

**Files:**
- Create: `docs/phases/c2-embed-finetune/report_c2_embed_finetune.md`
- Create (untracked outputs): `docintel/eval_rag_finetuned.json`, `docintel/eval_rag_finetuned_norerank.json`, `docintel/eval_ragas_finetuned.json`, `docintel/eval_ragas_finetuned_samples.csv`

- [ ] **Step 1: Fine-tuned retrieval eval (full stack + no-rerank ablation)**

Run (PowerShell, from `docintel/`; the env var routes `build_embedder` to the bundle):

```powershell
$env:DOCINTEL_RAG_EMBEDDING_LOCAL_PATH = "D:/AI Document Understanding/models/rag-embed-cuad"
uv run python -m docintel.scripts.eval_rag --sample 40 --seed 0 --top-ks 1,3,5,30 --out eval_rag_finetuned.json
uv run python -m docintel.scripts.eval_rag --sample 40 --seed 0 --top-ks 1,3,5,30 --no-rerank --out eval_rag_finetuned_norerank.json
```

Expected: recall@5 ≥ 0.65 (the phase's success bar). Record per-category recall@5 — check the three formerly-zero categories (Non-Compete, Price Restrictions, Unlimited/All-You-Can-Eat-License).

- [ ] **Step 2: RAGAS re-run (closing measurement; needs the Colab LLM endpoint up)**

```powershell
$env:DOCINTEL_RAG_EMBEDDING_LOCAL_PATH = "D:/AI Document Understanding/models/rag-embed-cuad"
$env:DOCINTEL_LLM_BASE_URL = "<current ngrok URL>"
uv run python -m docintel.scripts.eval_ragas --contracts 5 --questions 40 --seed 0 --max-workers 1 --samples-csv eval_ragas_finetuned_samples.csv
```

Expected: refusal rate < 0.375 (retrieval misses drop); answered-only metrics steady or better. ~3 h wall (C1 extraction dominates) — run in the background.

- [ ] **Step 3: Write the phase report**

Create `docs/phases/c2-embed-finetune/report_c2_embed_finetune.md` following `docs/phases/c2-retrieval-boost/report_c2_retrieval_boost.md`'s structure (Why / What changed / Headline metrics / RAGAS / Operational notes / Verification), with:

- Headline ablation table: stock vs fine-tuned × {full stack, no-rerank} — recall@1/3/5/30 + MRR (stock numbers from `eval_rag_stock_k30.json` and the retrieval-boost report).
- Reranker-headroom analysis: recall@30 before vs after (did the pool improve, the ranking, or both).
- Per-category recall@5 movement, explicitly covering the three formerly-zero categories.
- RAGAS before/after table (faithfulness, answer_relevancy all + answered-only, refusal rate) with the n=40 caveat.
- Dev IR metrics from the notebook (baseline vs fine-tuned).
- Success-bar verdict: recall@5 ≥ 0.65 met or not, stated plainly; if not, name the next lever (hard negatives / query expansion).
- Operational notes: re-index required (new embedding space — delete the Qdrant collection / compose volume and re-run `/extract`), bundle location + env var, training reproducibility (pair builder command + notebook + seeds).

- [ ] **Step 4: Full-suite + gates, then commit the report**

Run: `cd docintel && uv run pytest && uv run ruff check && uv run mypy src`
Expected: green (modulo the two pre-existing issues noted in the retrieval-boost report).

```bash
git add docs/phases/c2-embed-finetune/report_c2_embed_finetune.md
git commit -m "docs(eval): C2 embedder fine-tune report (CUAD-tuned bge-small)"
```

---

## Self-Review Notes

- **Spec coverage:** Decision 1–7 → Tasks 1 (setting + serving), 2 (pairs + holdout), 4–6 (training, export, tokenizer.json, parity), 5/7 (eval protocol incl. recall@30 pre-measure, ablations, RAGAS, success bar). Testing section → Tasks 1–4 unit tests; default-path regression covered by the full suite in Task 5 Step 1.
- **Type consistency:** pair dict keys `query`/`positive`/`title` are identical in Task 2 (producer), Task 4 Cells 3–5 (consumer), and Task 5 Step 4 (verifier). `parity.json` schema identical in Task 3 (consumer) and Task 4 Cell 7 (producer). `_CUSTOM_MODEL_NAME` and `model.onnx` consistent between Task 1 and Task 4 Cell 6.
- **Known judgment calls:** `sentence-transformers` is installed unpinned in Colab (latest v3+ trainer API used); if Colab's version predates `SentenceTransformerTrainer`, upgrade in Cell 2 rather than falling back to the legacy `fit` API.
