# Phase 2 — KIE Fine-tune (LayoutLMv3) + MLflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the laptop-side code artifacts (typed, CPU-tested `src/docintel/kie/` modules + a Colab notebook + a laptop-side import script) that fine-tune LayoutLMv3 on CORD, track the run in MLflow, and register the resulting model in the local MLflow + MinIO.

**Architecture:** A thin Colab notebook `pip install`s this repo and orchestrates: load CORD → convert to LayoutLMv3 token-classification features → fine-tune → log params/metrics/model to a **local MLflow file-store on the Colab VM** → evaluate seqeval F1 → save a self-contained model bundle. The user downloads the bundle; a committed laptop-side `docintel-import-kie` script ingests it into the local docker-compose MLflow + MinIO and registers it. All real logic lives in `src/docintel/kie/` and is unit-tested on CPU with synthetic fixtures; only the GPU train loop runs on Colab.

**Tech Stack:** Python 3.12, Hugging Face `transformers` (LayoutLMv3 + `Trainer` + `LayoutLMv3Processor`), `datasets` (CORD), `seqeval`, `mlflow`, `boto3` (MinIO artifact store), pytest, ruff, mypy, uv.

## Global Constraints

These bind every task. Copied from `docs/superpowers/specs/2026-06-21-phase-2-kie-layoutlmv3-design.md`.

- **Python `>=3.12`**; target `py312`.
- **mypy strict** over `src` only (`warn_return_any`, `warn_unused_ignores` are on — annotate intermediate locals, don't leave unneeded `# type: ignore`).
- **Functional over classes.** `TrainingConfig` is a plain data `@dataclass` (data, not behavior) — that is allowed; no service/behavior classes.
- **No hardcoded constants.** Training hyperparameters live in `TrainingConfig` (named defaults). Service-relevant values (`kie_model_name`, `kie_registered_model_name`) live in `Settings` (env prefix `DOCINTEL_`). The CORD label set is **data-derived** (never a hardcoded field list).
- **Model = `microsoft/layoutlmv3-base`**; **registered model name = `cord-layoutlmv3`** (both as `Settings` defaults).
- **Boxes normalized to integer `0–1000`** (LayoutLMv3 coordinate space), clamped to `[0, 1000]`; BIO tagging (`O`, `B-<cat>`, `I-<cat>`); subword label alignment (label first subword, `-100` for continuations) is delegated to `LayoutLMv3Processor(word_labels=...)`.
- **Minimal changes.** Only touch `src/docintel/kie/`, `src/docintel/config.py`, `pyproject.toml`, `.env.example`, `docintel/notebooks/`, and `docintel/tests/`. Do **not** modify `pipeline/`, `validation/`, `storage/`, `api/`, or the CPU serving Dockerfile.
- **No serving / no API change this phase.** Nothing is exposed over HTTP.
- **Tests hermetic on CPU.** Synthetic fixtures only; no network in the fast suite. Anything that downloads weights/data or needs a GPU is excluded from the fast run via the existing `slow` pytest marker (registered in Phase 1).
- **Commit trailer** on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- Full type hints and a docstring on every public function. Run commands from `docintel/` via `uv run`.

---

## File Structure

```
docintel/
  pyproject.toml                      # MODIFY: add [train] extra + kie deps; entry point
  .env.example                        # MODIFY: add DOCINTEL_KIE_* lines
  src/docintel/
    config.py                         # MODIFY: + kie_model_name, kie_registered_model_name
    kie/
      __init__.py                     # exists (empty stub)
      labels.py                       # CREATE: BIO label list/maps from categories
      config.py                       # CREATE: TrainingConfig dataclass
      dataset.py                      # CREATE: CORD parse -> words/boxes/labels + encode
      metrics.py                      # CREATE: seqeval entity-level F1 (overall + per-field)
      train.py                        # CREATE: training builder + bundle saver (Colab-run)
      import_run.py                   # CREATE: laptop-side import + register (CLI)
  notebooks/
    phase2_kie_layoutlmv3.ipynb       # CREATE: thin Colab orchestrator
  tests/
    test_kie_labels.py                # CREATE
    test_kie_config.py                # CREATE
    test_kie_dataset.py               # CREATE
    test_kie_metrics.py               # CREATE
    test_kie_train.py                 # CREATE (pure helpers only)
    test_kie_import.py                # CREATE (temp MLflow file-store)
```

---

## Task 1: Dependencies, Settings, and `TrainingConfig`

**Files:**
- Modify: `docintel/pyproject.toml`
- Modify: `docintel/.env.example`
- Modify: `docintel/src/docintel/config.py`
- Create: `docintel/src/docintel/kie/config.py`
- Test: `docintel/tests/test_kie_config.py`

**Interfaces:**
- Produces:
  - `Settings.kie_model_name: str = "microsoft/layoutlmv3-base"`
  - `Settings.kie_registered_model_name: str = "cord-layoutlmv3"`
  - `docintel.kie.config.TrainingConfig` — frozen dataclass with fields:
    `model_name: str`, `num_train_epochs: float = 4.0`, `learning_rate: float = 1e-5`,
    `train_batch_size: int = 2`, `eval_batch_size: int = 2`, `weight_decay: float = 0.01`,
    `warmup_ratio: float = 0.1`, `seed: int = 42`, `max_seq_length: int = 512`,
    `eval_strategy: str = "epoch"`, `save_strategy: str = "epoch"`.
    Plus `TrainingConfig.from_settings(settings: Settings) -> TrainingConfig` (sets `model_name` from `settings.kie_model_name`, other defaults unchanged).

- [ ] **Step 1: Add dependencies and entry point to `pyproject.toml`**

In `docintel/pyproject.toml`, add an optional `train` extra (heavy, Colab-only) and the laptop-side import deps. Locate `[project.optional-dependencies]` (the `data` extra already exists there from Phase 0) and add:

```toml
[project.optional-dependencies]
# ... existing groups (e.g. data) unchanged ...
train = [
    "transformers>=4.40",
    "datasets>=2.18",
    "seqeval>=1.2",
    "Pillow>=10.3",
]
kie = [
    "mlflow>=2.12",
    "boto3>=1.34",
]
```

Add the import-script entry point. Locate `[project.scripts]` (the `docintel-download-data` script exists there) and add the line:

```toml
[project.scripts]
# ... existing scripts unchanged ...
docintel-import-kie = "docintel.kie.import_run:main"
```

- [ ] **Step 2: Add the KIE settings to `.env.example`**

Append to `docintel/.env.example` (after the existing OCR lines):

```dotenv
# KIE (Phase 2)
DOCINTEL_KIE_MODEL_NAME=microsoft/layoutlmv3-base
DOCINTEL_KIE_REGISTERED_MODEL_NAME=cord-layoutlmv3
```

- [ ] **Step 3: Write the failing test for Settings + TrainingConfig**

Create `docintel/tests/test_kie_config.py`:

```python
"""Tests for KIE settings and the TrainingConfig dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from docintel.config import Settings
from docintel.kie.config import TrainingConfig


def test_settings_have_kie_defaults() -> None:
    settings = Settings()
    assert settings.kie_model_name == "microsoft/layoutlmv3-base"
    assert settings.kie_registered_model_name == "cord-layoutlmv3"


def test_training_config_is_frozen_dataclass() -> None:
    config = TrainingConfig(model_name="microsoft/layoutlmv3-base")
    assert dataclasses.is_dataclass(config)
    assert config.seed == 42
    assert config.num_train_epochs == 4.0


def test_training_config_from_settings_uses_model_name() -> None:
    settings = Settings(kie_model_name="microsoft/layoutlmv3-base")
    config = TrainingConfig.from_settings(settings)
    assert config.model_name == settings.kie_model_name
    assert config.learning_rate == 1e-5


def test_env_example_lists_every_kie_setting() -> None:
    # Drift guard: every DOCINTEL_KIE_* setting must be documented in .env.example.
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    text = env_example.read_text(encoding="utf-8")
    assert "DOCINTEL_KIE_MODEL_NAME=" in text
    assert "DOCINTEL_KIE_REGISTERED_MODEL_NAME=" in text
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cd docintel && uv run pytest tests/test_kie_config.py -v`
Expected: FAIL — `ImportError` (no `docintel.kie.config`) and `AttributeError` on `kie_model_name`.

- [ ] **Step 5: Add the settings fields**

In `docintel/src/docintel/config.py`, add after the OCR block (`max_upload_mb: float = 10.0`):

```python
    # KIE (Phase 2)
    kie_model_name: str = "microsoft/layoutlmv3-base"
    kie_registered_model_name: str = "cord-layoutlmv3"
```

- [ ] **Step 6: Create `TrainingConfig`**

Create `docintel/src/docintel/kie/config.py`:

```python
"""Training hyperparameters for KIE fine-tuning.

A plain data container (not service behavior) holding the knobs for a
LayoutLMv3 fine-tune. Defaults are named here rather than scattered as
literals; the Colab notebook may override individual fields.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from docintel.config import Settings


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for one LayoutLMv3 fine-tuning run."""

    model_name: str
    num_train_epochs: float = 4.0
    learning_rate: float = 1e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    max_seq_length: int = 512
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"

    @classmethod
    def from_settings(cls, settings: Settings) -> "TrainingConfig":
        """Build a config whose model name comes from service settings."""
        return cls(model_name=settings.kie_model_name)

    def with_overrides(self, **changes: object) -> "TrainingConfig":
        """Return a copy with the given fields replaced (notebook convenience)."""
        return replace(self, **changes)
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `cd docintel && uv run pytest tests/test_kie_config.py -v`
Expected: PASS (4 passed).

- [ ] **Step 8: Run the full gate**

Run: `cd docintel && uv run ruff check . && uv run mypy src && uv run pytest`
Expected: ruff clean, mypy clean, all tests pass (Phase 1 suite + the 4 new).

- [ ] **Step 9: Commit**

```bash
git add docintel/pyproject.toml docintel/.env.example \
        docintel/src/docintel/config.py docintel/src/docintel/kie/config.py \
        docintel/tests/test_kie_config.py
git commit -m "feat(kie): add KIE settings, train/kie deps, and TrainingConfig

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Label schema (`kie/labels.py`)

The CORD label set is derived from the categories present in the data — never hardcoded — so full-schema fidelity is preserved without guessing the field list. This module turns a set of category strings into a deterministic BIO label list and the `id2label`/`label2id` maps the model is trained with.

**Files:**
- Create: `docintel/src/docintel/kie/labels.py`
- Test: `docintel/tests/test_kie_labels.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `OUTSIDE_LABEL: str = "O"`
  - `build_label_list(categories: Iterable[str]) -> list[str]` — returns `["O", "B-<cat>", "I-<cat>", ...]` with categories sorted deterministically; `"O"` always at index 0.
  - `build_label_maps(label_list: Sequence[str]) -> tuple[dict[int, str], dict[str, int]]` — `(id2label, label2id)`.
  - `bio_labels_for_category(category: str) -> tuple[str, str]` — `("B-<cat>", "I-<cat>")`.

- [ ] **Step 1: Write the failing test**

Create `docintel/tests/test_kie_labels.py`:

```python
"""Tests for the CORD BIO label schema."""

from __future__ import annotations

from docintel.kie.labels import (
    OUTSIDE_LABEL,
    bio_labels_for_category,
    build_label_list,
    build_label_maps,
)


def test_build_label_list_puts_outside_first_and_is_sorted() -> None:
    labels = build_label_list(["total.total_price", "menu.nm"])
    assert labels[0] == OUTSIDE_LABEL
    assert labels == [
        "O",
        "B-menu.nm",
        "I-menu.nm",
        "B-total.total_price",
        "I-total.total_price",
    ]


def test_build_label_list_deduplicates_categories() -> None:
    labels = build_label_list(["menu.nm", "menu.nm"])
    assert labels == ["O", "B-menu.nm", "I-menu.nm"]


def test_bio_labels_for_category() -> None:
    assert bio_labels_for_category("menu.nm") == ("B-menu.nm", "I-menu.nm")


def test_build_label_maps_round_trips() -> None:
    labels = build_label_list(["menu.nm"])
    id2label, label2id = build_label_maps(labels)
    assert id2label[0] == "O"
    assert label2id["B-menu.nm"] == 1
    for index, name in id2label.items():
        assert label2id[name] == index
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd docintel && uv run pytest tests/test_kie_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: docintel.kie.labels`.

- [ ] **Step 3: Implement `labels.py`**

Create `docintel/src/docintel/kie/labels.py`:

```python
"""CORD key-information-extraction label schema.

The set of CORD field categories is derived from the dataset at training
time (see :mod:`docintel.kie.dataset`), never hardcoded. This module turns
those categories into the deterministic BIO label list and id/label maps a
LayoutLMv3 token-classification head is trained with, and that Phase 3/4
reuse via the saved model config.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

OUTSIDE_LABEL = "O"


def bio_labels_for_category(category: str) -> tuple[str, str]:
    """Return the ``(B-, I-)`` label pair for a CORD category."""
    return f"B-{category}", f"I-{category}"


def build_label_list(categories: Iterable[str]) -> list[str]:
    """Build the BIO label list: ``O`` first, then sorted ``B-/I-`` pairs."""
    unique = sorted(set(categories))
    labels = [OUTSIDE_LABEL]
    for category in unique:
        begin, inside = bio_labels_for_category(category)
        labels.append(begin)
        labels.append(inside)
    return labels


def build_label_maps(
    label_list: Sequence[str],
) -> tuple[dict[int, str], dict[str, int]]:
    """Return ``(id2label, label2id)`` for a label list."""
    id2label = {index: name for index, name in enumerate(label_list)}
    label2id = {name: index for index, name in id2label.items()}
    return id2label, label2id
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd docintel && uv run pytest tests/test_kie_labels.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the gate and commit**

```bash
cd docintel && uv run ruff check . && uv run mypy src && uv run pytest tests/test_kie_labels.py
git add docintel/src/docintel/kie/labels.py docintel/tests/test_kie_labels.py
git commit -m "feat(kie): add data-derived CORD BIO label schema

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: CORD → features converter (`kie/dataset.py`)

**This is the research task — do the verification step first.** The converter turns one CORD example into the `(words, boxes, labels)` LayoutLMv3 needs, then a thin `encode_example` hands those to `LayoutLMv3Processor` for subword tokenization + label alignment.

**Files:**
- Create: `docintel/src/docintel/kie/dataset.py`
- Test: `docintel/tests/test_kie_dataset.py`

**Interfaces:**
- Consumes: `docintel.kie.labels` (BIO label strings).
- Produces:
  - `normalize_box(box: Sequence[int], width: int, height: int) -> list[int]` — `[x_min, y_min, x_max, y_max]` scaled to `0–1000`, clamped to `[0, 1000]`.
  - `parse_cord_example(ground_truth: dict) -> tuple[list[str], list[list[int]], list[str]]` — returns `(words, boxes_0_1000, bio_labels)` read from `ground_truth["valid_line"]` and `ground_truth["meta"]["image_size"]`.
  - `collect_categories(ground_truths: Iterable[dict]) -> list[str]` — every distinct `valid_line` category across a split (for `build_label_list`).
  - `encode_example(words, boxes, bio_labels, processor, label2id) -> dict` — calls the processor with `apply_ocr=False` and `word_labels`, returning the encoding dict.

- [ ] **Step 1: VERIFY the CORD-v2 structure before writing code**

`download_data.py` fetches `naver-clova-ix/cord-v2` (Donut format). Confirm its `ground_truth` shape before implementing. If `datasets` is installed and the data is available, run:

```python
from datasets import load_dataset
import json
ds = load_dataset("naver-clova-ix/cord-v2", split="train")
gt = json.loads(ds[0]["ground_truth"])
print(sorted(gt.keys()))                       # expect: meta, valid_line, (gt_parse, ...)
print(gt["meta"]["image_size"])                # {"width": ..., "height": ...}
print(gt["valid_line"][0]["category"])         # e.g. "menu.nm"
print(gt["valid_line"][0]["words"][0].keys())  # expect: quad, text, ...
print(gt["valid_line"][0]["words"][0]["quad"]) # {"x1":..,"y1":..,..,"x4":..,"y4":..}
```

**Expected structure** (the converter below assumes this): `ground_truth` is a JSON string; parsed, it has `meta.image_size.{width,height}` and `valid_line: list`, where each line has a `category` (the CORD field) and `words: list`, each word having `text` and a `quad` with eight corner coords `x1..x4`, `y1..y4`.

**If the real structure differs** (e.g. no `valid_line`, or boxes stored differently): STOP and report `DONE_WITH_CONCERNS` with the actual structure printed above, so the controller can adjust this task. Do not silently invent a different parse. The synthetic fixture in the test encodes the expected structure, so the unit test stays hermetic either way.

- [ ] **Step 2: Write the failing test**

Create `docintel/tests/test_kie_dataset.py`:

```python
"""Tests for CORD -> LayoutLMv3 feature conversion."""

from __future__ import annotations

from typing import Any

from docintel.kie.dataset import (
    collect_categories,
    encode_example,
    normalize_box,
    parse_cord_example,
)


def _word(text: str, x1: int, y1: int, x3: int, y3: int) -> dict[str, Any]:
    # quad corners: (x1,y1) top-left ... (x3,y3) bottom-right
    return {
        "text": text,
        "quad": {
            "x1": x1, "y1": y1, "x2": x3, "y2": y1,
            "x3": x3, "y3": y3, "x4": x1, "y4": y3,
        },
    }


def _ground_truth() -> dict[str, Any]:
    return {
        "meta": {"image_size": {"width": 100, "height": 200}},
        "valid_line": [
            {
                "category": "menu.nm",
                "words": [_word("Latte", 10, 20, 30, 40), _word("Grande", 35, 20, 60, 40)],
            },
            {
                "category": "total.total_price",
                "words": [_word("12.99", 10, 160, 50, 180)],
            },
        ],
    }


def test_normalize_box_scales_to_0_1000_and_clamps() -> None:
    assert normalize_box([10, 20, 30, 40], width=100, height=200) == [100, 100, 300, 200]
    # Out-of-range coords clamp to [0, 1000].
    assert normalize_box([-5, 0, 150, 250], width=100, height=200) == [0, 0, 1000, 1000]


def test_parse_cord_example_yields_words_boxes_bio() -> None:
    words, boxes, labels = parse_cord_example(_ground_truth())
    assert words == ["Latte", "Grande", "12.99"]
    assert boxes == [[100, 100, 300, 200], [350, 100, 600, 200], [100, 800, 500, 900]]
    # First word of a line's category is B-, subsequent words I-.
    assert labels == ["B-menu.nm", "I-menu.nm", "B-total.total_price"]


def test_collect_categories_is_sorted_and_unique() -> None:
    cats = collect_categories([_ground_truth(), _ground_truth()])
    assert cats == ["menu.nm", "total.total_price"]


def test_encode_example_calls_processor_without_ocr() -> None:
    captured: dict[str, Any] = {}

    class FakeProcessor:
        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"input_ids": [1, 2], "labels": [0, -100]}

    label2id = {"O": 0, "B-menu.nm": 1, "I-menu.nm": 2}
    words = ["Latte", "Grande"]
    boxes = [[100, 100, 300, 200], [350, 100, 600, 200]]
    labels = ["B-menu.nm", "I-menu.nm"]
    out = encode_example(words, boxes, labels, FakeProcessor(), label2id)

    assert captured["boxes"] == boxes
    assert captured["text"] == words
    assert captured["word_labels"] == [1, 2]  # mapped through label2id
    assert out["labels"] == [0, -100]
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd docintel && uv run pytest tests/test_kie_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: docintel.kie.dataset`.

- [ ] **Step 4: Implement `dataset.py`**

Create `docintel/src/docintel/kie/dataset.py`:

```python
"""Convert CORD (Donut format) examples into LayoutLMv3 features.

Each CORD example carries a ``ground_truth`` JSON with ``valid_line`` entries;
each line has a field ``category`` and ``words`` with quad boxes. This module
flattens those into ``(words, boxes, bio_labels)`` with boxes normalized to
LayoutLMv3's 0-1000 space, then ``encode_example`` defers subword tokenization
and label alignment to ``LayoutLMv3Processor``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from docintel.kie.labels import bio_labels_for_category

_COORD_MIN = 0
_COORD_MAX = 1000


def normalize_box(box: Sequence[int], width: int, height: int) -> list[int]:
    """Scale ``[x_min, y_min, x_max, y_max]`` to 0-1000 and clamp."""
    x_min, y_min, x_max, y_max = box
    scaled = [
        round(_COORD_MAX * x_min / width),
        round(_COORD_MAX * y_min / height),
        round(_COORD_MAX * x_max / width),
        round(_COORD_MAX * y_max / height),
    ]
    return [max(_COORD_MIN, min(_COORD_MAX, value)) for value in scaled]


def _quad_to_box(quad: Mapping[str, int]) -> list[int]:
    xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
    ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_cord_example(
    ground_truth: Mapping[str, Any],
) -> tuple[list[str], list[list[int]], list[str]]:
    """Flatten one CORD ground truth into ``(words, boxes_0_1000, bio_labels)``."""
    size = ground_truth["meta"]["image_size"]
    width, height = int(size["width"]), int(size["height"])

    words: list[str] = []
    boxes: list[list[int]] = []
    labels: list[str] = []
    for line in ground_truth["valid_line"]:
        category = line["category"]
        begin, inside = bio_labels_for_category(category)
        for position, word in enumerate(line["words"]):
            text = word["text"]
            if not text:
                continue
            words.append(text)
            boxes.append(normalize_box(_quad_to_box(word["quad"]), width, height))
            labels.append(begin if position == 0 else inside)
    return words, boxes, labels


def collect_categories(ground_truths: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return every distinct ``valid_line`` category across the examples, sorted."""
    categories: set[str] = set()
    for ground_truth in ground_truths:
        for line in ground_truth["valid_line"]:
            categories.add(line["category"])
    return sorted(categories)


def encode_example(
    words: Sequence[str],
    boxes: Sequence[Sequence[int]],
    bio_labels: Sequence[str],
    processor: Any,
    label2id: Mapping[str, int],
) -> dict[str, Any]:
    """Tokenize one example, letting the processor align labels to subwords."""
    word_label_ids = [label2id[label] for label in bio_labels]
    encoding: dict[str, Any] = processor(
        text=list(words),
        boxes=[list(box) for box in boxes],
        word_labels=word_label_ids,
        truncation=True,
        padding="max_length",
    )
    return encoding
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd docintel && uv run pytest tests/test_kie_dataset.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Run the gate and commit**

```bash
cd docintel && uv run ruff check . && uv run mypy src && uv run pytest tests/test_kie_dataset.py
git add docintel/src/docintel/kie/dataset.py docintel/tests/test_kie_dataset.py
git commit -m "feat(kie): add CORD->LayoutLMv3 feature converter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Evaluation metrics (`kie/metrics.py`)

**Files:**
- Create: `docintel/src/docintel/kie/metrics.py`
- Test: `docintel/tests/test_kie_metrics.py`

**Interfaces:**
- Consumes: `id2label: Mapping[int, str]` (from Task 2).
- Produces:
  - `align_predictions(predictions, label_ids, id2label) -> tuple[list[list[str]], list[list[str]]]` — drops `-100` positions, maps ids to BIO strings, returns `(true_labels, pred_labels)` per sequence.
  - `compute_seqeval_metrics(predictions, label_ids, id2label) -> dict[str, float]` — overall `precision`/`recall`/`f1`/`accuracy` plus `f1_<category>` per field.

- [ ] **Step 1: Write the failing test**

Create `docintel/tests/test_kie_metrics.py`:

```python
"""Tests for seqeval-based KIE metrics."""

from __future__ import annotations

import numpy as np

from docintel.kie.metrics import align_predictions, compute_seqeval_metrics

ID2LABEL = {0: "O", 1: "B-menu.nm", 2: "I-menu.nm", 3: "B-total.total_price"}


def test_align_predictions_drops_ignored_index() -> None:
    # logits over 4 classes for 3 tokens; second token is masked with -100.
    predictions = np.array([[[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 0, 9]]], dtype=float)
    label_ids = np.array([[0, -100, 3]])
    true_labels, pred_labels = align_predictions(predictions, label_ids, ID2LABEL)
    assert true_labels == [["O", "B-total.total_price"]]
    assert pred_labels == [["O", "B-total.total_price"]]


def test_compute_metrics_perfect_prediction() -> None:
    predictions = np.array([[[0, 9, 0, 0], [0, 0, 9, 0]]], dtype=float)  # B-menu.nm, I-menu.nm
    label_ids = np.array([[1, 2]])
    metrics = compute_seqeval_metrics(predictions, label_ids, ID2LABEL)
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_menu.nm"] == 1.0


def test_compute_metrics_includes_per_field_keys() -> None:
    predictions = np.array([[[0, 9, 0, 0], [0, 0, 9, 0]]], dtype=float)
    label_ids = np.array([[1, 2]])
    metrics = compute_seqeval_metrics(predictions, label_ids, ID2LABEL)
    assert "f1_menu.nm" in metrics
    assert all(isinstance(value, float) for value in metrics.values())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd docintel && uv run pytest tests/test_kie_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: docintel.kie.metrics`.

- [ ] **Step 3: Implement `metrics.py`**

Create `docintel/src/docintel/kie/metrics.py`:

```python
"""seqeval entity-level metrics for CORD token classification.

Produces the overall precision/recall/F1/accuracy plus a per-field F1 for
every CORD category, ready to log to MLflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

_IGNORE_INDEX = -100


def align_predictions(
    predictions: Any,
    label_ids: Any,
    id2label: Mapping[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Argmax predictions, drop ``-100`` positions, map ids to BIO strings."""
    preds = np.asarray(predictions).argmax(axis=-1)
    labels = np.asarray(label_ids)

    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []
    for pred_row, label_row in zip(preds, labels, strict=True):
        true_seq: list[str] = []
        pred_seq: list[str] = []
        for pred_id, label_id in zip(pred_row, label_row, strict=True):
            if int(label_id) == _IGNORE_INDEX:
                continue
            true_seq.append(id2label[int(label_id)])
            pred_seq.append(id2label[int(pred_id)])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    return true_labels, pred_labels


def compute_seqeval_metrics(
    predictions: Any,
    label_ids: Any,
    id2label: Mapping[int, str],
) -> dict[str, float]:
    """Return overall + per-field entity-level metrics."""
    true_labels, pred_labels = align_predictions(predictions, label_ids, id2label)

    metrics: dict[str, float] = {
        "precision": float(precision_score(true_labels, pred_labels)),
        "recall": float(recall_score(true_labels, pred_labels)),
        "f1": float(f1_score(true_labels, pred_labels)),
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
    }

    report: dict[str, Any] = classification_report(
        true_labels, pred_labels, output_dict=True, zero_division=0
    )
    for field, scores in report.items():
        if field in {"micro avg", "macro avg", "weighted avg"}:
            continue
        if isinstance(scores, Mapping):
            metrics[f"f1_{field}"] = float(scores["f1-score"])
    return metrics
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd docintel && uv run pytest tests/test_kie_metrics.py -v`
Expected: PASS (3 passed). Note: this task needs `seqeval` (the `train` extra). If the environment lacks it, install with `uv pip install seqeval` before running.

- [ ] **Step 5: Run the gate and commit**

```bash
cd docintel && uv run ruff check . && uv run mypy src && uv run pytest tests/test_kie_metrics.py
git add docintel/src/docintel/kie/metrics.py docintel/tests/test_kie_metrics.py
git commit -m "feat(kie): add seqeval entity-level metrics (overall + per-field)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Training builder + bundle saver (`kie/train.py`)

The full train loop only runs on Colab GPU, so this task unit-tests the **pure helpers** (`build_training_arguments`, `collect_repro_params`, `save_bundle`) and assembles a thin `run_training` orchestrator that the notebook calls. Heavy imports (`transformers`, `torch`) are done inside the functions so the module imports cheaply on CPU.

**Files:**
- Create: `docintel/src/docintel/kie/train.py`
- Test: `docintel/tests/test_kie_train.py`

**Interfaces:**
- Consumes: `TrainingConfig` (Task 1), `build_label_maps` (Task 2), `compute_seqeval_metrics` (Task 4).
- Produces:
  - `build_training_arguments(config, output_dir) -> TrainingArguments` — maps the dataclass to HF args.
  - `collect_repro_params(config, dataset_revision, git_sha) -> dict[str, str]` — flat string params for MLflow.
  - `save_bundle(model, processor, id2label, metrics, bundle_dir) -> Path` — writes `model/`, `processor/`, `label_map.json`, `metrics.json` under `bundle_dir`.
  - `run_training(...) -> Path` — Colab orchestrator (not unit-tested); returns the bundle path.

- [ ] **Step 1: Write the failing test (pure helpers only)**

Create `docintel/tests/test_kie_train.py`:

```python
"""Tests for the pure, CPU-testable helpers in kie.train."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docintel.kie.config import TrainingConfig
from docintel.kie.train import collect_repro_params, save_bundle


def test_collect_repro_params_is_flat_string_map() -> None:
    config = TrainingConfig(model_name="microsoft/layoutlmv3-base", seed=7)
    params = collect_repro_params(config, dataset_revision="abc123", git_sha="deadbee")
    assert params["seed"] == "7"
    assert params["model_name"] == "microsoft/layoutlmv3-base"
    assert params["dataset_revision"] == "abc123"
    assert params["git_sha"] == "deadbee"
    assert all(isinstance(value, str) for value in params.values())


def test_save_bundle_writes_expected_layout(tmp_path: Path) -> None:
    saved: dict[str, Path] = {}

    class FakeSaveable:
        def __init__(self, name: str) -> None:
            self._name = name

        def save_pretrained(self, path: str) -> None:
            saved[self._name] = Path(path)
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "marker").write_text(self._name, encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    out = save_bundle(
        model=FakeSaveable("model"),
        processor=FakeSaveable("processor"),
        id2label={0: "O", 1: "B-menu.nm"},
        metrics={"f1": 0.95},
        bundle_dir=bundle_dir,
    )

    assert out == bundle_dir
    assert (bundle_dir / "model" / "marker").read_text(encoding="utf-8") == "model"
    assert (bundle_dir / "processor" / "marker").read_text(encoding="utf-8") == "processor"
    assert json.loads((bundle_dir / "label_map.json").read_text(encoding="utf-8")) == {
        "0": "O",
        "1": "B-menu.nm",
    }
    assert json.loads((bundle_dir / "metrics.json").read_text(encoding="utf-8")) == {"f1": 0.95}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd docintel && uv run pytest tests/test_kie_train.py -v`
Expected: FAIL — `ModuleNotFoundError: docintel.kie.train`.

- [ ] **Step 3: Implement `train.py`**

Create `docintel/src/docintel/kie/train.py`:

```python
"""LayoutLMv3 fine-tuning builder and bundle saver.

The pure helpers (training-args mapping, repro params, bundle writing) are
CPU-testable. ``run_training`` assembles them with the Hugging Face Trainer and
runs the actual fine-tune — that executes on Colab GPU. Heavy libraries are
imported inside functions so this module loads cheaply on the laptop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from docintel.kie.config import TrainingConfig


def build_training_arguments(config: TrainingConfig, output_dir: str) -> Any:
    """Map ``TrainingConfig`` to Hugging Face ``TrainingArguments``."""
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
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
    )


def collect_repro_params(
    config: TrainingConfig,
    dataset_revision: str,
    git_sha: str,
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
        "dataset_revision": dataset_revision,
        "git_sha": git_sha,
    }


def save_bundle(
    model: Any,
    processor: Any,
    id2label: Mapping[int, str],
    metrics: Mapping[str, float],
    bundle_dir: Path,
) -> Path:
    """Write a self-contained model bundle for download + import."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(bundle_dir / "model"))
    processor.save_pretrained(str(bundle_dir / "processor"))
    (bundle_dir / "label_map.json").write_text(
        json.dumps({str(k): v for k, v in id2label.items()}), encoding="utf-8"
    )
    (bundle_dir / "metrics.json").write_text(json.dumps(dict(metrics)), encoding="utf-8")
    return bundle_dir


def run_training(
    config: TrainingConfig,
    train_dataset: Any,
    eval_dataset: Any,
    id2label: Mapping[int, str],
    label2id: Mapping[str, int],
    processor: Any,
    bundle_dir: Path,
    dataset_revision: str,
    git_sha: str,
    output_dir: str = "outputs",
) -> Path:
    """Fine-tune LayoutLMv3, log to MLflow, and save a bundle. Runs on Colab.

    Assumes ``mlflow.set_tracking_uri`` has been pointed at the Colab file-store
    by the caller (the notebook). Returns the bundle directory.
    """
    import mlflow
    from transformers import (
        AutoModelForTokenClassification,
        Trainer,
        set_seed,
    )

    from docintel.kie.metrics import compute_seqeval_metrics

    set_seed(config.seed)
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(id2label),
        id2label=dict(id2label),
        label2id=dict(label2id),
    )
    args = build_training_arguments(config, output_dir)

    def _metrics(eval_pred: Any) -> dict[str, float]:
        predictions, labels = eval_pred
        return compute_seqeval_metrics(predictions, labels, id2label)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_metrics,
    )

    with mlflow.start_run():
        mlflow.log_params(collect_repro_params(config, dataset_revision, git_sha))
        trainer.train()
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        bundle = save_bundle(model, processor, id2label, eval_metrics, bundle_dir)
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
    return bundle
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd docintel && uv run pytest tests/test_kie_train.py -v`
Expected: PASS (2 passed). The test never imports `transformers` (only the pure helpers), so it runs on CPU without the `train` extra.

- [ ] **Step 5: Run the gate and commit**

```bash
cd docintel && uv run ruff check . && uv run mypy src && uv run pytest tests/test_kie_train.py
git add docintel/src/docintel/kie/train.py docintel/tests/test_kie_train.py
git commit -m "feat(kie): add LayoutLMv3 training builder and bundle saver

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Import + register script (`kie/import_run.py`)

The laptop-side handoff. Ingests a downloaded bundle into the local MLflow + MinIO and registers the model. Tested against a temporary MLflow file-store (no live MinIO needed).

**Files:**
- Create: `docintel/src/docintel/kie/import_run.py`
- Test: `docintel/tests/test_kie_import.py`

**Interfaces:**
- Consumes: `Settings.mlflow_tracking_uri`, `Settings.kie_registered_model_name` (Task 1), the bundle layout from `save_bundle` (Task 5).
- Produces:
  - `import_bundle(bundle_dir, settings, tracking_uri=None) -> str` — logs params (from `metrics.json` + `label_map.json`), logs metrics, logs the bundle artifacts, registers the model under `settings.kie_registered_model_name`, returns the registered model **version** as a string.
  - `main() -> None` — CLI entry point (`docintel-import-kie`), `--bundle-dir` arg.

- [ ] **Step 1: Write the failing test**

Create `docintel/tests/test_kie_import.py`:

```python
"""Tests for importing a bundle into MLflow + registering it."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from docintel.config import Settings
from docintel.kie.import_run import import_bundle


def _make_bundle(bundle_dir: Path) -> Path:
    (bundle_dir / "model").mkdir(parents=True)
    (bundle_dir / "model" / "config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "processor").mkdir(parents=True)
    (bundle_dir / "processor" / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "label_map.json").write_text(
        json.dumps({"0": "O", "1": "B-menu.nm"}), encoding="utf-8"
    )
    (bundle_dir / "metrics.json").write_text(json.dumps({"f1": 0.95}), encoding="utf-8")
    return bundle_dir


def test_import_bundle_registers_model(tmp_path: Path) -> None:
    tracking_uri = (tmp_path / "mlruns").as_uri()
    bundle = _make_bundle(tmp_path / "bundle")
    settings = Settings(kie_registered_model_name="cord-layoutlmv3")

    version = import_bundle(bundle, settings, tracking_uri=tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model = client.get_registered_model("cord-layoutlmv3")
    assert model.name == "cord-layoutlmv3"
    assert version == "1"

    # The run logged the f1 metric and the label count param.
    run_id = client.get_model_version("cord-layoutlmv3", version).run_id
    run = client.get_run(run_id)
    assert run.data.metrics["f1"] == 0.95
    assert run.data.params["num_labels"] == "2"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd docintel && uv run pytest tests/test_kie_import.py -v`
Expected: FAIL — `ModuleNotFoundError: docintel.kie.import_run`.

- [ ] **Step 3: Implement `import_run.py`**

Create `docintel/src/docintel/kie/import_run.py`:

```python
"""Import a Colab-trained model bundle into the local MLflow + MinIO registry.

Run on the laptop after downloading the bundle from Colab::

    docintel-import-kie --bundle-dir ./cord-layoutlmv3-bundle

Logs the run's params + metrics, uploads the bundle artifacts to the MLflow
artifact store (MinIO in docker-compose), and registers the model under the
configured registry name.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow

from docintel.config import Settings, get_settings
from docintel.logging_config import configure_logging

logger = logging.getLogger("docintel.kie.import")


def import_bundle(
    bundle_dir: Path,
    settings: Settings,
    tracking_uri: str | None = None,
) -> str:
    """Log + register a downloaded bundle; return the new model version."""
    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("cord-kie")

    label_map: dict[str, str] = json.loads(
        (bundle_dir / "label_map.json").read_text(encoding="utf-8")
    )
    metrics: dict[str, float] = json.loads(
        (bundle_dir / "metrics.json").read_text(encoding="utf-8")
    )

    with mlflow.start_run() as run:
        mlflow.log_param("num_labels", str(len(label_map)))
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
        model_uri = f"runs:/{run.info.run_id}/bundle"
        registered = mlflow.register_model(model_uri, settings.kie_registered_model_name)

    logger.info(
        "kie.import.done",
        extra={"model": settings.kie_registered_model_name, "version": registered.version},
    )
    return str(registered.version)


def main() -> None:
    """CLI entry point for ``docintel-import-kie``."""
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Import a KIE bundle into MLflow.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Path to the downloaded model bundle directory.",
    )
    args = parser.parse_args()
    version = import_bundle(args.bundle_dir, settings)
    logger.info("kie.import.registered", extra={"version": version})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd docintel && uv run pytest tests/test_kie_import.py -v`
Expected: PASS (1 passed). Needs the `kie` extra (`mlflow`); install with `uv pip install mlflow` if missing.

- [ ] **Step 5: Run the gate and commit**

```bash
cd docintel && uv run ruff check . && uv run mypy src && uv run pytest tests/test_kie_import.py
git add docintel/src/docintel/kie/import_run.py docintel/tests/test_kie_import.py
git commit -m "feat(kie): add bundle import + MLflow model registration CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Colab orchestration notebook (`notebooks/phase2_kie_layoutlmv3.ipynb`)

A thin notebook that runs on Colab and calls the `src/docintel/kie/` functions. It is not unit-tested (it executes on Colab GPU); the deliverable is a valid, well-structured notebook whose cells use only the public interfaces built in Tasks 1–5. Verification is structural: it parses as a valid notebook and references the right symbols.

**Files:**
- Create: `docintel/notebooks/phase2_kie_layoutlmv3.ipynb`
- Verify: a small inline check (below) — no pytest test file.

**Interfaces:**
- Consumes: all of `docintel.kie.*` built in Tasks 1–6.
- Produces: nothing other code imports.

- [ ] **Step 1: Author the notebook**

Create `docintel/notebooks/phase2_kie_layoutlmv3.ipynb` as a valid nbformat-4 JSON notebook with these cells in order. Each fenced block is one code cell (preceded by a markdown cell with the stated heading).

**Markdown:** `# Phase 2 — Fine-tune LayoutLMv3 on CORD (Colab GPU)` with a sentence: *Runs on Colab with a GPU runtime. Trains, tracks to a local MLflow file-store, and saves a bundle to download for `docintel-import-kie`.*

**Cell — install (markdown heading `## 1. Install`):**
```python
# Use a GPU runtime (Runtime > Change runtime type > GPU).
!pip install -q "git+https://github.com/<your-org>/docintel.git@worktree-phase-2-kie#subdirectory=docintel[train,kie]"
# If installing from a local upload instead, pip install the uploaded wheel with the same extras.
```

**Cell — imports + seed (`## 2. Setup`):**
```python
import json, subprocess
from pathlib import Path

import mlflow
from datasets import load_dataset
from transformers import LayoutLMv3Processor

from docintel.config import get_settings
from docintel.kie.config import TrainingConfig
from docintel.kie.labels import build_label_list, build_label_maps
from docintel.kie.dataset import collect_categories, encode_example, parse_cord_example
from docintel.kie.train import run_training

settings = get_settings()
config = TrainingConfig.from_settings(settings)
mlflow.set_tracking_uri("file:./mlruns")   # local file-store on the Colab VM
mlflow.set_experiment("cord-kie")
```

**Cell — load CORD + build labels (`## 3. Data`):**
```python
DATASET_REVISION = "main"  # pin to a specific revision for reproducibility
raw = load_dataset("naver-clova-ix/cord-v2", revision=DATASET_REVISION)
train_gt = [json.loads(ex["ground_truth"]) for ex in raw["train"]]
label_list = build_label_list(collect_categories(train_gt))
id2label, label2id = build_label_maps(label_list)
processor = LayoutLMv3Processor.from_pretrained(config.model_name, apply_ocr=False)
```

**Cell — encode splits (`## 4. Features`):**
```python
def _encode_split(split):
    def _map(ex):
        gt = json.loads(ex["ground_truth"])
        words, boxes, bio = parse_cord_example(gt)
        enc = encode_example(words, boxes, bio, processor, label2id)
        enc["pixel_values"] = processor.image_processor(ex["image"], return_tensors="np")["pixel_values"][0]
        return enc
    return split.map(_map, remove_columns=split.column_names)

train_ds = _encode_split(raw["train"])
eval_ds = _encode_split(raw["validation"])
```

**Cell — train + log (`## 5. Train & track`):**
```python
git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
bundle = run_training(
    config=config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    id2label=id2label,
    label2id=label2id,
    processor=processor,
    bundle_dir=Path("cord-layoutlmv3-bundle"),
    dataset_revision=DATASET_REVISION,
    git_sha=git_sha,
)
print("Bundle saved to:", bundle)
```

**Cell — zip for download (`## 6. Download`):**
```python
import shutil
from google.colab import files  # type: ignore
archive = shutil.make_archive("cord-layoutlmv3-bundle", "zip", "cord-layoutlmv3-bundle")
files.download(archive)
```

**Markdown — final (`## 7. Next: import on the laptop`):** *Download the zip, unzip it, then on the laptop (with docker-compose MLflow + MinIO up) run `docintel-import-kie --bundle-dir ./cord-layoutlmv3-bundle`. Then smoke-test: `from transformers import AutoModelForTokenClassification, LayoutLMv3Processor; AutoModelForTokenClassification.from_pretrained("cord-layoutlmv3-bundle/model")` loads on CPU and a single forward pass returns logits of shape `[1, seq, num_labels]`.*

- [ ] **Step 2: Verify the notebook is structurally valid**

Run:
```bash
cd docintel && uv run python -c "import nbformat; nb = nbformat.read('notebooks/phase2_kie_layoutlmv3.ipynb', as_version=4); nbformat.validate(nb); src = '\n'.join(c.source for c in nb.cells); assert 'run_training' in src and 'build_label_list' in src and 'parse_cord_example' in src and 'file:./mlruns' in src; print('notebook OK,', len(nb.cells), 'cells')"
```
Expected: `notebook OK, N cells` (no validation error). `nbformat` ships with Jupyter; if absent, `uv pip install nbformat` first.

- [ ] **Step 3: Commit**

```bash
git add docintel/notebooks/phase2_kie_layoutlmv3.ipynb
git commit -m "feat(kie): add Colab notebook orchestrating CORD LayoutLMv3 fine-tune

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Manual Colab Run (after the branch is built — the user's step)

Not an automated task. Once Tasks 1–7 are merged-ready:

1. **User:** open `notebooks/phase2_kie_layoutlmv3.ipynb` in Colab on a GPU runtime, run all cells, download `cord-layoutlmv3-bundle.zip`.
2. **Together (laptop):** `docker compose up -d mlflow minio`, unzip the bundle, run `docintel-import-kie --bundle-dir ./cord-layoutlmv3-bundle`.
3. **Smoke test (laptop, CPU):** load `cord-layoutlmv3-bundle/model` with `AutoModelForTokenClassification.from_pretrained` and run one forward pass — confirm logits shape `[1, seq, num_labels]`.
4. **Verify done-when:** the model `cord-layoutlmv3` appears in the MLflow registry with `f1` and per-field metrics logged.

---

## Self-Review (completed during planning)

- **Spec coverage:** full CORD schema → Tasks 2+3; `layoutlmv3-base` → Task 1 (Settings) + Task 5; Colab file-store + import script → Task 5 (`file:./mlruns`) + Task 6; thin notebook + `src/kie` → Tasks 2–6 build `src/kie`, Task 7 is the thin notebook; `TrainingConfig` → Task 1; `kie_model_name`/`kie_registered_model_name` → Task 1; labels/dataset/metrics/import modules → Tasks 2/3/4/6; per-field F1 → Task 4; reproducibility (seed, git sha, dataset revision) → Task 5 `collect_repro_params` + notebook; deps split (Colab `train` vs laptop `kie`) → Task 1; CPU smoke test → Manual Colab Run section. No uncovered spec requirement.
- **Placeholder scan:** none — every code step shows complete, runnable code with no TODO/TBD markers.
- **Type consistency:** `build_label_list`/`build_label_maps` signatures match across Tasks 2/5/7; `parse_cord_example` returns `(words, boxes, bio_labels)` consumed identically in Tasks 3/7; `encode_example(words, boxes, bio_labels, processor, label2id)` arg order matches Task 3 test, impl, and notebook; `save_bundle`/`collect_repro_params` signatures match Task 5 test + `run_training`; `import_bundle(bundle_dir, settings, tracking_uri=None) -> str` matches Task 6 test + CLI.
```
