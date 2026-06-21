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
