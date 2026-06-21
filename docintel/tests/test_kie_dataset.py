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
