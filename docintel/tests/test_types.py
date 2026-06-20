"""Tests for the OCR domain models."""

from __future__ import annotations

from docintel.pipeline.types import OCRResult, OCRWord


def test_ocr_result_shape() -> None:
    result = OCRResult(
        text="TOTAL 12.99",
        words=[OCRWord(text="TOTAL", bbox=(1, 2, 3, 4), confidence=0.9)],
        confidence=0.9,
        image_width=100,
        image_height=200,
    )
    dumped = result.model_dump()
    assert set(dumped) == {"text", "words", "confidence", "image_width", "image_height"}
    assert dumped["words"][0]["bbox"] == (1, 2, 3, 4)
    assert dumped["confidence"] == 0.9
