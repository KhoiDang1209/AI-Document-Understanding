"""Domain models for the OCR pipeline stage.

These models are returned directly as the ``/extract`` response, so the
pipeline's internal contract and the API contract are one and the same.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OCRWord(BaseModel):
    """A single recognized word with its pixel bounding box and confidence."""

    text: str
    bbox: tuple[int, int, int, int] = Field(
        description="Pixel box [x_min, y_min, x_max, y_max], top-left origin."
    )
    confidence: float


class OCRResult(BaseModel):
    """Full OCR output for one image: document text, per-word boxes, and dimensions."""

    text: str
    words: list[OCRWord]
    confidence: float = Field(description="Mean of word confidences; 0.0 if none.")
    image_width: int
    image_height: int
