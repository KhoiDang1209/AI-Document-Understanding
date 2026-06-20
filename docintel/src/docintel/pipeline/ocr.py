"""OCR engine interface and the default docTR implementation."""

from __future__ import annotations

from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from docintel.config import Settings
from docintel.pipeline.types import OCRResult, OCRWord

Image = NDArray[np.uint8]


class OCREngine(Protocol):
    """A callable that turns a BGR image array into an :class:`OCRResult`."""

    def __call__(self, image: Image) -> OCRResult: ...


def _ocr_result_from_export(export: dict[str, Any], width: int, height: int) -> OCRResult:
    """Map a docTR ``Document.export()`` dict to an :class:`OCRResult`."""
    words: list[OCRWord] = []
    lines_text: list[str] = []
    pages = export.get("pages", [])
    if pages:
        for block in pages[0].get("blocks", []):
            for line in block.get("lines", []):
                line_words: list[str] = []
                for word in line.get("words", []):
                    (x0, y0), (x1, y1) = word["geometry"]
                    bbox = (
                        round(x0 * width),
                        round(y0 * height),
                        round(x1 * width),
                        round(y1 * height),
                    )
                    words.append(
                        OCRWord(
                            text=word["value"],
                            bbox=bbox,
                            confidence=float(word["confidence"]),
                        )
                    )
                    line_words.append(word["value"])
                if line_words:
                    lines_text.append(" ".join(line_words))
    confidence = sum(w.confidence for w in words) / len(words) if words else 0.0
    return OCRResult(
        text="\n".join(lines_text),
        words=words,
        confidence=confidence,
        image_width=width,
        image_height=height,
    )


def load_doctr_engine(settings: Settings) -> OCREngine:
    """Load docTR's pretrained predictor once and return an :class:`OCREngine`.

    ``settings`` is accepted for interface symmetry with future engines (which
    will read their own knobs); the docTR default needs none today.
    """
    from doctr.models import ocr_predictor

    predictor = ocr_predictor(pretrained=True)

    def _run(image: Image) -> OCRResult:
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        document = predictor([rgb])
        export: dict[str, Any] = document.export()
        return _ocr_result_from_export(export, width, height)

    return _run
