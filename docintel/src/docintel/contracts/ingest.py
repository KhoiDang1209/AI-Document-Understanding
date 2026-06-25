"""Dual-path contract ingestion: born-digital text (liteparse) or scanned OCR (docTR).

A born-digital PDF carries an extractable text layer; a scanned PDF does not, so
its pages are rasterized and sent through the existing OCR engine. Heavy imports
(liteparse, fitz) live inside functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from docintel.config import Settings
from docintel.pipeline.ocr import Image, OCREngine

_MIN_DIGITAL_CHARS = 20  # below this, treat the document as scanned


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
    from liteparse import LiteParse

    parser = LiteParse(ocr_enabled=False, output_format="text")
    result = parser.parse(data)
    return [page.text for page in result.pages]


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
