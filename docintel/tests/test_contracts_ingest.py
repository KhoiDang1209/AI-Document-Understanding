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
        return OCRResult(
            text="SCANNED CLAUSE", words=[], confidence=0.0, image_width=4, image_height=4
        )

    doc = ingest_pdf(b"%PDF-fake", ocr_engine=_fake_ocr, settings=settings)
    assert doc.source == "ocr"
    assert "SCANNED CLAUSE" in doc.text
