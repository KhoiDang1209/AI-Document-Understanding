"""Slow integration test: the real docTR engine reads known text."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from docintel.config import Settings
from docintel.pipeline.ocr import load_doctr_engine


@pytest.mark.slow
def test_doctr_reads_known_text() -> None:
    canvas = Image.new("RGB", (600, 200), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 70), "TOTAL 12.99", fill="black", font=ImageFont.load_default(size=48))
    bgr = np.array(canvas)[:, :, ::-1].copy()  # RGB -> BGR

    engine = load_doctr_engine(Settings())
    result = engine(bgr)

    assert "total" in result.text.lower()
    assert result.confidence > 0.0
