"""Tests for the optional OpenCV preprocessing step."""

from __future__ import annotations

import cv2
import numpy as np

from docintel.config import Settings
from docintel.pipeline.preprocess import _deskew, _resize_longest, preprocess


def _row_sum_variance(image: np.ndarray) -> float:
    """Higher when horizontal structure is well aligned with image rows."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.var(gray.sum(axis=1)))


def _striped_image() -> np.ndarray:
    img = np.full((400, 400, 3), 255, np.uint8)
    for y in range(40, 360, 60):
        cv2.rectangle(img, (40, y), (360, y + 12), (0, 0, 0), -1)
    return img


def test_deskew_reduces_skew() -> None:
    base = _striped_image()
    matrix = cv2.getRotationMatrix2D((200, 200), 10, 1.0)
    skewed = cv2.warpAffine(base, matrix, (400, 400), borderValue=(255, 255, 255))
    deskewed = _deskew(skewed)
    assert _row_sum_variance(deskewed) > _row_sum_variance(skewed)


def test_resize_caps_longest_side() -> None:
    img = np.full((1000, 3000, 3), 255, np.uint8)
    out = _resize_longest(img, 2000)
    assert max(out.shape[:2]) == 2000


def test_resize_does_not_upscale() -> None:
    img = np.full((100, 200, 3), 255, np.uint8)
    out = _resize_longest(img, 2000)
    assert out.shape[:2] == (100, 200)


def test_preprocess_returns_capped_ndarray() -> None:
    img = np.full((1000, 3000, 3), 255, np.uint8)
    out = preprocess(img, Settings(preprocess_max_dim=2000))
    assert isinstance(out, np.ndarray)
    assert max(out.shape[:2]) <= 2000
