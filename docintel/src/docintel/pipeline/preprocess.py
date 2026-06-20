"""Optional OpenCV preprocessing (deskew, denoise, resize) for OCR input.

Pure functions over BGR image arrays — no I/O, no global state. Applied only
when ``settings.preprocess_enabled`` is true; the default ``/extract`` path
skips it entirely.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from docintel.config import Settings

Image = NDArray[np.uint8]

# Fixed algorithm parameter (not deployment config): Non-Local-Means denoise
# filter strength. The latency-relevant knob (max longest side) is a setting.
_DENOISE_STRENGTH = 10


def _estimate_skew_angle(image: Image) -> float:
    """Estimate a small correction angle (degrees) to level the text."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 10:
        return 0.0
    angle = float(cv2.minAreaRect(coords.astype(np.float32))[-1])
    # OpenCV >=4.5 returns angle in [0, 90); map to a small signed correction.
    if angle > 45:
        angle -= 90
    return -angle


def _deskew(image: Image) -> Image:
    angle = _estimate_skew_angle(image)
    if abs(angle) < 0.1:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    # Annotate the local: cv2 is untyped (Any), and strict mypy's
    # warn_return_any rejects returning Any from an -> Image function.
    rotated: Image = cv2.warpAffine(  # type: ignore[assignment]
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _denoise(image: Image) -> Image:
    result: Image = cv2.fastNlMeansDenoisingColored(  # type: ignore[assignment]
        image, None, _DENOISE_STRENGTH, _DENOISE_STRENGTH, 7, 21
    )
    return result


def _resize_longest(image: Image, max_dim: int) -> Image:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_size = (round(width * scale), round(height * scale))
    resized: Image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)  # type: ignore[assignment]
    return resized


def preprocess(image: Image, settings: Settings) -> Image:
    """Deskew, denoise, and downscale a BGR image for OCR."""
    result = _deskew(image)
    result = _denoise(result)
    return _resize_longest(result, settings.preprocess_max_dim)
