"""The /extract endpoint: image upload -> OCR result."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from docintel.config import Settings, get_settings
from docintel.pipeline.ocr import Image, OCREngine, load_doctr_engine
from docintel.pipeline.preprocess import preprocess
from docintel.pipeline.types import OCRResult

logger = logging.getLogger("docintel.api.extract")
router = APIRouter(tags=["pipeline"])

_ACCEPTED_TYPES = {"image/png", "image/jpeg"}


def get_ocr_engine(request: Request) -> OCREngine:
    """Return the process-wide docTR engine, loading it once on first use."""
    # Annotate the local: getattr returns Any, and strict mypy's
    # warn_return_any rejects returning Any from an -> OCREngine function.
    engine: OCREngine | None = getattr(request.app.state, "ocr_engine", None)
    if engine is None:
        engine = load_doctr_engine(get_settings())
        request.app.state.ocr_engine = engine
    return engine


@router.post("/extract", response_model=OCRResult, summary="OCR an uploaded image")
async def extract(
    file: UploadFile,
    settings: Settings = Depends(get_settings),  # noqa: B008
    engine: OCREngine = Depends(get_ocr_engine),  # noqa: B008
) -> OCRResult:
    """Run OCR on an uploaded PNG/JPEG and return text, word boxes, and confidence."""
    if file.content_type not in _ACCEPTED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type!r}; use PNG or JPEG.",
        )

    data = await file.read()
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
        )

    # Annotate: cv2 is untyped (Any); strict mypy's warn_return_any needs the explicit cast.
    image: Image | None = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # type: ignore[assignment]
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the uploaded bytes as an image.",
        )

    if settings.preprocess_enabled:
        image = preprocess(image, settings)

    start = time.perf_counter()
    result = engine(image)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "extract.complete",
        extra={
            "latency_ms": round(latency_ms, 2),
            "word_count": len(result.words),
            "mean_confidence": round(result.confidence, 4),
        },
    )
    return result
