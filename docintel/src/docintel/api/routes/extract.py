"""The /extract endpoint: image upload -> OCR -> KIE -> validate -> persist -> Document."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from docintel.api.metrics import Metrics, record_extraction
from docintel.config import Settings, get_settings
from docintel.kie.backend import KIEBackend, LayoutLMv3OnnxBackend
from docintel.kie.decode import build_document
from docintel.pipeline.ocr import Image, OCREngine, load_doctr_engine
from docintel.pipeline.preprocess import preprocess
from docintel.schema import Document
from docintel.storage.db import init_db, save_document
from docintel.storage.objects import ensure_bucket, make_s3_client, put_image
from docintel.validation.rules import validate

logger = logging.getLogger("docintel.api.extract")
router = APIRouter(tags=["pipeline"])

_ACCEPTED_TYPES = {"image/png": "png", "image/jpeg": "jpg"}


def get_ocr_engine(request: Request) -> OCREngine:
    """Return the process-wide docTR engine, loading it once on first use."""
    engine: OCREngine | None = getattr(request.app.state, "ocr_engine", None)
    if engine is None:
        engine = load_doctr_engine(get_settings())
        request.app.state.ocr_engine = engine
    return engine


def get_kie_backend(request: Request) -> KIEBackend:
    """Return the process-wide KIE backend, loading it once on first use."""
    backend: KIEBackend | None = getattr(request.app.state, "kie_backend", None)
    if backend is None:
        backend = LayoutLMv3OnnxBackend.load(get_settings())
        request.app.state.kie_backend = backend
    return backend


def get_s3_client(request: Request) -> Any:
    """Return the process-wide MinIO/S3 client, building it once on first use."""
    client = getattr(request.app.state, "s3_client", None)
    if client is None:
        client = make_s3_client(get_settings())
        request.app.state.s3_client = client
    return client


def get_metrics(request: Request) -> Metrics:
    """Return the per-app metrics set built in create_app."""
    metrics: Metrics = request.app.state.metrics
    return metrics


@router.post("/extract", response_model=Document, summary="Extract structured fields from an image")
async def extract(
    file: UploadFile,
    settings: Settings = Depends(get_settings),  # noqa: B008
    engine: OCREngine = Depends(get_ocr_engine),  # noqa: B008
    backend: KIEBackend = Depends(get_kie_backend),  # noqa: B008
    s3: Any = Depends(get_s3_client),  # noqa: B008
    metrics: Metrics = Depends(get_metrics),  # noqa: B008
) -> Document:
    """Run the full pipeline, persist the result, and return a validated Document."""
    if file.content_type not in _ACCEPTED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type!r}; use PNG or JPEG.",
        )
    data = await file.read()
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
        )
    image: Image | None = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # type: ignore[assignment]
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the uploaded bytes as an image.",
        )
    if settings.preprocess_enabled:
        image = preprocess(image, settings)

    start = time.perf_counter()
    ocr = engine(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictions = backend.predict(ocr, rgb)
    document = build_document(predictions, settings.default_currency)
    document = document.model_copy(
        update={
            "id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    document = document.model_copy(update={"validation": validate(document, settings)})
    latency_ms = (time.perf_counter() - start) * 1000

    image_key = f"{document.id}.{_ACCEPTED_TYPES[file.content_type]}"
    ensure_bucket(s3, settings.minio_bucket)
    put_image(s3, settings.minio_bucket, image_key, data, file.content_type)
    init_db(settings.sqlite_path)
    save_document(settings.sqlite_path, document, image_key)

    logger.info(
        "extract.complete",
        extra={
            "document_id": document.id,
            "latency_ms": round(latency_ms, 2),
            "line_items": len(document.line_items),
            "validation_ok": document.validation.ok,
        },
    )
    record_extraction(metrics, document)
    return document
