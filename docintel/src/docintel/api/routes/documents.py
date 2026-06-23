"""Retrieval endpoints for persisted documents."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response

from docintel.api.routes.extract import get_s3_client
from docintel.config import Settings, get_settings
from docintel.schema import Document
from docintel.storage.db import get_document, init_db
from docintel.storage.objects import get_image

router = APIRouter(tags=["documents"])


def _lookup(settings: Settings, document_id: str) -> tuple[Document, str]:
    init_db(settings.sqlite_path)
    found = get_document(settings.sqlite_path, document_id)
    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return found


@router.get("/documents/{document_id}", response_model=Document, summary="Retrieve a document")
def read_document(
    document_id: str,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Document:
    """Return a previously extracted document by id."""
    document, _ = _lookup(settings, document_id)
    return document


@router.get("/documents/{document_id}/image", summary="Retrieve a document's source image")
def read_document_image(
    document_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
    s3: Any = Depends(get_s3_client),  # noqa: B008
) -> Response:
    """Stream the stored source image for a document by id."""
    _, image_key = _lookup(settings, document_id)
    data = get_image(s3, settings.minio_bucket, image_key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")
    media_type = "image/png" if image_key.endswith(".png") else "image/jpeg"
    return Response(content=data, media_type=media_type)
