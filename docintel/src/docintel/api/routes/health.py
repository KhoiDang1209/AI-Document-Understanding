"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from docintel import __version__
from docintel.config import get_settings

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str
    service: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    """Return service liveness and identity.

    Readiness checks for backing services (MLflow, Qdrant, MinIO) are added in
    later phases as those dependencies come online.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
