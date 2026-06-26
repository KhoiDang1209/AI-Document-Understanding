"""FastAPI application factory and ASGI entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator

from docintel import __version__
from docintel.api.metrics import build_metrics
from docintel.api.routes import ask, contracts, documents, extract, health
from docintel.config import get_settings
from docintel.logging_config import configure_logging

logger = logging.getLogger("docintel.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging on startup and emit lifecycle events."""
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.ocr_engine = None
    app.state.kie_backend = None
    app.state.s3_client = None
    app.state.contract_extractor = None
    app.state.rag_store = None
    app.state.rag_llm = None
    logger.info(
        "service.startup",
        extra={"environment": settings.environment, "version": __version__},
    )
    yield
    logger.info("service.shutdown")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        summary="Document AI: images to validated, queryable structured data.",
        lifespan=lifespan,
    )
    registry = CollectorRegistry()
    app.state.metrics = build_metrics(registry)
    Instrumentator(registry=registry).instrument(app).expose(app)
    app.include_router(health.router)
    app.include_router(extract.router)
    app.include_router(documents.router)
    app.include_router(contracts.router)
    app.include_router(ask.router)
    return app


app = create_app()
