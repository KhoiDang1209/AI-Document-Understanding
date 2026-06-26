"""The /contracts endpoints: PDF -> ingest -> QA extract -> persist -> ContractDocument."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from docintel.api.metrics import Metrics, record_contract_extraction
from docintel.api.routes.ask import get_rag_store_optional
from docintel.api.routes.extract import get_metrics, get_ocr_engine, get_s3_client
from docintel.config import Settings, get_settings
from docintel.contracts.extractor import ContractExtractor, CuadQaOnnxExtractor
from docintel.contracts.ingest import ingest_pdf
from docintel.contracts.schema import ContractDocument, build_derived
from docintel.pipeline.ocr import OCREngine
from docintel.rag.index import index_contract
from docintel.storage.contracts_db import get_contract, init_contracts_db, save_contract
from docintel.storage.objects import ensure_bucket, put_image

logger = logging.getLogger("docintel.api.contracts")
router = APIRouter(tags=["contracts"])

_PDF_TYPE = "application/pdf"


def get_contract_extractor(request: Request) -> ContractExtractor:
    """Return the process-wide contract extractor, loading it once on first use."""
    backend: ContractExtractor | None = getattr(request.app.state, "contract_extractor", None)
    if backend is None:
        backend = CuadQaOnnxExtractor.load(get_settings())
        request.app.state.contract_extractor = backend
    return backend


@router.post(
    "/contracts/extract",
    response_model=ContractDocument,
    summary="Extract structured clauses from a contract PDF",
)
async def extract_contract(
    file: UploadFile,
    settings: Settings = Depends(get_settings),  # noqa: B008
    engine: OCREngine = Depends(get_ocr_engine),  # noqa: B008
    extractor: ContractExtractor = Depends(get_contract_extractor),  # noqa: B008
    s3: Any = Depends(get_s3_client),  # noqa: B008
    metrics: Metrics = Depends(get_metrics),  # noqa: B008
    rag_store: Any = Depends(get_rag_store_optional),  # noqa: B008
) -> ContractDocument:
    """Ingest a PDF, extract clauses, persist, and return a ContractDocument."""
    if file.content_type != _PDF_TYPE:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type!r}; use application/pdf.",
        )
    data = await file.read()
    max_bytes = int(settings.contract_max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Upload exceeds the {settings.contract_max_upload_mb} MB limit.",
        )

    start = time.perf_counter()
    ingested = ingest_pdf(data, engine, settings)
    clauses = extractor.extract(ingested.text)
    doc = ContractDocument(
        id=uuid.uuid4().hex,
        source=ingested.source,
        clauses=clauses,
        derived=build_derived(clauses),
        page_count=ingested.page_count,
        created_at=datetime.now(UTC).isoformat(),
    )
    latency_ms = (time.perf_counter() - start) * 1000

    pdf_key = f"{doc.id}.pdf"
    ensure_bucket(s3, settings.minio_bucket)
    put_image(s3, settings.minio_bucket, pdf_key, data, _PDF_TYPE)
    init_contracts_db(settings.sqlite_path)
    save_contract(settings.sqlite_path, doc, pdf_key)

    chunks_indexed = 0
    if rag_store is not None:
        try:
            chunks_indexed = index_contract(doc.id, ingested.text, doc.clauses, rag_store, settings)
        except Exception:
            logger.warning(
                "contracts.extract.index_failed",
                extra={"contract_id": doc.id},
                exc_info=True,
            )

    logger.info(
        "contracts.extract.complete",
        extra={
            "contract_id": doc.id,
            "latency_ms": round(latency_ms, 2),
            "clauses": len(doc.clauses),
            "source": doc.source,
            "chunks_indexed": chunks_indexed,
        },
    )
    record_contract_extraction(metrics, doc)
    return doc


@router.get(
    "/contracts/{contract_id}",
    response_model=ContractDocument,
    summary="Retrieve an extracted contract",
)
def read_contract(
    contract_id: str,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> ContractDocument:
    """Return a previously extracted contract by id."""
    init_contracts_db(settings.sqlite_path)
    found = get_contract(settings.sqlite_path, contract_id)
    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found.")
    return found[0]
