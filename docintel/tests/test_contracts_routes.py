# tests/test_contracts_routes.py
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docintel.api.main import create_app
from docintel.api.routes.contracts import get_contract_extractor
from docintel.api.routes.extract import get_ocr_engine, get_s3_client
from docintel.config import Settings, get_settings
from docintel.contracts.schema import ExtractedClause
from tests.test_documents import _FakeS3


class _StubExtractor:
    def extract(self, text: str) -> list[ExtractedClause]:
        return [
            ExtractedClause(
                clause_type="Parties", answer_text="Acme", char_start=0, char_end=4, confidence=0.9
            )
        ]


@pytest.fixture
def client(tmp_path: Any) -> Iterator[TestClient]:
    app = create_app()
    settings = Settings(sqlite_path=str(tmp_path / "c.db"))
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_ocr_engine] = lambda: lambda image: None
    app.dependency_overrides[get_s3_client] = lambda: _FakeS3()
    app.dependency_overrides[get_contract_extractor] = lambda: _StubExtractor()
    # ingest is monkeypatched per-test to avoid building a real PDF
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_extract_rejects_non_pdf(client: TestClient) -> None:
    resp = client.post("/contracts/extract", files={"file": ("x.txt", b"hi", "text/plain")})
    assert resp.status_code == 415


def test_extract_and_retrieve(client: TestClient, monkeypatch: Any) -> None:
    from docintel.contracts.ingest import IngestedDoc

    monkeypatch.setattr(
        "docintel.api.routes.contracts.ingest_pdf",
        lambda data, ocr_engine, settings: IngestedDoc(
            text="Acme and Globex", page_count=1, source="digital"
        ),
    )
    resp = client.post(
        "/contracts/extract", files={"file": ("c.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "digital"
    assert body["derived"]["Parties"] == ["Acme"]
    cid = body["id"]

    got = client.get(f"/contracts/{cid}")
    assert got.status_code == 200
    assert got.json()["id"] == cid

    assert client.get("/contracts/missing").status_code == 404
