"""Tests for POST /extract (engine stubbed; no model load)."""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docintel.api.main import create_app
from docintel.api.routes.extract import get_kie_backend, get_ocr_engine, get_s3_client
from docintel.config import Settings, get_settings
from docintel.pipeline.types import OCRResult, OCRWord
from docintel.schema import WordPrediction
from tests.test_documents import _FakeS3


def _png_bytes(size: tuple[int, int] = (16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def _stub_result() -> OCRResult:
    return OCRResult(
        text="HELLO",
        words=[OCRWord(text="HELLO", bbox=(1, 2, 3, 4), confidence=0.9)],
        confidence=0.9,
        image_width=16,
        image_height=16,
    )


class _NoOpBackend:
    def predict(self, ocr: OCRResult, image: Any) -> list[WordPrediction]:
        return []


def _make_stubbed_app(settings: Settings | None = None) -> Any:
    """Build an app with all pipeline dependencies stubbed."""
    app = create_app()
    app.dependency_overrides[get_ocr_engine] = lambda: lambda image: _stub_result()
    app.dependency_overrides[get_kie_backend] = lambda: _NoOpBackend()
    app.dependency_overrides[get_s3_client] = lambda: _FakeS3()
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings
    return app


@pytest.fixture
def client(tmp_path: Any) -> Iterator[TestClient]:
    # Shadows the conftest `client` fixture: overrides the engine so no model loads.
    app = _make_stubbed_app(Settings(sqlite_path=str(tmp_path / "db.sqlite")))
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_extract_returns_document(client: TestClient) -> None:
    resp = client.post("/extract", files={"file": ("r.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "id" in body and "validation" in body and body["currency"]


def test_unsupported_type_returns_415(client: TestClient) -> None:
    resp = client.post("/extract", files={"file": ("r.txt", b"hello", "text/plain")})
    assert resp.status_code == 415


def test_undecodable_image_returns_400(client: TestClient) -> None:
    resp = client.post("/extract", files={"file": ("r.png", b"not-an-image", "image/png")})
    assert resp.status_code == 400


def test_missing_file_returns_422(client: TestClient) -> None:
    resp = client.post("/extract")
    assert resp.status_code == 422


def test_oversize_returns_413(tmp_path: Any) -> None:
    app = _make_stubbed_app(
        Settings(max_upload_mb=0.00001, sqlite_path=str(tmp_path / "db.sqlite"))
    )
    with TestClient(app) as test_client:
        resp = test_client.post(
            "/extract", files={"file": ("r.png", _png_bytes((64, 64)), "image/png")}
        )
    assert resp.status_code == 413


def test_preprocess_branch_runs(tmp_path: Any) -> None:
    app = _make_stubbed_app(
        Settings(preprocess_enabled=True, sqlite_path=str(tmp_path / "db.sqlite"))
    )
    with TestClient(app) as test_client:
        resp = test_client.post(
            "/extract", files={"file": ("r.png", _png_bytes((64, 64)), "image/png")}
        )
    assert resp.status_code == 200
