"""Tests for /extract (full pipeline) and /documents retrieval, fully stubbed."""

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


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    return buf.getvalue()


def _ocr() -> OCRResult:
    return OCRResult(
        text="Coke 3.000",
        words=[
            OCRWord(text="Coke", bbox=(0, 0, 4, 2), confidence=0.9),
            OCRWord(text="3.000", bbox=(0, 3, 4, 5), confidence=0.9),
        ],
        confidence=0.9,
        image_width=16,
        image_height=16,
    )


class _FakeBackend:
    def predict(self, ocr: OCRResult, image: Any) -> list[WordPrediction]:
        return [
            WordPrediction(text="Coke", box=(0, 0, 4, 2), label="B-menu.nm", confidence=0.9),
            WordPrediction(text="3.000", box=(0, 3, 4, 5), label="B-menu.price", confidence=0.9),
        ]


class _FakeS3:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.buckets: set[str] = {"documents"}

    def head_bucket(self, Bucket: str) -> None:
        return None

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.store[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.store:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        class _Body:
            def __init__(self, d: bytes) -> None:
                self._d = d

            def read(self) -> bytes:
                return self._d

        return {"Body": _Body(self.store[(Bucket, Key)])}


@pytest.fixture
def client(tmp_path: Any) -> Iterator[TestClient]:
    app = create_app()
    s3 = _FakeS3()
    app.dependency_overrides[get_ocr_engine] = lambda: lambda image: _ocr()
    app.dependency_overrides[get_kie_backend] = lambda: _FakeBackend()
    app.dependency_overrides[get_s3_client] = lambda: s3
    app.dependency_overrides[get_settings] = lambda: Settings(
        sqlite_path=str(tmp_path / "db.sqlite")
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_extract_returns_document_and_persists(client: TestClient) -> None:
    resp = client.post("/extract", files={"file": ("r.png", _png(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["line_items"][0]["name"] == "Coke"
    assert body["line_items"][0]["price"] == 3000.0
    assert "validation" in body
    doc_id = body["id"]

    got = client.get(f"/documents/{doc_id}")
    assert got.status_code == 200
    assert got.json()["id"] == doc_id

    img = client.get(f"/documents/{doc_id}/image")
    assert img.status_code == 200
    assert img.content == _png() or len(img.content) > 0


def test_get_unknown_document_404(client: TestClient) -> None:
    assert client.get("/documents/nope").status_code == 404
    assert client.get("/documents/nope/image").status_code == 404
