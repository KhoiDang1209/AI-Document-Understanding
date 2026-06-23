"""MinIO/S3 object helpers, exercised against a fake client."""

from __future__ import annotations

from typing import Any

from docintel.storage.objects import ensure_bucket, get_image, put_image


class _FakeS3:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.buckets: set[str] = set()

    def head_bucket(self, Bucket: str) -> None:
        if Bucket not in self.buckets:
            raise RuntimeError("missing")

    def create_bucket(self, Bucket: str) -> None:
        self.buckets.add(Bucket)

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.store[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.store:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(self.store[(Bucket, Key)])}


def test_ensure_bucket_creates_when_missing() -> None:
    client = _FakeS3()
    ensure_bucket(client, "documents")
    assert "documents" in client.buckets


def test_put_then_get_round_trips() -> None:
    client = _FakeS3()
    ensure_bucket(client, "documents")
    put_image(client, "documents", "a.png", b"bytes", "image/png")
    assert get_image(client, "documents", "a.png") == b"bytes"


def test_get_missing_returns_none() -> None:
    client = _FakeS3()
    ensure_bucket(client, "documents")
    assert get_image(client, "documents", "nope.png") is None


def test_ensure_bucket_noop_when_exists() -> None:
    client = _FakeS3()
    client.buckets.add("documents")
    ensure_bucket(client, "documents")
    assert client.buckets == {"documents"}
