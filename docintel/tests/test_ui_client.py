"""Unit tests for the UI client helpers."""

from __future__ import annotations

import httpx
import pytest

from docintel.ui.client import (
    ExtractError,
    extract_receipt,
    format_money,
    line_item_rows,
)

_BASE_URL = "http://api:8000"


def _client_extract(handler: httpx.MockTransport) -> dict:
    """Call extract_receipt with httpx.post patched to use a mock transport."""
    real_post = httpx.post

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        with httpx.Client(transport=handler) as client:
            return client.post(url, **kwargs)  # type: ignore[arg-type]

    httpx.post = fake_post  # type: ignore[assignment]
    try:
        return extract_receipt(_BASE_URL, 5.0, "r.png", "image/png", b"bytes")
    finally:
        httpx.post = real_post  # type: ignore[assignment]


def test_format_money_none() -> None:
    assert format_money(None, "IDR") == "—"


def test_format_money_thousands() -> None:
    assert format_money(10000.0, "IDR") == "10,000.00 IDR"


def test_line_item_rows_maps_fields_and_missing() -> None:
    document = {
        "line_items": [
            {"name": "Coffee", "qty": 2, "unit_price": 15000.0, "price": 30000.0},
            {"name": None, "qty": None, "unit_price": None, "price": None},
        ]
    }
    rows = line_item_rows(document)
    assert rows[0] == {"Name": "Coffee", "Qty": 2, "Unit price": 15000.0, "Price": 30000.0}
    assert rows[1]["Name"] == "—"


def test_extract_receipt_success() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"id": "abc"}))
    result = _client_extract(transport)
    assert result == {"id": "abc"}


def test_extract_receipt_unsupported_media_type() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(415, json={"detail": "Unsupported content type"})
    )
    with pytest.raises(ExtractError, match=r"415.*Unsupported content type"):
        _client_extract(transport)


def test_extract_receipt_server_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(ExtractError, match="500"):
        _client_extract(transport)


def test_extract_receipt_timeout() -> None:
    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    with pytest.raises(ExtractError, match="timed out"):
        _client_extract(httpx.MockTransport(raise_timeout))


def test_extract_receipt_connection_error() -> None:
    def raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ExtractError, match="Could not reach"):
        _client_extract(httpx.MockTransport(raise_connect))
