"""Pure, testable helpers for the Streamlit UI.

HTTP access to ``/extract`` and the small formatting functions live here so they
can be unit-tested; ``app.py`` keeps only the thin Streamlit rendering layer.
"""

from __future__ import annotations

from typing import Any, cast

import httpx


class ExtractError(Exception):
    """Raised when an extraction request fails, carrying a user-facing message."""


def extract_receipt(
    base_url: str,
    timeout_s: float,
    filename: str,
    content_type: str,
    data: bytes,
) -> dict[str, Any]:
    """POST an image to ``/extract`` and return the parsed Document JSON.

    Raises :class:`ExtractError` with a user-facing message on any non-2xx
    response or transport failure (connection refused, timeout).
    """
    url = f"{base_url.rstrip('/')}/extract"
    files = {"file": (filename, data, content_type)}
    try:
        response = httpx.post(url, files=files, timeout=timeout_s)
    except httpx.TimeoutException as exc:
        raise ExtractError("The API timed out while processing the receipt.") from exc
    except httpx.HTTPError as exc:
        raise ExtractError(f"Could not reach the API at {base_url}.") from exc

    if response.is_success:
        return cast("dict[str, Any]", response.json())

    detail = _error_detail(response)
    raise ExtractError(f"Extraction failed ({response.status_code}): {detail}")


def _error_detail(response: httpx.Response) -> str:
    """Pull a human-readable ``detail`` from an error response, falling back to text."""
    try:
        payload = response.json()
    except ValueError:
        return response.text or "no detail"
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)


def format_money(value: float | None, currency: str) -> str:
    """Format a money amount with its currency, or an em dash when absent."""
    if value is None:
        return "—"
    return f"{value:,.2f} {currency}"


def line_item_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a document's line items into rows for a table widget."""
    rows: list[dict[str, Any]] = []
    for item in document.get("line_items", []):
        rows.append(
            {
                "Name": item.get("name") or "—",
                "Qty": item.get("qty"),
                "Unit price": item.get("unit_price"),
                "Price": item.get("price"),
            }
        )
    return rows
