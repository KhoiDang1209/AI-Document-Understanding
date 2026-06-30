"""Pure, testable helpers for the contract-intelligence Streamlit page.

HTTP access to ``/contracts/extract``, ``/ask``, ``/agent`` plus small
formatting helpers live here so they can be unit-tested; the page module keeps
only the thin Streamlit rendering layer (mirrors the receipt UI split this
replaces).
"""

from __future__ import annotations

from typing import Any, cast

import httpx

_PDF_TYPE = "application/pdf"


class ContractApiError(Exception):
    """Raised when a contract API request fails, carrying a user-facing message."""


def _error_detail(response: httpx.Response) -> str:
    """Pull a human-readable ``detail`` from an error response, falling back to text."""
    try:
        payload = response.json()
    except ValueError:
        return response.text or "no detail"
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)


def _post(url: str, timeout_s: float, **kwargs: Any) -> dict[str, Any]:
    """POST and return parsed JSON, mapping transport/HTTP errors to ContractApiError."""
    try:
        response = httpx.post(url, timeout=timeout_s, **kwargs)
    except httpx.TimeoutException as exc:
        raise ContractApiError("The API timed out while processing the request.") from exc
    except httpx.HTTPError as exc:
        raise ContractApiError(f"Could not reach the API at {url}.") from exc
    if response.is_success:
        return cast("dict[str, Any]", response.json())
    raise ContractApiError(f"Request failed ({response.status_code}): {_error_detail(response)}")


def extract_contract(
    base_url: str, timeout_s: float, filename: str, data: bytes
) -> dict[str, Any]:
    """POST a PDF to ``/contracts/extract`` and return the parsed ContractDocument JSON."""
    url = f"{base_url.rstrip('/')}/contracts/extract"
    return _post(url, timeout_s, files={"file": (filename, data, _PDF_TYPE)})


def ask_question(
    base_url: str, timeout_s: float, question: str, contract_id: str | None
) -> dict[str, Any]:
    """POST ``/ask`` (scoped to one contract when given) and return the AskResponse JSON."""
    payload: dict[str, Any] = {"question": question}
    if contract_id:
        payload["contract_id"] = contract_id
    return _post(f"{base_url.rstrip('/')}/ask", timeout_s, json=payload)


def run_agent(
    base_url: str, timeout_s: float, task: str, contract_id: str | None
) -> dict[str, Any]:
    """POST ``/agent`` (scoped to one contract when given) and return the AgentResponse JSON."""
    payload: dict[str, Any] = {"task": task}
    if contract_id:
        payload["contract_id"] = contract_id
    return _post(f"{base_url.rstrip('/')}/agent", timeout_s, json=payload)


def clause_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a ContractDocument's clauses into table rows."""
    rows: list[dict[str, Any]] = []
    for clause in document.get("clauses", []):
        rows.append(
            {
                "Type": clause.get("clause_type") or "—",
                "Text": clause.get("answer_text") or "—",
                "Confidence": clause.get("confidence"),
            }
        )
    return rows


def citation_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten an Ask/Agent response's citations into table rows."""
    rows: list[dict[str, Any]] = []
    for chunk in response.get("citations", []):
        rows.append(
            {
                "Contract": chunk.get("contract_id") or "—",
                "Clause": chunk.get("clause_type") or "—",
                "Score": chunk.get("score"),
                "Text": chunk.get("text") or "—",
            }
        )
    return rows
