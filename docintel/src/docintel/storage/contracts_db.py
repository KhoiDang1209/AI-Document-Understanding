"""SQLite persistence for ContractDocument records (stdlib sqlite3, no ORM)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from docintel.contracts.schema import ContractDocument

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contracts (
    id TEXT PRIMARY KEY,
    contract_json TEXT NOT NULL,
    pdf_key TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


@contextmanager
def _connect(path: str) -> Iterator[sqlite3.Connection]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_contracts_db(path: str) -> None:
    """Create the contracts table if it does not exist."""
    with _connect(path) as conn:
        conn.execute(_SCHEMA)


def save_contract(path: str, doc: ContractDocument, pdf_key: str) -> None:
    """Upsert one contract record by id."""
    with _connect(path) as conn:
        conn.execute(
            "INSERT INTO contracts (id, contract_json, pdf_key, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "contract_json=excluded.contract_json, pdf_key=excluded.pdf_key",
            (doc.id, doc.model_dump_json(), pdf_key, doc.created_at),
        )


def get_contract(path: str, contract_id: str) -> tuple[ContractDocument, str] | None:
    """Return ``(ContractDocument, pdf_key)`` for an id, or None if absent."""
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT contract_json, pdf_key FROM contracts WHERE id = ?", (contract_id,)
        ).fetchone()
    if row is None:
        return None
    return ContractDocument.model_validate_json(row[0]), row[1]
