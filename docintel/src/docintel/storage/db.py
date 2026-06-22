"""SQLite persistence for extracted Document metadata (stdlib sqlite3, no ORM)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from docintel.schema import Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    document_json TEXT NOT NULL,
    image_key TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def _connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_db(path: str) -> None:
    """Create the documents table if it does not exist."""
    with _connect(path) as conn:
        conn.execute(_SCHEMA)


def save_document(path: str, document: Document, image_key: str) -> None:
    """Upsert one document's metadata by id."""
    with _connect(path) as conn:
        conn.execute(
            "INSERT INTO documents (id, document_json, image_key, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "document_json=excluded.document_json, image_key=excluded.image_key",
            (document.id, document.model_dump_json(), image_key, document.created_at),
        )


def get_document(path: str, document_id: str) -> tuple[Document, str] | None:
    """Return ``(Document, image_key)`` for an id, or None if absent."""
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT document_json, image_key FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    if row is None:
        return None
    return Document.model_validate_json(row[0]), row[1]
