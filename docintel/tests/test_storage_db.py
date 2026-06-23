"""SQLite metadata persistence round-trip."""

from __future__ import annotations

from pathlib import Path

from docintel.schema import Document, LineItem
from docintel.storage.db import get_document, init_db, save_document


def _doc() -> Document:
    return Document(
        id="doc1",
        line_items=[LineItem(name="Coke", price=3000.0, confidence=0.9)],
        subtotal=3000.0,
        total=3000.0,
        currency="IDR",
        created_at="2026-06-22T00:00:00+00:00",
    )


def test_save_then_get_round_trips(tmp_path: Path) -> None:
    db = str(tmp_path / "docintel.db")
    init_db(db)
    save_document(db, _doc(), image_key="doc1.png")
    fetched = get_document(db, "doc1")
    assert fetched is not None
    document, image_key = fetched
    assert document.id == "doc1"
    assert document.line_items[0].name == "Coke"
    assert image_key == "doc1.png"


def test_get_unknown_returns_none(tmp_path: Path) -> None:
    db = str(tmp_path / "docintel.db")
    init_db(db)
    assert get_document(db, "missing") is None


def test_upsert_updates_existing(tmp_path: Path) -> None:
    db = str(tmp_path / "docintel.db")
    init_db(db)
    save_document(db, _doc(), image_key="v1.png")
    save_document(db, _doc(), image_key="v2.png")
    fetched = get_document(db, "doc1")
    assert fetched is not None
    _, image_key = fetched
    assert image_key == "v2.png"
