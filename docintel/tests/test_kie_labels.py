"""Tests for the CORD BIO label schema."""

from __future__ import annotations

from docintel.kie.labels import (
    OUTSIDE_LABEL,
    bio_labels_for_category,
    build_label_list,
    build_label_maps,
)


def test_build_label_list_puts_outside_first_and_is_sorted() -> None:
    labels = build_label_list(["total.total_price", "menu.nm"])
    assert labels[0] == OUTSIDE_LABEL
    assert labels == [
        "O",
        "B-menu.nm",
        "I-menu.nm",
        "B-total.total_price",
        "I-total.total_price",
    ]


def test_build_label_list_deduplicates_categories() -> None:
    labels = build_label_list(["menu.nm", "menu.nm"])
    assert labels == ["O", "B-menu.nm", "I-menu.nm"]


def test_bio_labels_for_category() -> None:
    assert bio_labels_for_category("menu.nm") == ("B-menu.nm", "I-menu.nm")


def test_build_label_maps_round_trips() -> None:
    labels = build_label_list(["menu.nm"])
    id2label, label2id = build_label_maps(labels)
    assert id2label[0] == "O"
    assert label2id["B-menu.nm"] == 1
    for index, name in id2label.items():
        assert label2id[name] == index
