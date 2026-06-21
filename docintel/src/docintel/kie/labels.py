"""CORD key-information-extraction label schema.

The set of CORD field categories is derived from the dataset at training
time (see :mod:`docintel.kie.dataset`), never hardcoded. This module turns
those categories into the deterministic BIO label list and id/label maps a
LayoutLMv3 token-classification head is trained with, and that Phase 3/4
reuse via the saved model config.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

OUTSIDE_LABEL = "O"


def bio_labels_for_category(category: str) -> tuple[str, str]:
    """Return the ``(B-, I-)`` label pair for a CORD category."""
    return f"B-{category}", f"I-{category}"


def build_label_list(categories: Iterable[str]) -> list[str]:
    """Build the BIO label list: ``O`` first, then sorted ``B-/I-`` pairs."""
    unique = sorted(set(categories))
    labels = [OUTSIDE_LABEL]
    for category in unique:
        begin, inside = bio_labels_for_category(category)
        labels.append(begin)
        labels.append(inside)
    return labels


def build_label_maps(
    label_list: Sequence[str],
) -> tuple[dict[int, str], dict[str, int]]:
    """Return ``(id2label, label2id)`` for a label list."""
    id2label = dict(enumerate(label_list))
    label2id = {name: index for index, name in id2label.items()}
    return id2label, label2id
