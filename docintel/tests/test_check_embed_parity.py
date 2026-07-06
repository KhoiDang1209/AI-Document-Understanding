"""Unit tests for the embedding parity gate's pure comparison logic."""

from __future__ import annotations

from docintel.scripts.check_embed_parity import check

_REFERENCE = {
    "sentences": ["alpha", "beta"],
    "vectors": [[1.0, 0.0], [0.6, 0.8]],
}


def test_check_passes_on_identical_vectors() -> None:
    assert check(_REFERENCE, [[1.0, 0.0], [0.6, 0.8]], threshold=0.999) == []


def test_check_flags_diverging_vector_with_its_sentence() -> None:
    failures = check(_REFERENCE, [[1.0, 0.0], [0.8, 0.6]], threshold=0.999)
    assert [sentence for sentence, _ in failures] == ["beta"]
    assert all(cos < 0.999 for _, cos in failures)


def test_check_is_scale_invariant() -> None:
    # cosine ignores magnitude: a scaled copy still passes
    assert check(_REFERENCE, [[2.0, 0.0], [1.2, 1.6]], threshold=0.999) == []
