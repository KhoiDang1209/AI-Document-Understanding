"""Unit tests for the CUAD embedding-pair builder (fake dataset, no downloads)."""

from __future__ import annotations

from docintel.scripts.build_embed_pairs import build_pairs, split_dev

_CONTEXT = "".join(f"sentence {i:03d}. " for i in range(40))  # 560 chars


def _row(title: str, category: str, starts: list[int]) -> dict[str, object]:
    question = (
        f'Highlight the parts (if any) of this contract related to "{category}" '
        f"that should be reviewed by a lawyer. Details: some details about {category}"
    )
    return {
        "title": title,
        "context": _CONTEXT,
        "question": question,
        "answers": {"text": ["x"] * len(starts), "answer_start": starts},
    }


def _dataset() -> list[dict[str, object]]:
    return [
        _row("A", "Governing Law", [10]),
        _row("A", "Insurance", []),  # unanswered -> no pair
        _row("B", "Non-Compete", [10, 400]),  # two spans -> two windows
        _row("HELD-OUT", "Governing Law", [10]),
    ]


def test_build_pairs_excludes_holdout_and_unanswered() -> None:
    pairs = build_pairs(_dataset(), {"HELD-OUT"}, size=100, overlap=20)
    assert {p["title"] for p in pairs} == {"A", "B"}


def test_build_pairs_query_is_focused() -> None:
    pairs = build_pairs(_dataset(), set(), size=100, overlap=20)
    queries = {p["query"] for p in pairs if p["title"] == "A"}
    assert queries == {"Governing Law: some details about Governing Law"}


def test_build_pairs_positive_is_covering_window() -> None:
    pairs = build_pairs(_dataset(), set(), size=100, overlap=20)
    for pair in pairs:
        assert pair["positive"] in _CONTEXT  # a real window of the contract text
        assert len(pair["positive"]) <= 100


def test_build_pairs_multi_span_yields_multiple_windows() -> None:
    pairs = build_pairs(_dataset(), set(), size=100, overlap=20)
    b_pairs = [p for p in pairs if p["title"] == "B"]
    assert len(b_pairs) >= 2
    assert len({p["positive"] for p in b_pairs}) >= 2  # distinct windows for distant spans


def test_split_dev_is_contract_disjoint_and_deterministic() -> None:
    pairs = build_pairs(_dataset(), set(), size=100, overlap=20)
    train, dev = split_dev(pairs, dev_contracts=1, seed=0)
    train_titles = {p["title"] for p in train}
    dev_titles = {p["title"] for p in dev}
    assert dev_titles and not (train_titles & dev_titles)
    train2, dev2 = split_dev(pairs, dev_contracts=1, seed=0)
    assert (train, dev) == (train2, dev2)
