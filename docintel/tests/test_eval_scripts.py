"""Unit tests for the pure (no network) helpers in the C2/C3 eval runners."""

from __future__ import annotations

from datetime import date

from docintel.config import Settings
from docintel.scripts import eval_graph, eval_rag, eval_ragas


def test_rag_category_parsing() -> None:
    q = 'Highlight the parts (if any) of this contract related to "Governing Law" that ...'
    assert eval_rag._category(q) == "Governing Law"
    assert eval_rag._category("no quoted category here") is None


class _Chunk:
    def __init__(self, idx: int, start: int, end: int) -> None:
        self.chunk_index, self.char_start, self.char_end = idx, start, end


def test_rag_covering_chunk_indices_handles_overlap() -> None:
    chunks = [_Chunk(0, 0, 100), _Chunk(1, 80, 180), _Chunk(2, 160, 260)]
    assert eval_rag._covering_chunk_indices(chunks, 90) == {0, 1}  # in the overlap
    assert eval_rag._covering_chunk_indices(chunks, 200) == {2}
    assert eval_rag._covering_chunk_indices(chunks, 999) == set()


def test_ragas_refusal_detection() -> None:
    assert eval_ragas._is_refusal("I do not have enough information to answer.")
    assert eval_ragas._is_refusal("Sorry, I don't have enough information in the context.")
    assert not eval_ragas._is_refusal(
        "This Agreement is governed by the laws of the State of New York."
    )


def _fake_cuad() -> list[dict[str, object]]:
    """Two contracts: one with a parsable expiration + renewal, one unparsable."""

    def ex(title: str, category: str, text: str) -> dict[str, object]:
        return {
            "title": title,
            "question": f'... related to "{category}" that ...',
            "answers": {"text": [text] if text else [], "answer_start": [0] if text else []},
        }

    return [
        ex("A", "Expiration Date", "expires on March 1, 2020"),
        ex("A", "Renewal Term", "renews for one year"),
        ex("B", "Expiration Date", "ten (10) years from the Effective Date"),
    ]


def test_graph_build_gold_graph_projection() -> None:
    _, projection = eval_graph.build_gold_graph(_fake_cuad())
    assert projection == {"A": ("2020-03-01", True)}  # B's relative date is unparsable


def test_graph_expected_window_and_auto_renew() -> None:
    projection = {"A": ("2020-03-01", True), "C": ("2020-03-01", False)}
    # ref 2019-01-01 + 730d = 2020-12-31 -> both in window
    assert eval_graph._expected(projection, 730, auto_renew=False) == {"A", "C"}
    assert eval_graph._expected(projection, 730, auto_renew=True) == {"A"}  # only A renews
    assert eval_graph._expected(projection, 10, auto_renew=False) == set()  # window too short


def test_graph_build_cases_shape() -> None:
    projection = {"A": ("2020-03-01", True)}
    cases = eval_graph.build_cases(projection)
    assert cases and all(isinstance(q, str) and isinstance(s, set) for q, s in cases)
    assert date(2019, 1, 1) == eval_graph._REFERENCE_DATE
    assert isinstance(Settings().graph_default_within_days, int)


def test_rag_parse_top_ks() -> None:
    assert eval_rag._parse_top_ks("1,3,5,30") == (1, 3, 5, 30)
    assert eval_rag._parse_top_ks("5") == (5,)
