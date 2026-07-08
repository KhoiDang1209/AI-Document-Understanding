from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from docintel.config import Settings
from docintel.contracts.extractor import CuadQaOnnxExtractor, resolve_contract_bundle


class _InputMeta(NamedTuple):
    name: str


def test_resolve_contract_bundle_prefers_local_path() -> None:
    settings = Settings(contract_onnx_local_path="/models/cuad")
    out = resolve_contract_bundle(settings, download=_unused_download)
    assert out == Path("/models/cuad")


def _unused_download(name: str, version: str, dest: Path, uri: str | None) -> Path:
    raise AssertionError("download must not run when a local path is set")


class _FakeEncoding:
    """Mimics a single-window transformers BatchEncoding with offsets + sequence ids.

    ``seq_ids`` mirrors ``BatchEncoding.sequence_ids(i)``: ``None`` for special
    tokens, ``0`` for question tokens, ``1`` for context tokens. Defaults describe
    ``[special, "alpha", "beta"]`` (both context) so no-question callers are unchanged.
    """

    def __init__(
        self,
        input_ids: list[int] | None = None,
        offsets: list[tuple[int, int]] | None = None,
        seq_ids: list[int | None] | None = None,
    ) -> None:
        if offsets is None:
            offsets = [(0, 0), (0, 5), (6, 10)]
        if input_ids is None:
            input_ids = list(range(len(offsets)))
        if seq_ids is None:
            seq_ids = [None, *([1] * (len(offsets) - 1))]
        self._seq_ids = [seq_ids]
        self.data = {
            "input_ids": np.array([input_ids]),
            "attention_mask": np.array([[1] * len(offsets)]),
            "offset_mapping": np.array([offsets]),
        }

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def sequence_ids(self, i: int) -> list[int | None]:
        return self._seq_ids[i]


def test_extract_decodes_one_clause_per_strong_question(monkeypatch: Any) -> None:
    settings = Settings(contract_no_answer_threshold=0.0, contract_n_best=1)

    class _FakeSession:
        def get_inputs(self) -> list[_InputMeta]:
            return [_InputMeta("input_ids"), _InputMeta("attention_mask")]

        def run(self, _: Any, feeds: dict[str, Any]) -> list[Any]:
            # start favors token 1, end favors token 2 -> span "alpha beta"? offsets (0,10)
            start = np.array([[0.0, 5.0, 0.0]])
            end = np.array([[0.0, 0.0, 5.0]])
            return [start, end]

    # Only the first question yields a clause; force a single question for the test.
    monkeypatch.setattr(
        "docintel.contracts.extractor.all_questions", lambda: [("Parties", "Who are the parties?")]
    )
    extractor = CuadQaOnnxExtractor.__new__(CuadQaOnnxExtractor)
    fake_session = _FakeSession()
    extractor._session = fake_session  # type: ignore[attr-defined]
    extractor._input_names = frozenset(inp.name for inp in fake_session.get_inputs())  # type: ignore[attr-defined]
    extractor._settings = settings  # type: ignore[attr-defined]
    extractor._encode = lambda question, text: _FakeEncoding()  # type: ignore[attr-defined]

    clauses = extractor.extract("alpha beta")
    assert len(clauses) == 1
    assert clauses[0].clause_type == "Parties"
    assert clauses[0].answer_text == "alpha beta"


def test_extract_excludes_question_tokens_from_spans(monkeypatch: Any) -> None:
    """Question tokens carry real offsets into the *question* string; if the decode
    honored them it would slice a garbage span out of the contract text. They must
    be masked out via ``sequence_ids`` before span selection."""
    settings = Settings(contract_no_answer_threshold=0.0, contract_n_best=1)
    # tokens: [CLS], "who"(q), "are"(q), [SEP], "alpha"(ctx), "beta"(ctx), [SEP]
    input_ids = [0, 10, 11, 2, 20, 21, 2]
    offsets = [(0, 0), (0, 3), (4, 7), (0, 0), (0, 5), (6, 10), (0, 0)]
    seq_ids: list[int | None] = [None, 0, 0, None, 1, 1, None]

    class _FakeSession:
        def get_inputs(self) -> list[_InputMeta]:
            return [_InputMeta("input_ids"), _InputMeta("attention_mask")]

        def run(self, _: Any, feeds: dict[str, Any]) -> list[Any]:
            # Highest start/end logits sit on the QUESTION tokens (idx 1, 2). Honoring
            # them yields text[0:7] == "alpha b"; the context span must win instead.
            start = np.array([[0.0, 10.0, 0.0, 0.0, 5.0, 0.0, 0.0]])
            end = np.array([[0.0, 0.0, 10.0, 0.0, 0.0, 5.0, 0.0]])
            return [start, end]

    monkeypatch.setattr(
        "docintel.contracts.extractor.all_questions", lambda: [("Parties", "Who are the parties?")]
    )
    extractor = CuadQaOnnxExtractor.__new__(CuadQaOnnxExtractor)
    session = _FakeSession()
    extractor._session = session  # type: ignore[attr-defined]
    extractor._input_names = frozenset(inp.name for inp in session.get_inputs())  # type: ignore[attr-defined]
    extractor._settings = settings  # type: ignore[attr-defined]
    extractor._encode = lambda question, text: _FakeEncoding(input_ids, offsets, seq_ids)  # type: ignore[attr-defined]

    clauses = extractor.extract("alpha beta")
    assert len(clauses) == 1
    assert clauses[0].answer_text == "alpha beta"
