from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from docintel.config import Settings
from docintel.contracts.extractor import CuadQaOnnxExtractor, resolve_contract_bundle


def test_resolve_contract_bundle_prefers_local_path() -> None:
    settings = Settings(contract_onnx_local_path="/models/cuad")
    out = resolve_contract_bundle(settings, download=_unused_download)
    assert out == Path("/models/cuad")


def _unused_download(name: str, version: str, dest: Path, uri: str | None) -> Path:
    raise AssertionError("download must not run when a local path is set")


class _FakeEncoding:
    """Mimics a transformers BatchEncoding with overflow windows + offsets."""

    def __init__(self) -> None:
        # one window, 3 tokens: [special, "alpha", "beta"]
        self._offsets = [[(0, 0), (0, 5), (6, 10)]]
        self.data = {
            "input_ids": np.array([[0, 1, 2]]),
            "attention_mask": np.array([[1, 1, 1]]),
            "offset_mapping": np.array([[(0, 0), (0, 5), (6, 10)]]),
        }

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def num_windows(self) -> int:
        return len(self._offsets)

    def offsets(self, i: int) -> list[tuple[int, int]]:
        return self._offsets[i]


def test_extract_decodes_one_clause_per_strong_question(monkeypatch: Any) -> None:
    settings = Settings(contract_no_answer_threshold=0.0, contract_n_best=1)

    class _FakeSession:
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
    extractor._session = _FakeSession()  # type: ignore[attr-defined]
    extractor._settings = settings  # type: ignore[attr-defined]
    extractor._encode = lambda question, text: _FakeEncoding()  # type: ignore[attr-defined]

    clauses = extractor.extract("alpha beta")
    assert len(clauses) == 1
    assert clauses[0].clause_type == "Parties"
    assert clauses[0].answer_text == "alpha beta"
