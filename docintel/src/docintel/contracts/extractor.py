"""Serve the ONNX-INT8 CUAD extractive-QA model on CPU.

For each of the 41 clause questions, the (question, contract) pair is tokenized
with a sliding window (doc stride). Each window runs through a raw
``onnxruntime.InferenceSession`` (start/end logits); spans are decoded to
document char offsets and aggregated per clause. Heavy imports live in functions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from docintel.config import Settings
from docintel.contracts.aggregate import WindowSpan, aggregate_clause, best_spans_from_window
from docintel.contracts.questions import all_questions
from docintel.contracts.schema import ExtractedClause


def resolve_contract_bundle(
    settings: Settings,
    download: Callable[[str, str, Path, str | None], Path],
    tracking_uri: str | None = None,
) -> Path:
    """Return the ONNX bundle dir: the local override if set, else MLflow download."""
    if settings.contract_onnx_local_path:
        return Path(settings.contract_onnx_local_path)
    dest = Path(settings.data_dir) / "models" / settings.contract_onnx_registered_model_name
    return download(
        settings.contract_onnx_registered_model_name,
        settings.contract_onnx_model_version,
        dest,
        tracking_uri or settings.mlflow_tracking_uri,
    )


class ContractExtractor(Protocol):
    """Maps contract text to extracted clause spans."""

    def extract(self, text: str) -> list[ExtractedClause]: ...


class CuadQaOnnxExtractor:
    """ONNX-INT8 extractive-QA model served via onnxruntime."""

    def __init__(self, session: Any, tokenizer: Any, settings: Settings) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._settings = settings

    @classmethod
    def load(cls, settings: Settings, tracking_uri: str | None = None) -> CuadQaOnnxExtractor:
        """Load the INT8 model (local override or MLflow) and build the backend."""
        import onnxruntime as ort
        from transformers import AutoTokenizer

        from docintel.optimize.export import download_registered_model

        bundle = resolve_contract_bundle(settings, download_registered_model, tracking_uri)
        onnx_path = next(Path(bundle).rglob("*quantized*.onnx"), None) or next(
            Path(bundle).rglob("*.onnx"), None
        )
        if onnx_path is None:
            raise FileNotFoundError(f"No .onnx file found under bundle: {bundle}")
        session = ort.InferenceSession(str(onnx_path))
        tokenizer = AutoTokenizer.from_pretrained(settings.contract_model_name)  # type: ignore[no-untyped-call]
        return cls(session, tokenizer, settings)

    def _encode(self, question: str, text: str) -> Any:
        """Tokenize (question, text) into overflowing windows with offset maps."""
        return self._tokenizer(
            question,
            text,
            truncation="only_second",
            max_length=self._settings.contract_max_seq_length,
            stride=self._settings.contract_doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="np",
        )

    def _run_question(self, question: str, clause_type: str, text: str) -> list[ExtractedClause]:
        encoding = self._encode(question, text)
        spans: list[WindowSpan] = []
        num_windows = encoding["input_ids"].shape[0]
        for i in range(num_windows):
            feeds = {
                "input_ids": encoding["input_ids"][i : i + 1].astype(np.int64),
                "attention_mask": encoding["attention_mask"][i : i + 1].astype(np.int64),
            }
            start_logits, end_logits = self._session.run(None, feeds)
            offsets = [tuple(o) for o in encoding["offset_mapping"][i]]
            spans.extend(
                best_spans_from_window(
                    start_logits[0],
                    end_logits[0],
                    offsets,
                    self._settings.contract_n_best,
                    self._settings.contract_max_answer_length,
                )
            )
        return aggregate_clause(
            clause_type,
            text,
            spans,
            self._settings.contract_n_best,
            self._settings.contract_no_answer_threshold,
        )

    def extract(self, text: str) -> list[ExtractedClause]:
        """Run all 41 clause questions over the contract text."""
        clauses: list[ExtractedClause] = []
        for clause_type, question in all_questions():
            clauses.extend(self._run_question(question, clause_type, text))
        return clauses
