"""Serve the registered ONNX-INT8 LayoutLMv3 KIE model on CPU.

The quantized graph is pulled from MLflow and run via a raw
``onnxruntime.InferenceSession`` feeding all four inputs (input_ids,
attention_mask, bbox as int64; pixel_values as float32) — the Optimum wrapper
silently drops bbox/pixel_values (Phase 3 deviation). The processor is the
un-fine-tuned base processor (apply_ocr=False); id2label comes from the model
config. Heavy libraries are imported inside functions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from docintel.config import Settings
from docintel.kie.dataset import normalize_box
from docintel.pipeline.types import OCRResult
from docintel.schema import WordPrediction

_ONNX_INPUTS = ("input_ids", "attention_mask", "bbox", "pixel_values")


def _softmax_max(row: Any) -> tuple[int, float]:
    exp = np.exp(row - np.max(row))
    probs = exp / exp.sum()
    best = int(np.argmax(probs))
    return best, float(probs[best])


def words_from_token_logits(
    logits: Any,
    word_ids: Sequence[int | None],
    words: Sequence[str],
    boxes_pixel: Sequence[tuple[int, int, int, int]],
    id2label: Mapping[int, str],
) -> list[WordPrediction]:
    """Reduce per-token logits to one prediction per word (first subword wins)."""
    array = np.asarray(logits)
    seen: set[int] = set()
    predictions: list[WordPrediction] = []
    for position, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen or word_id >= len(words):
            continue
        seen.add(word_id)
        label_id, confidence = _softmax_max(array[position])
        predictions.append(
            WordPrediction(
                text=words[word_id],
                box=boxes_pixel[word_id],
                label=id2label[label_id],
                confidence=confidence,
            )
        )
    return predictions


class KIEBackend(Protocol):
    """Maps an OCR result to per-word BIO predictions."""

    def predict(self, ocr: OCRResult) -> list[WordPrediction]: ...


class LayoutLMv3OnnxBackend:
    """ONNX-INT8 LayoutLMv3 token classifier served via onnxruntime."""

    def __init__(self, session: Any, processor: Any, id2label: Mapping[int, str]) -> None:
        self._session = session
        self._processor = processor
        self._id2label = id2label

    @classmethod
    def load(
        cls,
        settings: Settings,
        tracking_uri: str | None = None,
    ) -> LayoutLMv3OnnxBackend:
        """Pull the INT8 model from MLflow and build a ready-to-serve backend."""
        import onnxruntime as ort
        from transformers import LayoutLMv3Processor

        from docintel.optimize.export import download_registered_model

        dest = Path(settings.data_dir) / "models" / settings.kie_onnx_registered_model_name
        bundle = download_registered_model(
            settings.kie_onnx_registered_model_name,
            settings.kie_onnx_model_version,
            dest,
            tracking_uri or settings.mlflow_tracking_uri,
        )
        onnx_path = next(Path(bundle).rglob("*quantized*.onnx"), None) or next(
            Path(bundle).rglob("*.onnx"), None
        )
        if onnx_path is None:
            raise FileNotFoundError(f"No .onnx file found under bundle: {bundle}")
        config = json.loads((onnx_path.parent / "config.json").read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in config["id2label"].items()}
        session = ort.InferenceSession(str(onnx_path))
        processor = LayoutLMv3Processor.from_pretrained(settings.kie_model_name, apply_ocr=False)
        return cls(session, processor, id2label)

    def predict(self, ocr: OCRResult) -> list[WordPrediction]:
        """Run OCR words through the ONNX KIE graph and label each word."""
        words = [w.text for w in ocr.words]
        boxes_pixel: list[tuple[int, int, int, int]] = [
            (w.bbox[0], w.bbox[1], w.bbox[2], w.bbox[3]) for w in ocr.words
        ]
        if not words:
            return []
        boxes_1000 = [
            normalize_box(list(box), ocr.image_width, ocr.image_height) for box in boxes_pixel
        ]
        encoding = self._processor(
            ocr_image_placeholder(ocr),
            text=words,
            boxes=boxes_1000,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        feeds = {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
            "bbox": encoding["bbox"].astype(np.int64),
            "pixel_values": encoding["pixel_values"].astype(np.float32),
        }
        logits = self._session.run(None, feeds)[0][0]
        word_ids = encoding.word_ids(batch_index=0)
        return words_from_token_logits(
            logits,
            word_ids,
            words,
            boxes_pixel,
            self._id2label,
        )


def ocr_image_placeholder(ocr: OCRResult) -> Any:
    """Build a white RGB PIL image of the OCR's dimensions for the processor.

    The KIE model attends to ``pixel_values``; serving feeds the page geometry
    via boxes and a blank canvas of the correct size (the real pixels are not
    re-decoded here — the API passes the decoded image in Task 6).
    """
    from PIL import Image

    return Image.new("RGB", (ocr.image_width, ocr.image_height), "white")
