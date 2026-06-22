"""Dynamic-INT8 quantization of an exported ONNX model via Optimum."""

from __future__ import annotations

from pathlib import Path


def quantize_dynamic_int8(onnx_dir: Path, out_dir: Path) -> Path:
    """Dynamic-INT8 quantize the ONNX model in ``onnx_dir`` into ``out_dir``."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(quantization_config=qconfig, save_dir=str(out_dir))
    return out_dir
