"""Fast tests for the docTR-export -> OCRResult mapping (no model load)."""

from __future__ import annotations

from docintel.pipeline.ocr import _ocr_result_from_export


def test_maps_export_to_pixel_boxes() -> None:
    export = {
        "pages": [
            {
                "blocks": [
                    {
                        "lines": [
                            {
                                "words": [
                                    {
                                        "value": "TOTAL",
                                        "confidence": 0.9,
                                        "geometry": ((0.1, 0.2), (0.3, 0.25)),
                                    },
                                    {
                                        "value": "12.99",
                                        "confidence": 0.8,
                                        "geometry": ((0.35, 0.2), (0.5, 0.25)),
                                    },
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    result = _ocr_result_from_export(export, width=1000, height=2000)
    assert result.text == "TOTAL 12.99"
    assert result.words[0].bbox == (100, 400, 300, 500)
    assert result.words[0].confidence == 0.9
    assert abs(result.confidence - 0.85) < 1e-9
    assert result.image_width == 1000
    assert result.image_height == 2000


def test_empty_export_is_zeroed() -> None:
    result = _ocr_result_from_export({"pages": []}, width=10, height=20)
    assert result.text == ""
    assert result.words == []
    assert result.confidence == 0.0
    assert result.image_width == 10
