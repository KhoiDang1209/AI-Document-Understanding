from __future__ import annotations

from docintel.contracts.eval import anls, average_precision, normalize_answer, token_f1
from docintel.contracts.ocr_cer import cer


def test_normalize_answer_strips_articles_punctuation_case() -> None:
    assert normalize_answer("The  Agreement.") == "agreement"


def test_token_f1_exact_and_partial() -> None:
    assert token_f1("new york law", "new york law") == 1.0
    assert 0.0 < token_f1("new york", "new york law") < 1.0
    assert token_f1("", "anything") == 0.0


def test_anls_identical_is_one_and_far_is_zero() -> None:
    assert anls("acme corp", "acme corp") == 1.0
    assert anls("acme", "zzzzzzzz") == 0.0


def test_average_precision_perfect_ranking() -> None:
    ap = average_precision(scores=[0.9, 0.8, 0.1], labels=[1, 1, 0])
    assert ap == 1.0


def test_cer_basic() -> None:
    assert cer("contract", "contract") == 0.0
    assert cer("contract", "contracX") == 1 / 8
    assert cer("", "") == 0.0
