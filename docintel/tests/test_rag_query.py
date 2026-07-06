from __future__ import annotations

from docintel.rag.query import focus_query

_CUAD_TEMPLATE = (
    'Highlight the parts (if any) of this contract related to "Governing Law" that should be '
    "reviewed by a lawyer. Details: Which state/country's law governs the interpretation of "
    "the contract?"
)


def test_rewrites_cuad_template_to_category_and_details() -> None:
    assert focus_query(_CUAD_TEMPLATE) == (
        "Governing Law: Which state/country's law governs the interpretation of the contract?"
    )


def test_natural_question_passes_through_unchanged() -> None:
    question = "What law governs this agreement?"
    assert focus_query(question) == question


def test_category_only_when_details_missing() -> None:
    question = (
        'Highlight the parts (if any) of this contract related to "Non-Compete" that should '
        "be reviewed by a lawyer."
    )
    assert focus_query(question) == "Non-Compete"
