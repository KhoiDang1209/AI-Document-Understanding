from __future__ import annotations

from docintel.contracts.questions import CLAUSE_CATEGORIES, all_questions, question_for


def test_there_are_41_unique_categories() -> None:
    assert len(CLAUSE_CATEGORIES) == 41
    assert len(set(CLAUSE_CATEGORIES)) == 41


def test_every_question_mentions_its_category_and_is_nonempty() -> None:
    pairs = all_questions()
    assert len(pairs) == 41
    for category, question in pairs:
        assert category in question
        assert question == question_for(category)
        assert len(question) > len(category)
