from __future__ import annotations

import pytest

from docintel.graph.templates import TEMPLATES, get_template


def test_templates_registry_has_both_patterns() -> None:
    assert set(TEMPLATES) == {"expiring_within", "auto_renewing_expiring_within"}


def test_expiring_template_filters_on_date_bounds() -> None:
    cypher = get_template("expiring_within")
    assert "EXPIRES_ON" in cypher and "$lower" in cypher and "$upper" in cypher
    assert "contract_id" in cypher


def test_auto_renew_template_requires_has_clause() -> None:
    cypher = get_template("auto_renewing_expiring_within")
    assert "HAS_CLAUSE" in cypher and "ren_answer" in cypher


def test_get_template_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_template("nope")
