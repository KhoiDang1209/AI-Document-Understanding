"""The 41 CUAD clause categories and their span-selection questions.

The category list is reference data from CUAD (Hendrycks et al., 2021). The
question follows CUAD's per-category template, so questions stay DRY.
"""

from __future__ import annotations

CLAUSE_CATEGORIES: tuple[str, ...] = (
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "Competitive Restriction Exception",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
)


def question_for(category: str) -> str:
    """Return the CUAD-style question for one clause category."""
    return (
        f'Highlight the parts (if any) of this contract related to "{category}" '
        "that should be reviewed by a lawyer."
    )


def all_questions() -> list[tuple[str, str]]:
    """Return ``(category, question)`` for all 41 clause categories."""
    return [(category, question_for(category)) for category in CLAUSE_CATEGORIES]
