"""CUAD extraction metrics: normalized token-F1, ANLS, and AUPR.

Pure functions over strings/score lists so they are CPU-testable and reusable by
the build-time eval notebook. AUPR wraps scikit-learn.
"""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles, and collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1 between two normalized answer strings."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    for tok in pred_tokens:
        if tok in gold_tokens:
            common[tok] = min(pred_tokens.count(tok), gold_tokens.count(tok))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity for a single (pred, gold) pair."""
    pred = prediction.strip().lower()
    gold = ground_truth.strip().lower()
    if not pred and not gold:
        return 1.0
    longest = max(len(pred), len(gold))
    if longest == 0:
        return 1.0
    similarity = 1.0 - _levenshtein(pred, gold) / longest
    return similarity if similarity >= threshold else 0.0


def average_precision(scores: list[float], labels: list[int]) -> float:
    """Area under the precision-recall curve (AUPR) via scikit-learn.

    Returns 0.0 when ``labels`` contains fewer than two distinct classes, since
    AUPR is undefined in single-class scenarios and scikit-learn would warn.
    """
    if len(set(labels)) < 2:
        return 0.0
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))
