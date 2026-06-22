"""Tests for the pure token->word aggregation seam."""

from __future__ import annotations

import numpy as np

from docintel.kie.backend import words_from_token_logits


def test_first_subword_token_drives_word_label() -> None:
    # 2 words; word 0 -> tokens 1,2 ; word 1 -> token 3. Token 0 is a special token.
    id2label = {0: "O", 1: "B-menu.nm", 2: "I-menu.nm"}
    # logits shape (seq=4, num_labels=3)
    logits = np.array(
        [
            [9.0, 0.0, 0.0],  # special (word_id None)
            [0.0, 9.0, 0.0],  # word 0 first token -> B-menu.nm
            [0.0, 0.0, 9.0],  # word 0 second token (ignored)
            [9.0, 0.0, 0.0],  # word 1 -> O
        ],
        dtype=np.float32,
    )
    word_ids = [None, 0, 0, 1]
    preds = words_from_token_logits(
        logits, word_ids, ["Coke", "x"], [(0, 0, 1, 1), (2, 2, 3, 3)], id2label
    )
    assert [p.label for p in preds] == ["B-menu.nm", "O"]
    assert preds[0].text == "Coke"
    assert 0.99 < preds[0].confidence <= 1.0
