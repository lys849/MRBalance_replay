"""Tests for option parsing, frequency voting and label extraction."""

from __future__ import annotations

import pytest

from mrbalance_replay.parsing import extract_final_label, most_frequent, parse_single_choice


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("(A)", "A"),
        ("The answer is (B).", "B"),
        ("(c)", "C"),
        ("(A) cause, but actually (C) none", "C"),
        ("A) cause", "A"),
        ("answer: b)", "B"),
        ("no option here", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_single_choice(reply: str | None, expected: str | None) -> None:
    assert parse_single_choice(reply) == expected


def test_most_frequent_returns_plurality() -> None:
    candidates = ["A", "B", "B", "C", "B"]

    winner, count = most_frequent(candidates, lambda left, right: left == right)

    assert winner == "B"
    assert count == 3


def test_most_frequent_keeps_first_candidate_on_tie() -> None:
    candidates = ["A", "B", "A", "B"]

    winner, count = most_frequent(candidates, lambda left, right: left == right)

    assert winner == "A"
    assert count == 2


def test_most_frequent_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        most_frequent([], lambda left, right: left == right)


def test_naive_mode_follows_first_letter_token() -> None:
    assert extract_final_label("(A) cause", mode="naive") == 1
    assert extract_final_label("(B) caused by", mode="naive") == 1
    assert extract_final_label("(C) none", mode="naive") == 0
    assert extract_final_label(None, mode="naive") == 0


def test_naive_mode_reproduces_article_defect() -> None:
    # The correct option is (C), but the first letter token is the article "a".
    response = "A careful reading suggests the answer is (C)."

    assert extract_final_label(response, mode="naive") == 1
    assert extract_final_label(response, mode="fixed") == 0


def test_fixed_mode_handles_no_causal_relationship_wording() -> None:
    response = "There is no causal relationship between the two events."

    assert extract_final_label(response, mode="fixed") == 0


def test_fixed_mode_uses_last_parenthesized_option() -> None:
    response = "Option (A) is tempting, but the answer is (B) caused by."

    assert extract_final_label(response, mode="fixed") == 1


def test_fixed_mode_recognizes_textual_positive_relation() -> None:
    assert extract_final_label("The flood was caused by heavy rain.", mode="fixed") == 1


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        extract_final_label("(A)", mode="experimental")
