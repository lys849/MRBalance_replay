"""Fidelity tests for the verbatim MRBalance prompt assets."""

import re

import pytest

from mrbalance_replay.prompts import (
    DEFAULT_AGENT_ROLES,
    DEFAULT_PROMPT_STYLES,
    PROMPT_STYLES,
    ROLES,
    SYSTEM_PROMPT,
    construct_message,
    construct_ranking_message,
)

QUESTION = (
    "Input: The heavy rain caused flooding in the city.\n"
    'Question: is there a causal relationship between "heavy rain" and '
    '"flooding in the city" ?\n'
    "(A) cause\n"
    "(B) caused by\n"
    "(C) none"
)

STYLE_FIRST_LINE = {
    "positive": "\nWhen analyzing a sentence, consider the following questions:",
    "normal": "\n1.Event Clarity",
    "normal1": "\nRefined Rules for Identifying Causal Relationships:",
    "negative": "\nthere are some examples for ECI:",
}

OPTION_INSTRUCTION = (
    "Put your answer in the form (X) at the end of your response. "
    "(X) represents choice (A), (B), or (C)."
)

CONCISE_INSTRUCTION = (
    "Your answer should be concise and your reasons should be condensed into one sentence."
)


@pytest.mark.parametrize("style", ["positive", "normal", "normal1", "negative"])
def test_each_prompt_style_is_verbatim(style: str) -> None:
    """Each style keeps its opening, closing separator, and option wording."""
    message = construct_message([], QUESTION, style)
    content = message["content"]

    assert message["role"] == "user"
    assert content.startswith(STYLE_FIRST_LINE[style])
    assert PROMPT_STYLES[style] in content
    # The reference literals for ``normal1`` and ``negative`` keep the four
    # spaces that precede their closing triple quotes, so only strip them here.
    assert PROMPT_STYLES[style].rstrip(" ").endswith("---\n")
    assert re.search(r"---\n\s*\nHere is the question:\n", content)
    assert OPTION_INSTRUCTION in content
    assert CONCISE_INSTRUCTION in content
    assert content.endswith(CONCISE_INSTRUCTION)


def test_none_style_has_no_style_prefix() -> None:
    """The ``none`` style starts directly with the question block."""
    content = construct_message([], QUESTION, "none")["content"]

    assert PROMPT_STYLES["none"] == ""
    assert content.startswith("\nHere is the question:\n")
    assert "---" not in content
    assert OPTION_INSTRUCTION in content


@pytest.mark.parametrize("style", ["positive", "normal", "normal1", "negative"])
def test_prompt_style_closes_with_separator_before_question(style: str) -> None:
    """The reference always emits ``---`` followed by the question block."""
    content = construct_message([], QUESTION, style)["content"]

    assert PROMPT_STYLES[style].rstrip(" ").endswith("---\n")
    assert re.search(r"---\n\s*\nHere is the question:\n", content)


def test_round_zero_message_is_exact() -> None:
    """The first-round message matches the reference concatenation exactly."""
    expected = (
        PROMPT_STYLES["normal"]
        + "\nHere is the question:\n"
        + QUESTION
        + "\n"
        + OPTION_INSTRUCTION
        + " "
        + CONCISE_INSTRUCTION
    )

    assert construct_message([], QUESTION, "normal") == {
        "role": "user",
        "content": expected,
    }


def test_multi_agent_message_keeps_reference_ordering_and_scores() -> None:
    """Replies are numbered in order and the score instruction uses the count."""
    responses = ["answer one (A)", "answer two (B)", "answer three (C)"]
    content = construct_message(responses, QUESTION, "positive")["content"]

    assert content.startswith("\nHere is the question:\n" + QUESTION)
    assert content.index("Agent solution 1") < content.index("Agent solution 2")
    assert content.index("Agent solution 2") < content.index("Agent solution 3")
    assert content.count("Agent solution") == len(responses)
    assert (
        "Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents."
    ) in content
    assert ("Put all 3 scores in the form like [[1, 5, 2, ...]].") in content
    # The reference ignores prompt_type once predecessor replies exist.
    assert PROMPT_STYLES["positive"] not in content


def test_force_style_is_opt_in_for_later_rounds() -> None:
    """``force_style`` prepends the style without changing anything else."""
    responses = ["answer one (A)"]
    default_content = construct_message(responses, QUESTION, "normal1")["content"]
    forced_content = construct_message(responses, QUESTION, "normal1", force_style=True)["content"]

    assert default_content.startswith("\nHere is the question:\n")
    assert forced_content == PROMPT_STYLES["normal1"] + default_content


def test_ranking_message_is_exact() -> None:
    """The listwise ranking prompt matches the reference byte for byte."""
    expected = (
        "Here is the question:\n"
        + QUESTION
        + "\n\nThese are the solutions to the problem from other agents: "
        + "\n\nAgent solution 1: ```alpha (A)```"
        + "\n\nAgent solution 2: ```beta (B)```"
        + "\n\nAgent solution 3: ```gamma (C)```"
        + "\n\nPlease choose the best 2 solutions and think step by step. "
        "Put your answer in the form like [1,2] or [3,4] at the end of your response."
    )

    assert construct_ranking_message(["alpha (A)", "beta (B)", "gamma (C)"], QUESTION) == {
        "role": "user",
        "content": expected,
    }


def test_personas_and_system_prompt_are_verbatim() -> None:
    """All four personas keep the reference text and keys."""
    assert set(ROLES) == {
        "knowledge graphs expert",
        "causal analysis expert",
        "NLPexpert",
        "none",
    }
    assert ROLES["none"] == ""
    assert ROLES["knowledge graphs expert"].startswith("You are an expert in knowledge graphs")
    assert ROLES["causal analysis expert"].startswith("You are a causal analysis expert")
    assert ROLES["NLPexpert"].startswith("You are an expert in natural language processing")
    assert SYSTEM_PROMPT == "Can you answer the following question as accurately as possible?"


def test_defaults_mirror_the_reference_demo() -> None:
    """Default roles/styles mirror the reference debate script."""
    assert DEFAULT_AGENT_ROLES == ("knowledge graphs expert",) * 4
    assert DEFAULT_PROMPT_STYLES == ("positive", "normal", "normal", "negative")
    assert set(DEFAULT_PROMPT_STYLES) <= set(PROMPT_STYLES)
