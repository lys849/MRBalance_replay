"""Answer parsing helpers, faithfully ported from the official implementation.

``parse_single_choice`` and ``most_frequent`` mirror
``reference/MRBalance/CausalDebate/util.py``.  ``extract_final_label`` provides
two modes:

``naive``
    The ECI.py baseline tokenizer: ``re.split("[^a-z]", ...)`` and the first
    ``a``/``b``/``c`` token.  This is *not* how the official debate pipeline
    derives labels -- it reads the consensus letter (A/B/C) before tokenizing.
    On free-form prose the tokenizer also mistakes articles such as "a" for
    option A.

``fixed``
    Prefers the last parenthesized option in prose and otherwise recognizes the
    "No causal relationship" phrasing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

_OPTION_PATTERN = re.compile(r"\(([ABCabc])\)")
_ALT_OPTION_PATTERN = re.compile(r"([ABCabc])\)")
_NO_CAUSAL_PATTERN = re.compile(r"\bno\s+causal\s+relationship\b")


def parse_single_choice(reply: str | None) -> str | None:
    """Return the last ``(A)``/``(B)``/``(C)`` option in ``reply``, if any."""

    if not reply:
        return None

    matches: list[str] = _OPTION_PATTERN.findall(reply)
    if matches:
        return matches[-1].upper()

    trailing_matches: list[str] = _ALT_OPTION_PATTERN.findall(reply)
    if trailing_matches:
        return trailing_matches[-1].upper()
    return None


def most_frequent(candidates: Sequence[T], cmp_func: Callable[[T, T], bool]) -> tuple[T, int]:
    """Return the most frequent candidate and its count, keeping first on ties."""

    if not candidates:
        raise ValueError("most_frequent requires at least one candidate")

    counter = 0
    num = candidates[0]
    for candidate in candidates:
        current_frequency = sum(cmp_func(candidate, item) for item in candidates)
        if current_frequency > counter:
            counter = current_frequency
            num = candidate
    return num, counter


def extract_final_label(response: str | None, mode: str = "naive") -> int:
    """Convert a free-form response into a binary causal label (1/0).

    ``naive`` mirrors the ECI.py baseline tokenizer and is intended for prose
    replies.  ``fixed`` prefers the last explicit option.  Neither mode is the
    official debate label: that is derived from the consensus letter itself.
    """

    if mode == "naive":
        text = "C" if response is None else response
        for word in re.split("[^a-z]", text.strip().lower()):
            if word in ("a", "b"):
                return 1
            if word == "c":
                return 0
        return 0

    if mode == "fixed":
        option = parse_single_choice(response)
        if option is not None:
            return 1 if option in ("A", "B") else 0

        lowered = (response or "").lower()
        if _NO_CAUSAL_PATTERN.search(lowered):
            return 0
        if re.search(r"\b(caused\s+by|cause)\b", lowered):
            return 1
        if re.search(r"\bcausal\s+relationship\b", lowered):
            return 1
        return 0

    raise ValueError(f"Unknown extract_final_label mode: {mode!r}")
