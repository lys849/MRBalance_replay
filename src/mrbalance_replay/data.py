"""Dataset loading and question construction for the MRBalance replay.

The prompt template is reproduced verbatim from the official implementation in
``reference/MRBalance/CausalDebate/Muty_agent_debate.py`` (including its
leading newline and absence of a trailing newline).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

QUESTION_TEMPLATE = (
    "\nInput: %s\n"
    'Question: is there a causal relationship between "%s" and "%s" ?\n'
    "(A) cause\n"
    "(B) caused by\n"
    "(C) none"
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines."""

    items: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            items.append(json.loads(stripped))
    return items


def build_question(item: Mapping[str, Any]) -> str:
    """Build the exactly-formatted single-item prompt used by the paper."""

    words = item["words"]
    sentence = " ".join(words)
    events = item["events"]
    event1 = " ".join(words[index] for index in events[0])
    event2 = " ".join(words[index] for index in events[1])
    return QUESTION_TEMPLATE % (sentence, event1, event2)


def gold_label(item: Mapping[str, Any]) -> int:
    """Return the binary causal label, preferring ``bi_causal_label``.

    The official scoring script reads ``labels`` while the debate output keeps
    ``bi_causal_label``; accepting both makes the data layer usable with either
    representation.
    """

    label = item.get("bi_causal_label")
    if label is None:
        label = item.get("labels")
    if label is None:
        raise KeyError(
            "Item has neither 'bi_causal_label' nor 'labels'; cannot determine gold label."
        )
    return int(label)
