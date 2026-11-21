"""Canonical ECI request contract shared by every model adapter.

The canonical request contains only the inference-time information: the
sentence, the two events in display order and the fixed typed-Choice question.
Gold labels and document metadata must never be part of it.

The SHA-256 digest is computed over deterministic JSON: UTF-8 bytes,
``sort_keys=True``, ``separators=(",", ":")`` and ``ensure_ascii=False``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, SupportsIndex, cast

SCHEMA_VERSION: Final = "1.0.0"
PROMPT_VERSION: Final = "eci-canonical-prompt-v1"


class Choice(str, Enum):
    """Stable internal identifiers for the three-way causal choice."""

    cause = "cause"
    caused_by = "caused_by"
    none = "none"


QUESTION_INSTRUCTIONS: Final = (
    "Read the sentence and the two events in the order given. Event 1 is the "
    "first event and event 2 is the second event. Choose exactly one option: "
    "(A) event 1 causes event 2, (B) event 1 is caused by event 2, or (C) there "
    "is no causal relationship between the events."
)

CHOICE_IDS: Final[tuple[str, str, str]] = (
    Choice.cause.value,
    Choice.caused_by.value,
    Choice.none.value,
)
CHOICE_LETTERS: Final[tuple[str, str, str]] = ("A", "B", "C")
CHOICE_DESCRIPTIONS: Final[tuple[str, str, str]] = (
    "event 1 causes event 2",
    "event 1 is caused by event 2",
    "no causal relationship",
)


def _build_options() -> tuple[dict[str, str], ...]:
    """Return a fresh, ordered option list for a single request."""
    return tuple(
        {
            "id": choice_id,
            "letter": letter,
            "description": description,
        }
        for choice_id, letter, description in zip(
            CHOICE_IDS,
            CHOICE_LETTERS,
            CHOICE_DESCRIPTIONS,
            strict=True,
        )
    )


def _build_payload(
    sentence: str,
    event1: str,
    event2: str,
    options: tuple[dict[str, str], ...],
) -> dict[str, object]:
    """Return a fresh canonical payload with its own option containers."""
    question_options = [dict(option) for option in options]
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "state": {
            "sentence": sentence,
            "event1": event1,
            "event2": event2,
        },
        "question": {
            "type": "choice",
            "instructions": QUESTION_INSTRUCTIONS,
            "options": question_options,
        },
    }


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class _FrozenDict(dict[str, Any]):
    """A dict subclass that keeps the JSON shape but rejects every mutation."""

    _frozen_initialized: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_frozen_initialized", False):
            raise TypeError("canonical payload is immutable")
        super().__init__(*args, **kwargs)
        self._frozen_initialized = True

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("canonical payload is immutable")

    def __ior__(self, other: Any) -> _FrozenDict:  # type: ignore[override, misc]
        raise TypeError("canonical payload is immutable")

    def __delitem__(self, key: str) -> None:
        raise TypeError("canonical payload is immutable")

    def clear(self) -> None:
        raise TypeError("canonical payload is immutable")

    def pop(self, key: str, default: Any = None) -> Any:
        raise TypeError("canonical payload is immutable")

    def popitem(self) -> tuple[str, Any]:
        raise TypeError("canonical payload is immutable")

    def setdefault(self, key: str, default: Any = None) -> Any:
        raise TypeError("canonical payload is immutable")

    def update(self, *args: object, **kwargs: object) -> None:
        raise TypeError("canonical payload is immutable")


class _FrozenList(list[Any]):
    """A list subclass that keeps the JSON shape but rejects every mutation."""

    _frozen_initialized: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_frozen_initialized", False):
            raise TypeError("canonical payload is immutable")
        super().__init__(*args, **kwargs)
        self._frozen_initialized = True

    def __setitem__(self, index: Any, value: Any) -> None:
        raise TypeError("canonical payload is immutable")

    def __ior__(self, other: Any) -> _FrozenList:
        raise TypeError("canonical payload is immutable")

    def __delitem__(self, index: Any) -> None:
        raise TypeError("canonical payload is immutable")

    def append(self, value: Any) -> None:
        raise TypeError("canonical payload is immutable")

    def extend(self, values: Any) -> None:
        raise TypeError("canonical payload is immutable")

    def insert(self, index: SupportsIndex, value: Any) -> None:
        raise TypeError("canonical payload is immutable")

    def remove(self, value: Any) -> None:
        raise TypeError("canonical payload is immutable")

    def pop(self, index: SupportsIndex = -1) -> Any:
        raise TypeError("canonical payload is immutable")

    def clear(self) -> None:
        raise TypeError("canonical payload is immutable")

    def sort(self, *args: object, **kwargs: object) -> None:
        raise TypeError("canonical payload is immutable")

    def reverse(self) -> None:
        raise TypeError("canonical payload is immutable")

    def __iadd__(self, values: Any) -> _FrozenList:  # type: ignore[misc]
        raise TypeError("canonical payload is immutable")

    def __imul__(self, count: SupportsIndex) -> Any:
        raise TypeError("canonical payload is immutable")


def _deep_freeze(value: Any) -> Any:
    """Recursively rebuild dict/list containers as immutable equivalents."""
    if isinstance(value, dict):
        frozen = _FrozenDict()
        for key, item in value.items():
            dict.__setitem__(frozen, key, _deep_freeze(item))
        return frozen
    if isinstance(value, list):
        frozen_list = _FrozenList()
        for item in value:
            list.append(frozen_list, _deep_freeze(item))
        return frozen_list
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CanonicalRequest:
    """A frozen canonical request together with its deterministic digest."""

    payload: dict[str, object]
    options: tuple[dict[str, str], ...]
    sha256: str


def build_canonical_request(sentence: str, event1: str, event2: str) -> CanonicalRequest:
    """Build the canonical request and its SHA-256 digest.

    Every call allocates fresh, deeply immutable payload and option containers,
    so callers cannot mutate shared state that would affect later requests.
    """
    raw_options = _build_options()
    raw_payload = _build_payload(sentence, event1, event2, raw_options)
    options = cast("tuple[dict[str, str], ...]", _deep_freeze(raw_options))
    payload = cast("dict[str, object]", _deep_freeze(raw_payload))
    serialized = _canonical_json(payload)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return CanonicalRequest(payload=payload, options=options, sha256=digest)
