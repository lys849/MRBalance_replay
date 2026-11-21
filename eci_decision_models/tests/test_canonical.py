"""Tests for the canonical ECI request contract.

The canonical payload under test is intentionally explicit::

    {
        "schema_version": <str>,
        "prompt_version": <str>,
        "state": {
            "sentence": <str>,
            "event1": <str>,
            "event2": <str>,
        },
        "question": {
            "type": "choice",
            "options": [
                {"id": "cause", "letter": "A",
                 "description": "event 1 causes event 2"},
                {"id": "caused_by", "letter": "B",
                 "description": "event 1 is caused by event 2"},
                {"id": "none", "letter": "C",
                 "description": "no causal relationship"},
            ],
        },
    }

The hash is SHA-256 over canonical deterministic JSON: UTF-8 bytes,
``sort_keys=True``, ``separators=(",", ":")`` and ``ensure_ascii=False``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, is_dataclass
from typing import cast

import pytest

from eci_decision_models.canonical import CanonicalRequest, build_canonical_request

CHOICE_IDS: tuple[str, str, str] = ("cause", "caused_by", "none")
CHOICE_LETTERS: tuple[str, str, str] = ("A", "B", "C")
CHOICE_DESCRIPTIONS: tuple[str, str, str] = (
    "event 1 causes event 2",
    "event 1 is caused by event 2",
    "no causal relationship",
)
EXPECTED_OPTIONS: tuple[dict[str, str], ...] = tuple(
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
FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "bi_causal_label",
        "doc_metadata",
        "doc_name",
        "ESC_rel",
        "events_id",
        "gold",
        "gold_choice",
        "gold_choice_reconstructed",
        "item_id",
        "label",
        "labels",
        "metadata",
        "topic",
        "tri_causal_label",
    }
)


def _sample_request() -> CanonicalRequest:
    return build_canonical_request(
        "The storm destroyed the house.",
        "the storm",
        "the house was destroyed",
    )


def _as_mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return cast("Mapping[str, object]", value)


def _as_sequence(value: object) -> tuple[object, ...]:
    assert isinstance(value, (list, tuple))
    return tuple(value)


def _options_from(value: object) -> tuple[dict[str, object], ...]:
    return tuple(dict(_as_mapping(option)) for option in _as_sequence(value))


def _collect_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys |= _collect_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys |= _collect_keys(child)
    return keys


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def test_canonical_request_is_a_frozen_dataclass() -> None:
    request = _sample_request()

    assert isinstance(request, CanonicalRequest)
    assert is_dataclass(request)
    params = type(request).__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen is True

    with pytest.raises(FrozenInstanceError):
        request.sha256 = "not-the-real-hash"  # type: ignore[misc]


def test_payload_contains_canonical_state_and_single_choice_question() -> None:
    request = _sample_request()
    payload = request.payload

    assert isinstance(payload, dict)

    state = _as_mapping(payload["state"])
    assert dict(state) == {
        "sentence": "The storm destroyed the house.",
        "event1": "the storm",
        "event2": "the house was destroyed",
    }

    question = _as_mapping(payload["question"])
    assert question["type"] == "choice"
    assert _options_from(question["options"]) == EXPECTED_OPTIONS

    assert isinstance(payload["schema_version"], str)
    assert payload["schema_version"]
    assert isinstance(payload["prompt_version"], str)
    assert payload["prompt_version"]


def test_request_options_have_exact_ids_letters_descriptions_and_order() -> None:
    request = _sample_request()

    assert isinstance(request.options, tuple)
    assert len(request.options) == 3
    assert tuple(dict(_as_mapping(option)) for option in request.options) == EXPECTED_OPTIONS
    assert tuple(option["id"] for option in EXPECTED_OPTIONS) == CHOICE_IDS
    assert tuple(option["letter"] for option in EXPECTED_OPTIONS) == CHOICE_LETTERS
    assert tuple(option["description"] for option in EXPECTED_OPTIONS) == CHOICE_DESCRIPTIONS


def test_payload_options_match_request_options() -> None:
    request = _sample_request()
    payload = request.payload
    question = _as_mapping(payload["question"])

    assert _options_from(question["options"]) == tuple(
        dict(_as_mapping(option)) for option in request.options
    )


def test_payload_contains_no_gold_labels_or_document_metadata() -> None:
    request = _sample_request()

    assert _collect_keys(request.payload).isdisjoint(FORBIDDEN_KEYS)


def test_sha256_is_recomputable_from_canonical_payload_json() -> None:
    request = _sample_request()

    serialized = _canonical_json(request.payload)
    expected = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    assert request.sha256 == expected
    assert len(request.sha256) == 64
    assert set(request.sha256) <= set("0123456789abcdef")
    assert json.loads(serialized) == request.payload


def test_sha256_uses_utf8_without_ascii_escaping() -> None:
    request = build_canonical_request("台风摧毁了房屋。", "台风", "房屋被摧毁")
    state = _as_mapping(request.payload["state"])

    assert state["sentence"] == "台风摧毁了房屋。"
    assert state["event1"] == "台风"
    assert state["event2"] == "房屋被摧毁"

    serialized = _canonical_json(request.payload)
    assert "台风" in serialized
    expected = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    assert request.sha256 == expected


def test_equivalent_construction_is_deterministic() -> None:
    first = _sample_request()
    second = _sample_request()

    assert first == second
    assert first.payload == second.payload
    assert first.options == second.options
    assert first.sha256 == second.sha256


def test_swapping_event_order_changes_the_hash() -> None:
    first = build_canonical_request(
        "The storm destroyed the house.",
        "the storm",
        "the house was destroyed",
    )
    second = build_canonical_request(
        "The storm destroyed the house.",
        "the house was destroyed",
        "the storm",
    )

    assert first.payload != second.payload
    assert first.sha256 != second.sha256
