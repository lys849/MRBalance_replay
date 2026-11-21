"""Tests for the standardized prediction artifact schema and JSONL handling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from eci_decision_models.artifacts import (
    PredictionRecord,
    atomic_write_text,
    read_predictions_jsonl,
    redact_secrets,
    sha256_file,
    write_predictions_jsonl,
)

EXPECTED_FIELDS: frozenset[str] = frozenset(
    {
        "run_id",
        "item_id",
        "record_id",
        "model",
        "attempt",
        "raw_request_id",
        "request_hash",
        "response_hash",
        "adapter_version",
        "parser_version",
        "gold_choice",
        "prediction",
        "status",
        "prediction_valid",
        "probability_valid",
        "probabilities",
        "error",
        "control_status",
    }
)

REDACTED = "[REDACTED]"


def _record(
    *,
    record_id: str = "r1",
    item_id: int = 1,
    gold_choice: str = "cause",
    prediction: str | None = "cause",
    status: str = "ok",
    prediction_valid: bool = True,
    probability_valid: bool = True,
    probabilities: tuple[float, float, float] | None = (0.6, 0.3, 0.1),
    error: str | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        run_id="run-1",
        item_id=item_id,
        record_id=record_id,
        model="test-model",
        attempt=1,
        raw_request_id=f"request-{record_id}",
        request_hash="request-hash",
        response_hash="response-hash",
        adapter_version="adapter-v1",
        parser_version="parser-v1",
        gold_choice=gold_choice,
        prediction=prediction,
        status=status,
        prediction_valid=prediction_valid,
        probability_valid=probability_valid,
        probabilities=probabilities,
        error=error,
    )


def _consistent_record(status: str) -> PredictionRecord:
    if status == "ok":
        return _record()
    if status == "missing_probability":
        return _record(
            status=status,
            probability_valid=False,
            probabilities=None,
            error="missing probability",
        )
    if status == "invalid_probability":
        return _record(
            status=status,
            probability_valid=False,
            probabilities=None,
            error="invalid probability",
        )
    return _record(
        status=status,
        prediction=None,
        prediction_valid=False,
        probability_valid=False,
        probabilities=None,
        error="no usable response",
    )


@pytest.mark.parametrize(
    "status",
    [
        "ok",
        "missing_probability",
        "invalid_probability",
        "invalid_response",
        "api_error",
    ],
)
def test_from_dict_accepts_every_consistent_status(status: str) -> None:
    record = _consistent_record(status)

    assert PredictionRecord.from_dict(record.to_dict()) == record


def test_prediction_record_is_frozen_and_has_standardized_fields() -> None:
    record = _record()

    assert is_dataclass(record)
    assert {field.name for field in fields(PredictionRecord)} == EXPECTED_FIELDS

    with pytest.raises(FrozenInstanceError):
        record.status = "api_error"  # type: ignore[misc]


def test_to_dict_uses_canonical_probability_list_and_fresh_containers() -> None:
    record = _record()
    data = record.to_dict()

    assert set(data) == EXPECTED_FIELDS
    assert data["probabilities"] == [0.6, 0.3, 0.1]
    assert isinstance(data["probabilities"], list)

    data["record_id"] = "mutated"
    assert record.record_id == "r1"


def test_to_dict_from_dict_round_trip_preserves_record() -> None:
    record = _record(record_id="r7", item_id=7, gold_choice="caused_by")

    assert PredictionRecord.from_dict(record.to_dict()) == record


def test_from_dict_round_trips_missing_probabilities() -> None:
    record = _record(
        status="missing_probability",
        probability_valid=False,
        probabilities=None,
        error="missing probability",
    )

    restored = PredictionRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.probabilities is None


def _api_error_record() -> PredictionRecord:
    return _record(
        status="api_error",
        prediction=None,
        prediction_valid=False,
        probability_valid=False,
        probabilities=None,
        error="no usable response",
    )


def test_control_status_round_trips_none_and_budget_exhausted() -> None:
    no_control = _record()
    budget_exhausted = replace(_api_error_record(), control_status="budget_exhausted")

    for record in (no_control, budget_exhausted):
        restored = PredictionRecord.from_dict(record.to_dict())
        assert restored == record
        assert restored.control_status == record.control_status

    assert no_control.control_status is None
    assert budget_exhausted.control_status == "budget_exhausted"
    assert budget_exhausted.to_dict()["control_status"] == "budget_exhausted"


def test_control_status_requires_api_error_status() -> None:
    with pytest.raises(ValueError):
        replace(_record(), control_status="budget_exhausted")


@pytest.mark.parametrize("control_status", ["", "   ", "\t\n"])
def test_control_status_must_be_a_non_empty_string(control_status: str) -> None:
    with pytest.raises(ValueError):
        replace(_api_error_record(), control_status=control_status)


def test_from_dict_requires_the_control_status_field() -> None:
    data = _record().to_dict()
    del data["control_status"]

    with pytest.raises(ValueError):
        PredictionRecord.from_dict(data)


@pytest.mark.parametrize(
    "updates",
    [
        {"prediction": None},
        {"prediction_valid": False},
        {"probability_valid": False},
        {"probabilities": None},
        {"probabilities": [0.6, 0.3]},
        {"status": "missing_probability"},
        {"status": "invalid_probability", "probability_valid": True},
        {"status": "invalid_response", "prediction": "cause"},
        {"status": "invalid_response", "prediction_valid": True},
        {"status": "api_error", "prediction_valid": True},
        {"status": "not-a-status"},
    ],
)
def test_from_dict_rejects_status_boolean_and_probability_inconsistency(
    updates: dict[str, object],
) -> None:
    data = _record().to_dict()
    data.update(updates)

    with pytest.raises(ValueError):
        PredictionRecord.from_dict(data)


def test_constructing_ok_record_rejects_prediction_outside_probability_argmax() -> None:
    with pytest.raises(ValueError):
        _record(prediction="none", probabilities=(0.6, 0.3, 0.1))


def test_write_and_read_predictions_jsonl_is_deterministic_and_round_trips(
    tmp_path: Path,
) -> None:
    first = _record(record_id="r1", item_id=1)
    second = _record(
        record_id="r2",
        item_id=2,
        gold_choice="none",
        prediction="none",
        probabilities=(0.1, 0.2, 0.7),
    )
    path = tmp_path / "predictions.jsonl"

    write_predictions_jsonl([first, second], path)

    expected = "".join(
        json.dumps(
            record.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for record in (first, second)
    )
    assert path.read_text(encoding="utf-8") == expected
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert read_predictions_jsonl(path) == [first, second]

    write_predictions_jsonl([first, second], path)
    assert path.read_text(encoding="utf-8") == expected


def test_read_predictions_jsonl_accepts_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert read_predictions_jsonl(path) == []


def test_atomic_writes_preserve_content_and_leave_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_text("old\n", encoding="utf-8")

    atomic_write_text(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["artifact.jsonl"]

    records_path = tmp_path / "predictions.jsonl"
    write_predictions_jsonl([_record()], records_path)
    assert read_predictions_jsonl(records_path) == [_record()]
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "artifact.jsonl",
        "predictions.jsonl",
    ]
    assert not any(item.name.endswith(".tmp") for item in tmp_path.iterdir())


def test_sha256_file_matches_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    payload = b'{"answer":1}\n\xe4\xb8\xad'
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_redact_secrets_is_recursive_case_insensitive_and_preserves_lineage() -> None:
    payload: dict[str, object] = {
        "record_id": "r-1",
        "item_id": 7,
        "API_KEY": "sk-live-should-not-survive",
        "Authorization": "Bearer auth-token-should-not-survive",
        "nested": {
            "access_token": "token-value-should-not-survive",
            "keep": "visible",
            "client_secret": "secret-value-should-not-survive",
        },
        "items": [
            {
                "Password": "password-should-not-survive",
                "item_id": 8,
                "note": "lineage",
            },
        ],
    }

    redacted = redact_secrets(payload)

    assert isinstance(redacted, dict)
    assert redacted["record_id"] == "r-1"
    assert redacted["item_id"] == 7
    assert redacted["API_KEY"] == REDACTED
    assert redacted["Authorization"] == REDACTED
    assert redacted["nested"] == {
        "access_token": REDACTED,
        "keep": "visible",
        "client_secret": REDACTED,
    }
    assert redacted["items"] == [
        {
            "Password": REDACTED,
            "item_id": 8,
            "note": "lineage",
        },
    ]

    serialized = json.dumps(redacted, ensure_ascii=False)
    for secret in (
        "sk-live-should-not-survive",
        "auth-token-should-not-survive",
        "token-value-should-not-survive",
        "secret-value-should-not-survive",
        "password-should-not-survive",
    ):
        assert secret not in serialized
    assert "Bearer" not in serialized

    assert payload["API_KEY"] == "sk-live-should-not-survive"
    nested = cast("dict[str, object]", payload["nested"])
    assert nested["access_token"] == "token-value-should-not-survive"
