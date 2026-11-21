"""Standardized prediction artifacts and JSONL persistence.

``PredictionRecord`` is the frozen, deterministic representation of one model
attempt.  It keeps every failure visible: the five statuses are validated
against the ``prediction_valid`` / ``probability_valid`` booleans both when a
record is constructed and when it is restored from JSON.  Probability vectors
are stored in canonical ``[cause, caused_by, none]`` order, and a record whose
probabilities are invalid always carries ``probabilities=None``.

The JSONL helpers use deterministic JSON (``sort_keys=True``,
``separators=(",", ":")``, ``ensure_ascii=False``) so that writing the same
records always produces byte-identical output.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from eci_decision_models.canonical import CHOICE_IDS
from eci_decision_models.probability import PredictionStatus, validate_prediction

REDACTED: Final = "[REDACTED]"

SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "authorization",
        "access_token",
        "client_secret",
        "secret",
        "password",
        "token",
    },
)

_STATUS_VALUES: Final[frozenset[str]] = frozenset(status.value for status in PredictionStatus)

_EXPECTED_FIELDS: Final[frozenset[str]] = frozenset(
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
    },
)

_PROBABILITY_SUM_TOLERANCE: Final = 1e-9


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _require_optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name)


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _validate_probabilities(
    probabilities: tuple[float, float, float],
) -> tuple[float, float, float]:
    for value in probabilities:
        if not math.isfinite(value):
            raise ValueError("probabilities must be finite")
        if value < 0.0:
            raise ValueError("probabilities must be non-negative")
    total = probabilities[0] + probabilities[1] + probabilities[2]
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_SUM_TOLERANCE):
        raise ValueError("probabilities must be normalized to sum to one")
    return probabilities


@dataclass(frozen=True)
class PredictionRecord:
    """One standardized prediction attempt with lineage and failure status."""

    run_id: str
    item_id: int
    record_id: str
    model: str
    attempt: int
    raw_request_id: str
    request_hash: str
    response_hash: str
    adapter_version: str
    parser_version: str
    gold_choice: str
    prediction: str | None
    status: str
    prediction_valid: bool
    probability_valid: bool
    probabilities: tuple[float, float, float] | None
    error: str | None
    control_status: str | None = None

    def __post_init__(self) -> None:
        _require_str(self.run_id, "run_id")
        _require_int(self.item_id, "item_id")
        _require_str(self.record_id, "record_id")
        _require_str(self.model, "model")
        _require_int(self.attempt, "attempt")
        _require_str(self.raw_request_id, "raw_request_id")
        _require_str(self.request_hash, "request_hash")
        _require_str(self.response_hash, "response_hash")
        _require_str(self.adapter_version, "adapter_version")
        _require_str(self.parser_version, "parser_version")
        _require_optional_str(self.error, "error")
        _require_bool(self.prediction_valid, "prediction_valid")
        _require_bool(self.probability_valid, "probability_valid")
        control_status = _require_optional_str(self.control_status, "control_status")
        if control_status is not None:
            if not control_status.strip():
                raise ValueError("control_status must be a non-empty string")
            if self.status != PredictionStatus.API_ERROR.value:
                raise ValueError("control_status requires status 'api_error'")

        if self.gold_choice not in CHOICE_IDS:
            raise ValueError(f"gold_choice must be one of {list(CHOICE_IDS)!r}")
        if self.prediction is not None and self.prediction not in CHOICE_IDS:
            raise ValueError(f"prediction must be one of {list(CHOICE_IDS)!r} or None")
        if self.status not in _STATUS_VALUES:
            raise ValueError(f"status must be one of {sorted(_STATUS_VALUES)!r}")

        probabilities = self.probabilities
        if probabilities is not None:
            if isinstance(probabilities, bool) or not isinstance(probabilities, (list, tuple)):
                raise ValueError("probabilities must be a three-element sequence or None")
            if len(probabilities) != 3:
                raise ValueError("probabilities must contain exactly three values")
            values: list[float] = []
            for value in probabilities:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("probabilities must be numeric")
                values.append(float(value))
            normalized: tuple[float, float, float] = (values[0], values[1], values[2])
            object.__setattr__(self, "probabilities", _validate_probabilities(normalized))

        self._validate_status_consistency()

    def _validate_status_consistency(self) -> None:
        prediction_present = self.prediction is not None
        probabilities_present = self.probabilities is not None

        if self.probability_valid and not self.prediction_valid:
            raise ValueError("probability_valid requires prediction_valid")
        if not self.prediction_valid and prediction_present:
            raise ValueError("prediction must be None when prediction_valid is false")
        if not self.probability_valid and probabilities_present:
            raise ValueError("probabilities must be None when probability_valid is false")
        if self.probability_valid and not probabilities_present:
            raise ValueError("probabilities are required when probability_valid is true")

        if self.status == PredictionStatus.OK.value:
            if not (self.prediction_valid and self.probability_valid):
                raise ValueError("status 'ok' requires valid prediction and probabilities")
            prediction = self.prediction
            probabilities = self.probabilities
            if prediction is None or probabilities is None:
                raise ValueError("status 'ok' requires prediction and probabilities")
            validated = validate_prediction(
                prediction,
                dict(zip(CHOICE_IDS, probabilities, strict=True)),
            )
            if validated.status is not PredictionStatus.OK:
                raise ValueError(
                    validated.error
                    or "status 'ok' requires a prediction consistent with probabilities",
                )
            return

        if self.status in (
            PredictionStatus.MISSING_PROBABILITY.value,
            PredictionStatus.INVALID_PROBABILITY.value,
        ):
            if not self.prediction_valid or self.probability_valid:
                raise ValueError(
                    f"status {self.status!r} requires prediction_valid=true and "
                    "probability_valid=false",
                )
            if not prediction_present or probabilities_present:
                raise ValueError(
                    f"status {self.status!r} requires a prediction and no probabilities",
                )
            return

        if self.prediction_valid or self.probability_valid:
            raise ValueError(
                f"status {self.status!r} requires both validity flags to be false",
            )
        if prediction_present or probabilities_present:
            raise ValueError(
                f"status {self.status!r} requires prediction and probabilities to be None",
            )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh, JSON-serializable mapping of the record."""
        probabilities = self.probabilities
        return {
            "run_id": self.run_id,
            "item_id": self.item_id,
            "record_id": self.record_id,
            "model": self.model,
            "attempt": self.attempt,
            "raw_request_id": self.raw_request_id,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "adapter_version": self.adapter_version,
            "parser_version": self.parser_version,
            "gold_choice": self.gold_choice,
            "prediction": self.prediction,
            "status": self.status,
            "prediction_valid": self.prediction_valid,
            "probability_valid": self.probability_valid,
            "probabilities": list(probabilities) if probabilities is not None else None,
            "error": self.error,
            "control_status": self.control_status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PredictionRecord:
        """Restore a record, rejecting unknown fields and inconsistent states."""
        if not isinstance(data, Mapping):
            raise ValueError("record must be a mapping")
        if set(data) != _EXPECTED_FIELDS:
            missing = sorted(_EXPECTED_FIELDS - set(data))
            unexpected = sorted(set(data) - _EXPECTED_FIELDS)
            raise ValueError(
                f"record fields mismatch: missing={missing!r} unexpected={unexpected!r}",
            )

        return cls(
            run_id=_require_str(data["run_id"], "run_id"),
            item_id=_require_int(data["item_id"], "item_id"),
            record_id=_require_str(data["record_id"], "record_id"),
            model=_require_str(data["model"], "model"),
            attempt=_require_int(data["attempt"], "attempt"),
            raw_request_id=_require_str(data["raw_request_id"], "raw_request_id"),
            request_hash=_require_str(data["request_hash"], "request_hash"),
            response_hash=_require_str(data["response_hash"], "response_hash"),
            adapter_version=_require_str(data["adapter_version"], "adapter_version"),
            parser_version=_require_str(data["parser_version"], "parser_version"),
            gold_choice=_require_str(data["gold_choice"], "gold_choice"),
            prediction=_require_optional_str(data["prediction"], "prediction"),
            status=_require_str(data["status"], "status"),
            prediction_valid=_require_bool(data["prediction_valid"], "prediction_valid"),
            probability_valid=_require_bool(
                data["probability_valid"],
                "probability_valid",
            ),
            probabilities=_parse_probabilities(data["probabilities"]),
            error=_require_optional_str(data["error"], "error"),
            control_status=_require_optional_str(data["control_status"], "control_status"),
        )


def _parse_probabilities(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("probabilities must be a three-element sequence or None")
    if len(value) != 3:
        raise ValueError("probabilities must contain exactly three values")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("probabilities must be numeric")
        numbers.append(float(item))
    return (numbers[0], numbers[1], numbers[2])


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` via a same-directory temporary file and atomic replace.

    The temporary file is flushed and fsynced before ``os.replace``.  On failure
    the temporary file is removed, so a caller never observes a partial write or
    a leftover sibling temp file.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_predictions_jsonl(
    records: Iterable[PredictionRecord],
    path: Path,
) -> None:
    """Write records as deterministic JSONL, one record per line."""
    content = "".join(
        json.dumps(
            record.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    )
    atomic_write_text(path, content)


def read_predictions_jsonl(path: Path) -> list[PredictionRecord]:
    """Read a JSONL file, skipping blank lines and restoring each record."""
    records: list[PredictionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError("each JSONL line must contain a JSON object")
            records.append(PredictionRecord.from_dict(payload))
    return records


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CANONICAL_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    "".join(character for character in key.lower() if character.isalnum()) for key in SENSITIVE_KEYS
)


def _canonical_key(key: str) -> str:
    """Flatten snake_case, hyphenated and camelCase names to one form."""
    return "".join(character for character in key.lower() if character.isalnum())


def _is_sensitive_key(key: str) -> bool:
    normalized = _canonical_key(key)
    return any(normalized.endswith(sensitive) for sensitive in _CANONICAL_SENSITIVE_KEYS)


def _redact_value(value: object) -> object:
    if isinstance(value, Mapping):
        redacted: dict[object, object] = {}
        for key, item in value.items():
            if isinstance(key, str) and _is_sensitive_key(key):
                redacted[key] = REDACTED
            else:
                redacted[key] = _redact_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, set):
        return {_redact_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_redact_value(item) for item in value)
    return value


def redact_secrets(payload: object) -> object:
    """Return a deep copy with sensitive values replaced by ``[REDACTED]``.

    Matching is case-insensitive and normalizes snake_case, hyphenated and
    camelCase names to a single form, accepting exact sensitive names or names
    ending in a sensitive component (``api_key``, ``API-KEY``, ``apiKey``,
    ``X-Access-Token``).  Usage counters such as ``state_tokens`` and
    ``input_tokens`` are intentionally preserved.  The input object is never
    mutated.
    """
    return _redact_value(payload)
