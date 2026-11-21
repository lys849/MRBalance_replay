"""Probability validation and prediction status semantics.

A prediction is valid only when the interface returns a known choice.  A
probability vector is valid only when it contains exactly the three canonical
keys with finite, non-negative values whose sum is greater than zero.  Values
are normalized into the fixed ``[cause, caused_by, none]`` order.

The interface choice is authoritative, but it must be within ``1e-12`` of the
maximum normalized probability.  On an exact or near tie the interface choice
is preserved as the top label.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from eci_decision_models.canonical import CHOICE_IDS

TOLERANCE: Final = 1e-12


class PredictionStatus(str, Enum):
    """Fixed per-record status values used by every report."""

    OK = "ok"
    MISSING_PROBABILITY = "missing_probability"
    INVALID_PROBABILITY = "invalid_probability"
    INVALID_RESPONSE = "invalid_response"
    API_ERROR = "api_error"


class InvalidProbabilityError(Exception):
    """Raised when a probability vector cannot be normalized."""


@dataclass(frozen=True)
class ValidatedPrediction:
    """Immutable validation outcome for a single model response."""

    status: PredictionStatus
    prediction_valid: bool
    probability_valid: bool
    choice: str | None
    probabilities: tuple[float, float, float] | None
    error: str | None


def validate_probabilities(raw: Mapping[str, float]) -> tuple[float, float, float]:
    """Validate and normalize a three-way probability vector.

    The returned tuple is always in canonical ``[cause, caused_by, none]``
    order and sums to one.
    """
    if set(raw) != set(CHOICE_IDS):
        raise InvalidProbabilityError(
            f"probabilities must contain exactly the keys {list(CHOICE_IDS)!r}",
        )

    try:
        values: dict[str, float] = {}
        for choice_id in CHOICE_IDS:
            value = raw[choice_id]
            if isinstance(value, bool):
                raise InvalidProbabilityError(
                    f"probability for {choice_id!r} must not be a boolean",
                )
            values[choice_id] = float(value)
    except InvalidProbabilityError:
        raise
    except (TypeError, ValueError) as exc:
        raise InvalidProbabilityError("probabilities must be numeric") from exc

    for choice_id, value in values.items():
        if not math.isfinite(value):
            raise InvalidProbabilityError(
                f"probability for {choice_id!r} must be finite",
            )
        if value < 0.0:
            raise InvalidProbabilityError(
                f"probability for {choice_id!r} must be non-negative",
            )

    total = values[CHOICE_IDS[0]] + values[CHOICE_IDS[1]] + values[CHOICE_IDS[2]]
    if total <= 0.0:
        raise InvalidProbabilityError("probabilities must have a positive sum")

    return (
        values[CHOICE_IDS[0]] / total,
        values[CHOICE_IDS[1]] / total,
        values[CHOICE_IDS[2]] / total,
    )


def validate_prediction(
    choice: str | None,
    probabilities: Mapping[str, float] | None,
) -> ValidatedPrediction:
    """Validate an interface choice and its probability vector.

    Check order is deliberate: a bad or missing choice is ``invalid_response``
    even when probabilities are also absent, because the choice determines
    whether a prediction exists at all.
    """
    if choice is None or choice not in CHOICE_IDS:
        return ValidatedPrediction(
            status=PredictionStatus.INVALID_RESPONSE,
            prediction_valid=False,
            probability_valid=False,
            choice=None,
            probabilities=None,
            error="response did not contain a valid choice",
        )

    if probabilities is None:
        return ValidatedPrediction(
            status=PredictionStatus.MISSING_PROBABILITY,
            prediction_valid=True,
            probability_valid=False,
            choice=choice,
            probabilities=None,
            error="response did not contain a complete probability distribution",
        )

    if not isinstance(probabilities, Mapping):
        return ValidatedPrediction(
            status=PredictionStatus.INVALID_PROBABILITY,
            prediction_valid=True,
            probability_valid=False,
            choice=choice,
            probabilities=None,
            error="probabilities must be a mapping",
        )

    missing_keys = [choice_id for choice_id in CHOICE_IDS if choice_id not in probabilities]
    if missing_keys:
        return ValidatedPrediction(
            status=PredictionStatus.MISSING_PROBABILITY,
            prediction_valid=True,
            probability_valid=False,
            choice=choice,
            probabilities=None,
            error=(
                "response did not contain a complete probability distribution; "
                f"missing keys {missing_keys!r}"
            ),
        )

    try:
        normalized = validate_probabilities(probabilities)
    except InvalidProbabilityError as exc:
        return ValidatedPrediction(
            status=PredictionStatus.INVALID_PROBABILITY,
            prediction_valid=True,
            probability_valid=False,
            choice=choice,
            probabilities=None,
            error=str(exc),
        )

    choice_index = CHOICE_IDS.index(choice)
    max_probability = max(normalized)
    if max_probability - normalized[choice_index] > TOLERANCE:
        return ValidatedPrediction(
            status=PredictionStatus.INVALID_PROBABILITY,
            prediction_valid=True,
            probability_valid=False,
            choice=choice,
            probabilities=None,
            error="choice is inconsistent with the probability argmax",
        )

    return ValidatedPrediction(
        status=PredictionStatus.OK,
        prediction_valid=True,
        probability_valid=True,
        choice=choice,
        probabilities=normalized,
        error=None,
    )
