"""Tests for probability validation and prediction status semantics.

Tie rule under test: a choice is accepted when

    max(probabilities) - probabilities[choice] <= 1e-12

The interface choice is preserved as the top label, so exact or near ties may
resolve to any option that is within tolerance of the maximum.  A choice that
is outside that tolerance is rejected even if all probabilities are otherwise
well formed.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, is_dataclass
from enum import Enum

import pytest

from eci_decision_models.probability import (
    InvalidProbabilityError,
    PredictionStatus,
    ValidatedPrediction,
    validate_prediction,
    validate_probabilities,
)

TOLERANCE = 1e-12


def test_prediction_status_has_exact_public_values() -> None:
    assert issubclass(PredictionStatus, Enum)
    assert len(PredictionStatus) == 5
    assert {status.value for status in PredictionStatus} == {
        "ok",
        "missing_probability",
        "invalid_probability",
        "invalid_response",
        "api_error",
    }


def test_invalid_probability_error_is_an_exception() -> None:
    assert issubclass(InvalidProbabilityError, Exception)


def test_validate_probabilities_normalizes_and_orders_canonically() -> None:
    result = validate_probabilities({"none": 0.1, "cause": 0.3, "caused_by": 0.6})

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result == pytest.approx((0.3, 0.6, 0.1))
    assert math.isclose(sum(result), 1.0, abs_tol=TOLERANCE)


def test_validate_probabilities_normalizes_unnormalized_values() -> None:
    result = validate_probabilities({"cause": 3.0, "caused_by": 6.0, "none": 1.0})

    assert result == pytest.approx((0.3, 0.6, 0.1))
    assert math.isclose(sum(result), 1.0, abs_tol=TOLERANCE)


def test_zero_probability_is_a_valid_input() -> None:
    result = validate_probabilities({"cause": 0.0, "caused_by": 1.0, "none": 0.0})

    assert result == pytest.approx((0.0, 1.0, 0.0))


@pytest.mark.parametrize(
    "values",
    [
        {"cause": 0.5, "caused_by": 0.5},
        {"cause": 0.5, "caused_by": 0.25, "none": 0.25, "extra": 0.0},
        {"cause": 0.5, "other": 0.25, "none": 0.25},
    ],
)
def test_validate_probabilities_requires_exactly_three_canonical_keys(
    values: dict[str, float],
) -> None:
    with pytest.raises(InvalidProbabilityError):
        validate_probabilities(values)


def test_validate_probabilities_rejects_negative_values() -> None:
    with pytest.raises(InvalidProbabilityError):
        validate_probabilities({"cause": -0.1, "caused_by": 0.6, "none": 0.5})


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_validate_probabilities_rejects_non_finite_values(bad_value: float) -> None:
    with pytest.raises(InvalidProbabilityError):
        validate_probabilities({"cause": bad_value, "caused_by": 0.5, "none": 0.5})


def test_validate_probabilities_rejects_zero_sum() -> None:
    with pytest.raises(InvalidProbabilityError):
        validate_probabilities({"cause": 0.0, "caused_by": 0.0, "none": 0.0})


def test_validated_prediction_is_a_frozen_dataclass() -> None:
    result = validate_prediction("cause", {"cause": 1.0, "caused_by": 0.0, "none": 0.0})

    assert isinstance(result, ValidatedPrediction)
    assert is_dataclass(result)
    params = type(result).__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen is True

    with pytest.raises(FrozenInstanceError):
        result.status = PredictionStatus("ok")  # type: ignore[misc]


def test_valid_prediction_is_ok_with_canonical_probability_order() -> None:
    result = validate_prediction(
        "cause",
        {"none": 0.2, "caused_by": 0.3, "cause": 0.5},
    )

    assert isinstance(result, ValidatedPrediction)
    assert isinstance(result.status, PredictionStatus)
    assert result.status.value == "ok"
    assert result.prediction_valid is True
    assert result.probability_valid is True
    assert result.choice == "cause"
    assert result.probabilities == pytest.approx((0.5, 0.3, 0.2))
    assert result.error is None


def test_missing_choice_is_invalid_response() -> None:
    result = validate_prediction(
        None,
        {"cause": 0.5, "caused_by": 0.3, "none": 0.2},
    )

    assert result.status.value == "invalid_response"
    assert result.prediction_valid is False
    assert result.probability_valid is False
    assert result.choice is None
    assert result.probabilities is None
    assert isinstance(result.error, str)
    assert result.error


def test_missing_choice_takes_priority_over_missing_probabilities() -> None:
    result = validate_prediction(None, None)

    assert result.status.value == "invalid_response"
    assert result.prediction_valid is False
    assert result.probability_valid is False
    assert result.choice is None
    assert result.probabilities is None
    assert isinstance(result.error, str)
    assert result.error


def test_choice_without_probabilities_is_missing_probability() -> None:
    result = validate_prediction("cause", None)

    assert result.status.value == "missing_probability"
    assert result.prediction_valid is True
    assert result.probability_valid is False
    assert result.choice == "cause"
    assert result.probabilities is None
    assert isinstance(result.error, str)
    assert result.error


@pytest.mark.parametrize(
    "probabilities",
    [
        {"cause": -0.1, "caused_by": 0.6, "none": 0.5},
        {"cause": math.nan, "caused_by": 0.5, "none": 0.5},
        {"cause": math.inf, "caused_by": 0.5, "none": 0.5},
        {"cause": -math.inf, "caused_by": 0.5, "none": 0.5},
        {"cause": 0.0, "caused_by": 0.0, "none": 0.0},
        {"cause": 0.5, "caused_by": 0.25, "none": 0.25, "extra": 0.0},
    ],
)
def test_malformed_probabilities_is_invalid_probability(
    probabilities: dict[str, float],
) -> None:
    result = validate_prediction("cause", probabilities)

    assert result.status.value == "invalid_probability"
    assert result.prediction_valid is True
    assert result.probability_valid is False
    assert result.choice == "cause"
    assert result.probabilities is None
    assert isinstance(result.error, str)
    assert result.error


def test_missing_probability_keys_is_missing_probability() -> None:
    result = validate_prediction("cause", {"cause": 0.5, "caused_by": 0.5})

    assert result.status.value == "missing_probability"
    assert result.prediction_valid is True
    assert result.probability_valid is False
    assert result.choice == "cause"
    assert result.probabilities is None
    assert isinstance(result.error, str)
    assert result.error


def test_choice_inconsistent_with_argmax_is_invalid_probability() -> None:
    result = validate_prediction(
        "none",
        {"cause": 0.6, "caused_by": 0.3, "none": 0.1},
    )

    assert result.status.value == "invalid_probability"
    assert result.prediction_valid is True
    assert result.probability_valid is False
    assert result.choice == "none"
    assert result.probabilities is None
    assert isinstance(result.error, str)
    assert result.error


@pytest.mark.parametrize("choice", ["cause", "caused_by"])
def test_exact_tie_at_max_accepts_either_interface_choice(choice: str) -> None:
    result = validate_prediction(
        choice,
        {"cause": 0.5, "caused_by": 0.5, "none": 0.0},
    )

    assert result.status.value == "ok"
    assert result.prediction_valid is True
    assert result.probability_valid is True
    assert result.choice == choice
    assert result.probabilities == pytest.approx((0.5, 0.5, 0.0))
    assert result.error is None


def test_near_tie_within_tolerance_accepts_non_argmax_interface_choice() -> None:
    probabilities = {
        "cause": 0.500_000_000_000_25,
        "caused_by": 0.499_999_999_999_75,
        "none": 0.0,
    }
    assert probabilities["cause"] - probabilities["caused_by"] < TOLERANCE

    result = validate_prediction("caused_by", probabilities)

    assert result.status.value == "ok"
    assert result.prediction_valid is True
    assert result.probability_valid is True
    assert result.choice == "caused_by"
    assert result.probabilities == pytest.approx((0.500_000_000_000_25, 0.499_999_999_999_75, 0.0))
    assert result.error is None


def test_choice_outside_tolerance_of_max_is_rejected() -> None:
    probabilities = {
        "cause": 0.500_000_000_002,
        "caused_by": 0.499_999_999_998,
        "none": 0.0,
    }
    assert probabilities["cause"] - probabilities["caused_by"] > TOLERANCE

    result = validate_prediction("caused_by", probabilities)

    assert result.status.value == "invalid_probability"
    assert result.prediction_valid is True
    assert result.probability_valid is False
    assert result.choice == "caused_by"
    assert result.probabilities is None
    assert isinstance(result.error, str)
    assert result.error


def test_zero_probability_is_valid_when_choice_is_argmax() -> None:
    result = validate_prediction(
        "caused_by",
        {"cause": 0.0, "caused_by": 1.0, "none": 0.0},
    )

    assert result.status.value == "ok"
    assert result.prediction_valid is True
    assert result.probability_valid is True
    assert result.probabilities == pytest.approx((0.0, 1.0, 0.0))


def test_validate_prediction_returns_probabilities_in_canonical_order() -> None:
    result = validate_prediction(
        "none",
        {"none": 0.7, "cause": 0.2, "caused_by": 0.1},
    )

    assert result.status.value == "ok"
    assert result.probabilities == pytest.approx((0.2, 0.1, 0.7))
