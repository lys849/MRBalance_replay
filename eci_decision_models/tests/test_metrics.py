"""Tests for classification, probability and binary metrics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import cast

import pytest

from eci_decision_models.artifacts import PredictionRecord
from eci_decision_models.metrics import (
    compute_binary_metrics,
    compute_classification_metrics,
    compute_paired_binary_metrics,
    compute_paired_classification_metrics,
    compute_probability_metrics,
)

TOLERANCE = 1e-12


def _record(
    *,
    record_id: str,
    item_id: int,
    gold_choice: str,
    prediction: str | None,
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


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return cast("Mapping[str, object]", value)


def _sequence(value: object) -> Sequence[object]:
    assert isinstance(value, (list, tuple))
    return cast("Sequence[object]", value)


def _number(value: object) -> float:
    assert isinstance(value, (int, float))
    assert not isinstance(value, bool)
    return float(value)


def _integer(value: object) -> int:
    assert isinstance(value, int)
    assert not isinstance(value, bool)
    return value


def _assert_counts(
    report: Mapping[str, object],
    *,
    n: int,
    valid: int,
    fail: int,
) -> None:
    assert _integer(report["N"]) == n
    assert _integer(report["valid"]) == valid
    assert _integer(report["fail"]) == fail
    expected_coverage = valid / n if n else 0.0
    assert _number(report["coverage"]) == pytest.approx(expected_coverage)


def _per_class(report: Mapping[str, object], choice: str) -> Mapping[str, object]:
    return _mapping(_mapping(report["per_class"])[choice])


def _ece_bins(report: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _sequence(report["ece_bins"]))


def _classification_fixture() -> list[PredictionRecord]:
    return [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r2",
            item_id=2,
            gold_choice="caused_by",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r3",
            item_id=3,
            gold_choice="none",
            prediction=None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
            probabilities=None,
            error="bad response",
        ),
        _record(
            record_id="r4",
            item_id=4,
            gold_choice="cause",
            prediction="none",
            status="missing_probability",
            probability_valid=False,
            probabilities=None,
            error="missing probability",
        ),
    ]


def test_full_classification_matches_hand_computed_fixture() -> None:
    records = _classification_fixture()
    before = list(records)

    report = compute_classification_metrics(records)

    assert records == before
    assert json.dumps(report)
    _assert_counts(report, n=4, valid=3, fail=1)
    assert _number(report["accuracy"]) == pytest.approx(0.25)
    assert _number(report["macro_f1"]) == pytest.approx(1 / 6)

    assert _number(_per_class(report, "cause")["precision"]) == pytest.approx(0.5)
    assert _number(_per_class(report, "cause")["recall"]) == pytest.approx(0.5)
    assert _number(_per_class(report, "cause")["f1"]) == pytest.approx(0.5)
    assert _number(_per_class(report, "caused_by")["f1"]) == pytest.approx(0.0)
    assert _number(_per_class(report, "none")["f1"]) == pytest.approx(0.0)

    assert report["confusion_columns"] == [
        "cause",
        "caused_by",
        "none",
        "invalid prediction",
    ]
    matrix = _mapping(report["confusion_matrix"])
    assert matrix["cause"] == [1, 0, 1, 0]
    assert matrix["caused_by"] == [1, 0, 0, 0]
    assert matrix["none"] == [0, 0, 0, 1]

    valid_subset = _mapping(report["valid_subset"])
    _assert_counts(valid_subset, n=3, valid=3, fail=0)
    assert _number(valid_subset["accuracy"]) == pytest.approx(1 / 3)
    assert _number(valid_subset["macro_f1"]) == pytest.approx(1 / 6)
    valid_matrix = _mapping(valid_subset["confusion_matrix"])
    assert valid_matrix["none"] == [0, 0, 0, 0]


def test_paired_classification_uses_common_both_valid_record_ids() -> None:
    first = _classification_fixture()
    second = [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r2",
            item_id=2,
            gold_choice="caused_by",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r3",
            item_id=3,
            gold_choice="none",
            prediction=None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
            probabilities=None,
            error="bad response",
        ),
        _record(
            record_id="r4",
            item_id=4,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
    ]

    report = compute_paired_classification_metrics(first, second)

    assert json.dumps(report)
    _assert_counts(report, n=3, valid=3, fail=0)
    assert _number(_mapping(report["first"])["accuracy"]) == pytest.approx(1 / 3)
    assert _number(_mapping(report["second"])["accuracy"]) == pytest.approx(2 / 3)


def test_paired_classification_hard_fails_on_duplicate_record_id() -> None:
    first = _classification_fixture()
    second = _classification_fixture()
    duplicated = [first[0], *first]

    with pytest.raises(ValueError):
        compute_paired_classification_metrics(duplicated, second)


def test_paired_classification_hard_fails_on_gold_mismatch() -> None:
    first = _classification_fixture()
    mismatched = [
        *first[:3],
        _record(
            record_id="r4",
            item_id=4,
            gold_choice="caused_by",
            prediction="none",
            status="missing_probability",
            probability_valid=False,
            probabilities=None,
            error="missing probability",
        ),
    ]

    with pytest.raises(ValueError):
        compute_paired_classification_metrics(first, mismatched)


def _probability_fixtures() -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    first = [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r2",
            item_id=2,
            gold_choice="cause",
            prediction="caused_by",
            probabilities=(0.1, 0.8, 0.1),
        ),
        _record(
            record_id="r3",
            item_id=3,
            gold_choice="none",
            prediction="none",
            status="missing_probability",
            probability_valid=False,
            probabilities=None,
            error="missing probability",
        ),
        _record(
            record_id="r4",
            item_id=4,
            gold_choice="caused_by",
            prediction="caused_by",
            probabilities=(0.2, 0.7, 0.1),
        ),
    ]
    second = [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.9, 0.05, 0.05),
        ),
        _record(
            record_id="r2",
            item_id=2,
            gold_choice="cause",
            prediction="caused_by",
            probabilities=(0.2, 0.7, 0.1),
        ),
        _record(
            record_id="r3",
            item_id=3,
            gold_choice="none",
            prediction="none",
            status="missing_probability",
            probability_valid=False,
            probabilities=None,
            error="missing probability",
        ),
        _record(
            record_id="r4",
            item_id=4,
            gold_choice="caused_by",
            prediction="caused_by",
            status="invalid_probability",
            probability_valid=False,
            probabilities=None,
            error="invalid probability",
        ),
    ]
    return first, second


def test_probability_metrics_match_hand_computed_common_set() -> None:
    first, second = _probability_fixtures()
    before = list(first)

    report = compute_probability_metrics(first, second)

    assert first == before
    assert json.dumps(report)
    first_report = _mapping(report["first"])
    second_report = _mapping(report["second"])
    _assert_counts(first_report, n=4, valid=3, fail=1)
    _assert_counts(second_report, n=4, valid=2, fail=2)
    assert _integer(report["common_N"]) == 2

    assert _number(first_report["brier"]) == pytest.approx(0.86)
    assert _number(first_report["nll"]) == pytest.approx(
        -(math.log(0.6) + math.log(0.1)) / 2,
    )
    assert _integer(first_report["nll_affected"]) == 0
    assert _number(first_report["ece"]) == pytest.approx(0.6)

    first_bins = _ece_bins(first_report)
    assert len(first_bins) == 10
    assert _number(first_bins[6]["lower"]) == pytest.approx(0.6)
    assert _number(first_bins[6]["upper"]) == pytest.approx(0.7)
    assert _integer(first_bins[6]["count"]) == 1
    assert _number(first_bins[6]["accuracy"]) == pytest.approx(1.0)
    assert _number(first_bins[6]["confidence"]) == pytest.approx(0.6)
    assert _number(first_bins[6]["contribution"]) == pytest.approx(0.2)
    assert _integer(first_bins[8]["count"]) == 1
    assert _number(first_bins[8]["accuracy"]) == pytest.approx(0.0)
    assert _number(first_bins[8]["confidence"]) == pytest.approx(0.8)
    assert _number(first_bins[8]["contribution"]) == pytest.approx(0.4)
    assert _integer(first_bins[0]["count"]) == 0
    assert first_bins[0]["accuracy"] is None
    assert first_bins[0]["confidence"] is None
    assert _number(first_bins[0]["contribution"]) == pytest.approx(0.0)
    assert _integer(first_bins[9]["count"]) == 0

    assert _number(second_report["brier"]) == pytest.approx(0.5775)
    assert _number(second_report["nll"]) == pytest.approx(
        -(math.log(0.9) + math.log(0.2)) / 2,
    )
    assert _number(second_report["ece"]) == pytest.approx(0.4)
    second_bins = _ece_bins(second_report)
    assert _integer(second_bins[9]["count"]) == 1
    assert _integer(second_bins[7]["count"]) == 1


def test_ece_last_bin_contains_confidence_one() -> None:
    record = _record(
        record_id="r1",
        item_id=1,
        gold_choice="cause",
        prediction="cause",
        probabilities=(1.0, 0.0, 0.0),
    )
    report = compute_probability_metrics([record], [record])
    first_report = _mapping(report["first"])
    bins = _ece_bins(first_report)

    assert len(bins) == 10
    assert all(_integer(bin_report["count"]) == 0 for bin_report in bins[:9])
    assert _integer(bins[9]["count"]) == 1
    assert _number(bins[9]["accuracy"]) == pytest.approx(1.0)
    assert _number(bins[9]["confidence"]) == pytest.approx(1.0)
    assert _number(first_report["ece"]) == pytest.approx(0.0)
    assert _number(first_report["brier"]) == pytest.approx(0.0)
    assert _number(first_report["nll"]) == pytest.approx(0.0)


def _zero_probability_fixture() -> list[PredictionRecord]:
    return [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="caused_by",
            probabilities=(0.0, 0.5, 0.5),
        ),
    ]


def test_zero_probability_error_policy_reports_nll_unavailable() -> None:
    records = _zero_probability_fixture()
    report = compute_probability_metrics(records, records)
    first_report = _mapping(report["first"])

    assert first_report["nll"] is None
    assert _integer(first_report["nll_affected"]) == 1
    assert _number(first_report["brier"]) == pytest.approx(1.5)
    assert _number(first_report["ece"]) == pytest.approx(0.5)


def test_zero_probability_clip_policy_requires_and_uses_explicit_epsilon() -> None:
    records = _zero_probability_fixture()
    report = compute_probability_metrics(
        records,
        records,
        zero_policy="clip",
        nll_epsilon=1e-3,
    )
    first_report = _mapping(report["first"])

    assert _integer(first_report["nll_affected"]) == 1
    assert _number(first_report["nll"]) == pytest.approx(-math.log(1e-3))
    assert _number(first_report["brier"]) == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("zero_policy", "nll_epsilon"),
    [
        ("clip", None),
        ("clip", 0.0),
        ("clip", -1e-12),
        ("unknown", None),
    ],
)
def test_probability_metrics_reject_unsafe_zero_policies(
    zero_policy: str,
    nll_epsilon: float | None,
) -> None:
    records = _zero_probability_fixture()

    with pytest.raises(ValueError):
        compute_probability_metrics(
            records,
            records,
            zero_policy=zero_policy,
            nll_epsilon=nll_epsilon,
        )


def test_probability_metrics_hard_fail_on_gold_mismatch() -> None:
    first, second = _probability_fixtures()
    mismatched = [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="none",
            prediction="cause",
            probabilities=(0.9, 0.05, 0.05),
        ),
        *second[1:],
    ]

    with pytest.raises(ValueError):
        compute_probability_metrics(first, mismatched)


def _binary_fixture() -> list[PredictionRecord]:
    return [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r2",
            item_id=2,
            gold_choice="caused_by",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r3",
            item_id=3,
            gold_choice="none",
            prediction=None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
            probabilities=None,
            error="bad response",
        ),
        _record(
            record_id="r4",
            item_id=4,
            gold_choice="cause",
            prediction="none",
            status="missing_probability",
            probability_valid=False,
            probabilities=None,
            error="missing probability",
        ),
        _record(
            record_id="r5",
            item_id=5,
            gold_choice="none",
            prediction="cause",
            probabilities=(0.7, 0.2, 0.1),
        ),
    ]


def test_binary_full_view_counts_invalid_predictions_as_wrong() -> None:
    records = _binary_fixture()
    report = compute_binary_metrics(records)

    assert json.dumps(report)
    _assert_counts(report, n=5, valid=4, fail=1)
    assert _number(report["accuracy"]) == pytest.approx(0.4)
    assert _number(report["precision"]) == pytest.approx(2 / 3)
    assert _number(report["recall"]) == pytest.approx(2 / 3)
    assert _number(report["f1"]) == pytest.approx(2 / 3)
    assert _number(report["positive_recall"]) == pytest.approx(2 / 3)
    assert _number(report["negative_recall"]) == pytest.approx(0.0)


def _paired_binary_fixtures() -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    first = [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r2",
            item_id=2,
            gold_choice="none",
            prediction="none",
            probabilities=(0.1, 0.1, 0.8),
        ),
        _record(
            record_id="r3",
            item_id=3,
            gold_choice="cause",
            prediction=None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
            probabilities=None,
            error="bad response",
        ),
        _record(
            record_id="r4",
            item_id=4,
            gold_choice="none",
            prediction="cause",
            probabilities=(0.7, 0.2, 0.1),
        ),
    ]
    second = [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r2",
            item_id=2,
            gold_choice="none",
            prediction="cause",
            probabilities=(0.7, 0.2, 0.1),
        ),
        _record(
            record_id="r3",
            item_id=3,
            gold_choice="cause",
            prediction=None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
            probabilities=None,
            error="bad response",
        ),
        _record(
            record_id="r4",
            item_id=4,
            gold_choice="none",
            prediction="cause",
            probabilities=(0.7, 0.2, 0.1),
        ),
    ]
    return first, second


def test_paired_binary_uses_common_both_valid_record_ids() -> None:
    first, second = _paired_binary_fixtures()
    report = compute_paired_binary_metrics(first, second)

    assert json.dumps(report)
    _assert_counts(report, n=3, valid=3, fail=0)
    first_report = _mapping(report["first"])
    second_report = _mapping(report["second"])
    assert _number(first_report["accuracy"]) == pytest.approx(2 / 3)
    assert _number(first_report["precision"]) == pytest.approx(0.5)
    assert _number(first_report["recall"]) == pytest.approx(1.0)
    assert _number(first_report["f1"]) == pytest.approx(2 / 3)
    assert _number(first_report["positive_recall"]) == pytest.approx(1.0)
    assert _number(first_report["negative_recall"]) == pytest.approx(0.5)

    assert _number(second_report["accuracy"]) == pytest.approx(1 / 3)
    assert _number(second_report["precision"]) == pytest.approx(1 / 3)
    assert _number(second_report["recall"]) == pytest.approx(1.0)
    assert _number(second_report["f1"]) == pytest.approx(0.5)
    assert _number(second_report["positive_recall"]) == pytest.approx(1.0)
    assert _number(second_report["negative_recall"]) == pytest.approx(0.0)


def _single_probability_metrics(
    records: Sequence[PredictionRecord],
    *,
    zero_policy: str = "error",
    nll_epsilon: float | None = None,
) -> Mapping[str, object]:
    from eci_decision_models.metrics import compute_probability_metrics_single

    return compute_probability_metrics_single(
        records,
        zero_policy=zero_policy,
        nll_epsilon=nll_epsilon,
    )


def _single_probability_fixture() -> list[PredictionRecord]:
    return [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.6, 0.3, 0.1),
        ),
        _record(
            record_id="r2",
            item_id=2,
            gold_choice="cause",
            prediction="caused_by",
            probabilities=(0.1, 0.8, 0.1),
        ),
        _record(
            record_id="r3",
            item_id=3,
            gold_choice="none",
            prediction="none",
            status="missing_probability",
            probability_valid=False,
            probabilities=None,
            error="missing probability",
        ),
        _record(
            record_id="r4",
            item_id=4,
            gold_choice="caused_by",
            prediction="caused_by",
            probabilities=(0.2, 0.7, 0.1),
        ),
    ]


def test_single_probability_metrics_match_hand_computed_fixture() -> None:
    records = _single_probability_fixture()
    before = list(records)

    report = _single_probability_metrics(records)

    assert records == before
    assert json.dumps(report)
    _assert_counts(report, n=4, valid=3, fail=1)
    assert _integer(report["common_N"]) == 3
    assert report["zero_policy"] == "error"
    assert report["nll_epsilon"] is None

    assert _number(report["brier"]) == pytest.approx(0.62)
    assert _number(report["nll"]) == pytest.approx(
        -(math.log(0.6) + math.log(0.1) + math.log(0.7)) / 3,
    )
    assert _integer(report["nll_affected"]) == 0
    assert _number(report["ece"]) == pytest.approx(0.5)

    bins = _ece_bins(report)
    assert len(bins) == 10
    assert _integer(bins[6]["count"]) == 1
    assert _number(bins[6]["accuracy"]) == pytest.approx(1.0)
    assert _number(bins[6]["confidence"]) == pytest.approx(0.6)
    assert _number(bins[6]["contribution"]) == pytest.approx(0.4 / 3)
    assert _integer(bins[7]["count"]) == 1
    assert _number(bins[7]["accuracy"]) == pytest.approx(1.0)
    assert _number(bins[7]["confidence"]) == pytest.approx(0.7)
    assert _number(bins[7]["contribution"]) == pytest.approx(0.3 / 3)
    assert _integer(bins[8]["count"]) == 1
    assert _number(bins[8]["accuracy"]) == pytest.approx(0.0)
    assert _number(bins[8]["confidence"]) == pytest.approx(0.8)
    assert _number(bins[8]["contribution"]) == pytest.approx(0.8 / 3)
    assert _integer(bins[0]["count"]) == 0
    assert bins[0]["accuracy"] is None
    assert bins[0]["confidence"] is None
    assert _number(bins[0]["contribution"]) == pytest.approx(0.0)


def test_single_probability_metrics_ignore_invalid_records_but_count_them() -> None:
    records = [
        *_single_probability_fixture(),
        _record(
            record_id="r5",
            item_id=5,
            gold_choice="cause",
            prediction=None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
            probabilities=None,
            error="invalid response",
        ),
    ]

    report = _single_probability_metrics(records)

    _assert_counts(report, n=5, valid=3, fail=2)
    assert _integer(report["common_N"]) == 3
    assert _number(report["brier"]) == pytest.approx(0.62)
    assert _number(report["nll"]) == pytest.approx(
        -(math.log(0.6) + math.log(0.1) + math.log(0.7)) / 3,
    )
    assert _number(report["ece"]) == pytest.approx(0.5)


def test_single_probability_metrics_zero_error_policy_reports_nll_unavailable() -> None:
    report = _single_probability_metrics(_zero_probability_fixture())

    _assert_counts(report, n=1, valid=1, fail=0)
    assert _integer(report["common_N"]) == 1
    assert _number(report["brier"]) == pytest.approx(1.5)
    assert report["nll"] is None
    assert _integer(report["nll_affected"]) == 1
    assert _number(report["ece"]) == pytest.approx(0.5)


def test_single_probability_metrics_zero_clip_policy_uses_explicit_epsilon() -> None:
    report = _single_probability_metrics(
        _zero_probability_fixture(),
        zero_policy="clip",
        nll_epsilon=1e-3,
    )

    assert _integer(report["nll_affected"]) == 1
    assert _number(report["nll"]) == pytest.approx(-math.log(1e-3))
    assert _number(report["brier"]) == pytest.approx(1.5)
    assert report["nll_epsilon"] == pytest.approx(1e-3)


def test_single_probability_metrics_clip_policy_still_requires_epsilon() -> None:
    with pytest.raises(ValueError):
        _single_probability_metrics(_zero_probability_fixture(), zero_policy="clip")


def test_single_probability_metrics_empty_valid_set() -> None:
    empty = _single_probability_metrics([])
    _assert_counts(empty, n=0, valid=0, fail=0)
    assert _integer(empty["common_N"]) == 0
    assert empty["brier"] is None
    assert empty["nll"] is None
    assert empty["ece"] is None
    empty_bins = _ece_bins(empty)
    assert len(empty_bins) == 10
    assert all(_integer(bin_report["count"]) == 0 for bin_report in empty_bins)

    invalid_only = _single_probability_metrics(
        [
            _record(
                record_id="r1",
                item_id=1,
                gold_choice="cause",
                prediction=None,
                status="invalid_response",
                prediction_valid=False,
                probability_valid=False,
                probabilities=None,
                error="invalid response",
            ),
        ],
    )
    _assert_counts(invalid_only, n=1, valid=0, fail=1)
    assert _integer(invalid_only["common_N"]) == 0
    assert invalid_only["brier"] is None
    assert invalid_only["nll"] is None
    assert invalid_only["ece"] is None
    assert len(_ece_bins(invalid_only)) == 10


def test_metrics_tolerate_valid_probability_vectors_on_a_tie() -> None:
    first = [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.5, 0.5, 0.0),
        ),
    ]
    second = [
        _record(
            record_id="r1",
            item_id=1,
            gold_choice="cause",
            prediction="cause",
            probabilities=(0.5, 0.5, 0.0),
        ),
    ]

    report = compute_probability_metrics(first, second)
    first_report = _mapping(report["first"])

    assert _number(first_report["brier"]) == pytest.approx(0.5)
    assert _number(first_report["nll"]) == pytest.approx(-math.log(0.5))
    assert _number(first_report["ece"]) == pytest.approx(0.5)
    assert _number(first_report["ece"]) >= 0.0
    assert abs(_number(first_report["ece"]) - 0.5) < TOLERANCE
