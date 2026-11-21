"""Classification, binary and probability metrics over prediction records.

Every report keeps failures visible.  The primary classification and binary
views count an invalid prediction as wrong, while a separate valid-only subset
is reported alongside them.  Paired reports align two runs on the
``record_id`` values for which both runs have a valid prediction (or valid
probabilities), and they reject duplicate IDs or disagreements about the gold
choice.

Probability metrics are computed only on the common set of records for which
both models returned a valid probability vector.  NLL never silently applies a
fixed epsilon: the caller chooses ``zero_policy="error"`` (NLL unavailable) or
``zero_policy="clip"`` with an explicit positive ``nll_epsilon``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from eci_decision_models.artifacts import PredictionRecord
from eci_decision_models.canonical import CHOICE_IDS

_CHOICE_INDEX: Final[dict[str, int]] = {
    choice_id: index for index, choice_id in enumerate(CHOICE_IDS)
}
_POSITIVE_CHOICES: Final[frozenset[str]] = frozenset({"cause", "caused_by"})
_ZERO_POLICIES: Final[frozenset[str]] = frozenset({"error", "clip"})
_NUM_BINS: Final = 10


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _coverage(valid: int, total: int) -> float:
    return valid / total if total else 0.0


def _counts(records: Sequence[PredictionRecord], *, valid: int) -> dict[str, object]:
    total = len(records)
    return {
        "N": total,
        "valid": valid,
        "fail": total - valid,
        "coverage": _coverage(valid, total),
    }


def _classification_core(records: Sequence[PredictionRecord]) -> dict[str, object]:
    total = len(records)
    valid = sum(1 for record in records if record.prediction_valid)
    columns = [*CHOICE_IDS, "invalid prediction"]
    matrix: dict[str, list[int]] = {choice_id: [0, 0, 0, 0] for choice_id in CHOICE_IDS}

    correct = 0
    for record in records:
        if record.prediction_valid:
            prediction = record.prediction
            if prediction is None:
                raise ValueError("prediction_valid record is missing a prediction")
            column = _CHOICE_INDEX[prediction]
            if prediction == record.gold_choice:
                correct += 1
        else:
            column = len(CHOICE_IDS)
        matrix[record.gold_choice][column] += 1

    per_class: dict[str, dict[str, object]] = {}
    f1_scores: list[float] = []
    for index, choice_id in enumerate(CHOICE_IDS):
        true_positive = matrix[choice_id][index]
        predicted = sum(matrix[row][index] for row in CHOICE_IDS)
        support = sum(matrix[choice_id])
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_f1(precision, recall)
        per_class[choice_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
        }
        f1_scores.append(f1)

    report = _counts(records, valid=valid)
    report.update(
        {
            "accuracy": correct / total if total else 0.0,
            "macro_f1": sum(f1_scores) / len(f1_scores),
            "per_class": per_class,
            "confusion_columns": columns,
            "confusion_matrix": matrix,
        },
    )
    return report


def _classification_report(records: Sequence[PredictionRecord]) -> dict[str, object]:
    report = _classification_core(records)
    valid_records = [record for record in records if record.prediction_valid]
    report["valid_subset"] = _classification_core(valid_records)
    return report


def compute_classification_metrics(
    records: Sequence[PredictionRecord],
) -> dict[str, object]:
    """Return the full failure-counted three-class classification report."""
    return _classification_report(records)


def _index_records(
    records: Sequence[PredictionRecord],
    *,
    run_label: str,
) -> dict[str, PredictionRecord]:
    indexed: dict[str, PredictionRecord] = {}
    for record in records:
        if record.record_id in indexed:
            raise ValueError(f"duplicate record_id {record.record_id!r} in {run_label}")
        indexed[record.record_id] = record
    return indexed


def _validate_paired_records(
    first: Sequence[PredictionRecord],
    second: Sequence[PredictionRecord],
) -> tuple[dict[str, PredictionRecord], dict[str, PredictionRecord]]:
    first_index = _index_records(first, run_label="first run")
    second_index = _index_records(second, run_label="second run")
    for record_id in sorted(first_index.keys() & second_index.keys()):
        if first_index[record_id].gold_choice != second_index[record_id].gold_choice:
            raise ValueError(f"gold_choice mismatch for shared record_id {record_id!r}")
    return first_index, second_index


def _paired_top_level(count: int) -> dict[str, object]:
    return {
        "N": count,
        "valid": count,
        "fail": 0,
        "coverage": 1.0 if count else 0.0,
    }


def compute_paired_classification_metrics(
    first: Sequence[PredictionRecord],
    second: Sequence[PredictionRecord],
) -> dict[str, object]:
    """Compare two runs on record IDs where both predictions are valid."""
    first_index, second_index = _validate_paired_records(first, second)
    common_ids = [
        record.record_id
        for record in first
        if record.record_id in second_index
        and record.prediction_valid
        and second_index[record.record_id].prediction_valid
    ]

    first_records = [first_index[record_id] for record_id in common_ids]
    second_records = [second_index[record_id] for record_id in common_ids]
    report = _paired_top_level(len(common_ids))
    report["first"] = _classification_core(first_records)
    report["second"] = _classification_core(second_records)
    return report


def _binary_label(choice: str) -> int:
    return 1 if choice in _POSITIVE_CHOICES else 0


def _binary_core(records: Sequence[PredictionRecord]) -> dict[str, object]:
    total = len(records)
    valid = sum(1 for record in records if record.prediction_valid)
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    invalid_positive = 0
    invalid_negative = 0

    for record in records:
        gold = _binary_label(record.gold_choice)
        if not record.prediction_valid:
            if gold:
                invalid_positive += 1
            else:
                invalid_negative += 1
            continue
        prediction = record.prediction
        if prediction is None:
            raise ValueError("prediction_valid record is missing a prediction")
        predicted = _binary_label(prediction)
        if predicted == gold:
            correct += 1
            if gold:
                true_positive += 1
            else:
                true_negative += 1
        elif predicted:
            false_positive += 1
        else:
            false_negative += 1

    positive_support = true_positive + false_negative + invalid_positive
    negative_support = true_negative + false_positive + invalid_negative
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, positive_support)
    f1 = _safe_f1(precision, recall)

    report = _counts(records, valid=valid)
    report.update(
        {
            "accuracy": correct / total if total else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positive_recall": recall,
            "negative_recall": _safe_ratio(true_negative, negative_support),
            "confusion_matrix": {
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
                "tn": true_negative,
                "invalid_positive": invalid_positive,
                "invalid_negative": invalid_negative,
            },
            "positive_support": positive_support,
            "negative_support": negative_support,
        },
    )
    return report


def compute_binary_metrics(records: Sequence[PredictionRecord]) -> dict[str, object]:
    """Return derived binary metrics with invalid predictions counted wrong."""
    return _binary_core(records)


def compute_paired_binary_metrics(
    first: Sequence[PredictionRecord],
    second: Sequence[PredictionRecord],
) -> dict[str, object]:
    """Compare two runs on record IDs where both predictions are valid."""
    first_index, second_index = _validate_paired_records(first, second)
    common_ids = [
        record.record_id
        for record in first
        if record.record_id in second_index
        and record.prediction_valid
        and second_index[record.record_id].prediction_valid
    ]

    first_records = [first_index[record_id] for record_id in common_ids]
    second_records = [second_index[record_id] for record_id in common_ids]
    report = _paired_top_level(len(common_ids))
    report["first"] = _binary_core(first_records)
    report["second"] = _binary_core(second_records)
    return report


def _resolve_zero_policy(
    zero_policy: str,
    nll_epsilon: float | None,
) -> float | None:
    if zero_policy not in _ZERO_POLICIES:
        raise ValueError(f"zero_policy must be one of {sorted(_ZERO_POLICIES)!r}")
    if zero_policy == "error":
        return None
    if isinstance(nll_epsilon, bool) or not isinstance(nll_epsilon, (int, float)):
        raise ValueError("zero_policy='clip' requires an explicit positive nll_epsilon")
    epsilon = float(nll_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1.0:
        raise ValueError("nll_epsilon must be finite and in the interval (0, 1]")
    return epsilon


def _bin_index(confidence: float) -> int:
    if confidence >= 1.0:
        return _NUM_BINS - 1
    index = int(confidence * _NUM_BINS)
    if index < 0:
        return 0
    if index >= _NUM_BINS:
        return _NUM_BINS - 1
    return index


def _ece_report(
    confidences: Sequence[float],
    correctness: Sequence[int],
) -> tuple[float | None, list[dict[str, object]]]:
    total = len(confidences)
    counts = [0] * _NUM_BINS
    correct_counts = [0] * _NUM_BINS
    confidence_sums = [0.0] * _NUM_BINS

    for confidence, correct in zip(confidences, correctness, strict=True):
        index = _bin_index(confidence)
        counts[index] += 1
        correct_counts[index] += correct
        confidence_sums[index] += confidence

    bins: list[dict[str, object]] = []
    contributions: list[float] = []
    for index in range(_NUM_BINS):
        lower = index / _NUM_BINS
        upper = (index + 1) / _NUM_BINS
        count = counts[index]
        if count == 0:
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "accuracy": None,
                    "confidence": None,
                    "contribution": 0.0,
                },
            )
            continue
        accuracy = correct_counts[index] / count
        confidence = confidence_sums[index] / count
        contribution = (count / total) * abs(accuracy - confidence)
        contributions.append(contribution)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
                "contribution": contribution,
            },
        )

    ece = sum(contributions) if total else None
    return ece, bins


def _probability_model_report(
    all_records: Sequence[PredictionRecord],
    common_records: Sequence[PredictionRecord],
    *,
    common_n: int,
    zero_policy: str,
    nll_epsilon: float | None,
) -> dict[str, object]:
    valid = sum(1 for record in all_records if record.probability_valid)

    brier_values: list[float] = []
    nll_values: list[float] = []
    nll_affected = 0
    confidences: list[float] = []
    correctness: list[int] = []

    for record in common_records:
        probabilities = record.probabilities
        prediction = record.prediction
        if probabilities is None or prediction is None:
            raise ValueError("probability-valid record is missing probabilities or prediction")
        gold_index = _CHOICE_INDEX[record.gold_choice]
        gold_probability = probabilities[gold_index]

        brier_values.append(
            sum(
                (value - (1.0 if index == gold_index else 0.0)) ** 2
                for index, value in enumerate(probabilities)
            ),
        )

        if zero_policy == "clip":
            epsilon = nll_epsilon
            if epsilon is None:
                raise ValueError("zero_policy='clip' requires nll_epsilon")
            if gold_probability < epsilon:
                nll_affected += 1
            nll_values.append(-math.log(max(gold_probability, epsilon)))
        elif gold_probability <= 0.0:
            nll_affected += 1
        else:
            nll_values.append(-math.log(gold_probability))

        confidence = probabilities[_CHOICE_INDEX[prediction]]
        confidences.append(confidence)
        correctness.append(1 if prediction == record.gold_choice else 0)

    brier = sum(brier_values) / len(brier_values) if brier_values else None
    if not common_records:
        nll: float | None = None
    elif zero_policy == "error" and nll_affected > 0:
        nll = None
    else:
        nll = sum(nll_values) / len(nll_values) if nll_values else None

    ece, bins = _ece_report(confidences, correctness)

    report = _counts(all_records, valid=valid)
    report.update(
        {
            "common_N": common_n,
            "brier": brier,
            "nll": nll,
            "nll_affected": nll_affected,
            "zero_policy": zero_policy,
            "nll_epsilon": nll_epsilon,
            "ece": ece,
            "ece_bins": bins,
        },
    )
    return report


def compute_probability_metrics_single(
    records: Sequence[PredictionRecord],
    *,
    zero_policy: str = "error",
    nll_epsilon: float | None = None,
) -> dict[str, object]:
    """Return Brier, NLL and top-label ECE for a single run.

    Only records with ``probability_valid=True`` contribute to the metrics;
    invalid records still count in ``N``/``fail`` and therefore reduce
    ``coverage``.  ``common_N`` mirrors ``valid`` to keep the per-model schema
    consistent with the paired report.
    """
    epsilon = _resolve_zero_policy(zero_policy, nll_epsilon)
    valid_records = [record for record in records if record.probability_valid]
    return _probability_model_report(
        records,
        valid_records,
        common_n=len(valid_records),
        zero_policy=zero_policy,
        nll_epsilon=epsilon,
    )


def compute_probability_metrics(
    first: Sequence[PredictionRecord],
    second: Sequence[PredictionRecord],
    *,
    zero_policy: str = "error",
    nll_epsilon: float | None = None,
) -> dict[str, object]:
    """Compare Brier, NLL and top-label ECE on the common probability set.

    ``zero_policy='error'`` leaves NLL unavailable (``None``) and reports the
    number of records with a zero gold probability, without suppressing Brier
    or ECE.  ``zero_policy='clip'`` requires an explicit positive
    ``nll_epsilon`` and reports how many records were clipped.
    """
    epsilon = _resolve_zero_policy(zero_policy, nll_epsilon)
    first_index, second_index = _validate_paired_records(first, second)
    common_ids = [
        record.record_id
        for record in first
        if record.record_id in second_index
        and record.probability_valid
        and second_index[record.record_id].probability_valid
    ]

    first_records = [first_index[record_id] for record_id in common_ids]
    second_records = [second_index[record_id] for record_id in common_ids]
    common_n = len(common_ids)

    report: dict[str, object] = {"common_N": common_n}
    report["first"] = _probability_model_report(
        first,
        first_records,
        common_n=common_n,
        zero_policy=zero_policy,
        nll_epsilon=epsilon,
    )
    report["second"] = _probability_model_report(
        second,
        second_records,
        common_n=common_n,
        zero_policy=zero_policy,
        nll_epsilon=epsilon,
    )
    return report
