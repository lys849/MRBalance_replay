"""ECI evaluation metrics compatible with the official MRBalance scorer.

The official ``ECI_compute_score.py`` reports, for the overall / inter / intra
splits, a ``P,R,F`` triple and a ``Pos:Neg:Full`` triple.  This module keeps
that layout while fixing the official script's three defects:

* gold labels are read from ``bi_causal_label`` (with a ``labels`` fallback);
* ``type`` is optional and defaults to ``"intra"``;
* every division is guarded, including empty inter/intra subsets.

Semantics note: ``pos_recall`` is the recall of the positive class and
``neg_recall`` is the recall of the negative class.  The official script
prints these two recalls under the names ``Pos`` and ``Neg``; they are *not*
per-class accuracies.  ``accuracy`` is the share of exactly matching labels
and is what the official script prints as ``Full``.

Run as a script::

    python -m mrbalance_replay.evaluation.metrics <response.jsonl>
"""

from __future__ import annotations

import json
import sys
import warnings
from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
BINARY_LABELS = (NEGATIVE_LABEL, POSITIVE_LABEL)
PREDICTION_KEYS = ("gpt_pred_bi_causal_label", "pred", "prediction")
GOLD_KEYS = ("bi_causal_label", "labels")

_MISSING = object()


@dataclass(frozen=True)
class Metrics:
    """Metrics for one split.

    Attributes:
        precision: Precision of the positive class.
        recall: Recall of the positive class.
        f1: Harmonic mean of ``precision`` and ``recall``.
        pos_recall: Recall of the positive class (official ``Pos``).
        neg_recall: Recall of the negative class (official ``Neg``).
        accuracy: Exact-match accuracy over all labels (official ``Full``).
        n_samples: Number of valid records in this split.
    """

    precision: float
    recall: float
    f1: float
    pos_recall: float
    neg_recall: float
    accuracy: float
    n_samples: int


@dataclass(frozen=True)
class MetricsReport:
    """Metrics for the overall, intra and inter splits."""

    overall: Metrics
    intra: Metrics
    inter: Metrics
    skipped: int = 0


_EMPTY_METRICS = Metrics(
    precision=0.0,
    recall=0.0,
    f1=0.0,
    pos_recall=0.0,
    neg_recall=0.0,
    accuracy=0.0,
    n_samples=0,
)


def _lookup(record: dict[str, Any], keys: tuple[str, ...]) -> object:
    """Return the first present value whose key exists, else a sentinel."""
    for key in keys:
        if key in record:
            return record[key]
    return _MISSING


def _is_binary_label(value: object) -> TypeGuard[int]:
    """Return whether ``value`` is an integer 0 or 1 (booleans excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value in BINARY_LABELS


def _coerce_gold(value: object) -> int | None:
    """Accept integer 0/1 plus the string forms ``"0"``/``"1"``."""
    if _is_binary_label(value):
        return value
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    return None


def _split_of(record: dict[str, Any]) -> str:
    """Return ``"inter"`` only for records explicitly marked as inter."""
    raw_type = record.get("type", "intra")
    return "inter" if str(raw_type).lower() == "inter" else "intra"


def _metrics_from_pairs(pairs: list[tuple[int, int]]) -> Metrics:
    """Compute metrics from ``(gold, prediction)`` pairs with zero guards."""
    n_samples = len(pairs)
    if n_samples == 0:
        return _EMPTY_METRICS

    true_positive = sum(
        1 for gold, pred in pairs if gold == POSITIVE_LABEL and pred == POSITIVE_LABEL
    )
    predicted_positive = sum(1 for _, pred in pairs if pred == POSITIVE_LABEL)
    gold_positive = sum(1 for gold, _ in pairs if gold == POSITIVE_LABEL)
    gold_negative = n_samples - gold_positive
    true_negative = sum(
        1 for gold, pred in pairs if gold == NEGATIVE_LABEL and pred == NEGATIVE_LABEL
    )

    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / gold_positive if gold_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    neg_recall = true_negative / gold_negative if gold_negative else 0.0
    accuracy = (true_positive + true_negative) / n_samples

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        pos_recall=recall,
        neg_recall=neg_recall,
        accuracy=accuracy,
        n_samples=n_samples,
    )


def compute_metrics(records: list[dict[str, Any]]) -> MetricsReport:
    """Compute overall/intra/inter metrics for ECI response records.

    Prediction values are read from ``gpt_pred_bi_causal_label``, falling back
    to ``pred`` and then ``prediction`` when the key is absent.  Gold labels
    are read from ``bi_causal_label``, falling back to ``labels``.  Records
    without an explicit ``type`` are treated as ``"intra"``.

    Gold labels may be integers 0/1 or the strings ``"0"``/``"1"``.  Records
    whose prediction or gold label is missing or otherwise invalid are excluded
    from every split and reported through ``MetricsReport.skipped``.  Invalid
    gold labels additionally emit a single ``RuntimeWarning`` per call so an
    entirely unlabelled file cannot silently score zero.
    """
    overall_pairs: list[tuple[int, int]] = []
    intra_pairs: list[tuple[int, int]] = []
    inter_pairs: list[tuple[int, int]] = []
    skipped = 0
    invalid_golds = 0

    for record in records:
        raw_prediction = _lookup(record, PREDICTION_KEYS)
        raw_gold = _lookup(record, GOLD_KEYS)
        gold = _coerce_gold(raw_gold)
        if gold is None:
            invalid_golds += 1
            skipped += 1
            continue
        if not _is_binary_label(raw_prediction):
            skipped += 1
            continue

        pair = (gold, raw_prediction)
        overall_pairs.append(pair)
        if _split_of(record) == "inter":
            inter_pairs.append(pair)
        else:
            intra_pairs.append(pair)

    if invalid_golds:
        warnings.warn(
            f"skipped {invalid_golds} record(s) with an invalid gold label; "
            'expected 0, 1, "0" or "1"',
            RuntimeWarning,
            stacklevel=2,
        )

    return MetricsReport(
        overall=_metrics_from_pairs(overall_pairs),
        intra=_metrics_from_pairs(intra_pairs),
        inter=_metrics_from_pairs(inter_pairs),
        skipped=skipped,
    )


def _format_metrics(metrics: Metrics) -> str:
    return (
        f"P,R,F= & {metrics.precision * 100:.1f} & {metrics.recall * 100:.1f}"
        f" & {metrics.f1 * 100:.1f}\n"
        f"Pos:Neg:Full= & {metrics.pos_recall * 100:.1f} & {metrics.neg_recall * 100:.1f}"
        f" & {metrics.accuracy * 100:.1f}"
    )


def format_report(report: MetricsReport) -> str:
    """Render a report in the official ``All / inter / intra`` layout."""
    sections = (
        ("All test", report.overall),
        ("inter test", report.inter),
        ("intra test", report.intra),
    )
    lines: list[str] = []
    for title, metrics in sections:
        lines.append(title)
        lines.append(_format_metrics(metrics))
    return "\n".join(lines)


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object per line")
            records.append(item)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: compute and print metrics for a JSONL response file."""
    parser = ArgumentParser(description="Compute ECI metrics for a JSONL response file.")
    parser.add_argument("response_jsonl", type=Path, help="path to response.jsonl")
    args = parser.parse_args(argv)

    report = compute_metrics(_load_records(args.response_jsonl))
    print(args.response_jsonl)
    print(format_report(report))
    print(f"Skipped: {report.skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
