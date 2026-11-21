"""Tests for the ECI evaluation metrics module."""

import json
from pathlib import Path
from typing import Any

import pytest

from mrbalance_replay.evaluation.metrics import (
    Metrics,
    MetricsReport,
    compute_metrics,
    format_report,
    main,
)

ZERO_METRICS = Metrics(
    precision=0.0,
    recall=0.0,
    f1=0.0,
    pos_recall=0.0,
    neg_recall=0.0,
    accuracy=0.0,
    n_samples=0,
)


def make_record(
    gold: object,
    prediction: object,
    split: object = "intra",
    *,
    gold_key: str = "bi_causal_label",
    prediction_key: str = "gpt_pred_bi_causal_label",
) -> dict[str, Any]:
    """Build a response record with configurable field names."""
    record: dict[str, Any] = {gold_key: gold, prediction_key: prediction}
    if split is not None:
        record["type"] = split
    return record


def test_all_correct() -> None:
    records = [
        make_record(1, 1, "intra"),
        make_record(0, 0, "intra"),
        make_record(1, 1, "inter"),
        make_record(0, 0, "inter"),
    ]

    report = compute_metrics(records)

    perfect_intra = Metrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2)
    assert report.overall == Metrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4)
    assert report.intra == perfect_intra
    assert report.inter == perfect_intra
    assert report.skipped == 0


def test_all_wrong() -> None:
    records = [
        make_record(1, 0, "intra"),
        make_record(1, 0, "intra"),
        make_record(0, 1, "inter"),
        make_record(0, 1, "inter"),
    ]

    report = compute_metrics(records)

    assert report.overall == Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4)
    assert report.intra == Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2)
    assert report.inter == Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2)


def test_all_predicted_positive_has_low_precision_high_recall() -> None:
    records = [
        make_record(1, 1),
        make_record(1, 1),
        make_record(0, 1),
        make_record(0, 1),
    ]

    report = compute_metrics(records)

    assert report.overall.precision == pytest.approx(0.5)
    assert report.overall.recall == pytest.approx(1.0)
    assert report.overall.f1 == pytest.approx(2 / 3)
    assert report.overall.pos_recall == pytest.approx(1.0)
    assert report.overall.neg_recall == pytest.approx(0.0)
    assert report.overall.accuracy == pytest.approx(0.5)
    assert report.overall.n_samples == 4


def test_all_predicted_negative_gives_zero_recall_without_crashing() -> None:
    records = [
        make_record(1, 0),
        make_record(1, 0),
        make_record(0, 0),
        make_record(0, 0),
    ]

    report = compute_metrics(records)

    assert report.overall.precision == 0.0
    assert report.overall.recall == 0.0
    assert report.overall.f1 == 0.0
    assert report.overall.pos_recall == 0.0
    assert report.overall.neg_recall == 1.0
    assert report.overall.accuracy == pytest.approx(0.5)
    assert report.overall.n_samples == 4


def test_empty_input_returns_zeroed_report() -> None:
    report = compute_metrics([])

    assert report == MetricsReport(overall=ZERO_METRICS, intra=ZERO_METRICS, inter=ZERO_METRICS)
    assert report.overall.n_samples == 0
    assert report.intra.n_samples == 0
    assert report.inter.n_samples == 0
    assert report.skipped == 0


def test_only_intra_records_leave_inter_zeroed() -> None:
    records = [
        make_record(1, 1),
        make_record(0, 1),
        make_record(0, 0),
    ]

    report = compute_metrics(records)

    assert report.intra.n_samples == 3
    assert report.intra.precision == pytest.approx(1 / 2)
    assert report.intra.recall == pytest.approx(1.0)
    assert report.intra.accuracy == pytest.approx(2 / 3)
    assert report.inter == ZERO_METRICS
    assert report.overall.n_samples == 3


def test_falls_back_to_alternative_field_names_and_default_intra_type() -> None:
    records = [
        make_record(1, 1, None, gold_key="labels", prediction_key="pred"),
        make_record(
            0,
            0,
            "inter",
            gold_key="labels",
            prediction_key="prediction",
        ),
    ]

    report = compute_metrics(records)

    assert report.skipped == 0
    assert report.overall.n_samples == 2
    assert report.intra == Metrics(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1)
    assert report.inter == Metrics(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1)


def test_invalid_predictions_and_missing_gold_are_counted_as_skipped() -> None:
    records: list[dict[str, Any]] = [
        make_record(1, 1),
        make_record(1, None),
        make_record(1, 2),
        make_record(1, "1"),
        {"bi_causal_label": 1},
        {"gpt_pred_bi_causal_label": 1},
    ]

    with pytest.warns(RuntimeWarning, match="invalid gold"):
        report = compute_metrics(records)

    assert report.skipped == 5
    assert report.overall.n_samples == 1
    assert report.overall.precision == 1.0
    assert report.intra == report.overall
    assert report.inter == ZERO_METRICS


def test_string_gold_labels_are_coerced() -> None:
    records = [make_record("1", 1), make_record("0", 0)]

    report = compute_metrics(records)

    assert report.skipped == 0
    assert report.overall.n_samples == 2
    assert report.overall.accuracy == 1.0


def test_invalid_gold_labels_warn_once() -> None:
    records = [
        make_record("NONE", 1),
        make_record(2, 1),
        make_record(None, 1),
        make_record(1, 1),
    ]

    with pytest.warns(RuntimeWarning, match="invalid gold") as caught:
        report = compute_metrics(records)

    assert len(caught) == 1
    assert report.skipped == 3
    assert report.overall.n_samples == 1


def test_mixed_inter_and_intra_grouping() -> None:
    records = [
        make_record(1, 1, "intra"),
        make_record(0, 0, "intra"),
        make_record(1, 1, "inter"),
        make_record(1, 1, "inter"),
        make_record(0, 1, "inter"),
        make_record(0, 1, "inter"),
    ]

    report = compute_metrics(records)

    assert report.intra == Metrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2)
    assert report.inter.precision == pytest.approx(0.5)
    assert report.inter.recall == pytest.approx(1.0)
    assert report.inter.f1 == pytest.approx(2 / 3)
    assert report.inter.neg_recall == pytest.approx(0.0)
    assert report.inter.accuracy == pytest.approx(0.5)
    assert report.inter.n_samples == 4

    assert report.overall.precision == pytest.approx(0.6)
    assert report.overall.recall == pytest.approx(1.0)
    assert report.overall.f1 == pytest.approx(0.75)
    assert report.overall.neg_recall == pytest.approx(1 / 3)
    assert report.overall.accuracy == pytest.approx(4 / 6)
    assert report.overall.n_samples == 6


def test_format_report_matches_official_layout() -> None:
    report = MetricsReport(
        overall=Metrics(0.5, 1.0, 2 / 3, 1.0, 0.0, 0.5, 4),
        intra=Metrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4),
        inter=Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
    )

    assert format_report(report) == (
        "All test\n"
        "P,R,F= & 50.0 & 100.0 & 66.7\n"
        "Pos:Neg:Full= & 100.0 & 0.0 & 50.0\n"
        "inter test\n"
        "P,R,F= & 0.0 & 0.0 & 0.0\n"
        "Pos:Neg:Full= & 0.0 & 0.0 & 0.0\n"
        "intra test\n"
        "P,R,F= & 100.0 & 100.0 & 100.0\n"
        "Pos:Neg:Full= & 100.0 & 100.0 & 100.0"
    )


def test_cli_prints_report_and_skipped_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    response_path = tmp_path / "response.jsonl"
    response_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                make_record(1, 1),
                make_record(0, 0),
                make_record(0, None),
            ]
        ),
        encoding="utf-8",
    )

    assert main([str(response_path)]) == 0

    captured = capsys.readouterr()
    assert "All test" in captured.out
    assert "P,R,F= & 100.0 & 100.0 & 100.0" in captured.out
    assert "Skipped: 1" in captured.out
