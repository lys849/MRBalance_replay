"""Public contracts for the single-model ESC evaluation report."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from eci_decision_models.artifacts import PredictionRecord


def _record(
    record_id: str,
    item_id: int,
    gold_choice: str,
    prediction: str | None,
    probabilities: tuple[float, float, float] | None = (0.6, 0.3, 0.1),
    *,
    status: str = "ok",
    prediction_valid: bool = True,
    probability_valid: bool = True,
) -> PredictionRecord:
    return PredictionRecord(
        run_id="run-1",
        item_id=item_id,
        record_id=record_id,
        model="laya-test",
        attempt=1,
        raw_request_id=f"request-{record_id}",
        request_hash="request-hash",
        response_hash="response-hash",
        adapter_version="laya-adapter-v1",
        parser_version="eci-parser-v1",
        gold_choice=gold_choice,
        prediction=prediction,
        status=status,
        prediction_valid=prediction_valid,
        probability_valid=probability_valid,
        probabilities=probabilities,
        error=None if status == "ok" else status,
    )


def _report_module() -> Any:
    from eci_decision_models import report

    return report


def _fixture_records() -> list[PredictionRecord]:
    return [
        _record("r1", 1, "cause", "cause", (0.6, 0.3, 0.1)),
        _record("r2", 2, "cause", "caused_by", (0.1, 0.8, 0.1)),
        _record("r3", 3, "caused_by", "cause", (0.6, 0.3, 0.1)),
        _record("r4", 4, "caused_by", "none", (0.1, 0.2, 0.7)),
        _record("r5", 5, "none", "cause", (0.7, 0.2, 0.1)),
        _record("r6", 6, "none", "caused_by", (0.1, 0.8, 0.1)),
        _record(
            "r7",
            7,
            "cause",
            None,
            None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
        ),
        _record(
            "r8",
            8,
            "caused_by",
            None,
            None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
        ),
        _record(
            "r9",
            9,
            "none",
            None,
            None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
        ),
        _record(
            "r10",
            10,
            "cause",
            None,
            None,
            status="api_error",
            prediction_valid=False,
            probability_valid=False,
        ),
    ]


def _meta() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "model": "laya-test",
        "data": {"path": "esc.jsonl", "sha256": "data-sha"},
        "gold": {"path": "gold.jsonl", "sha256": "gold-sha", "version": "gold-v1"},
    }


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return cast("Mapping[str, object]", value)


def test_build_single_model_metrics_has_frozen_shape_and_direction_counts() -> None:
    report = _report_module()
    records = _fixture_records()

    metrics = report.build_single_model_metrics(records)

    assert set(metrics) == {
        "definition_version",
        "classification",
        "binary",
        "probability",
        "direction_errors",
        "status_counts",
        "control_status_counts",
    }
    assert metrics["definition_version"] == "eci-metrics-v1"

    classification = _mapping(metrics["classification"])
    assert classification["N"] == 10
    assert classification["valid"] == 6
    assert classification["fail"] == 4
    assert classification["coverage"] == 0.6
    assert classification["accuracy"] == 0.1
    matrix = _mapping(classification["confusion_matrix"])
    assert matrix["cause"] == [1, 1, 0, 2]
    assert matrix["caused_by"] == [1, 0, 1, 1]
    assert matrix["none"] == [1, 1, 0, 1]

    binary = _mapping(metrics["binary"])
    assert binary["N"] == 10
    assert binary["valid"] == 6
    assert binary["fail"] == 4

    probability = _mapping(metrics["probability"])
    assert probability["N"] == 10
    assert probability["valid"] == 6
    assert probability["fail"] == 4
    assert probability["coverage"] == 0.6
    assert probability["common_N"] == 6
    assert len(cast("list[object]", probability["ece_bins"])) == 10

    assert metrics["direction_errors"] == {
        "gold_cause_pred_caused_by": 1,
        "gold_caused_by_pred_cause": 1,
        "gold_causal_pred_none": 1,
        "gold_none_pred_causal": 2,
        "invalid_causal": 3,
        "invalid_none": 1,
    }
    assert metrics["status_counts"] == {
        "ok": 6,
        "invalid_response": 3,
        "api_error": 1,
    }
    assert metrics["control_status_counts"] == {}


def test_build_single_model_metrics_reports_only_non_none_control_statuses() -> None:
    report = _report_module()
    records = _fixture_records()
    records[-1] = replace(records[-1], control_status="budget_exhausted")

    metrics = report.build_single_model_metrics(records)

    assert metrics["control_status_counts"] == {"budget_exhausted": 1}


def test_render_metrics_markdown_is_deterministic_and_has_required_sections() -> None:
    report = _report_module()
    metrics = report.build_single_model_metrics(_fixture_records())
    meta = _meta()

    first = report.render_metrics_markdown(metrics, meta)
    second = report.render_metrics_markdown(metrics, meta)

    assert first == second
    for heading in (
        "# ECI Evaluation Report",
        "## Caveats",
        "## Classification",
        "## Confusion Matrix",
        "## Per-class",
        "## Derived Binary",
        "## Probability",
        "## Direction Errors",
        "## Status",
    ):
        assert heading in first
    for identity in ("run-1", "laya-test", "esc.jsonl", "gold-v1"):
        assert identity in first
    assert "invalid prediction" in first
    assert "0.1000" in first

    caveats = first.split("## Caveats", 1)[1].split("## Classification", 1)[0].lower()
    for needle in (
        "laya-only",
        "jev",
        "zero-shot",
        "canonical",
        "baseline",
        "unlabeled",
        "absence",
        "phase 2",
        "comparison",
    ):
        assert needle in caveats


def test_render_metrics_markdown_marks_missing_identity_as_not_available() -> None:
    report = _report_module()
    metrics = report.build_single_model_metrics([])

    text = report.render_metrics_markdown(metrics, {})

    assert "# ECI Evaluation Report" in text
    assert "N/A" in text
    assert "## Probability" in text


def test_render_metrics_markdown_includes_valid_subset_and_binary_counts() -> None:
    report = _report_module()
    metrics = report.build_single_model_metrics(_fixture_records())
    text = report.render_metrics_markdown(metrics, _meta())

    classification = _mapping(metrics["classification"])
    valid_subset = _mapping(classification["valid_subset"])
    classification_section = text.split("## Classification", 1)[1].split(
        "## Confusion Matrix",
        1,
    )[0]
    assert "### Valid subset" in classification_section
    assert f"| N | {valid_subset['N']} |" in classification_section
    assert f"| accuracy | {valid_subset['accuracy']:.4f} |" in classification_section
    assert f"| macro-F1 | {valid_subset['macro_f1']:.4f} |" in classification_section

    binary = _mapping(metrics["binary"])
    binary_section = text.split("## Derived Binary", 1)[1].split("## Probability", 1)[0]
    for key in ("N", "valid", "fail"):
        assert f"| {key} | {binary[key]} |" in binary_section
    coverage = binary["coverage"]
    assert isinstance(coverage, float)
    assert f"| coverage | {coverage:.4f} |" in binary_section


def test_write_metrics_files_is_deterministic_json_and_markdown(tmp_path: Path) -> None:
    report = _report_module()
    metrics = report.build_single_model_metrics(_fixture_records())
    meta = _meta()

    json_path, markdown_path = report.write_metrics_files(tmp_path, metrics, meta)

    assert json_path == tmp_path / "metrics.json"
    assert markdown_path == tmp_path / "metrics.md"
    expected_json = json.dumps(metrics, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert json_path.read_text(encoding="utf-8") == expected_json
    assert markdown_path.read_text(encoding="utf-8") == report.render_metrics_markdown(
        metrics,
        meta,
    )
    assert sorted(item.name for item in tmp_path.iterdir()) == ["metrics.json", "metrics.md"]


def test_build_single_model_metrics_empty_records() -> None:
    report = _report_module()

    metrics = report.build_single_model_metrics([])

    classification = _mapping(metrics["classification"])
    assert classification["N"] == 0
    assert classification["valid"] == 0
    assert classification["fail"] == 0
    assert classification["accuracy"] == 0.0
    assert classification["macro_f1"] == 0.0

    probability = _mapping(metrics["probability"])
    assert probability["N"] == 0
    assert probability["valid"] == 0
    assert probability["fail"] == 0
    assert probability["coverage"] == 0.0
    assert probability["common_N"] == 0
    assert probability["brier"] is None
    assert probability["nll"] is None
    assert probability["ece"] is None
    bins = cast("list[Mapping[str, object]]", probability["ece_bins"])
    assert len(bins) == 10
    assert all(bin_report["count"] == 0 for bin_report in bins)

    assert set(cast("Mapping[str, object]", metrics["direction_errors"]).values()) == {0}
    assert metrics["status_counts"] == {}
    assert metrics["control_status_counts"] == {}

    text = report.render_metrics_markdown(metrics, _meta())
    assert "# ECI Evaluation Report" in text
    assert "N/A" in text
