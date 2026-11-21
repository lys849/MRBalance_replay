"""Single-model evaluation report assembly and rendering.

This module combines the existing classification, binary and probability
metrics into one frozen report shape and renders it as deterministic Markdown.
It is intentionally focused on a single Laya run: there is no Jev/Laya
comparison here.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from eci_decision_models.artifacts import PredictionRecord, atomic_write_text
from eci_decision_models.canonical import CHOICE_IDS
from eci_decision_models.metrics import (
    compute_binary_metrics,
    compute_classification_metrics,
    compute_probability_metrics_single,
)

_DEFINITION_VERSION = "eci-metrics-v1"
_CONFUSION_COLUMNS = [*CHOICE_IDS, "invalid prediction"]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast("Mapping[str, object]", value)


def _matrix_row(matrix: Mapping[str, object], row: str) -> list[int]:
    value = matrix[row]
    if not isinstance(value, list):
        raise ValueError(f"confusion matrix row {row!r} must be a list")
    return cast("list[int]", value)


def _status_counts(records: Sequence[PredictionRecord]) -> dict[str, int]:
    counts = Counter(record.status for record in records)
    return {status: counts[status] for status in sorted(counts)}


def _control_status_counts(records: Sequence[PredictionRecord]) -> dict[str, int]:
    counts = Counter(
        record.control_status for record in records if record.control_status is not None
    )
    return {label: counts[label] for label in sorted(counts)}


def build_single_model_metrics(
    records: Sequence[PredictionRecord],
    *,
    zero_policy: str = "error",
    nll_epsilon: float | None = None,
) -> dict[str, object]:
    """Build the frozen single-model metrics report shape."""
    classification = compute_classification_metrics(records)
    matrix = _mapping(classification["confusion_matrix"], "confusion_matrix")
    cause = _matrix_row(matrix, "cause")
    caused_by = _matrix_row(matrix, "caused_by")
    none = _matrix_row(matrix, "none")

    return {
        "definition_version": _DEFINITION_VERSION,
        "classification": classification,
        "binary": compute_binary_metrics(records),
        "probability": compute_probability_metrics_single(
            records,
            zero_policy=zero_policy,
            nll_epsilon=nll_epsilon,
        ),
        "direction_errors": {
            "gold_cause_pred_caused_by": cause[1],
            "gold_caused_by_pred_cause": caused_by[0],
            "gold_causal_pred_none": cause[2] + caused_by[2],
            "gold_none_pred_causal": none[0] + none[1],
            "invalid_causal": cause[3] + caused_by[3],
            "invalid_none": none[3],
        },
        "status_counts": _status_counts(records),
        "control_status_counts": _control_status_counts(records),
    }


def _lookup(meta: Mapping[str, object], *keys: str) -> object:
    current: object = meta
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = cast("Mapping[str, object]", current).get(key)
        if current is None:
            return None
    return current


def _format(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(cell) for cell in row) + " |")
    return "\n".join(lines)


def _identity_lines(meta: Mapping[str, object]) -> list[str]:
    return [
        f"- run_id: {_format(_lookup(meta, 'run_id'))}",
        f"- model: {_format(_lookup(meta, 'model'))}",
        f"- data path: {_format(_lookup(meta, 'data', 'path'))}",
        f"- data sha256: {_format(_lookup(meta, 'data', 'sha256'))}",
        f"- gold path: {_format(_lookup(meta, 'gold', 'path'))}",
        f"- gold sha256: {_format(_lookup(meta, 'gold', 'sha256'))}",
        f"- gold version: {_format(_lookup(meta, 'gold', 'version'))}",
    ]


def _render_caveats() -> list[str]:
    bullets = [
        "Laya-only evaluation: only the Laya model was run; Jev was not evaluated.",
        "Zero-shot evaluation: no task-specific fine-tuning was applied.",
        (
            "The canonical prompt and option order were fixed for every item; "
            "no prompt or ordering variants were sampled."
        ),
        "No extra baselines were run; results describe the single model alone.",
        (
            "The `none` label means ESC was left unlabeled, not that the absence "
            "of a causal relation was proven."
        ),
        "The Phase 2 gate remains open; this pilot does not close it.",
        "These results are not a Jev/Laya comparison.",
    ]
    return [f"- {bullet}" for bullet in bullets]


def _render_classification(classification: Mapping[str, object]) -> list[str]:
    rows: list[Sequence[object]] = [
        ("N", classification.get("N")),
        ("valid", classification.get("valid")),
        ("fail", classification.get("fail")),
        ("coverage", classification.get("coverage")),
        ("accuracy", classification.get("accuracy")),
        ("macro-F1", classification.get("macro_f1")),
    ]
    lines = [_table(("Metric", "Value"), rows), "", "### Valid subset", ""]
    valid_subset = _mapping(classification.get("valid_subset"), "valid_subset")
    subset_rows: list[Sequence[object]] = [
        ("N", valid_subset.get("N")),
        ("accuracy", valid_subset.get("accuracy")),
        ("macro-F1", valid_subset.get("macro_f1")),
    ]
    lines.append(_table(("Metric", "Value"), subset_rows))
    return lines


def _render_confusion(classification: Mapping[str, object]) -> list[str]:
    matrix = _mapping(classification["confusion_matrix"], "confusion_matrix")
    rows: list[Sequence[object]] = [(row, *_matrix_row(matrix, row)) for row in CHOICE_IDS]
    return [_table(("Gold \\ Predicted", *_CONFUSION_COLUMNS), rows)]


def _render_per_class(classification: Mapping[str, object]) -> list[str]:
    per_class = _mapping(classification["per_class"], "per_class")
    rows: list[Sequence[object]] = []
    for choice_id in CHOICE_IDS:
        values = _mapping(per_class[choice_id], f"per_class[{choice_id!r}]")
        rows.append(
            (
                choice_id,
                values.get("precision"),
                values.get("recall"),
                values.get("f1"),
                values.get("support"),
                values.get("predicted"),
            ),
        )
    return [_table(("Class", "Precision", "Recall", "F1", "Support", "Predicted"), rows)]


def _render_binary(binary: Mapping[str, object]) -> list[str]:
    rows: list[Sequence[object]] = [
        ("N", binary.get("N")),
        ("valid", binary.get("valid")),
        ("fail", binary.get("fail")),
        ("coverage", binary.get("coverage")),
        ("accuracy", binary.get("accuracy")),
        ("precision", binary.get("precision")),
        ("recall", binary.get("recall")),
        ("F1", binary.get("f1")),
        ("positive recall", binary.get("positive_recall")),
        ("negative recall", binary.get("negative_recall")),
    ]
    return [_table(("Metric", "Value"), rows)]


def _render_probability(probability: Mapping[str, object]) -> list[str]:
    summary: list[Sequence[object]] = [
        ("Brier", probability.get("brier")),
        ("NLL", probability.get("nll")),
        ("zero policy", probability.get("zero_policy")),
        ("NLL epsilon", probability.get("nll_epsilon")),
        ("ECE", probability.get("ece")),
        ("valid coverage", probability.get("coverage")),
        ("valid", probability.get("valid")),
        ("N", probability.get("N")),
    ]
    lines = [_table(("Metric", "Value"), summary), ""]
    bins = probability.get("ece_bins")
    bin_rows: list[Sequence[object]] = []
    if isinstance(bins, Sequence):
        for item in bins:
            bin_report = _mapping(item, "ece_bins entry")
            bin_rows.append(
                (
                    bin_report.get("lower"),
                    bin_report.get("upper"),
                    bin_report.get("count"),
                    bin_report.get("accuracy"),
                    bin_report.get("confidence"),
                    bin_report.get("contribution"),
                ),
            )
    lines.append(
        _table(
            ("Lower", "Upper", "Count", "Accuracy", "Confidence", "Contribution"),
            bin_rows,
        ),
    )
    return lines


def _render_direction(direction: Mapping[str, object]) -> list[str]:
    rows: list[Sequence[object]] = [
        ("gold cause, predicted caused_by", direction.get("gold_cause_pred_caused_by")),
        ("gold caused_by, predicted cause", direction.get("gold_caused_by_pred_cause")),
        ("gold causal, predicted none", direction.get("gold_causal_pred_none")),
        ("gold none, predicted causal", direction.get("gold_none_pred_causal")),
        ("invalid prediction on causal gold", direction.get("invalid_causal")),
        ("invalid prediction on none gold", direction.get("invalid_none")),
    ]
    return [_table(("Direction error", "Count"), rows)]


def _render_counts_heading(title: str, counts: Mapping[str, object]) -> list[str]:
    rows: list[Sequence[object]] = [(label, counts[label]) for label in sorted(counts)]
    if not rows:
        rows = [("N/A", "N/A")]
    return [f"### {title}", _table(("Label", "Count"), rows)]


def render_metrics_markdown(
    metrics: Mapping[str, object],
    meta: Mapping[str, object],
) -> str:
    """Render the frozen report as deterministic Markdown."""
    classification = _mapping(metrics["classification"], "classification")
    binary = _mapping(metrics["binary"], "binary")
    probability = _mapping(metrics["probability"], "probability")
    direction = _mapping(metrics["direction_errors"], "direction_errors")
    status_counts = _mapping(metrics["status_counts"], "status_counts")
    control_counts = _mapping(metrics["control_status_counts"], "control_status_counts")

    lines: list[str] = ["# ECI Evaluation Report", ""]
    lines.extend(_identity_lines(meta))
    lines.extend(["", "## Caveats", ""])
    lines.extend(_render_caveats())
    lines.extend(["", "## Classification", ""])
    lines.extend(_render_classification(classification))
    lines.extend(["", "## Confusion Matrix", ""])
    lines.extend(_render_confusion(classification))
    lines.extend(["", "## Per-class", ""])
    lines.extend(_render_per_class(classification))
    lines.extend(["", "## Derived Binary", ""])
    lines.extend(_render_binary(binary))
    lines.extend(["", "## Probability", ""])
    lines.extend(_render_probability(probability))
    lines.extend(["", "## Direction Errors", ""])
    lines.extend(_render_direction(direction))
    lines.extend(["", "## Status", ""])
    lines.extend(_render_counts_heading("Status counts", status_counts))
    lines.append("")
    lines.extend(_render_counts_heading("Control status counts", control_counts))
    lines.append("")
    return "\n".join(lines)


def write_metrics_files(
    run_dir: Path,
    metrics: Mapping[str, object],
    meta: Mapping[str, object],
) -> tuple[Path, Path]:
    """Write ``metrics.json`` and ``metrics.md`` into ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = run_dir / "metrics.json"
    metrics_md_path = run_dir / "metrics.md"
    atomic_write_text(
        metrics_json_path,
        json.dumps(metrics, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(
        metrics_md_path,
        render_metrics_markdown(metrics, meta),
    )
    return metrics_json_path, metrics_md_path
