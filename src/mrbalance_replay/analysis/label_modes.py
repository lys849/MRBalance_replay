"""Compare consensus, naive-prose and fixed-prose label extraction.

The official debate pipeline stores the consensus *letter* (A/B/C) in
``gpt_response`` and tokenizes that letter, so ``consensus`` is the
official-equivalent label.  It is not a bug-fixed variant.

``naive`` applies the ECI.py baseline tokenizer (``re.split("[^a-z]", ...)``,
first ``a``/``b``/``c`` token) to the free-form reply retained for offline
analysis.  On prose, ordinary articles such as "a" are misread as option A.

``fixed`` applies the last-parenthesized-option heuristic to the same prose.

New runner outputs carry ``label_consensus``/``consensus_answer`` inline.  For
old outputs without those fields this tool falls back to the last bracketed
option in ``gpt_response`` and reports how many rows needed that fallback.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data import load_jsonl
from ..evaluation.metrics import MetricsReport, compute_metrics, format_report
from ..parsing import extract_final_label, parse_single_choice

MODES = ("consensus", "naive", "fixed")


@dataclass(frozen=True)
class ModeComparison:
    """Metrics for one label source plus disagreement vs ``consensus``."""

    mode: str
    report: MetricsReport
    changed_vs_consensus: int
    fallback_rows: int = 0


def _answer_to_label(answer: object) -> int | None:
    if answer in ("A", "B"):
        return 1
    if answer == "C":
        return 0
    return None


def _inline_binary(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    return None


def _consensus_labels(rows: Sequence[dict[str, Any]]) -> tuple[list[int | None], int]:
    """Extract consensus labels, counting rows that need the legacy fallback."""

    labels: list[int | None] = []
    fallback_rows = 0
    for row in rows:
        if "label_consensus" in row:
            labels.append(_inline_binary(row["label_consensus"]))
            continue
        if "consensus_answer" in row:
            labels.append(_answer_to_label(row["consensus_answer"]))
            continue

        fallback_rows += 1
        option = parse_single_choice(row.get("gpt_response"))
        labels.append(_answer_to_label(option))
    return labels, fallback_rows


def _prose_labels(rows: Sequence[dict[str, Any]], mode: str) -> list[int | None]:
    labels: list[int | None] = []
    for row in rows:
        response = row.get("gpt_response")
        if not isinstance(response, str) or not response:
            labels.append(None)
            continue
        labels.append(extract_final_label(response, mode))
    return labels


def _relabel(rows: Sequence[dict[str, Any]], labels: Sequence[int | None]) -> list[dict[str, Any]]:
    return [
        dict(row, gpt_pred_bi_causal_label=label) for row, label in zip(rows, labels, strict=True)
    ]


def compare_label_modes(rows: list[dict[str, Any]]) -> list[ModeComparison]:
    """Score every label source over ``rows`` and count disagreements."""

    consensus_labels, fallback_rows = _consensus_labels(rows)
    mode_labels = {
        "consensus": consensus_labels,
        "naive": _prose_labels(rows, "naive"),
        "fixed": _prose_labels(rows, "fixed"),
    }

    comparisons: list[ModeComparison] = []
    for mode in MODES:
        labels = mode_labels[mode]
        changed = sum(
            1
            for label, consensus in zip(labels, consensus_labels, strict=True)
            if label != consensus
        )
        comparisons.append(
            ModeComparison(
                mode=mode,
                report=compute_metrics(_relabel(rows, labels)),
                changed_vs_consensus=changed,
                fallback_rows=fallback_rows,
            )
        )
    return comparisons


def format_comparison(comparisons: Sequence[ModeComparison]) -> str:
    """Render a compact side-by-side table for reports."""

    header = f"{'mode':<10}{'P':>7}{'R':>7}{'F1':>7}{'Full':>7}{'n':>6}{'changed':>9}"
    lines = [header, "-" * len(header)]
    for comparison in comparisons:
        overall = comparison.report.overall
        lines.append(
            f"{comparison.mode:<10}"
            f"{overall.precision * 100:>7.1f}"
            f"{overall.recall * 100:>7.1f}"
            f"{overall.f1 * 100:>7.1f}"
            f"{overall.accuracy * 100:>7.1f}"
            f"{overall.n_samples:>6}"
            f"{comparison.changed_vs_consensus:>9}"
        )
    if comparisons:
        lines.append(f"fallback_rows: {comparisons[0].fallback_rows}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare consensus vs naive-prose vs fixed-prose labels"
    )
    parser.add_argument("path", type=Path, help="path to response.jsonl")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of the table",
    )
    args = parser.parse_args(argv)

    rows = load_jsonl(args.path)
    comparisons = compare_label_modes(rows)
    if args.json:
        payload = [
            {
                "mode": comparison.mode,
                "precision": comparison.report.overall.precision,
                "recall": comparison.report.overall.recall,
                "f1": comparison.report.overall.f1,
                "accuracy": comparison.report.overall.accuracy,
                "n_samples": comparison.report.overall.n_samples,
                "changed_vs_consensus": comparison.changed_vs_consensus,
                "fallback_rows": comparison.fallback_rows,
            }
            for comparison in comparisons
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(format_comparison(comparisons))
    print()
    for comparison in comparisons:
        print(f"=== {comparison.mode} ===")
        print(format_report(comparison.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
