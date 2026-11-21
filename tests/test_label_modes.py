"""Tests for the offline consensus/naive/fixed label comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrbalance_replay.analysis.label_modes import (
    compare_label_modes,
    format_comparison,
    main,
)
from mrbalance_replay.data import load_jsonl


def make_row(reply: str, gold: int, item_id: int = 0) -> dict[str, object]:
    return {
        "item_id": item_id,
        "words": ["rain", "flood"],
        "events": [[0], [1]],
        "bi_causal_label": gold,
        "type": "intra",
        "gpt_response": reply,
    }


def test_old_rows_fall_back_to_last_bracket_option() -> None:
    rows = [
        make_row("There is a causal link between them. (C)", gold=0),
        make_row("The storm caused the flood. (A)", gold=1),
    ]

    comparisons = {item.mode: item for item in compare_label_modes(rows)}

    assert comparisons["consensus"].report.overall.accuracy == 1.0
    assert comparisons["fixed"].report.overall.accuracy == 1.0
    assert comparisons["naive"].report.overall.accuracy == 0.5
    assert comparisons["consensus"].changed_vs_consensus == 0
    assert comparisons["naive"].changed_vs_consensus == 1
    assert comparisons["fixed"].changed_vs_consensus == 0
    assert all(comparison.fallback_rows == 2 for comparison in comparisons.values())


def test_inline_consensus_fields_take_precedence_over_prose() -> None:
    row = make_row("A careful reading suggests the answer is (C).", gold=0)
    row["consensus_answer"] = "C"
    row["label_consensus"] = 0

    comparisons = {item.mode: item for item in compare_label_modes([row])}

    # Consensus is exact even though the naive prose parser reads "A".
    assert comparisons["consensus"].report.overall.accuracy == 1.0
    assert comparisons["consensus"].fallback_rows == 0
    assert comparisons["naive"].report.overall.accuracy == 0.0
    assert comparisons["fixed"].report.overall.accuracy == 1.0


def test_consensus_answer_field_is_used_when_label_is_absent() -> None:
    row = make_row("noise (C)", gold=1)
    row["consensus_answer"] = "A"

    comparisons = {item.mode: item for item in compare_label_modes([row])}

    assert comparisons["consensus"].report.overall.precision == 1.0
    assert comparisons["consensus"].fallback_rows == 0


def test_format_comparison_lists_three_modes_and_fallbacks() -> None:
    rows = [make_row("No causal relationship. (C)", gold=0)]

    table = format_comparison(compare_label_modes(rows))

    assert "consensus" in table
    assert "naive" in table
    assert "fixed" in table
    assert "changed" in table.splitlines()[0]
    assert "fallback_rows: 1" in table


def test_cli_reads_jsonl_and_emits_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "response.jsonl"
    rows = [make_row("a thing happened. (C)", gold=0)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    exit_code = main([str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert [entry["mode"] for entry in payload] == ["consensus", "naive", "fixed"]
    assert payload[0]["fallback_rows"] == 1
    assert payload[1]["changed_vs_consensus"] == 1
    assert len(load_jsonl(path)) == 1
