"""Tests for JSONL loading, prompt construction and gold labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mrbalance_replay.data import build_question, gold_label, load_jsonl

# First record of reference/MRBalance/data/test/ESC_intra_data.json.
ESC_FIRST_ITEM: dict[str, Any] = {
    "item_id": 4713,
    "words": [
        "Breaking",
        "News",
        "-",
        "Three",
        "People",
        "Shot",
        "in",
        "Office",
        "Shooting",
        "in",
        "Suburban",
        "Detroit",
    ],
    "events": [[8], [5]],
    "tri_causal_label": 2,
    "bi_causal_label": 1,
    "doc_name": "data/EventStoryLine/annotated_data/v0.9/18/18_4ecbplus.xml.xml",
    "sent_in_doc_id": 1,
    "events_id": ["2", "1"],
    "topic": "18",
    "ESC_rel": "FALLING_ACTION",
}

EXPECTED_ESC_QUESTION = (
    "\nInput: Breaking News - Three People Shot in Office Shooting in Suburban Detroit\n"
    'Question: is there a causal relationship between "Shooting" and "Shot" ?\n'
    "(A) cause\n"
    "(B) caused by\n"
    "(C) none"
)


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text(
        '{"item_id": 1}\n\n   \n{"item_id": 2}\n',
        encoding="utf-8",
    )

    assert load_jsonl(path) == [{"item_id": 1}, {"item_id": 2}]


def test_load_jsonl_returns_empty_list_for_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    assert load_jsonl(path) == []


def test_build_question_matches_reference_record_snapshot() -> None:
    assert build_question(ESC_FIRST_ITEM) == EXPECTED_ESC_QUESTION


def test_build_question_joins_multi_token_events() -> None:
    item = {
        "words": ["The", "heavy", "rain", "caused", "the", "flood"],
        "events": [[1, 2], [4, 5]],
    }

    assert 'between "heavy rain" and "the flood" ?' in build_question(item)


def test_gold_label_prefers_bi_causal_label() -> None:
    assert gold_label({"bi_causal_label": 1}) == 1
    assert gold_label({"bi_causal_label": 0, "labels": 1}) == 0


def test_gold_label_falls_back_to_labels() -> None:
    assert gold_label({"labels": 1}) == 1
    assert gold_label({"bi_causal_label": None, "labels": 0}) == 0


def test_gold_label_missing_fields_raises() -> None:
    with pytest.raises(KeyError, match="bi_causal_label"):
        gold_label({"item_id": 1})


def test_load_jsonl_round_trips_reference_record(tmp_path: Path) -> None:
    path = tmp_path / "esc.jsonl"
    path.write_text(json.dumps(ESC_FIRST_ITEM, ensure_ascii=False) + "\n", encoding="utf-8")

    items = load_jsonl(path)

    assert items[0]["item_id"] == 4713
    assert gold_label(items[0]) == 1
