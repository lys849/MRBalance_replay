"""Public contract tests for the ESC JSONL data loader."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from eci_decision_models.data import (
    EscDataset,
    EscInventory,
    EscRecord,
    load_esc,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PATH = FIXTURES / "esc_sample.jsonl"


def _record(item_id: int) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "words": ["A", "caused", "B"],
        "events": [[0], [2]],
        "tri_causal_label": 1,
        "bi_causal_label": 1,
        "doc_name": "synthetic.xml",
        "sent_in_doc_id": 0,
        "events_id": ["evt1", "evt2"],
        "topic": "synthetic",
        "ESC_rel": "PRECONDITION",
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _assert_line_context(exc: pytest.ExceptionInfo[ValueError], line: int) -> None:
    assert re.search(rf"\b{line}\b", str(exc.value)) is not None


def test_load_esc_preserves_file_order_and_provenance() -> None:
    dataset = load_esc(str(SAMPLE_PATH))

    assert isinstance(dataset, EscDataset)
    assert [record.item_id for record in dataset.records] == [101, 102, 103, 104]
    assert dataset.source_path.resolve() == SAMPLE_PATH.resolve()
    assert dataset.source_sha256 == hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest()


def test_sentence_and_events_are_rebuilt_from_words_without_reordering() -> None:
    records = load_esc(SAMPLE_PATH).records

    assert records[0].sentence == "Heavy rain caused the flood"
    assert records[0].event1 == "Heavy rain"
    assert records[0].event2 == "flood"

    assert records[1].sentence == "The storm destroyed the house"
    assert records[1].event1 == "storm"
    assert records[1].event2 == "destroyed the house"

    # UTF-8 content is preserved and tokens are joined with exactly one space.
    assert records[3].sentence == "Café opens at noon"


def test_loaded_records_keep_audit_fields_and_optional_nulls() -> None:
    records = load_esc(SAMPLE_PATH).records

    first = records[0]
    assert first.doc_name == "doc_a.xml"
    assert first.events_id == ("e1", "e2")
    assert first.esc_rel == "PRECONDITION"
    assert first.tri_causal_label == 1
    assert first.bi_causal_label == 1
    assert first.sent_in_doc_id == 0
    assert first.topic == "weather"

    last = records[3]
    assert last.sent_in_doc_id is None
    assert last.topic is None


def test_record_id_is_nonempty_stable_and_content_based() -> None:
    first_load = load_esc(SAMPLE_PATH)
    second_load = load_esc(SAMPLE_PATH)

    first_ids = [record.record_id for record in first_load.records]
    second_ids = [record.record_id for record in second_load.records]

    assert all(isinstance(record_id, str) and record_id for record_id in first_ids)
    assert first_ids == second_ids
    assert len(set(first_ids)) == len(first_ids)


@pytest.mark.parametrize("events", [[[0], [1]], [[4], [0, 1]]])
def test_record_id_includes_event_spans(tmp_path: Path, events: list[list[int]]) -> None:
    original = load_esc(SAMPLE_PATH).records[0]
    changed_record = json.loads(SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0])
    changed_record["events"] = events

    path = tmp_path / "changed_spans.jsonl"
    _write_jsonl(path, [changed_record])
    changed = load_esc(path).records[0]

    assert changed.sentence == original.sentence
    assert changed.record_id != original.record_id


def test_record_id_does_not_depend_on_item_id(tmp_path: Path) -> None:
    original = load_esc(SAMPLE_PATH).records[0]
    changed_record = json.loads(SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0])
    changed_record["item_id"] = 999

    path = tmp_path / "changed_item_id.jsonl"
    _write_jsonl(path, [changed_record])
    changed = load_esc(path).records[0]

    assert changed.item_id == 999
    assert changed.record_id == original.record_id


@pytest.mark.parametrize(
    ("field", "first_value", "second_value"),
    [
        ("doc_name", "doc_a.xml", "doc_b.xml"),
        ("events_id", ["evt1", "evt2"], ["evt3", "evt4"]),
    ],
)
def test_record_id_distinguishes_same_sentence_and_spans_by_document_provenance(
    tmp_path: Path, field: str, first_value: Any, second_value: Any
) -> None:
    first = _record(201)
    first[field] = first_value
    second = _record(202)
    second[field] = second_value

    path = tmp_path / f"distinct_{field}.jsonl"
    _write_jsonl(path, [first, second])
    records = load_esc(path).records

    assert records[0].sentence == records[1].sentence
    assert records[0].event1 == records[1].event1
    assert records[0].event2 == records[1].event2
    assert records[0].record_id != records[1].record_id


def test_record_id_ignores_item_id_for_identical_sentence_spans_and_provenance(
    tmp_path: Path,
) -> None:
    first = _record(301)
    second = _record(302)

    path = tmp_path / "same_content_different_item_id.jsonl"
    _write_jsonl(path, [first, second])
    records = load_esc(path).records

    assert records[0].item_id != records[1].item_id
    assert records[0].record_id == records[1].record_id


def test_inventory_is_recomputed_from_records() -> None:
    inventory = load_esc(SAMPLE_PATH).inventory

    assert isinstance(inventory, EscInventory)
    assert inventory.total == 4
    assert inventory.unique_item_ids == 4
    assert dict(inventory.tri_counts) == {0: 2, 1: 1, 2: 1}
    assert dict(inventory.bi_counts) == {0: 2, 1: 2}
    assert inventory.causal_rows == 2
    assert dict(inventory.esc_rel_counts) == {
        "PRECONDITION": 1,
        "FALLING_ACTION": 1,
        "UN_LABEL": 2,
    }


def test_public_dataclasses_are_frozen_and_records_are_a_tuple() -> None:
    dataset = load_esc(SAMPLE_PATH)

    assert isinstance(dataset.records, tuple)
    assert isinstance(dataset.records[0], EscRecord)
    assert isinstance(dataset.inventory.tri_counts, Mapping)
    assert isinstance(dataset.inventory.bi_counts, Mapping)
    assert isinstance(dataset.inventory.esc_rel_counts, Mapping)

    with pytest.raises(FrozenInstanceError):
        dataset.records[0].item_id = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        dataset.inventory.total = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        dataset.records = ()  # type: ignore[misc]


def test_duplicate_item_id_is_rejected(tmp_path: Path) -> None:
    duplicate = _record(101)
    duplicate["words"] = ["Different", "content", "here"]
    path = tmp_path / "duplicate_item_id.jsonl"
    _write_jsonl(path, [_record(101), duplicate])

    with pytest.raises(ValueError):
        load_esc(path)


def test_empty_event_span_is_rejected(tmp_path: Path) -> None:
    record = _record(101)
    record["events"] = [[], [2]]
    path = tmp_path / "empty_span.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(ValueError):
        load_esc(path)


def test_out_of_bounds_event_span_is_rejected(tmp_path: Path) -> None:
    record = _record(101)
    record["events"] = [[0], [3]]
    path = tmp_path / "out_of_bounds_span.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(ValueError):
        load_esc(path)


def test_malformed_jsonl_raises_value_error_with_line_context(tmp_path: Path) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text(
        json.dumps(_record(101), ensure_ascii=False) + "\n{not valid json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_esc(path)

    _assert_line_context(exc_info, 2)


def test_missing_required_field_raises_value_error_with_line_context(tmp_path: Path) -> None:
    missing_words = _record(102)
    del missing_words["words"]

    path = tmp_path / "missing_field.jsonl"
    _write_jsonl(path, [_record(101), missing_words])

    with pytest.raises(ValueError) as exc_info:
        load_esc(path)

    _assert_line_context(exc_info, 2)


def test_real_esc_dataset_inventory_when_env_is_set() -> None:
    raw_path = os.environ.get("ECI_ESC_DATA")
    if not raw_path:
        pytest.skip("ECI_ESC_DATA is not set")

    dataset = load_esc(Path(raw_path))

    assert len(dataset.records) == 1600
    assert dataset.inventory.total == 1600
    assert dataset.inventory.unique_item_ids == 1600
    assert dict(dataset.inventory.tri_counts) == {0: 1100, 1: 238, 2: 262}
    assert dataset.inventory.causal_rows == 500


def test_real_esc_record_ids_are_unique_when_env_is_set() -> None:
    raw_path = os.environ.get("ECI_ESC_DATA")
    if not raw_path:
        pytest.skip("ECI_ESC_DATA is not set")

    dataset = load_esc(Path(raw_path))
    record_ids = [record.record_id for record in dataset.records]

    assert len(record_ids) == 1600
    assert len(set(record_ids)) == 1600
