"""Load ESC JSONL data and rebuild normalized records plus inventory.

The loader implements the shared-input contract in ``docs/application_plan.md``
sections 3.1 and 3.2. It reads the file without modifying it, preserves file
order, rebuilds event text from the original token spans without reordering,
and derives a content fingerprint that is independent of ``item_id``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_NAMESPACE = "esc"

_REQUIRED_FIELDS = (
    "item_id",
    "words",
    "events",
    "tri_causal_label",
    "bi_causal_label",
    "doc_name",
    "sent_in_doc_id",
    "events_id",
    "topic",
    "ESC_rel",
)


@dataclass(frozen=True)
class EscRecord:
    """One normalized ESC example with provenance for later auditing."""

    item_id: int
    record_id: str
    sentence: str
    event1: str
    event2: str
    doc_name: str
    sent_in_doc_id: int | None
    events_id: tuple[str, ...]
    topic: str | None
    esc_rel: str
    tri_causal_label: int
    bi_causal_label: int


@dataclass(frozen=True)
class EscInventory:
    """Recomputed counts for the loaded ESC dataset."""

    total: int
    unique_item_ids: int
    tri_counts: Mapping[int, int]
    bi_counts: Mapping[int, int]
    causal_rows: int
    esc_rel_counts: Mapping[str, int]


@dataclass(frozen=True)
class EscDataset:
    """An immutable ESC dataset with source path and content hash."""

    source_path: Path
    source_sha256: str
    records: tuple[EscRecord, ...]
    inventory: EscInventory


def _error(line_number: int, message: str) -> ValueError:
    return ValueError(f"line {line_number}: {message}")


def _field(record: Mapping[str, Any], name: str, line_number: int) -> Any:
    if name not in record:
        raise _error(line_number, f"missing required field {name!r}")
    return record[name]


def _int_field(record: Mapping[str, Any], name: str, line_number: int) -> int:
    value = _field(record, name, line_number)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(line_number, f"field {name!r} must be an integer")
    return value


def _str_field(record: Mapping[str, Any], name: str, line_number: int) -> str:
    value = _field(record, name, line_number)
    if not isinstance(value, str):
        raise _error(line_number, f"field {name!r} must be a string")
    return value


def _optional_int_field(record: Mapping[str, Any], name: str, line_number: int) -> int | None:
    value = _field(record, name, line_number)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(line_number, f"field {name!r} must be an integer or null")
    return value


def _optional_str_field(record: Mapping[str, Any], name: str, line_number: int) -> str | None:
    value = _field(record, name, line_number)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _error(line_number, f"field {name!r} must be a string or null")
    return value


def _str_list_field(record: Mapping[str, Any], name: str, line_number: int) -> list[str]:
    value = _field(record, name, line_number)
    if not isinstance(value, list):
        raise _error(line_number, f"field {name!r} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise _error(line_number, f"field {name!r} must be a list of strings")
        items.append(item)
    return items


def _event_spans(
    record: Mapping[str, Any], line_number: int, word_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    value = _field(record, "events", line_number)
    if not isinstance(value, list) or len(value) != 2:
        raise _error(line_number, "field 'events' must contain exactly two spans")

    spans: list[tuple[int, ...]] = []
    for span_index, span in enumerate(value):
        if not isinstance(span, list) or not span:
            raise _error(
                line_number,
                f"field 'events[{span_index}]' must be a non-empty list of token indexes",
            )
        positions: list[int] = []
        for position in span:
            if isinstance(position, bool) or not isinstance(position, int):
                raise _error(
                    line_number,
                    f"field 'events[{span_index}]' must contain integer token indexes",
                )
            if position < 0 or position >= word_count:
                raise _error(
                    line_number,
                    f"field 'events[{span_index}]' index {position} is out of bounds "
                    f"for {word_count} tokens",
                )
            positions.append(position)
        spans.append(tuple(positions))

    return spans[0], spans[1]


def _render(words: list[str], span: tuple[int, ...]) -> str:
    return " ".join(words[position] for position in span)


def _content_record_id(
    sentence: str,
    spans: tuple[tuple[int, ...], tuple[int, ...]],
    doc_name: str,
    events_id: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "namespace": _NAMESPACE,
            "sentence": sentence,
            "events": [list(spans[0]), list(spans[1])],
            "doc_name": doc_name,
            "events_id": list(events_id),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{_NAMESPACE}:{digest}"


def _parse_record(raw: object, line_number: int) -> EscRecord:
    if not isinstance(raw, dict):
        raise _error(line_number, "expected a JSON object")

    record: Mapping[str, Any] = raw
    words = _str_list_field(record, "words", line_number)
    span1, span2 = _event_spans(record, line_number, len(words))
    sentence = " ".join(words)
    event1 = _render(words, span1)
    event2 = _render(words, span2)
    doc_name = _str_field(record, "doc_name", line_number)
    events_id = tuple(_str_list_field(record, "events_id", line_number))

    return EscRecord(
        item_id=_int_field(record, "item_id", line_number),
        record_id=_content_record_id(sentence, (span1, span2), doc_name, events_id),
        sentence=sentence,
        event1=event1,
        event2=event2,
        doc_name=doc_name,
        sent_in_doc_id=_optional_int_field(record, "sent_in_doc_id", line_number),
        events_id=events_id,
        topic=_optional_str_field(record, "topic", line_number),
        esc_rel=_str_field(record, "ESC_rel", line_number),
        tri_causal_label=_int_field(record, "tri_causal_label", line_number),
        bi_causal_label=_int_field(record, "bi_causal_label", line_number),
    )


def _build_inventory(records: list[EscRecord], unique_item_ids: int) -> EscInventory:
    tri_counts = Counter(record.tri_causal_label for record in records)
    bi_counts = Counter(record.bi_causal_label for record in records)
    esc_rel_counts = Counter(record.esc_rel for record in records)
    return EscInventory(
        total=len(records),
        unique_item_ids=unique_item_ids,
        tri_counts=MappingProxyType(dict(tri_counts)),
        bi_counts=MappingProxyType(dict(bi_counts)),
        causal_rows=sum(1 for record in records if record.tri_causal_label in (1, 2)),
        esc_rel_counts=MappingProxyType(dict(esc_rel_counts)),
    )


def load_esc(path: str | Path) -> EscDataset:
    """Load an ESC JSONL dataset, validating every line with line context."""

    source_path = Path(path)
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read {source_path}: {exc}") from exc

    source_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"unable to decode {source_path} as UTF-8: {exc}") from exc

    records: list[EscRecord] = []
    seen_item_ids: set[int] = set()

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _error(line_number, f"invalid JSON: {exc.msg}") from exc

        record = _parse_record(raw, line_number)
        if record.item_id in seen_item_ids:
            raise _error(line_number, f"duplicate item_id {record.item_id}")
        seen_item_ids.add(record.item_id)
        records.append(record)

    return EscDataset(
        source_path=source_path,
        source_sha256=source_sha256,
        records=tuple(records),
        inventory=_build_inventory(records, len(seen_item_ids)),
    )


__all__ = ["EscDataset", "EscInventory", "EscRecord", "load_esc"]
