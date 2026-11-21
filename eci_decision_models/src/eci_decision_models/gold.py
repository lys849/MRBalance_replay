"""Reconstruct direction-aware ESC gold labels from raw and official sources.

The implementation follows ``docs/application_plan.md`` section 3.3 and ADR
0002.  Raw EventStoryLine XML remains the direction authority.  The official
``evaluation_format/full_corpus/v0.9`` export locates the corresponding mention
pair and supplies relation/direction only through the explicitly approved
fallbacks.  Legacy ``tri_causal_label`` values are used solely for audits and
never decide direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from eci_decision_models.canonical import Choice as Choice
from eci_decision_models.data import EscDataset, EscRecord, load_esc

_ESC_DOC_PREFIX = "data/EventStoryLine/"
_GOLD_FILENAME = "gold_choice_reconstructed.jsonl"
_MANIFEST_FILENAME = "manifest.json"
_AUDIT_FILENAME = "audit.json"
_SCHEMA_VERSION = "1"
_GOLD_VERSION = "gold_choice_reconstructed/v1"
_EVALUATION_PREFIX = "evaluation_format/full_corpus/v0.9"
_KNOWN_REL_TYPES = frozenset({"PRECONDITION", "FALLING_ACTION"})
_MISSING_RELATION_FALLBACK_ITEMS = frozenset({9775})
_SELF_COREF_FALLBACK_ITEMS = frozenset({2531, 10592, 12487})
_APPROVED_FALLBACK_ITEMS = {
    "missing_relation_fallback": _MISSING_RELATION_FALLBACK_ITEMS,
    "self_coref_evaluation_fallback": _SELF_COREF_FALLBACK_ITEMS,
}

_PairKey = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class GoldIssue:
    """A row-level hard failure, mismatch, or source discrepancy."""

    item_id: int
    record_id: str
    reason: str
    details: Mapping[str, object]


@dataclass(frozen=True)
class GoldRecord:
    """One reconstructed gold record with full provenance."""

    item_id: int
    record_id: str
    event1: str
    event2: str
    esc_rel: str
    choice: Choice
    plot_link_id: str | None
    plot_link_source: str | None
    plot_link_target: str | None
    xml_path: str | None
    xml_sha256: str | None
    xml_version: str
    relation_source: str
    direction_source: str
    resolution_method: str
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class GoldReconstruction:
    """A complete reconstruction result, including failures and audit data."""

    source_path: Path
    source_sha256: str
    xml_version: str
    records: tuple[GoldRecord, ...]
    hard_failures: tuple[GoldIssue, ...]
    mismatches: tuple[GoldIssue, ...]
    source_discrepancies: tuple[GoldIssue, ...]
    xml_files_sha256: Mapping[str, str]
    evaluation_files_sha256: Mapping[str, str]
    summary: Mapping[str, int]


@dataclass(frozen=True)
class GoldArtifactPaths:
    """Paths and content hashes of a finalized gold artifact."""

    gold_choice_jsonl: Path
    manifest_json: Path
    audit_json: Path
    artifact_sha256: Mapping[str, str]


class GoldReconstructionError(Exception):
    """Raised when an invalid reconstruction is finalized as an artifact."""


class _HardFailureError(Exception):
    def __init__(self, reason: str, details: Mapping[str, object]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = dict(details)


@dataclass(frozen=True)
class _PlotLink:
    link_id: str | None
    rel_type: str | None
    source: str | None
    target: str | None


@dataclass(frozen=True)
class _Document:
    markable_tags: Mapping[str, str]
    event_tokens: Mapping[str, tuple[int, ...]]
    plot_links: tuple[_PlotLink, ...]


@dataclass(frozen=True)
class _OfficialRow:
    source_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    relation: str


@dataclass(frozen=True)
class _OfficialIndex:
    rows: Mapping[_PairKey, _OfficialRow]
    ambiguous: frozenset[_PairKey]


@dataclass(frozen=True)
class _CorefRow:
    source_members: tuple[tuple[int, ...], ...]
    target_members: tuple[tuple[int, ...], ...]
    relation: str
    line_number: int


@dataclass(frozen=True)
class _Resolution:
    choice: Choice
    plot_link_ids: tuple[str, ...]
    plot_link_source: str | None
    plot_link_target: str | None
    relation_source: str
    direction_source: str
    resolution_method: str
    relation: str | None
    source_role: str
    coref_path: str | None


def _local_name(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _flip_relation(relation: str) -> str:
    if relation == "PRECONDITION":
        return "FALLING_ACTION"
    if relation == "FALLING_ACTION":
        return "PRECONDITION"
    return relation


def _first_token(tokens: tuple[int, ...]) -> int:
    if not tokens:
        return -1
    return tokens[0]


def _tokens_text(tokens: tuple[int, ...]) -> str:
    return "_".join(str(token) for token in tokens)


def _pair_key(first: tuple[int, ...], second: tuple[int, ...]) -> _PairKey:
    if _first_token(first) <= _first_token(second):
        return (first, second)
    return (second, first)


def _relation_in_order(
    relation: str,
    source_tokens: tuple[int, ...],
    target_tokens: tuple[int, ...],
) -> str:
    if _first_token(source_tokens) <= _first_token(target_tokens):
        return relation
    return _flip_relation(relation)


def _choice_for(relation: str, source_role: str) -> Choice:
    if relation == "PRECONDITION":
        return Choice.cause if source_role == "event1" else Choice.caused_by
    return Choice.cause if source_role == "event2" else Choice.caused_by


def _safe_relative_path(doc_name: str, xml_root: Path) -> str:
    prefix_length = len(_ESC_DOC_PREFIX)
    relative = doc_name[prefix_length:] if doc_name.startswith(_ESC_DOC_PREFIX) else doc_name
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise _HardFailureError(
            "unsafe doc_name",
            {"doc_name": doc_name, "relative": relative},
        )

    root = xml_root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise _HardFailureError(
            "doc_name escapes xml_root",
            {"doc_name": doc_name, "relative": relative, "resolved": str(resolved)},
        )
    return candidate.as_posix()


def _evaluation_relative(relative: str, kind: str) -> str:
    parts = Path(relative).parts
    try:
        version_index = parts.index("v0.9")
    except ValueError as exc:
        raise _HardFailureError(
            "cannot derive evaluation path from doc_name",
            {"xml_path": relative},
        ) from exc
    if version_index + 1 >= len(parts):
        raise _HardFailureError(
            "cannot derive evaluation topic from doc_name",
            {"xml_path": relative},
        )
    topic = parts[version_index + 1]
    stem = Path(relative).name.removesuffix(".xml").removesuffix(".xml")
    return f"{_EVALUATION_PREFIX}/{kind}/{topic}/{stem}.tab"


def _read_bytes(xml_root: Path, relative: str, hashes: dict[str, str]) -> bytes:
    root = xml_root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise _HardFailureError(
            "input file escapes xml_root",
            {"relative": relative, "resolved": str(path)},
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise _HardFailureError(
            "unable to read input file",
            {"relative": relative, "error": str(exc)},
        ) from exc
    hashes[relative] = hashlib.sha256(payload).hexdigest()
    return payload


def _parse_token_tuple(text: str) -> tuple[int, ...]:
    parts = text.split("_")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid token tuple {text!r}")
    return tuple(int(part) for part in parts)


def _endpoint(link: ET.Element, name: str) -> str | None:
    for child in link:
        if _local_name(child.tag) == name:
            return child.get("m_id")
    return None


def _parse_document(payload: bytes, relative: str) -> _Document:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise _HardFailureError(
            "invalid XML",
            {"xml_path": relative, "error": str(exc)},
        ) from exc

    markable_tags: dict[str, str] = {}
    event_tokens: dict[str, tuple[int, ...]] = {}
    plot_links: list[_PlotLink] = []
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "Markables":
            for markable in element:
                markable_id = markable.get("m_id")
                if markable_id is None:
                    continue
                tag = _local_name(markable.tag)
                markable_tags[markable_id] = tag
                if not (tag.startswith("ACTION_") or tag.startswith("NEG_ACTION_")):
                    continue
                token_ids: list[int] = []
                for child in markable:
                    if _local_name(child.tag) != "token_anchor":
                        continue
                    token_id = child.get("t_id")
                    if token_id is None:
                        raise _HardFailureError(
                            "event markable token_anchor missing t_id",
                            {"xml_path": relative, "m_id": markable_id},
                        )
                    try:
                        token_ids.append(int(token_id))
                    except ValueError as exc:
                        raise _HardFailureError(
                            "event markable token_anchor is not an integer",
                            {"xml_path": relative, "m_id": markable_id, "t_id": token_id},
                        ) from exc
                event_tokens[markable_id] = tuple(token_ids)
        elif name == "PLOT_LINK":
            plot_links.append(
                _PlotLink(
                    link_id=element.get("r_id"),
                    rel_type=element.get("relType"),
                    source=_endpoint(element, "source"),
                    target=_endpoint(element, "target"),
                )
            )

    return _Document(
        markable_tags=MappingProxyType(markable_tags),
        event_tokens=MappingProxyType(event_tokens),
        plot_links=tuple(plot_links),
    )


def _load_document(
    xml_root: Path,
    relative: str,
    cache: dict[str, _Document],
    hashes: dict[str, str],
) -> _Document:
    cached = cache.get(relative)
    if cached is not None:
        return cached
    payload = _read_bytes(xml_root, relative, hashes)
    document = _parse_document(payload, relative)
    cache[relative] = document
    return document


def _parse_official_index(payload: bytes, relative: str) -> _OfficialIndex:
    rows: dict[_PairKey, _OfficialRow] = {}
    ambiguous: set[_PairKey] = set()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _HardFailureError(
            "evaluation file is not UTF-8",
            {"evaluation_path": relative, "error": str(exc)},
        ) from exc

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 3:
            raise _HardFailureError(
                "invalid evaluation row",
                {"evaluation_path": relative, "line": line_number, "row": line},
            )
        source_members = columns[0].strip().split()
        target_members = columns[1].strip().split()
        relation = columns[2].strip()
        if len(source_members) != 1 or len(target_members) != 1:
            raise _HardFailureError(
                "evaluation pair row has multiple members",
                {"evaluation_path": relative, "line": line_number, "row": line},
            )
        try:
            source_tokens = _parse_token_tuple(source_members[0])
            target_tokens = _parse_token_tuple(target_members[0])
        except ValueError as exc:
            raise _HardFailureError(
                "invalid evaluation token anchor",
                {"evaluation_path": relative, "line": line_number, "row": line},
            ) from exc

        if _first_token(source_tokens) > _first_token(target_tokens):
            source_tokens, target_tokens = target_tokens, source_tokens
            relation = _flip_relation(relation)
        key = _pair_key(source_tokens, target_tokens)
        row = _OfficialRow(
            source_tokens=source_tokens, target_tokens=target_tokens, relation=relation
        )
        existing = rows.get(key)
        if existing is not None and existing.relation != relation:
            ambiguous.add(key)
            rows.pop(key, None)
            continue
        if key in ambiguous:
            continue
        rows[key] = row

    return _OfficialIndex(
        rows=MappingProxyType(rows),
        ambiguous=frozenset(ambiguous),
    )


def _load_official_index(
    xml_root: Path,
    relative: str,
    cache: dict[str, _OfficialIndex],
    hashes: dict[str, str],
) -> _OfficialIndex:
    evaluation_relative = _evaluation_relative(relative, "event_mentions_extended")
    cached = cache.get(evaluation_relative)
    if cached is not None:
        return cached
    payload = _read_bytes(xml_root, evaluation_relative, hashes)
    index = _parse_official_index(payload, evaluation_relative)
    cache[evaluation_relative] = index
    return index


def _parse_coref_rows(payload: bytes, relative: str) -> tuple[_CorefRow, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _HardFailureError(
            "coref file is not UTF-8",
            {"evaluation_path": relative, "error": str(exc)},
        ) from exc

    rows: list[_CorefRow] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 3:
            raise _HardFailureError(
                "invalid coref row",
                {"evaluation_path": relative, "line": line_number, "row": line},
            )
        try:
            source_members = tuple(
                _parse_token_tuple(member) for member in columns[0].strip().split()
            )
            target_members = tuple(
                _parse_token_tuple(member) for member in columns[1].strip().split()
            )
        except ValueError as exc:
            raise _HardFailureError(
                "invalid coref token anchor",
                {"evaluation_path": relative, "line": line_number, "row": line},
            ) from exc
        if not source_members or not target_members:
            raise _HardFailureError(
                "coref row has an empty member set",
                {"evaluation_path": relative, "line": line_number, "row": line},
            )
        rows.append(
            _CorefRow(
                source_members=source_members,
                target_members=target_members,
                relation=columns[2].strip(),
                line_number=line_number,
            )
        )
    return tuple(rows)


def _load_coref_rows(
    xml_root: Path,
    relative: str,
    cache: dict[str, tuple[_CorefRow, ...]],
    hashes: dict[str, str],
) -> tuple[_CorefRow, ...]:
    evaluation_relative = _evaluation_relative(relative, "coref_chain")
    cached = cache.get(evaluation_relative)
    if cached is not None:
        return cached
    payload = _read_bytes(xml_root, evaluation_relative, hashes)
    rows = _parse_coref_rows(payload, evaluation_relative)
    cache[evaluation_relative] = rows
    return rows


def _validate_row_labels(record: EscRecord) -> bool:
    tri = record.tri_causal_label
    bi = record.bi_causal_label
    esc_rel = record.esc_rel

    if tri == 0:
        if bi != 0 or esc_rel != "UN_LABEL":
            raise _HardFailureError(
                "inconsistent row labels for none row",
                {
                    "tri_causal_label": tri,
                    "bi_causal_label": bi,
                    "ESC_rel": esc_rel,
                },
            )
        return False

    if tri in (1, 2):
        expected_rel = "PRECONDITION" if tri == 1 else "FALLING_ACTION"
        if bi != 1 or esc_rel != expected_rel:
            raise _HardFailureError(
                "inconsistent row labels for causal row",
                {
                    "tri_causal_label": tri,
                    "bi_causal_label": bi,
                    "ESC_rel": esc_rel,
                    "expected_ESC_rel": expected_rel,
                },
            )
        return True

    raise _HardFailureError(
        "unsupported tri_causal_label",
        {"tri_causal_label": tri, "bi_causal_label": bi, "ESC_rel": esc_rel},
    )


def _none_record(record: EscRecord, xml_version: str) -> GoldRecord:
    return GoldRecord(
        item_id=record.item_id,
        record_id=record.record_id,
        event1=record.event1,
        event2=record.event2,
        esc_rel=record.esc_rel,
        choice=Choice.none,
        plot_link_id=None,
        plot_link_source=None,
        plot_link_target=None,
        xml_path=None,
        xml_sha256=None,
        xml_version=xml_version,
        relation_source="none",
        direction_source="none",
        resolution_method="none",
        provenance=MappingProxyType(
            {
                "doc_name": record.doc_name,
                "events_id": list(record.events_id),
                "tri_causal_label": record.tri_causal_label,
                "bi_causal_label": record.bi_causal_label,
                "plot_link_ids": [],
                "resolution": {"method": "none"},
            }
        ),
    )


def _event_token_tuple(
    document: _Document,
    markable_id: str,
    role: str,
    relative: str,
) -> tuple[int, ...]:
    tokens = document.event_tokens.get(markable_id)
    if tokens is None:
        if markable_id not in document.markable_tags:
            raise _HardFailureError(
                "event markable is missing",
                {"xml_path": relative, "role": role, "m_id": markable_id},
            )
        raise _HardFailureError(
            "event markable is not an event markable",
            {
                "xml_path": relative,
                "role": role,
                "m_id": markable_id,
                "markable_tag": document.markable_tags[markable_id],
            },
        )
    if not tokens:
        raise _HardFailureError(
            "event markable has no token anchors",
            {"xml_path": relative, "role": role, "m_id": markable_id},
        )
    return tokens


def _source_role(link: _PlotLink, event1_id: str, event2_id: str, relative: str) -> str:
    if link.source == event1_id:
        return "event1"
    if link.source == event2_id:
        return "event2"
    raise _HardFailureError(
        "PLOT_LINK source is not an events_id endpoint",
        {
            "xml_path": relative,
            "link_id": link.link_id,
            "source": link.source,
            "events_id": [event1_id, event2_id],
        },
    )


def _classify_direct_conflict(links: Sequence[_PlotLink]) -> str:
    directions = {(link.source, link.target) for link in links}
    rel_types = {link.rel_type for link in links}
    if len(directions) > 1:
        return "bidirectional PLOT_LINK matches"
    if len(rel_types) > 1:
        return "conflicting PLOT_LINK relType matches"
    return "nonidentical duplicate PLOT_LINK matches"


def _resolve_direct(
    links: Sequence[_PlotLink],
    event1_id: str,
    event2_id: str,
    event1_tokens: tuple[int, ...],
    event2_tokens: tuple[int, ...],
    official: _OfficialRow,
    relative: str,
) -> tuple[_Resolution, Mapping[str, object] | None]:
    signatures = {(link.source, link.target, link.rel_type) for link in links}
    if len(signatures) > 1:
        raise _HardFailureError(
            _classify_direct_conflict(links),
            {
                "xml_path": relative,
                "events_id": [event1_id, event2_id],
                "link_ids": [link.link_id for link in links],
                "rel_types": [link.rel_type for link in links],
                "sources": [link.source for link in links],
                "targets": [link.target for link in links],
            },
        )

    link = links[0]
    source_role = _source_role(link, event1_id, event2_id, relative)
    link_ids = tuple(link.link_id for link in links if link.link_id is not None)
    first = links[0]
    raw_source_tokens = event1_tokens if source_role == "event1" else event2_tokens
    raw_target_tokens = event2_tokens if source_role == "event1" else event1_tokens

    if link.rel_type is None:
        if official.relation not in _KNOWN_REL_TYPES:
            raise _HardFailureError(
                "official evaluation relation is not causal",
                {
                    "xml_path": relative,
                    "link_id": link.link_id,
                    "official_relation": official.relation,
                },
            )
        raw_relation = _relation_in_order(official.relation, raw_source_tokens, raw_target_tokens)
        return (
            _Resolution(
                choice=_choice_for(raw_relation, source_role),
                plot_link_ids=link_ids,
                plot_link_source=first.source,
                plot_link_target=first.target,
                relation_source="evaluation_format",
                direction_source="mixed",
                resolution_method="missing_relation_fallback",
                relation=raw_relation,
                source_role=source_role,
                coref_path=None,
            ),
            None,
        )

    if link.rel_type not in _KNOWN_REL_TYPES:
        raise _HardFailureError(
            "unknown PLOT_LINK relType",
            {
                "xml_path": relative,
                "link_id": link.link_id,
                "relType": link.rel_type,
            },
        )

    normalized_relation = _relation_in_order(link.rel_type, raw_source_tokens, raw_target_tokens)
    discrepancy: Mapping[str, object] | None = None
    if normalized_relation != official.relation:
        discrepancy = MappingProxyType(
            {
                "xml_path": relative,
                "link_ids": list(link_ids),
                "raw_relation": link.rel_type,
                "normalized_raw_relation": normalized_relation,
                "official_relation": official.relation,
                "events_id": [event1_id, event2_id],
            }
        )

    method = "exact_duplicate_collapse" if len(links) > 1 else "direct_xml"
    return (
        _Resolution(
            choice=_choice_for(link.rel_type, source_role),
            plot_link_ids=link_ids,
            plot_link_source=first.source,
            plot_link_target=first.target,
            relation_source="raw_xml",
            direction_source="raw_xml",
            resolution_method=method,
            relation=link.rel_type,
            source_role=source_role,
            coref_path=None,
        ),
        discrepancy,
    )


@dataclass(frozen=True)
class _CorefCandidate:
    row: _CorefRow
    choice: Choice
    raw_relation: str
    source_role: str
    self_coref: bool


def _coref_candidates(
    rows: Sequence[_CorefRow],
    event1_tokens: tuple[int, ...],
    event2_tokens: tuple[int, ...],
    official_relation: str,
) -> list[_CorefCandidate]:
    candidates: list[_CorefCandidate] = []
    for row in rows:
        orientations: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        if event1_tokens in row.source_members and event2_tokens in row.target_members:
            orientations.append(("event1", event1_tokens, event2_tokens))
        if event1_tokens in row.target_members and event2_tokens in row.source_members:
            orientations.append(("event2", event2_tokens, event1_tokens))
        for source_role, raw_source_tokens, raw_target_tokens in orientations:
            raw_relation = row.relation
            if raw_relation not in _KNOWN_REL_TYPES:
                continue
            normalized = _relation_in_order(raw_relation, raw_source_tokens, raw_target_tokens)
            if normalized != official_relation:
                continue
            candidates.append(
                _CorefCandidate(
                    row=row,
                    choice=_choice_for(raw_relation, source_role),
                    raw_relation=raw_relation,
                    source_role=source_role,
                    self_coref=set(row.source_members) == set(row.target_members),
                )
            )
    return candidates


def _anchor_coref_links(
    document: _Document,
    candidates: Sequence[_CorefCandidate],
) -> tuple[_PlotLink, ...]:
    matches: list[_PlotLink] = []
    for link in document.plot_links:
        if link.rel_type is None or link.source is None or link.target is None:
            continue
        source_tokens = document.event_tokens.get(link.source)
        target_tokens = document.event_tokens.get(link.target)
        if source_tokens is None or target_tokens is None:
            continue
        for candidate in candidates:
            if link.rel_type != candidate.row.relation:
                continue
            if source_tokens not in candidate.row.source_members:
                continue
            if target_tokens not in candidate.row.target_members:
                continue
            matches.append(link)
            break
    return tuple(matches)


def _resolve_coref(
    document: _Document,
    event1_tokens: tuple[int, ...],
    event2_tokens: tuple[int, ...],
    official: _OfficialRow,
    relative: str,
    coref_rows: Sequence[_CorefRow],
    coref_path: str,
) -> _Resolution:
    candidates = _coref_candidates(coref_rows, event1_tokens, event2_tokens, official.relation)
    if not candidates:
        raise _HardFailureError(
            "no coref expansion confirms the official relation",
            {
                "xml_path": relative,
                "events_id_tokens": [_tokens_text(event1_tokens), _tokens_text(event2_tokens)],
                "official_relation": official.relation,
                "coref_path": coref_path,
            },
        )

    choices = {candidate.choice for candidate in candidates}
    if len(choices) > 1:
        raise _HardFailureError(
            "conflicting coref expansion directions",
            {
                "xml_path": relative,
                "events_id_tokens": [
                    _tokens_text(event1_tokens),
                    _tokens_text(event2_tokens),
                ],
                "choices": sorted(choice.value for choice in choices),
                "coref_rows": sorted({candidate.row.line_number for candidate in candidates}),
                "coref_path": coref_path,
            },
        )

    self_coref = all(candidate.self_coref for candidate in candidates)
    links = _anchor_coref_links(document, candidates)
    if not links and not self_coref:
        raise _HardFailureError(
            "no raw parent PLOT_LINK for coref expansion",
            {
                "xml_path": relative,
                "events_id_tokens": [
                    _tokens_text(event1_tokens),
                    _tokens_text(event2_tokens),
                ],
                "coref_path": coref_path,
            },
        )

    selected = candidates[0]
    first = links[0] if links else None
    return _Resolution(
        choice=selected.choice,
        plot_link_ids=tuple(link.link_id for link in links if link.link_id is not None),
        plot_link_source=first.source if first is not None else None,
        plot_link_target=first.target if first is not None else None,
        relation_source="raw_xml",
        direction_source="evaluation_format" if self_coref else "raw_xml",
        resolution_method=("self_coref_evaluation_fallback" if self_coref else "coref_expansion"),
        relation=selected.raw_relation,
        source_role=selected.source_role,
        coref_path=coref_path,
    )


def _require_approved_fallback(
    item_id: int,
    resolution: _Resolution,
    relative: str,
) -> None:
    """Reject any evaluation-format fallback that is not explicitly approved."""
    method = resolution.resolution_method
    approved = _APPROVED_FALLBACK_ITEMS.get(method)
    if approved is None:
        return
    if item_id in approved:
        return
    raise _HardFailureError(
        "evaluation fallback is not approved for this item",
        {
            "item_id": item_id,
            "xml_path": relative,
            "resolution_method": method,
            "relation_source": resolution.relation_source,
            "direction_source": resolution.direction_source,
        },
    )


def _build_causal_record(
    record: EscRecord,
    xml_version: str,
    relative: str,
    xml_sha256: str,
    event1_tokens: tuple[int, ...],
    event2_tokens: tuple[int, ...],
    official: _OfficialRow,
    official_path: str,
    resolution: _Resolution,
) -> GoldRecord:
    evaluation_row: dict[str, object] = {
        "path": official_path,
        "source_token": _tokens_text(official.source_tokens),
        "target_token": _tokens_text(official.target_tokens),
        "relation": official.relation,
    }
    resolution_details: dict[str, object] = {
        "method": resolution.resolution_method,
        "relation": resolution.relation,
        "source_role": resolution.source_role,
        "relation_source": resolution.relation_source,
        "direction_source": resolution.direction_source,
    }
    if resolution.coref_path is not None:
        evaluation_row["coref_path"] = resolution.coref_path
        resolution_details["coref_path"] = resolution.coref_path

    provenance: dict[str, object] = {
        "doc_name": record.doc_name,
        "events_id": list(record.events_id),
        "xml_path": relative,
        "xml_version": xml_version,
        "tri_causal_label": record.tri_causal_label,
        "bi_causal_label": record.bi_causal_label,
        "plot_link_ids": list(resolution.plot_link_ids),
        "event1_token": _tokens_text(event1_tokens),
        "event2_token": _tokens_text(event2_tokens),
        "evaluation_row": evaluation_row,
        "resolution": resolution_details,
    }

    return GoldRecord(
        item_id=record.item_id,
        record_id=record.record_id,
        event1=record.event1,
        event2=record.event2,
        esc_rel=record.esc_rel,
        choice=resolution.choice,
        plot_link_id=(resolution.plot_link_ids[0] if resolution.plot_link_ids else None),
        plot_link_source=resolution.plot_link_source,
        plot_link_target=resolution.plot_link_target,
        xml_path=relative,
        xml_sha256=xml_sha256,
        xml_version=xml_version,
        relation_source=resolution.relation_source,
        direction_source=resolution.direction_source,
        resolution_method=resolution.resolution_method,
        provenance=MappingProxyType(provenance),
    )


def _causal_record(
    record: EscRecord,
    xml_root: Path,
    xml_version: str,
    documents: dict[str, _Document],
    xml_hashes: dict[str, str],
    official_indexes: dict[str, _OfficialIndex],
    coref_cache: dict[str, tuple[_CorefRow, ...]],
    evaluation_hashes: dict[str, str],
) -> tuple[GoldRecord, Mapping[str, object] | None]:
    if len(record.events_id) != 2:
        raise _HardFailureError(
            "causal row must have exactly two events_id endpoints",
            {"events_id": list(record.events_id)},
        )

    event1_id, event2_id = record.events_id
    relative = _safe_relative_path(record.doc_name, xml_root)
    document = _load_document(xml_root, relative, documents, xml_hashes)
    event1_tokens = _event_token_tuple(document, event1_id, "event1", relative)
    event2_tokens = _event_token_tuple(document, event2_id, "event2", relative)

    official_index = _load_official_index(xml_root, relative, official_indexes, evaluation_hashes)
    key = _pair_key(event1_tokens, event2_tokens)
    if key in official_index.ambiguous:
        raise _HardFailureError(
            "multiple official evaluation relations for token pair",
            {
                "xml_path": relative,
                "event1_token": _tokens_text(event1_tokens),
                "event2_token": _tokens_text(event2_tokens),
            },
        )
    official = official_index.rows.get(key)
    if official is None:
        raise _HardFailureError(
            "no official evaluation row for token pair",
            {
                "xml_path": relative,
                "event1_token": _tokens_text(event1_tokens),
                "event2_token": _tokens_text(event2_tokens),
            },
        )

    oriented_official = _relation_in_order(official.relation, event1_tokens, event2_tokens)
    if oriented_official != record.esc_rel:
        raise _HardFailureError(
            "ESC_rel does not match official evaluation relation",
            {
                "xml_path": relative,
                "event1_token": _tokens_text(event1_tokens),
                "event2_token": _tokens_text(event2_tokens),
                "official_relation": official.relation,
                "oriented_official_relation": oriented_official,
                "ESC_rel": record.esc_rel,
            },
        )
    if official.relation not in _KNOWN_REL_TYPES:
        raise _HardFailureError(
            "official evaluation relation is not causal",
            {
                "xml_path": relative,
                "official_relation": official.relation,
                "ESC_rel": record.esc_rel,
            },
        )

    official_path = _evaluation_relative(relative, "event_mentions_extended")
    endpoints = frozenset((event1_id, event2_id))
    direct_links = tuple(
        link
        for link in document.plot_links
        if link.source is not None
        and link.target is not None
        and frozenset((link.source, link.target)) == endpoints
    )

    discrepancy: Mapping[str, object] | None = None
    if direct_links:
        resolution, discrepancy = _resolve_direct(
            direct_links,
            event1_id,
            event2_id,
            event1_tokens,
            event2_tokens,
            official,
            relative,
        )
    else:
        coref_path = _evaluation_relative(relative, "coref_chain")
        coref_rows = _load_coref_rows(xml_root, relative, coref_cache, evaluation_hashes)
        resolution = _resolve_coref(
            document,
            event1_tokens,
            event2_tokens,
            official,
            relative,
            coref_rows,
            coref_path,
        )

    _require_approved_fallback(record.item_id, resolution, relative)

    gold_record = _build_causal_record(
        record,
        xml_version,
        relative,
        xml_hashes[relative],
        event1_tokens,
        event2_tokens,
        official,
        official_path,
        resolution,
    )
    return gold_record, discrepancy


def _legacy_choice(tri_causal_label: int) -> Choice:
    return Choice.cause if tri_causal_label == 1 else Choice.caused_by


def reconstruct_gold(
    dataset: EscDataset,
    xml_root: str | Path,
    xml_version: str,
) -> GoldReconstruction:
    """Reconstruct every row, collecting hard failures instead of skipping."""

    root = Path(xml_root)
    documents: dict[str, _Document] = {}
    xml_hashes: dict[str, str] = {}
    official_indexes: dict[str, _OfficialIndex] = {}
    coref_cache: dict[str, tuple[_CorefRow, ...]] = {}
    evaluation_hashes: dict[str, str] = {}

    records: list[GoldRecord] = []
    hard_failures: list[GoldIssue] = []
    mismatches: list[GoldIssue] = []
    source_discrepancies: list[GoldIssue] = []

    for source_record in dataset.records:
        try:
            is_causal = _validate_row_labels(source_record)
            if is_causal:
                record, discrepancy = _causal_record(
                    source_record,
                    root,
                    xml_version,
                    documents,
                    xml_hashes,
                    official_indexes,
                    coref_cache,
                    evaluation_hashes,
                )
            else:
                record = _none_record(source_record, xml_version)
                discrepancy = None
        except _HardFailureError as exc:
            hard_failures.append(
                GoldIssue(
                    item_id=source_record.item_id,
                    record_id=source_record.record_id,
                    reason=exc.reason,
                    details=MappingProxyType(dict(exc.details)),
                )
            )
            continue

        records.append(record)
        if discrepancy is not None:
            source_discrepancies.append(
                GoldIssue(
                    item_id=source_record.item_id,
                    record_id=source_record.record_id,
                    reason="raw PLOT_LINK relation disagrees with official evaluation relation",
                    details=discrepancy,
                )
            )
        if is_causal:
            legacy = _legacy_choice(source_record.tri_causal_label)
            if legacy is not record.choice:
                mismatches.append(
                    GoldIssue(
                        item_id=source_record.item_id,
                        record_id=source_record.record_id,
                        reason="legacy tri direction disagrees with reconstructed gold",
                        details=MappingProxyType(
                            {
                                "legacy_choice": legacy.value,
                                "reconstructed_choice": record.choice.value,
                                "tri_causal_label": source_record.tri_causal_label,
                                "ESC_rel": source_record.esc_rel,
                                "resolution_method": record.resolution_method,
                                "relation_source": record.relation_source,
                                "direction_source": record.direction_source,
                                "xml_path": record.xml_path,
                                "plot_link_id": record.plot_link_id,
                                "plot_link_source": record.plot_link_source,
                                "plot_link_target": record.plot_link_target,
                            }
                        ),
                    )
                )

    records_total = len(dataset.records)
    causal = sum(1 for record in dataset.records if record.tri_causal_label in (1, 2))
    summary = MappingProxyType(
        {
            "records": records_total,
            "causal": causal,
            "none": records_total - causal,
            "hard_failures": len(hard_failures),
            "mismatches": len(mismatches),
            "source_discrepancies": len(source_discrepancies),
            "direct_xml": sum(1 for record in records if record.resolution_method == "direct_xml"),
            "coref_expansion": sum(
                1 for record in records if record.resolution_method == "coref_expansion"
            ),
            "exact_duplicate_collapses": sum(
                1 for record in records if record.resolution_method == "exact_duplicate_collapse"
            ),
            "missing_relation_fallbacks": sum(
                1 for record in records if record.resolution_method == "missing_relation_fallback"
            ),
            "self_coref_evaluation_fallbacks": sum(
                1
                for record in records
                if record.resolution_method == "self_coref_evaluation_fallback"
            ),
        }
    )

    return GoldReconstruction(
        source_path=dataset.source_path,
        source_sha256=dataset.source_sha256,
        xml_version=xml_version,
        records=tuple(records),
        hard_failures=tuple(hard_failures),
        mismatches=tuple(mismatches),
        source_discrepancies=tuple(source_discrepancies),
        xml_files_sha256=MappingProxyType(dict(xml_hashes)),
        evaluation_files_sha256=MappingProxyType(dict(evaluation_hashes)),
        summary=summary,
    )


def _record_payload(record: GoldRecord) -> dict[str, Any]:
    return {
        "item_id": record.item_id,
        "record_id": record.record_id,
        "event1": record.event1,
        "event2": record.event2,
        "esc_rel": record.esc_rel,
        "choice": record.choice.value,
        "plot_link_id": record.plot_link_id,
        "plot_link_source": record.plot_link_source,
        "plot_link_target": record.plot_link_target,
        "xml_path": record.xml_path,
        "xml_sha256": record.xml_sha256,
        "xml_version": record.xml_version,
        "relation_source": record.relation_source,
        "direction_source": record.direction_source,
        "resolution_method": record.resolution_method,
        "provenance": dict(record.provenance),
    }


def _resolution_entry(record: GoldRecord) -> dict[str, Any]:
    """Return the frozen per-item provenance entry stored in the audit."""
    return {
        "item_id": record.item_id,
        "record_id": record.record_id,
        "choice": record.choice.value,
        "resolution_method": record.resolution_method,
        "relation_source": record.relation_source,
        "direction_source": record.direction_source,
        "xml_path": record.xml_path,
    }


def _issue_payload(issue: GoldIssue) -> dict[str, Any]:
    return {
        "item_id": issue.item_id,
        "record_id": issue.record_id,
        "reason": issue.reason,
        "details": dict(issue.details),
    }


def _json_bytes(payload: object) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _jsonl_bytes(records: Sequence[GoldRecord]) -> bytes:
    return b"".join(_json_bytes(_record_payload(record)) for record in records)


def write_gold_artifact(
    result: GoldReconstruction,
    output_dir: str | Path,
) -> GoldArtifactPaths:
    """Finalize a validated reconstruction into deterministic artifact files."""

    if result.hard_failures:
        reasons = ", ".join(f"{issue.item_id}: {issue.reason}" for issue in result.hard_failures)
        raise GoldReconstructionError(
            f"refusing to write gold artifact with hard failures: {reasons}"
        )

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "gold_version": _GOLD_VERSION,
        "xml_version": result.xml_version,
        "source_esc": {
            "path": result.source_path.as_posix(),
            "sha256": result.source_sha256,
        },
        "xml_files_sha256": dict(result.xml_files_sha256),
        "evaluation_files_sha256": dict(result.evaluation_files_sha256),
        "summary": dict(result.summary),
    }
    item_resolutions = sorted(
        (_resolution_entry(record) for record in result.records),
        key=lambda entry: (entry["item_id"], entry["record_id"]),
    )
    audit = {
        "hard_failures": [_issue_payload(issue) for issue in result.hard_failures],
        "mismatches": [_issue_payload(issue) for issue in result.mismatches],
        "source_discrepancies": [_issue_payload(issue) for issue in result.source_discrepancies],
        "item_resolutions": item_resolutions,
        "summary": dict(result.summary),
    }

    contents = {
        _GOLD_FILENAME: _jsonl_bytes(result.records),
        _MANIFEST_FILENAME: _json_bytes(manifest),
        _AUDIT_FILENAME: _json_bytes(audit),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for name, payload in contents.items():
        (destination / name).write_bytes(payload)

    artifact_sha256 = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in contents.items()
    }
    return GoldArtifactPaths(
        gold_choice_jsonl=destination / _GOLD_FILENAME,
        manifest_json=destination / _MANIFEST_FILENAME,
        audit_json=destination / _AUDIT_FILENAME,
        artifact_sha256=MappingProxyType(artifact_sha256),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct and finalize the ESC gold_choice artifact.",
    )
    parser.add_argument("--esc-data", required=True, help="Path to ESC JSONL data")
    parser.add_argument("--xml-root", required=True, help="EventStoryLine XML root")
    parser.add_argument("--xml-version", required=True, help="Frozen XML version label")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dataset = load_esc(args.esc_data)
    result = reconstruct_gold(dataset, args.xml_root, args.xml_version)

    for key in (
        "records",
        "causal",
        "none",
        "hard_failures",
        "mismatches",
        "source_discrepancies",
        "direct_xml",
        "coref_expansion",
        "exact_duplicate_collapses",
        "missing_relation_fallbacks",
        "self_coref_evaluation_fallbacks",
    ):
        print(f"{key}={result.summary[key]}")

    for issue in result.mismatches:
        print(f"mismatch item_id={issue.item_id} reason={issue.reason}")
    for issue in result.source_discrepancies:
        print(f"source_discrepancy item_id={issue.item_id} reason={issue.reason}")
    for issue in result.hard_failures:
        print(f"hard_failure item_id={issue.item_id} reason={issue.reason}")

    if result.hard_failures:
        print("refusing to finalize artifact while hard failures exist")
        return 1

    paths = write_gold_artifact(result, args.output_dir)
    print(f"artifact_dir={Path(args.output_dir)}")
    for name, sha256 in sorted(paths.artifact_sha256.items()):
        print(f"artifact_sha256 {name}={sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
