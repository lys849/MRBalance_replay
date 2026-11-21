"""Public contract tests for hybrid ESC direction-gold reconstruction and artifacts.

The tests encode the approved plan amendment (docs/application_plan.md section 3.3)
and ADR 0002: official ``full_corpus/v0.9`` evaluation export locates rows while raw
XML ``PLOT_LINK`` stays the direction authority.  Only approved evaluation fallbacks
may supply relation/direction, and every exception must be explicit in provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from eci_decision_models.data import EscDataset, load_esc
from eci_decision_models.gold import (
    Choice,
    GoldArtifactPaths,
    GoldIssue,
    GoldReconstruction,
    GoldReconstructionError,
    GoldRecord,
    reconstruct_gold,
    write_gold_artifact,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
XML_ROOT = FIXTURES / "EventStoryLine"
EVAL_ROOT = XML_ROOT / "evaluation_format/full_corpus/v0.9"
XML_VERSION = "fixture-v0.9"
ESC_PREFIX = "data/EventStoryLine/"
DEFAULT_XML_VERSION = "46edefee5e82e0917b823abe0a18bf8c7770f15c"

REL_10_1 = "annotated_data/v0.9/10/10_1ecbplus.xml.xml"
REL_10_2 = "annotated_data/v0.9/10/10_2ecbplus.xml.xml"
REL_10_3 = "annotated_data/v0.9/10/10_3ecbplus.xml.xml"
REL_10_4 = "annotated_data/v0.9/10/10_4ecbplus.xml.xml"
REL_11_1 = "annotated_data/v0.9/11/11_1ecbplus.xml.xml"
REL_11_2 = "annotated_data/v0.9/11/11_2ecbplus.xml.xml"
REL_11_3 = "annotated_data/v0.9/11/11_3ecbplus.xml.xml"
REL_11_4 = "annotated_data/v0.9/11/11_4ecbplus.xml.xml"
REL_11_5 = "annotated_data/v0.9/11/11_5ecbplus.xml.xml"
REL_11_6 = "annotated_data/v0.9/11/11_6ecbplus.xml.xml"
REL_11_7 = "annotated_data/v0.9/11/11_7ecbplus.xml.xml"
REL_11_8 = "annotated_data/v0.9/11/11_8ecbplus.xml.xml"
REL_12_1 = "annotated_data/v0.9/12/12_1ecbplus.xml.xml"  # coref expansion
REL_12_3 = "annotated_data/v0.9/12/12_3ecbplus.xml.xml"  # missing raw relType
REL_12_4 = "annotated_data/v0.9/12/12_4ecbplus.xml.xml"  # self-coref fallback
REL_12_7 = "annotated_data/v0.9/12/12_7ecbplus.xml.xml"  # conflicting coref
REL_12_8 = "annotated_data/v0.9/12/12_8ecbplus.xml.xml"  # missing official row
REL_12_11 = "annotated_data/v0.9/12/12_11ecbplus.xml.xml"  # raw/export discrepancy

NONE_REL = "annotated_data/v0.9/99/99_missing.xml.xml"
UNSAFE_REL = "../../escape.xml"

SUMMARY_KEYS = {
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
}


def _doc_name(rel_path: str) -> str:
    return ESC_PREFIX + rel_path


def _topic_of(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if "v0.9" in parts:
        index = parts.index("v0.9")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "fixture"


def _row(
    item_id: int,
    rel_path: str,
    *,
    events_id: tuple[str, str] = ("1", "2"),
    esc_rel: str = "PRECONDITION",
    tri: int = 1,
    bi: int = 1,
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "words": ["Alpha", "triggers", "Gamma", "today"],
        "events": [[0], [2]],
        "tri_causal_label": tri,
        "bi_causal_label": bi,
        "doc_name": _doc_name(rel_path),
        "sent_in_doc_id": 0,
        "events_id": list(events_id),
        "topic": _topic_of(rel_path),
        "ESC_rel": esc_rel,
    }


def _write_esc(path: Path, rows: list[dict[str, object]]) -> EscDataset:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return load_esc(path)


def _one_row_dataset(
    tmp_path: Path,
    *,
    item_id: int,
    rel_path: str,
    esc_rel: str = "PRECONDITION",
    tri: int = 1,
    bi: int = 1,
) -> EscDataset:
    return _write_esc(
        tmp_path / f"esc_{item_id}.jsonl",
        [_row(item_id, rel_path, esc_rel=esc_rel, tri=tri, bi=bi)],
    )


def _xml_sha(rel_path: str) -> str:
    return hashlib.sha256((XML_ROOT / rel_path).read_bytes()).hexdigest()


def _eval_sha(kind: str, rel_path: str) -> str:
    topic = _topic_of(rel_path)
    stem = Path(rel_path).name.removesuffix(".xml").removesuffix(".xml")
    path = EVAL_ROOT / kind / topic / f"{stem}.tab"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_names() -> set[str]:
    return {"gold_choice_reconstructed.jsonl", "manifest.json", "audit.json"}


def _assert_provenance(record: GoldRecord, *, link_ids: list[str]) -> None:
    provenance = record.provenance
    assert isinstance(provenance, Mapping)
    assert provenance["plot_link_ids"] == link_ids
    assert provenance["event1_token"] == "1"
    assert provenance["event2_token"] == "3"
    assert isinstance(provenance["evaluation_row"], Mapping)
    assert isinstance(provenance["resolution"], Mapping)
    assert provenance["resolution"]["method"] == record.resolution_method


def test_choice_enum_has_the_three_stable_values() -> None:
    assert issubclass(Choice, str)
    assert {choice.value for choice in Choice} == {"cause", "caused_by", "none"}
    assert Choice("cause") is Choice.cause
    assert Choice("caused_by") is Choice.caused_by
    assert Choice("none") is Choice.none


def test_none_rows_need_no_xml_or_evaluation_files(tmp_path: Path) -> None:
    dataset = _one_row_dataset(
        tmp_path,
        item_id=1,
        rel_path=NONE_REL,
        esc_rel="UN_LABEL",
        tri=0,
        bi=0,
    )

    result = reconstruct_gold(dataset, tmp_path / "xml-root-does-not-exist", XML_VERSION)

    assert isinstance(result, GoldReconstruction)
    assert result.xml_version == XML_VERSION
    assert result.hard_failures == ()
    assert result.mismatches == ()
    assert result.source_discrepancies == ()
    assert result.xml_files_sha256 == {}
    assert result.evaluation_files_sha256 == {}
    assert set(result.summary) == SUMMARY_KEYS
    assert dict(result.summary) == {
        "records": 1,
        "causal": 0,
        "none": 1,
        "hard_failures": 0,
        "mismatches": 0,
        "source_discrepancies": 0,
        "direct_xml": 0,
        "coref_expansion": 0,
        "exact_duplicate_collapses": 0,
        "missing_relation_fallbacks": 0,
        "self_coref_evaluation_fallbacks": 0,
    }

    assert len(result.records) == 1
    record = result.records[0]
    assert isinstance(record, GoldRecord)
    assert record.item_id == 1
    assert record.choice is Choice.none
    assert record.plot_link_id is None
    assert record.xml_path is None
    assert isinstance(record.relation_source, str)
    assert isinstance(record.direction_source, str)
    assert isinstance(record.resolution_method, str)
    assert record.provenance["doc_name"] == _doc_name(NONE_REL)
    assert record.provenance["events_id"] == ["1", "2"]


@pytest.mark.parametrize(
    (
        "rel_path",
        "link_id",
        "tri",
        "esc_rel",
        "expected_choice",
        "expected_source",
        "expected_target",
    ),
    [
        (REL_10_1, "1001", 1, "PRECONDITION", Choice.cause, "1", "2"),
        (REL_10_2, "1002", 2, "FALLING_ACTION", Choice.caused_by, "2", "1"),
        (REL_10_3, "1003", 1, "PRECONDITION", Choice.cause, "2", "1"),
        (REL_10_4, "1004", 2, "FALLING_ACTION", Choice.caused_by, "1", "2"),
    ],
)
def test_direct_unique_raw_link_uses_raw_direction_and_reltype(
    tmp_path: Path,
    rel_path: str,
    link_id: str,
    tri: int,
    esc_rel: str,
    expected_choice: Choice,
    expected_source: str,
    expected_target: str,
) -> None:
    dataset = _one_row_dataset(
        tmp_path,
        item_id=2,
        rel_path=rel_path,
        esc_rel=esc_rel,
        tri=tri,
        bi=1,
    )

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert result.hard_failures == ()
    assert result.source_discrepancies == ()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.choice is expected_choice
    assert record.plot_link_id == link_id
    assert record.plot_link_source == expected_source
    assert record.plot_link_target == expected_target
    assert record.xml_path == rel_path
    assert record.xml_sha256 == _xml_sha(rel_path)
    assert record.xml_version == XML_VERSION
    assert record.relation_source == "raw_xml"
    assert record.direction_source == "raw_xml"
    assert record.resolution_method == "direct_xml"

    assert result.xml_files_sha256 == {rel_path: _xml_sha(rel_path)}
    assert result.evaluation_files_sha256
    assert _eval_sha("event_mentions_extended", rel_path) in set(
        result.evaluation_files_sha256.values()
    )
    _assert_provenance(record, link_ids=[link_id])

    assert result.mismatches == ()
    assert dict(result.summary)["direct_xml"] == 1
    assert dict(result.summary)["mismatches"] == 0


def test_exact_duplicate_direct_links_collapse_but_keep_all_rids(tmp_path: Path) -> None:
    dataset = _one_row_dataset(tmp_path, item_id=3, rel_path=REL_11_2)

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert result.hard_failures == ()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.choice is Choice.cause
    assert record.resolution_method == "exact_duplicate_collapse"
    assert record.relation_source == "raw_xml"
    assert record.direction_source == "raw_xml"
    assert record.plot_link_source == "1"
    assert record.plot_link_target == "2"
    _assert_provenance(record, link_ids=["1102", "1103"])
    assert set(_provenance_rids(record)) == {"1102", "1103"}
    assert dict(result.summary)["exact_duplicate_collapses"] == 1


def _provenance_rids(record: GoldRecord) -> list[str]:
    ids = record.provenance["plot_link_ids"]
    assert isinstance(ids, list)
    return [str(item) for item in ids]


def test_coref_expansion_uses_raw_parent_link_direction(tmp_path: Path) -> None:
    dataset = _one_row_dataset(tmp_path, item_id=4, rel_path=REL_12_1)

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert result.hard_failures == ()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.choice is Choice.cause
    assert record.resolution_method == "coref_expansion"
    assert record.relation_source == "raw_xml"
    assert record.direction_source == "raw_xml"
    assert record.plot_link_id == "1201"
    assert record.plot_link_source == "5"
    assert record.plot_link_target == "6"
    _assert_provenance(record, link_ids=["1201"])
    assert dict(result.summary)["coref_expansion"] == 1
    assert set(result.evaluation_files_sha256.values()) >= {
        _eval_sha("coref_chain", REL_12_1),
        _eval_sha("event_mentions_extended", REL_12_1),
    }


def test_missing_relation_fallback_uses_evaluation_relation(tmp_path: Path) -> None:
    dataset = _one_row_dataset(tmp_path, item_id=9775, rel_path=REL_12_3)

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert result.hard_failures == ()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.choice is Choice.cause
    assert record.resolution_method == "missing_relation_fallback"
    assert record.relation_source == "evaluation_format"
    assert record.direction_source == "mixed"
    _assert_provenance(record, link_ids=["1203"])
    assert dict(result.summary)["missing_relation_fallbacks"] == 1


def test_self_coref_evaluation_fallback_selects_direction(tmp_path: Path) -> None:
    dataset = _one_row_dataset(tmp_path, item_id=2531, rel_path=REL_12_4)

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert result.hard_failures == ()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.choice is Choice.cause
    assert record.resolution_method == "self_coref_evaluation_fallback"
    assert record.direction_source == "evaluation_format"
    assert dict(result.summary)["self_coref_evaluation_fallbacks"] == 1


def test_source_discrepancy_keeps_raw_choice_and_is_audited(tmp_path: Path) -> None:
    dataset = _one_row_dataset(tmp_path, item_id=7, rel_path=REL_12_11)

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert result.hard_failures == ()
    assert len(result.records) == 1
    record = result.records[0]
    # raw FALLING_ACTION with source=event1 wins over the export PRECONDITION.
    assert record.choice is Choice.caused_by
    assert record.relation_source == "raw_xml"
    assert record.direction_source == "raw_xml"
    assert record.resolution_method == "direct_xml"

    assert len(result.source_discrepancies) == 1
    discrepancy = result.source_discrepancies[0]
    assert isinstance(discrepancy, GoldIssue)
    assert discrepancy.item_id == 7
    assert discrepancy.reason

    assert {issue.item_id for issue in result.mismatches} == {7}
    assert dict(result.summary)["source_discrepancies"] == 1
    assert dict(result.summary)["mismatches"] == 1


@pytest.mark.parametrize(
    ("rel_path", "esc_rel", "tri"),
    [
        (REL_11_1, "PRECONDITION", 1),
        (REL_11_3, "PRECONDITION", 1),
        (REL_11_4, "PRECONDITION", 1),
        (REL_11_5, "PRECONDITION", 1),
        (REL_11_6, "FALLING_ACTION", 2),
        (REL_11_7, "PRECONDITION", 1),
        (REL_11_8, "PRECONDITION", 1),
        (REL_12_7, "PRECONDITION", 1),
        (REL_12_8, "PRECONDITION", 1),
        (UNSAFE_REL, "PRECONDITION", 1),
    ],
    ids=[
        "missing_link",
        "bidirectional",
        "conflicting_reltype",
        "unknown_reltype",
        "esc_rel_mismatch_to_evaluation",
        "missing_markable",
        "non_event_markable",
        "conflicting_coref_expansion",
        "missing_official_row",
        "unsafe_path",
    ],
)
def test_unusable_or_conflicting_rows_are_explicit_hard_failures(
    tmp_path: Path,
    rel_path: str,
    esc_rel: str,
    tri: int,
) -> None:
    dataset = _one_row_dataset(
        tmp_path,
        item_id=300,
        rel_path=rel_path,
        esc_rel=esc_rel,
        tri=tri,
        bi=1,
    )

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert result.records == ()
    assert len(result.hard_failures) == 1
    issue = result.hard_failures[0]
    assert isinstance(issue, GoldIssue)
    assert issue.item_id == 300
    assert issue.record_id == dataset.records[0].record_id
    assert isinstance(issue.reason, str) and issue.reason
    assert isinstance(issue.details, Mapping)
    assert dict(result.summary)["hard_failures"] == 1
    assert dict(result.summary)["records"] == 1


@pytest.mark.parametrize(
    ("rel_path", "esc_rel", "tri", "bi"),
    [
        (REL_10_1, "PRECONDITION", 1, 0),
        (NONE_REL, "UN_LABEL", 0, 1),
        (REL_10_1, "PRECONDITION", 0, 0),
        (REL_10_1, "UN_LABEL", 1, 1),
        (NONE_REL, "UN_LABEL", 2, 1),
        (REL_10_1, "PRECONDITION", 2, 1),
    ],
    ids=[
        "causal_with_bi_zero",
        "none_with_bi_one",
        "tri_zero_with_precondition",
        "tri_one_with_un_label",
        "tri_two_with_un_label",
        "tri_two_with_precondition",
    ],
)
def test_row_label_consistency_violations_are_hard_failures(
    tmp_path: Path,
    rel_path: str,
    esc_rel: str,
    tri: int,
    bi: int,
) -> None:
    dataset = _one_row_dataset(
        tmp_path,
        item_id=400,
        rel_path=rel_path,
        esc_rel=esc_rel,
        tri=tri,
        bi=bi,
    )

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert result.records == ()
    assert len(result.hard_failures) == 1
    issue = result.hard_failures[0]
    assert issue.item_id == 400
    assert issue.reason
    assert dict(result.summary)["hard_failures"] == 1


def test_mixed_hybrid_dataset_summary_methods_and_choices(tmp_path: Path) -> None:
    rows = [
        _row(1001, REL_10_1),
        _row(1002, REL_10_2, esc_rel="FALLING_ACTION", tri=2),
        _row(1003, REL_11_2),
        _row(1004, REL_12_1),
        _row(9775, REL_12_3),
        _row(2531, REL_12_4),
        _row(1007, REL_12_11),
        _row(1008, NONE_REL, esc_rel="UN_LABEL", tri=0, bi=0),
    ]
    dataset = _write_esc(tmp_path / "mixed_hybrid.jsonl", rows)

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert result.hard_failures == ()
    assert [record.item_id for record in result.records] == [
        1001,
        1002,
        1003,
        1004,
        9775,
        2531,
        1007,
        1008,
    ]
    assert [record.choice for record in result.records] == [
        Choice.cause,
        Choice.caused_by,
        Choice.cause,
        Choice.cause,
        Choice.cause,
        Choice.cause,
        Choice.caused_by,
        Choice.none,
    ]
    methods = {record.item_id: record.resolution_method for record in result.records}
    assert methods[1001] == "direct_xml"
    assert methods[1002] == "direct_xml"
    assert methods[1003] == "exact_duplicate_collapse"
    assert methods[1004] == "coref_expansion"
    assert methods[9775] == "missing_relation_fallback"
    assert methods[2531] == "self_coref_evaluation_fallback"
    assert methods[1007] == "direct_xml"
    assert isinstance(methods[1008], str)

    assert set(result.summary) == SUMMARY_KEYS
    assert dict(result.summary) == {
        "records": 8,
        "causal": 7,
        "none": 1,
        "hard_failures": 0,
        "mismatches": 1,
        "source_discrepancies": 1,
        "direct_xml": 3,
        "coref_expansion": 1,
        "exact_duplicate_collapses": 1,
        "missing_relation_fallbacks": 1,
        "self_coref_evaluation_fallbacks": 1,
    }
    assert {issue.item_id for issue in result.mismatches} == {1007}
    assert {issue.item_id for issue in result.source_discrepancies} == {1007}
    assert result.evaluation_files_sha256
    assert isinstance(result.source_discrepancies, tuple)


def test_every_input_row_is_accounted_for_when_some_rows_hard_fail(tmp_path: Path) -> None:
    rows = [
        _row(501, REL_10_1),
        _row(502, NONE_REL, esc_rel="UN_LABEL", tri=0, bi=0),
        _row(503, REL_11_1),
    ]
    dataset = _write_esc(tmp_path / "partial.jsonl", rows)

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert {record.item_id for record in result.records} == {501, 502}
    assert {issue.item_id for issue in result.hard_failures} == {503}
    assert (
        {record.item_id for record in result.records}
        | {issue.item_id for issue in result.hard_failures}
    ) == {record.item_id for record in dataset.records}
    assert dict(result.summary) == {
        "records": 3,
        "causal": 2,
        "none": 1,
        "hard_failures": 1,
        "mismatches": 0,
        "source_discrepancies": 0,
        "direct_xml": 1,
        "coref_expansion": 0,
        "exact_duplicate_collapses": 0,
        "missing_relation_fallbacks": 0,
        "self_coref_evaluation_fallbacks": 0,
    }


def test_public_gold_dataclasses_are_frozen(tmp_path: Path) -> None:
    dataset = _one_row_dataset(tmp_path, item_id=601, rel_path=REL_10_1)
    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)
    record = result.records[0]

    with pytest.raises(FrozenInstanceError):
        record.__setattr__("item_id", 999)

    with pytest.raises(FrozenInstanceError):
        result.__setattr__("records", ())

    with pytest.raises(FrozenInstanceError):
        result.__setattr__("source_discrepancies", ())

    failing = reconstruct_gold(
        _one_row_dataset(tmp_path, item_id=602, rel_path=REL_11_1),
        XML_ROOT,
        XML_VERSION,
    )
    issue = failing.hard_failures[0]
    with pytest.raises(FrozenInstanceError):
        issue.__setattr__("reason", "changed")


def test_write_gold_artifact_is_deterministic_and_records_hybrid_provenance(
    tmp_path: Path,
) -> None:
    rows = [
        _row(701, REL_10_1),
        _row(702, NONE_REL, esc_rel="UN_LABEL", tri=0, bi=0),
        _row(703, REL_11_2),
        _row(704, REL_12_1),
        _row(705, REL_12_11),
    ]
    dataset = _write_esc(tmp_path / "artifact_source.jsonl", rows)
    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    first_dir = tmp_path / "gold-first"
    second_dir = tmp_path / "gold-second"
    first = write_gold_artifact(result, first_dir)
    second = write_gold_artifact(result, second_dir)

    assert isinstance(first, GoldArtifactPaths)
    assert first.gold_choice_jsonl == first_dir / "gold_choice_reconstructed.jsonl"
    assert first.manifest_json == first_dir / "manifest.json"
    assert first.audit_json == first_dir / "audit.json"
    assert set(first.artifact_sha256) == _artifact_names()
    with pytest.raises(FrozenInstanceError):
        first.__setattr__("gold_choice_jsonl", Path("elsewhere.jsonl"))

    for name in _artifact_names():
        first_bytes = (first_dir / name).read_bytes()
        assert hashlib.sha256(first_bytes).hexdigest() == first.artifact_sha256[name]
        assert first_bytes == (second_dir / name).read_bytes()
    assert first.artifact_sha256 == second.artifact_sha256

    manifest = json.loads(first.manifest_json.read_text(encoding="utf-8"))
    assert manifest["schema_version"]
    assert manifest["gold_version"]
    assert manifest["xml_version"] == XML_VERSION
    assert manifest["source_esc"]["sha256"] == dataset.source_sha256
    assert manifest["xml_files_sha256"] == dict(result.xml_files_sha256)
    assert manifest["evaluation_files_sha256"] == dict(result.evaluation_files_sha256)
    assert manifest["evaluation_files_sha256"]
    assert manifest["summary"] == dict(result.summary)

    audit = json.loads(first.audit_json.read_text(encoding="utf-8"))
    assert audit["hard_failures"] == []
    assert {entry["item_id"] for entry in audit["mismatches"]} == {705}
    assert {entry["item_id"] for entry in audit["source_discrepancies"]} == {705}
    assert audit["summary"] == dict(result.summary)

    gold_lines = [
        json.loads(line)
        for line in first.gold_choice_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [line["item_id"] for line in gold_lines] == [701, 702, 703, 704, 705]
    for line in gold_lines[:1] + gold_lines[2:]:
        assert line["relation_source"] in {"raw_xml", "evaluation_format"}
        assert line["direction_source"] in {"raw_xml", "evaluation_format", "mixed"}
        assert isinstance(line["resolution_method"], str)
    direct = gold_lines[0]
    assert direct["choice"] == "cause"
    assert direct["resolution_method"] == "direct_xml"
    assert direct["provenance"]["plot_link_ids"] == ["1001"]
    duplicate = gold_lines[2]
    assert duplicate["resolution_method"] == "exact_duplicate_collapse"
    assert set(duplicate["provenance"]["plot_link_ids"]) == {"1102", "1103"}
    discrepancy = gold_lines[4]
    assert discrepancy["choice"] == "caused_by"
    assert discrepancy["relation_source"] == "raw_xml"


def test_finalization_refuses_to_write_when_hard_failures_exist(tmp_path: Path) -> None:
    dataset = _one_row_dataset(tmp_path, item_id=801, rel_path=REL_11_1)
    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)
    assert result.hard_failures

    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(GoldReconstructionError):
        write_gold_artifact(result, output_dir)

    for name in _artifact_names():
        assert not (output_dir / name).exists()


def test_gold_reconstruction_uses_tuple_collections(tmp_path: Path) -> None:
    dataset = _one_row_dataset(tmp_path, item_id=901, rel_path=REL_10_1)
    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)

    assert isinstance(result, GoldReconstruction)
    assert isinstance(result.records, tuple)
    assert isinstance(result.hard_failures, tuple)
    assert isinstance(result.mismatches, tuple)
    assert isinstance(result.source_discrepancies, tuple)
    assert isinstance(result.evaluation_files_sha256, Mapping)


def test_real_esc_data_reconstructs_all_rows_with_hybrid_provenance(tmp_path: Path) -> None:
    raw_esc = os.environ.get("ECI_ESC_DATA")
    raw_xml_root = os.environ.get("ECI_EVENTSTORYLINE_ROOT")
    if not raw_esc or not raw_xml_root:
        pytest.skip("ECI_ESC_DATA and ECI_EVENTSTORYLINE_ROOT must both be set")

    xml_version = os.environ.get("ECI_EVENTSTORYLINE_VERSION", DEFAULT_XML_VERSION)
    dataset = load_esc(raw_esc)
    result = reconstruct_gold(dataset, raw_xml_root, xml_version)

    assert len(dataset.records) == 1600
    assert result.xml_version == xml_version
    assert len(result.records) + len(result.hard_failures) == 1600
    assert result.hard_failures == ()
    assert len(result.records) == 1600
    assert len({record.record_id for record in result.records}) == 1600
    assert set(result.summary) == SUMMARY_KEYS
    assert dict(result.summary)["records"] == 1600
    assert dict(result.summary)["causal"] == 500
    assert dict(result.summary)["none"] == 1100
    assert dict(result.summary)["hard_failures"] == 0
    assert dict(result.summary)["direct_xml"] == 448
    assert dict(result.summary)["coref_expansion"] == 46
    assert dict(result.summary)["exact_duplicate_collapses"] == 2
    assert dict(result.summary)["missing_relation_fallbacks"] == 1
    assert dict(result.summary)["self_coref_evaluation_fallbacks"] == 3
    assert dict(result.summary)["mismatches"] == 6
    assert dict(result.summary)["source_discrepancies"] == 6

    assert result.xml_files_sha256
    assert result.evaluation_files_sha256

    by_item_id = {record.item_id: record for record in result.records}
    assert by_item_id[4713].choice is Choice.cause
    assert by_item_id[4713].relation_source == "raw_xml"
    assert by_item_id[4713].direction_source == "raw_xml"
    assert by_item_id[4808].choice is Choice.cause
    assert by_item_id[4808].relation_source == "raw_xml"
    assert by_item_id[4808].direction_source == "raw_xml"

    assert by_item_id[9775].choice is Choice.cause
    assert by_item_id[9775].relation_source == "evaluation_format"
    assert by_item_id[9775].direction_source == "mixed"

    assert by_item_id[2531].choice is Choice.caused_by
    assert by_item_id[2531].direction_source == "evaluation_format"
    assert by_item_id[10592].choice is Choice.caused_by
    assert by_item_id[10592].direction_source == "evaluation_format"
    assert by_item_id[12487].choice is Choice.cause
    assert by_item_id[12487].direction_source == "evaluation_format"

    assert {issue.item_id for issue in result.mismatches} == {
        4713,
        9795,
        19833,
        19868,
        468,
        8984,
    }
    assert {issue.item_id for issue in result.source_discrepancies} == {
        4713,
        9795,
        19833,
        19868,
        468,
        8984,
    }

    expected_choices = {"cause": 0, "caused_by": 0, "none": 0}
    for record in result.records:
        expected_choices[record.choice.value] += 1
    assert expected_choices == {"cause": 244, "caused_by": 256, "none": 1100}

    output_dir = tmp_path / "real-gold"
    paths = write_gold_artifact(result, output_dir)
    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["evaluation_files_sha256"] == dict(result.evaluation_files_sha256)
    audit = json.loads(paths.audit_json.read_text(encoding="utf-8"))
    assert {entry["item_id"] for entry in audit["source_discrepancies"]} == {
        4713,
        9795,
        19833,
        19868,
        468,
        8984,
    }
