"""Contracts for the EventStoryLine intra-sentence pool and fold-0 builder.

The synthetic fixture exercises direct cause, direct caused_by (with source /
target reversal), coreference expansion, a non-causal pair, an excluded event
tag, and a conflicting-coref ambiguity.  The real-corpus test is skipped unless
``ECI_RUN_REAL_FINETUNE_DATA=1`` is set.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from eci_decision_models.finetune_data import (
    CHOICE_KEYS,
    EXCLUDED_TAG_FRAGMENTS,
    EXPECTED_VALUES,
    NAMESPACE,
    STATE_KEYS,
    PairRecord,
    PoolResult,
    build_datasets,
    build_pool,
    laya_row,
    plan_gao_folds,
    published_count_check,
    run_build,
    split_documents,
    validate_laya_row,
    verify_fold0_balance,
    write_fold_outputs,
    write_folds_index,
)

REAL_DATA_ENV = "ECI_RUN_REAL_FINETUNE_DATA"

SYNTHETIC_XML = """<?xml version='1.0' encoding='UTF-8'?>
<Document doc_name="99_synth.xml">
  <token t_id="1" sentence="0" number="0">alpha</token>
  <token t_id="2" sentence="0" number="1">beta</token>
  <token t_id="3" sentence="0" number="2">gamma</token>
  <token t_id="4" sentence="0" number="3">delta</token>
  <token t_id="5" sentence="0" number="4">epsilon</token>
  <token t_id="6" sentence="0" number="5">zeta</token>
  <token t_id="7" sentence="0" number="6">eta</token>
  <token t_id="8" sentence="0" number="7">theta</token>
  <token t_id="9" sentence="0" number="8">iota</token>
  <token t_id="10" sentence="0" number="9">kappa</token>
  <token t_id="11" sentence="0" number="10">lambda</token>
  <Markables>
    <ACTION_OCCURRENCE m_id="1"><token_anchor t_id="1"/></ACTION_OCCURRENCE>
    <ACTION_OCCURRENCE m_id="2"><token_anchor t_id="2"/></ACTION_OCCURRENCE>
    <ACTION_OCCURRENCE m_id="3"><token_anchor t_id="3"/></ACTION_OCCURRENCE>
    <ACTION_OCCURRENCE m_id="4"><token_anchor t_id="4"/></ACTION_OCCURRENCE>
    <ACTION_OCCURRENCE m_id="5"><token_anchor t_id="5"/></ACTION_OCCURRENCE>
    <ACTION_OCCURRENCE m_id="6"><token_anchor t_id="6"/></ACTION_OCCURRENCE>
    <ACTION_ASPECTUAL m_id="7"><token_anchor t_id="7"/></ACTION_ASPECTUAL>
    <ACTION_OCCURRENCE m_id="8"><token_anchor t_id="8"/></ACTION_OCCURRENCE>
    <ACTION_OCCURRENCE m_id="9"><token_anchor t_id="9"/></ACTION_OCCURRENCE>
  </Markables>
</Document>
"""

DIRECT_TAB = "1\t2\tPRECONDITION\n4\t3\tPRECONDITION\n"
COREF_TAB = "5 11\t6\tPRECONDITION\n8\t9\tPRECONDITION\n8\t9\tFALLING_ACTION\n"


def _write_synthetic_corpus(
    tmp_path: Path,
    *,
    include_dev: bool = False,
) -> tuple[Path, Path]:
    xml_root = tmp_path / "annotated_data"
    eval_root = tmp_path / "evaluation_format"
    topics = ["99", "37"] if include_dev else ["99"]
    for topic in topics:
        doc_dir = xml_root / topic
        doc_dir.mkdir(parents=True)
        (doc_dir / f"{topic}_synth.xml.xml").write_text(SYNTHETIC_XML, encoding="utf-8")

        direct_dir = eval_root / "event_mentions_extended" / topic
        direct_dir.mkdir(parents=True)
        (direct_dir / f"{topic}_synth.tab").write_text(DIRECT_TAB, encoding="utf-8")

        coref_dir = eval_root / "coref_chain" / topic
        coref_dir.mkdir(parents=True)
        (coref_dir / f"{topic}_synth.tab").write_text(COREF_TAB, encoding="utf-8")
    return xml_root, eval_root


def _pair_lookup(pool: PoolResult) -> dict[tuple[tuple[int, ...], tuple[int, ...]], PairRecord]:
    return {(pair.event1_tokens, pair.event2_tokens): pair for pair in pool.pairs}


def _assert_laya_schema(row: dict[str, object]) -> None:
    assert set(row) == {"record_id", "state", "questions", "expected"}
    assert isinstance(row["record_id"], str)
    assert row["record_id"]
    state = row["state"]
    assert isinstance(state, dict)
    assert set(state) == STATE_KEYS
    questions = row["questions"]
    assert isinstance(questions, dict)
    assert set(questions) == {"eci_relation"}
    question = questions["eci_relation"]
    assert isinstance(question, dict)
    assert set(question) == {"type", "instructions", "criteria"}
    assert question["type"] == "choice"
    criteria = question["criteria"]
    assert isinstance(criteria, dict)
    assert set(criteria) == CHOICE_KEYS
    expected = row["expected"]
    assert isinstance(expected, dict)
    assert set(expected) == {"eci_relation"}
    assert expected["eci_relation"] in CHOICE_KEYS
    assert expected["eci_relation"] in EXPECTED_VALUES


def test_synthetic_pool_covers_every_labelling_path(tmp_path: Path) -> None:
    xml_root, eval_root = _write_synthetic_corpus(tmp_path)
    pool = build_pool(xml_root, eval_root)
    pairs = _pair_lookup(pool)

    # 8 included events in one sentence -> 28 enumerated, 1 ambiguous excluded.
    assert pool.summary["enumerated_pairs"] == 28
    assert pool.summary["total_pairs"] == 27
    assert pool.summary["ambiguous_count"] == 1
    assert pool.summary["causal_count"] == 3
    assert pool.summary["none_count"] == 24
    assert pool.summary["direct_count"] == 2
    assert pool.summary["coref_count"] == 1
    assert pool.summary["excluded_event_count"] == 1
    assert pool.summary["docs_processed"] == 1
    assert pool.summary["skipped_no_anchor_count"] == 0
    assert pool.summary["skipped_multi_sentence_count"] == 0

    cause = pairs[((1,), (2,))]
    assert cause.label == "cause"
    assert cause.provenance == "direct"
    assert cause.relation == "PRECONDITION"

    caused_by = pairs[((3,), (4,))]
    assert caused_by.label == "caused_by"
    assert caused_by.provenance == "direct"
    assert caused_by.relation == "FALLING_ACTION"

    coref = pairs[((5,), (6,))]
    assert coref.label == "cause"
    assert coref.provenance == "coref"
    assert coref.relation == "PRECONDITION"

    non_causal = pairs[((1,), (3,))]
    assert non_causal.label == "none"
    assert non_causal.provenance == "none"
    assert non_causal.relation is None

    # The ASPECTUAL mention is excluded entirely, and the conflicting coref
    # pair is removed from the pool.
    assert not any((7,) in key for key in pairs)
    assert ((8,), (9,)) not in pairs

    assert cause.event1 == "alpha"
    assert cause.event2 == "beta"
    assert cause.sentence == "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    assert cause.doc_name == "99/99_synth.xml.xml"
    assert cause.topic == "99"
    assert cause.sent_in_doc_id == 0
    assert cause.record_id.startswith(NAMESPACE)
    assert len({pair.record_id for pair in pool.pairs}) == len(pool.pairs)

    topic_counts = cast(
        "Mapping[str, Mapping[str, int]]",
        pool.summary["per_topic_counts"],
    )
    assert dict(topic_counts["99"]) == {
        "pairs": 27,
        "causal": 3,
        "none": 24,
        "ambiguous": 1,
    }


def test_synthetic_writes_schema_valid_laya_jsonl(tmp_path: Path) -> None:
    xml_root, eval_root = _write_synthetic_corpus(tmp_path)
    outputs = run_build(
        xml_root,
        eval_root,
        tmp_path / "artifacts",
        tmp_path / "experiments",
        enforce_counts=False,
    )

    for name in ("train", "test", "dev"):
        rows = [
            json.loads(line)
            for line in outputs[name].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            _assert_laya_schema(row)

    test_rows = [
        json.loads(line)
        for line in outputs["test"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # A single CV document always lands in fold-0 test.
    assert len(test_rows) == 27
    assert {row["expected"]["eci_relation"] for row in test_rows} == {
        "cause",
        "caused_by",
        "none",
    }

    metadata = [
        json.loads(line)
        for line in outputs["test_pairs"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metadata) == len(test_rows)
    assert len({entry["record_id"] for entry in metadata}) == len(metadata)
    assert all(entry["split"] == "test" for entry in metadata)
    assert {row["record_id"] for row in test_rows} == {entry["record_id"] for entry in metadata}
    assert {entry["sentence"] for entry in metadata} == {
        row["state"]["sentence"] for row in test_rows
    }
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()

    # A manually built row must also pass the validator.
    sample = PairRecord(
        record_id=NAMESPACE + "0" * 64,
        doc_name="99/99_synth.xml.xml",
        topic="99",
        sent_in_doc_id=0,
        sentence="alpha beta",
        event1="alpha",
        event2="beta",
        event1_tokens=(1,),
        event2_tokens=(2,),
        label="cause",
        provenance="direct",
        relation="PRECONDITION",
    )
    validate_laya_row(laya_row(sample))


def test_test_and_dev_rows_join_to_their_pair_metadata(tmp_path: Path) -> None:
    xml_root, eval_root = _write_synthetic_corpus(tmp_path, include_dev=True)
    outputs = run_build(
        xml_root,
        eval_root,
        tmp_path / "artifacts",
        tmp_path / "experiments",
        enforce_counts=False,
    )

    for split_name in ("test", "dev"):
        laya_rows = [
            json.loads(line)
            for line in outputs[split_name].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pair_rows = [
            json.loads(line)
            for line in outputs[f"{split_name}_pairs"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert laya_rows
        assert pair_rows
        assert {row["record_id"] for row in laya_rows} == {
            entry["record_id"] for entry in pair_rows
        }
        assert len({row["record_id"] for row in laya_rows}) == len(laya_rows)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_real_data_gao_folds(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    xml_root = project_root / "data/EventStoryLine/annotated_data/v0.9"
    eval_root = project_root / "data/EventStoryLine/evaluation_format/full_corpus/v0.9"
    reference = project_root / "experiments/pilot-fold0/data/train-balanced.jsonl"
    if not (xml_root.exists() and eval_root.exists() and reference.exists()):
        pytest.skip("real EventStoryLine data or fold-0 reference is not available")

    pool = build_pool(xml_root, eval_root)
    plans = plan_gao_folds(pool)

    test_sets = [set(plan.split.test_docs) for plan in plans]
    cv = set(plans[0].split.cv_docs)
    assert len(cv) == 233
    assert set().union(*test_sets) == cv
    for left in range(len(test_sets)):
        for right in range(left + 1, len(test_sets)):
            assert not (test_sets[left] & test_sets[right])

    all_pairs = pool.pairs
    for plan in plans:
        fold = plan.split.fold
        train_docs = set(plan.split.train_docs)
        test_docs = set(plan.split.test_docs)
        assert not (train_docs & test_docs)
        assert train_docs | test_docs == cv

        causal_train = {
            pair.record_id
            for pair in all_pairs
            if pair.doc_name in train_docs and pair.label != "none"
        }
        train_ids = {pair.record_id for pair in plan.train_pairs}
        train_none = sum(1 for pair in plan.train_pairs if pair.label == "none")
        train_causal = len(plan.train_pairs) - train_none
        assert causal_train <= train_ids
        assert train_causal == len(causal_train)
        assert train_none == 2 * train_causal

        expected_test = {pair.record_id for pair in all_pairs if pair.doc_name in test_docs}
        assert {pair.record_id for pair in plan.test_pairs} == expected_test
        print(
            f"fold{fold} train_docs={len(train_docs)} test_docs={len(test_docs)} "
            f"train_pairs={len(plan.train_pairs)} train_none={train_none} "
            f"train_causal={train_causal} test_pairs={len(plan.test_pairs)}"
        )

    reference_ids = [row["record_id"] for row in _read_jsonl(reference)]
    fold0_ids = [laya_row(pair)["record_id"] for pair in plans[0].train_pairs]
    assert fold0_ids == reference_ids
    verify_fold0_balance(plans[0], reference)

    experiments_dir = tmp_path / "experiments"
    entries = [write_fold_outputs(plan, experiments_dir) for plan in plans]
    folds_path = write_folds_index(entries, experiments_dir, cv_doc_count=len(cv))
    folds_payload = json.loads(folds_path.read_text(encoding="utf-8"))
    assert set(folds_payload["folds"]) == {"0", "1", "2", "3", "4"}

    for plan in plans:
        fold_dir = experiments_dir / f"fold{plan.split.fold}"
        train_rows = _read_jsonl(fold_dir / "train.jsonl")
        test_rows = _read_jsonl(fold_dir / "test.jsonl")
        test_pairs = _read_jsonl(fold_dir / "test_pairs.jsonl")
        assert (fold_dir / "manifest.json").exists()
        for row in (*train_rows, *test_rows):
            _assert_laya_schema(row)
        assert len(train_rows) == len(plan.train_pairs)
        assert len(test_rows) == len(plan.test_pairs)
        assert {row["record_id"] for row in test_rows} == {
            entry["record_id"] for entry in test_pairs
        }

    written_fold0_ids = [
        row["record_id"] for row in _read_jsonl(experiments_dir / "fold0" / "train.jsonl")
    ]
    assert written_fold0_ids == reference_ids


def test_real_data_counts_and_fold0_sizes(capsys: pytest.CaptureFixture[str]) -> None:
    if os.environ.get(REAL_DATA_ENV) != "1":
        pytest.skip(f"set {REAL_DATA_ENV}=1 to run the real-corpus check")

    project_root = Path(__file__).resolve().parents[1]
    xml_root = project_root / "data/EventStoryLine/annotated_data/v0.9"
    eval_root = project_root / "data/EventStoryLine/evaluation_format/full_corpus/v0.9"
    pool = build_pool(xml_root, eval_root)

    check = published_count_check(pool.summary)
    split = split_documents(pool.doc_names)
    datasets = build_datasets(pool, split)

    print(
        "real_pool "
        f"total={pool.summary['total_pairs']} "
        f"causal={pool.summary['causal_count']} "
        f"none={pool.summary['none_count']} "
        f"direct={pool.summary['direct_count']} "
        f"coref={pool.summary['coref_count']} "
        f"ambiguous={pool.summary['ambiguous_count']} "
        f"excluded_events={pool.summary['excluded_event_count']}"
    )
    print(
        "fold0_docs "
        f"train={len(split.train_docs)} "
        f"test={len(split.test_docs)} "
        f"dev={len(split.dev_docs)}"
    )
    print(
        f"fold0_pairs train={len(datasets.train)} test={len(datasets.test)} dev={len(datasets.dev)}"
    )
    topic_counts = cast(
        "Mapping[str, Mapping[str, int]]",
        pool.summary["per_topic_counts"],
    )
    for topic, stats in topic_counts.items():
        print(f"topic {topic}: {dict(stats)}")

    assert pool.summary["docs_processed"] == 258
    assert pool.summary["total_pairs"] == 7805
    assert pool.summary["causal_count"] == 1740
    assert check["total_passes"] is True
    assert check["causal_passes"] is True

    assert len(split.test_docs) == 47
    assert len(split.train_docs) == 186
    assert len(split.dev_docs) == 25
    assert len(datasets.test) == 1645
    assert len(datasets.train) == 5350
    assert len(datasets.dev) == 810
    assert len(datasets.test) + len(datasets.train) + len(datasets.dev) == 7805


def test_exclusion_fragments_match_the_specification() -> None:
    assert EXCLUDED_TAG_FRAGMENTS == (
        "ASPECTUAL",
        "CAUSATIVE",
        "PERCEPTION",
        "REPORTING",
    )
