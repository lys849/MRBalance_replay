"""Phase 1 remediation contracts: shared Choice taxonomy and portable manifest path."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from eci_decision_models.artifacts import PredictionRecord, redact_secrets
from eci_decision_models.canonical import CanonicalRequest, build_canonical_request
from eci_decision_models.data import EscDataset, load_esc
from eci_decision_models.gold import GoldReconstruction, reconstruct_gold, write_gold_artifact
from eci_decision_models.metrics import compute_probability_metrics
from eci_decision_models.probability import (
    InvalidProbabilityError,
    PredictionStatus,
    validate_prediction,
    validate_probabilities,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
XML_ROOT = FIXTURES / "EventStoryLine"
XML_VERSION = "fixture-v0.9"
ESC_PREFIX = "data/EventStoryLine/"
EVAL_PREFIX = "evaluation_format/full_corpus/v0.9"
RELATIVE_REL_PATH = "annotated_data/v0.9/50/50_1ecbplus.xml.xml"
REDACTED = "[REDACTED]"


def _simple_valid_row() -> dict[str, object]:
    return {
        "item_id": 1,
        "words": ["Alpha", "triggers", "Gamma", "today"],
        "events": [[0], [2]],
        "tri_causal_label": 1,
        "bi_causal_label": 1,
        "doc_name": "data/EventStoryLine/annotated_data/v0.9/10/10_1ecbplus.xml.xml",
        "sent_in_doc_id": 0,
        "events_id": ["1", "2"],
        "topic": "10",
        "ESC_rel": "PRECONDITION",
    }


def test_choice_taxonomy_is_shared_between_canonical_and_gold() -> None:
    from eci_decision_models import canonical, gold

    assert issubclass(canonical.Choice, str)
    assert [member.value for member in canonical.Choice] == ["cause", "caused_by", "none"]
    assert canonical.CHOICE_IDS == ("cause", "caused_by", "none")
    assert tuple(member.value for member in canonical.Choice) == canonical.CHOICE_IDS
    assert gold.Choice is canonical.Choice


def test_gold_manifest_preserves_relative_source_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_esc = Path("esc_relative.jsonl")
    relative_esc.write_text(
        json.dumps(_simple_valid_row(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    dataset = load_esc(relative_esc)
    assert dataset.source_path == relative_esc
    assert dataset.source_path.is_absolute() is False

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)
    assert result.hard_failures == ()

    paths = write_gold_artifact(result, tmp_path / "gold")
    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))

    manifest_source_path = manifest["source_esc"]["path"]
    assert manifest_source_path == "esc_relative.jsonl"
    assert Path(manifest_source_path).is_absolute() is False


def _evaluation_path(rel_path: str, kind: str) -> Path:
    parts = Path(rel_path).parts
    version_index = parts.index("v0.9")
    topic = parts[version_index + 1]
    stem = Path(rel_path).name.removesuffix(".xml").removesuffix(".xml")
    return Path(EVAL_PREFIX) / kind / topic / f"{stem}.tab"


def _write_gold_case(
    xml_root: Path,
    rel_path: str,
    *,
    relation: str | None = "PRECONDITION",
    plot_link: bool = True,
    eval_relation: str = "PRECONDITION",
    coref_rows: tuple[str, ...] = (),
) -> None:
    rel_attr = "" if relation is None else f' relType="{relation}"'
    plot_links = (
        f'<PLOT_LINK r_id="9001" SIGNAL=""{rel_attr} CAUSES="FALSE" '
        'CAUSED_BY="FALSE"><source m_id="1"/><target m_id="2"/></PLOT_LINK>'
        if plot_link
        else ""
    )
    xml = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        f'<Document doc_name="{Path(rel_path).name}">\n'
        '  <token t_id="1" sentence="0" number="0">Alpha</token>\n'
        '  <token t_id="2" sentence="0" number="1">triggers</token>\n'
        '  <token t_id="3" sentence="0" number="2">Gamma</token>\n'
        '  <token t_id="4" sentence="0" number="3">today</token>\n'
        "  <Markables>\n"
        '    <ACTION_OCCURRENCE m_id="1" comment="" climaxEvent="FALSE">\n'
        '      <token_anchor t_id="1"/>\n'
        "    </ACTION_OCCURRENCE>\n"
        '    <ACTION_OCCURRENCE m_id="2" comment="" climaxEvent="FALSE">\n'
        '      <token_anchor t_id="3"/>\n'
        "    </ACTION_OCCURRENCE>\n"
        "  </Markables>\n"
        "  <Relations>\n"
        f"    {plot_links}\n"
        "  </Relations>\n"
        "</Document>\n"
    )
    xml_path = xml_root / rel_path
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml, encoding="utf-8")

    evaluation_path = xml_root / _evaluation_path(rel_path, "event_mentions_extended")
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(f"1\t3\t{eval_relation}\n", encoding="utf-8")

    if coref_rows:
        coref_path = xml_root / _evaluation_path(rel_path, "coref_chain")
        coref_path.parent.mkdir(parents=True, exist_ok=True)
        coref_path.write_text(
            "".join(f"{row}\n" for row in coref_rows),
            encoding="utf-8",
        )


def _esc_row(item_id: int, rel_path: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "words": ["Alpha", "triggers", "Gamma", "today"],
        "events": [[0], [2]],
        "tri_causal_label": 1,
        "bi_causal_label": 1,
        "doc_name": ESC_PREFIX + rel_path,
        "sent_in_doc_id": 0,
        "events_id": ["1", "2"],
        "topic": "50",
        "ESC_rel": "PRECONDITION",
    }


def _none_row(item_id: int) -> dict[str, object]:
    return {
        "item_id": item_id,
        "words": ["Alpha", "triggers", "Gamma", "today"],
        "events": [[0], [2]],
        "tri_causal_label": 0,
        "bi_causal_label": 0,
        "doc_name": ESC_PREFIX + "annotated_data/v0.9/99/99_none.xml.xml",
        "sent_in_doc_id": 0,
        "events_id": ["1", "2"],
        "topic": "99",
        "ESC_rel": "UN_LABEL",
    }


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> EscDataset:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return load_esc(path)


def _gold_result(
    tmp_path: Path,
    item_id: int,
    *,
    rel_path: str = RELATIVE_REL_PATH,
    **case: Any,
) -> GoldReconstruction:
    xml_root = tmp_path / "xml"
    _write_gold_case(xml_root, rel_path, **case)
    dataset = _write_dataset(tmp_path / "esc.jsonl", [_esc_row(item_id, rel_path)])
    return reconstruct_gold(dataset, xml_root, XML_VERSION)


# 1) Blocking: evaluation-format fallbacks are restricted to approved item IDs.


def test_missing_relation_fallback_rejects_unapproved_item(tmp_path: Path) -> None:
    result = _gold_result(tmp_path, 5001, relation=None)

    assert result.records == ()
    assert len(result.hard_failures) == 1
    assert result.hard_failures[0].item_id == 5001
    assert dict(result.summary)["missing_relation_fallbacks"] == 0


def test_self_coref_direction_fallback_rejects_unapproved_item(tmp_path: Path) -> None:
    result = _gold_result(
        tmp_path,
        5002,
        plot_link=False,
        coref_rows=("3 1\t3 1\tPRECONDITION",),
    )

    assert result.records == ()
    assert len(result.hard_failures) == 1
    assert result.hard_failures[0].item_id == 5002
    assert dict(result.summary)["self_coref_evaluation_fallbacks"] == 0


def test_item_9775_may_use_missing_relation_fallback(tmp_path: Path) -> None:
    result = _gold_result(tmp_path, 9775, relation=None)

    assert result.hard_failures == ()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.resolution_method == "missing_relation_fallback"
    assert record.relation_source == "evaluation_format"
    assert record.direction_source == "mixed"


@pytest.mark.parametrize("item_id", [2531, 10592, 12487])
def test_approved_self_coref_items_may_use_direction_fallback(
    tmp_path: Path,
    item_id: int,
) -> None:
    result = _gold_result(
        tmp_path,
        item_id,
        plot_link=False,
        coref_rows=("3 1\t3 1\tPRECONDITION",),
    )

    assert result.hard_failures == ()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.resolution_method == "self_coref_evaluation_fallback"
    assert record.direction_source == "evaluation_format"


@pytest.mark.parametrize("item_id", [2531, 10592, 12487])
def test_self_coref_approved_items_cannot_use_missing_relation_fallback(
    tmp_path: Path,
    item_id: int,
) -> None:
    result = _gold_result(tmp_path, item_id, relation=None)

    assert result.records == ()
    assert len(result.hard_failures) == 1
    assert result.hard_failures[0].item_id == item_id


def test_item_9775_cannot_use_self_coref_direction_fallback(tmp_path: Path) -> None:
    result = _gold_result(
        tmp_path,
        9775,
        plot_link=False,
        coref_rows=("3 1\t3 1\tPRECONDITION",),
    )

    assert result.records == ()
    assert len(result.hard_failures) == 1
    assert result.hard_failures[0].item_id == 9775


# 2) Blocking: partial probability is missing, booleans are not numeric.


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {"cause": 0.4},
        {"cause": 0.4, "caused_by": 0.6},
    ],
)
def test_partial_probability_is_missing_probability(
    probabilities: dict[str, float],
) -> None:
    result = validate_prediction("cause", probabilities)

    assert result.status is PredictionStatus.MISSING_PROBABILITY
    assert result.prediction_valid is True
    assert result.probability_valid is False
    assert result.choice == "cause"
    assert result.probabilities is None


@pytest.mark.parametrize(
    "probabilities",
    [
        {"cause": True, "caused_by": False, "none": False},
        {"cause": 1.0, "caused_by": False, "none": 0.0},
    ],
)
def test_boolean_probability_values_are_invalid_probability(
    probabilities: dict[str, float],
) -> None:
    result = validate_prediction("cause", probabilities)

    assert result.status is PredictionStatus.INVALID_PROBABILITY
    assert result.prediction_valid is True
    assert result.probability_valid is False
    assert result.probabilities is None


def test_validate_probabilities_rejects_boolean_values() -> None:
    with pytest.raises(InvalidProbabilityError):
        validate_probabilities({"cause": True, "caused_by": False, "none": False})


# 3) Material: the gold audit exposes per-item resolutions.


def test_gold_audit_lists_per_item_resolution_provenance(tmp_path: Path) -> None:
    xml_root = tmp_path / "xml"
    _write_gold_case(xml_root, RELATIVE_REL_PATH)
    dataset = _write_dataset(
        tmp_path / "esc.jsonl",
        [_esc_row(9001, RELATIVE_REL_PATH), _none_row(9002)],
    )
    result = reconstruct_gold(dataset, xml_root, XML_VERSION)
    assert result.hard_failures == ()

    paths = write_gold_artifact(result, tmp_path / "gold")
    audit = json.loads(paths.audit_json.read_text(encoding="utf-8"))

    assert "item_resolutions" in audit
    entries = audit["item_resolutions"]
    assert isinstance(entries, list)
    assert len(entries) == len(result.records)

    keys = [(entry["item_id"], entry["record_id"]) for entry in entries]
    assert len(set(keys)) == len(keys)
    assert keys == sorted(keys)
    assert set(keys) == {(record.item_id, record.record_id) for record in result.records}

    second_paths = write_gold_artifact(result, tmp_path / "gold-second")
    second_audit = json.loads(second_paths.audit_json.read_text(encoding="utf-8"))
    assert second_audit["item_resolutions"] == entries

    by_key = {(entry["item_id"], entry["record_id"]): entry for entry in entries}
    for record in result.records:
        entry = by_key[(record.item_id, record.record_id)]
        assert entry["item_id"] == record.item_id
        assert entry["record_id"] == record.record_id
        assert entry["choice"] == record.choice.value
        assert entry["resolution_method"] == record.resolution_method
        assert entry["relation_source"] == record.relation_source
        assert entry["direction_source"] == record.direction_source
        assert entry["xml_path"] == record.xml_path


# 4) Material: redaction handles sensitive key variants without touching usage fields.


def test_redaction_handles_hyphen_and_camel_case_variants_recursively() -> None:
    payload: dict[str, object] = {
        "apiKey": "camel-api-key",
        "API-KEY": "hyphen-api-key",
        "accessToken": "camel-access-token",
        "X-Access-Token": "hyphen-access-token",
        "clientSecret": "camel-client-secret",
        "userPassword": "camel-password",
        "nested": {
            "deep": [
                {"apiKey": "nested-api-key", "keep": "visible"},
            ],
        },
        "state_tokens": 11,
        "input_tokens": 22,
        "prompt_tokens": 33,
        "completion_tokens": 44,
    }
    original = copy.deepcopy(payload)

    redacted = redact_secrets(payload)
    assert isinstance(redacted, dict)
    redacted_any: Any = redacted

    assert redacted_any["apiKey"] == REDACTED
    assert redacted_any["API-KEY"] == REDACTED
    assert redacted_any["accessToken"] == REDACTED
    assert redacted_any["X-Access-Token"] == REDACTED
    assert redacted_any["clientSecret"] == REDACTED
    assert redacted_any["userPassword"] == REDACTED
    assert redacted_any["nested"]["deep"][0]["apiKey"] == REDACTED
    assert redacted_any["nested"]["deep"][0]["keep"] == "visible"
    assert redacted_any["state_tokens"] == 11
    assert redacted_any["input_tokens"] == 22
    assert redacted_any["prompt_tokens"] == 33
    assert redacted_any["completion_tokens"] == 44

    serialized = json.dumps(redacted, ensure_ascii=False)
    for secret in (
        "camel-api-key",
        "hyphen-api-key",
        "camel-access-token",
        "hyphen-access-token",
        "camel-client-secret",
        "camel-password",
        "nested-api-key",
    ):
        assert secret not in serialized
    assert payload == original

    redacted_any["nested"]["deep"][0]["keep"] = "mutated"
    assert payload["nested"] == {"deep": [{"apiKey": "nested-api-key", "keep": "visible"}]}


# 5) Material: the public canonical request must be deeply immutable.


def test_canonical_request_payload_is_deeply_immutable() -> None:
    request = build_canonical_request(
        "The storm destroyed the house.",
        "the storm",
        "the house was destroyed",
    )
    payload: Any = request.payload
    options: Any = request.options

    with pytest.raises(TypeError):
        payload["state"]["sentence"] = "tampered"

    with pytest.raises(TypeError):
        payload["question"]["options"][0]["id"] = "none"

    with pytest.raises(TypeError):
        options[0]["id"] = "none"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_snapshot(request: CanonicalRequest) -> tuple[str, str]:
    return (_canonical_json(request.payload), _canonical_json(request.options))


def _canonical_target(request: CanonicalRequest, name: str) -> Any:
    payload: Any = request.payload
    options: Any = request.options
    if name == "payload":
        return payload
    if name == "state":
        return payload["state"]
    if name == "payload_options":
        return payload["question"]["options"]
    if name == "option":
        return options[0]
    if name == "request_options":
        return options
    raise AssertionError(f"unknown canonical target: {name}")


@pytest.mark.parametrize(
    ("target_name", "replacement"),
    [
        ("payload", {"tampered": True}),
        ("state", {"sentence": "tampered"}),
        ("payload_options", [{"id": "none"}]),
        ("option", {"id": "none"}),
        ("request_options", ({"id": "none"},)),
    ],
)
def test_canonical_in_place_or_raises_and_preserves(
    target_name: str,
    replacement: object,
) -> None:
    request = build_canonical_request(
        "The storm destroyed the house.",
        "the storm",
        "the house was destroyed",
    )
    before = _canonical_snapshot(request)
    target = _canonical_target(request, target_name)

    with pytest.raises(TypeError):
        target |= replacement

    assert _canonical_snapshot(request) == before
    assert _canonical_sha256(request.payload) == request.sha256


@pytest.mark.parametrize(
    ("target_name", "replacement"),
    [
        ("payload", {"tampered": True}),
        ("state", {"sentence": "tampered"}),
        ("payload_options", [{"id": "none"}]),
    ],
)
def test_canonical_repeated_init_raises_and_preserves(
    target_name: str,
    replacement: object,
) -> None:
    request = build_canonical_request(
        "The storm destroyed the house.",
        "the storm",
        "the house was destroyed",
    )
    before = _canonical_snapshot(request)
    target = _canonical_target(request, target_name)

    with pytest.raises(TypeError):
        target.__init__(replacement)

    assert _canonical_snapshot(request) == before
    assert _canonical_sha256(request.payload) == request.sha256


def test_canonical_attempted_mutations_do_not_partially_modify_or_change_hash() -> None:
    request = build_canonical_request(
        "The storm destroyed the house.",
        "the storm",
        "the house was destroyed",
    )
    before = _canonical_snapshot(request)

    in_place_attempts = [
        ("payload", {"tampered": True}),
        ("state", {"sentence": "tampered"}),
        ("payload_options", [{"id": "none"}]),
        ("option", {"id": "none"}),
        ("request_options", ({"id": "none"},)),
    ]
    for target_name, replacement in in_place_attempts:
        target = _canonical_target(request, target_name)
        with contextlib.suppress(TypeError):
            target |= replacement

    init_attempts = [
        ("payload", {"tampered": True}),
        ("state", {"sentence": "tampered"}),
        ("payload_options", [{"id": "none"}]),
        ("request_options", ({"id": "none"},)),
    ]
    for target_name, replacement in init_attempts:
        target = _canonical_target(request, target_name)
        with contextlib.suppress(TypeError):
            target.__init__(replacement)

    assert _canonical_snapshot(request) == before
    assert _canonical_sha256(request.payload) == request.sha256


# 6) Low: NLL epsilon must be finite and 0 < epsilon <= 1.


def _probability_record() -> PredictionRecord:
    return PredictionRecord(
        run_id="run-1",
        item_id=1,
        record_id="r1",
        model="test-model",
        attempt=1,
        raw_request_id="request-r1",
        request_hash="request-hash",
        response_hash="response-hash",
        adapter_version="adapter-v1",
        parser_version="parser-v1",
        gold_choice="cause",
        prediction="caused_by",
        status="ok",
        prediction_valid=True,
        probability_valid=True,
        probabilities=(0.0, 0.5, 0.5),
        error=None,
    )


@pytest.mark.parametrize(
    "nll_epsilon",
    [1.0000001, 1.5, 2.0, math.inf, math.nan],
)
def test_nll_epsilon_must_be_finite_and_at_most_one(nll_epsilon: float) -> None:
    record = _probability_record()

    with pytest.raises(ValueError):
        compute_probability_metrics(
            [record],
            [record],
            zero_policy="clip",
            nll_epsilon=nll_epsilon,
        )


def test_nll_epsilon_accepts_one_as_upper_bound() -> None:
    record = _probability_record()

    report = compute_probability_metrics(
        [record],
        [record],
        zero_policy="clip",
        nll_epsilon=1.0,
    )
    first = report["first"]
    assert isinstance(first, dict)
    assert first["nll_epsilon"] == 1.0
    assert first["nll"] == 0.0


# 7) Low: manifest keeps the caller's relative input path.


def test_gold_manifest_preserves_nested_relative_source_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_esc = Path("inputs") / "esc_relative.jsonl"
    relative_esc.parent.mkdir()
    relative_esc.write_text(
        json.dumps(_simple_valid_row(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    dataset = load_esc(relative_esc)
    assert dataset.source_path.is_absolute() is False

    result = reconstruct_gold(dataset, XML_ROOT, XML_VERSION)
    assert result.hard_failures == ()

    paths = write_gold_artifact(result, tmp_path / "gold")
    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))

    manifest_source_path = manifest["source_esc"]["path"]
    assert manifest_source_path == "inputs/esc_relative.jsonl"
    assert Path(manifest_source_path).is_absolute() is False
    assert manifest_source_path != str(relative_esc.resolve())
