"""Public contracts for the Laya-only ESC evaluation runner."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from eci_decision_models.canonical import (
    CHOICE_DESCRIPTIONS,
    CHOICE_IDS,
    CHOICE_LETTERS,
    PROMPT_VERSION,
    SCHEMA_VERSION,
)
from eci_decision_models.data import EscDataset, load_esc
from eci_decision_models.laya import (
    LayaRequest,
    LayaResponse,
    LayaSettings,
    build_laya_request,
    parse_laya_response,
)

REDACTED = "[REDACTED]"
ENVIRONMENT_PACKAGES = {
    "laya",
    "torch",
    "transformers",
    "numpy",
    "huggingface-hub",
    "safetensors",
    "eci-decision-models",
}


def _esc_run_module() -> Any:
    return importlib.import_module("eci_decision_models.esc_run")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _esc_row(item_id: int) -> dict[str, object]:
    return {
        "item_id": item_id,
        "words": [f"event{item_id}a", "causes", f"event{item_id}b"],
        "events": [[0], [2]],
        "tri_causal_label": 1,
        "bi_causal_label": 1,
        "doc_name": f"doc{item_id}.xml",
        "sent_in_doc_id": 0,
        "events_id": [f"e{item_id}a", f"e{item_id}b"],
        "topic": "synthetic",
        "ESC_rel": "PRECONDITION",
    }


def _write_esc_dataset(path: Path, item_ids: Sequence[int]) -> EscDataset:
    _write_jsonl(path, [_esc_row(item_id) for item_id in item_ids])
    return load_esc(path)


def _gold_records(
    dataset: EscDataset,
    choices: Mapping[int, str] | None = None,
) -> list[dict[str, object]]:
    selected = choices or {}
    return [
        {
            "record_id": record.record_id,
            "item_id": record.item_id,
            "choice": selected.get(record.item_id, "cause"),
            "event1": record.event1,
            "event2": record.event2,
        }
        for record in dataset.records
    ]


def _write_gold(
    path: Path,
    dataset: EscDataset,
    choices: Mapping[int, str] | None = None,
) -> None:
    _write_jsonl(path, _gold_records(dataset, choices))


def _write_inputs(
    tmp_path: Path,
    item_ids: Sequence[int] = (1, 2),
    choices: Mapping[int, str] | None = None,
) -> tuple[Any, EscDataset, Path, Path]:
    esc_path = tmp_path / "esc.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    dataset = _write_esc_dataset(esc_path, item_ids)
    _write_gold(gold_path, dataset, choices)
    return _esc_run_module(), dataset, esc_path, gold_path


def _load_items(
    tmp_path: Path,
    item_ids: Sequence[int] = (1, 2),
    choices: Mapping[int, str] | None = None,
) -> tuple[Any, EscDataset, Path, Any]:
    module, dataset, _, gold_path = _write_inputs(tmp_path, item_ids, choices)
    gold = module.load_gold_artifact(gold_path)
    items = module.prepare_esc_items(dataset, gold)
    return module, dataset, gold_path, items


def _valid_raw(
    choice: str = "cause",
    *,
    model: str = "fake-laya",
    secret: bool = False,
) -> dict[str, object]:
    probabilities = {
        "cause": {"cause": 0.6, "caused_by": 0.3, "none": 0.1},
        "caused_by": {"cause": 0.1, "caused_by": 0.8, "none": 0.1},
        "none": {"cause": 0.1, "caused_by": 0.1, "none": 0.8},
    }[choice]
    raw: dict[str, object] = {
        "answers": {
            "eci_relation": {
                "choice": choice,
                "probabilities": probabilities,
            }
        },
        "usage": {"input_tokens": 4, "output_tokens": 1},
        "model": model,
    }
    if secret:
        raw["apiKey"] = "sk-should-not-survive"
    return raw


def _missing_raw(choice: str = "none") -> dict[str, object]:
    return {
        "answers": {"eci_relation": {"choice": choice}},
        "usage": {"input_tokens": 3, "output_tokens": 1},
        "model": "fake-laya",
    }


def _invalid_raw() -> dict[str, object]:
    return {
        "answers": "not-a-mapping",
        "usage": {"input_tokens": 1},
        "model": "fake-laya",
    }


def _response(raw: dict[str, object]) -> LayaResponse:
    model = raw.get("model")
    return LayaResponse(
        raw=raw,
        validated=parse_laya_response(raw),
        usage={},
        model=model if isinstance(model, str) else None,
    )


class _ScriptedAdapter:
    def __init__(self, outcomes: Sequence[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[LayaRequest] = []

    def predict(self, request: LayaRequest) -> LayaResponse:
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("scripted adapter ran out of outcomes")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, LayaResponse)
        return outcome


class _AdapterFactory:
    def __init__(self, adapter: object) -> None:
        self.adapter = adapter
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return self.adapter


class _ExplodingFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        raise AssertionError("dry-run must not create an adapter")


def _run(
    tmp_path: Path,
    items: Sequence[object],
    adapter: object,
    *,
    run_id: str = "esc-run",
    budget: object | None = None,
    resume: bool = False,
    code_metadata: Mapping[str, object] | None = None,
) -> Any:
    module = _esc_run_module()
    return module.run_laya_esc(
        items=items,
        adapter=adapter,
        output_dir=tmp_path / "runs",
        run_id=run_id,
        settings=LayaSettings.from_env({}),
        runtime_metadata={"device": "cpu"},
        budget=budget if budget is not None else module.EscRunBudget(max_requests=len(items)),
        resume=resume,
        code_metadata=code_metadata,
        command=["synthetic", "esc"],
    )


def test_load_gold_artifact_from_bare_jsonl(tmp_path: Path) -> None:
    module, dataset, _, gold_path = _write_inputs(tmp_path, (10, 11))

    artifact = module.load_gold_artifact(gold_path)

    assert artifact.path.resolve() == gold_path.resolve()
    assert artifact.sha256 == hashlib.sha256(gold_path.read_bytes()).hexdigest()
    assert artifact.version is None
    assert artifact.manifest is None
    assert set(artifact.records) == {record.record_id for record in dataset.records}
    first = artifact.records[dataset.records[0].record_id]
    assert first.record_id == dataset.records[0].record_id
    assert first.item_id == 10
    assert first.choice == "cause"
    assert first.event1 == dataset.records[0].event1
    assert first.event2 == dataset.records[0].event2


def test_load_gold_artifact_rejects_duplicate_record_id(tmp_path: Path) -> None:
    _, dataset, _, gold_path = _write_inputs(tmp_path, (10, 11))
    records = _gold_records(dataset)
    records[1]["record_id"] = records[0]["record_id"]
    _write_jsonl(gold_path, records)

    with pytest.raises(ValueError):
        _esc_run_module().load_gold_artifact(gold_path)


def test_load_gold_artifact_rejects_unknown_choice(tmp_path: Path) -> None:
    _, dataset, _, gold_path = _write_inputs(tmp_path, (10, 11))
    records = _gold_records(dataset)
    records[0]["choice"] = "maybe"
    _write_jsonl(gold_path, records)

    with pytest.raises(ValueError):
        _esc_run_module().load_gold_artifact(gold_path)


def test_load_gold_artifact_rejects_missing_or_wrong_typed_fields(tmp_path: Path) -> None:
    module = _esc_run_module()
    missing_path = tmp_path / "missing.jsonl"
    _write_jsonl(
        missing_path,
        [
            {
                "record_id": "r1",
                "item_id": 1,
                "choice": "cause",
                "event1": "a",
            }
        ],
    )
    with pytest.raises(ValueError):
        module.load_gold_artifact(missing_path)

    wrong_type_path = tmp_path / "wrong_type.jsonl"
    _write_jsonl(
        wrong_type_path,
        [
            {
                "record_id": "r1",
                "item_id": "1",
                "choice": "cause",
                "event1": "a",
                "event2": "b",
            }
        ],
    )
    with pytest.raises(ValueError):
        module.load_gold_artifact(wrong_type_path)


def test_load_gold_artifact_directory_mode_reads_manifest(tmp_path: Path) -> None:
    module, dataset, _, gold_path = _write_inputs(tmp_path, (10, 11))
    gold_dir = tmp_path / "gold-dir"
    gold_dir.mkdir()
    (gold_dir / "gold_choice_reconstructed.jsonl").write_bytes(gold_path.read_bytes())
    manifest = {
        "gold_version": "gold-v9",
        "summary": {"hard_failures": 0},
        "source_esc": {"sha256": dataset.source_sha256},
    }
    (gold_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )

    artifact = module.load_gold_artifact(gold_dir)

    assert artifact.path.name == "gold_choice_reconstructed.jsonl"
    assert artifact.version == "gold-v9"
    assert artifact.manifest is not None
    assert artifact.manifest["gold_version"] == "gold-v9"
    assert set(artifact.records) == {record.record_id for record in dataset.records}


def test_load_gold_artifact_rejects_manifest_hard_failures(tmp_path: Path) -> None:
    module, _, _, gold_path = _write_inputs(tmp_path, (10, 11))
    gold_dir = tmp_path / "gold-dir"
    gold_dir.mkdir()
    (gold_dir / "gold_choice_reconstructed.jsonl").write_bytes(gold_path.read_bytes())
    (gold_dir / "manifest.json").write_text(
        json.dumps({"summary": {"hard_failures": 1}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        module.load_gold_artifact(gold_dir)


def test_prepare_esc_items_returns_dataset_order_and_canonical_requests(
    tmp_path: Path,
) -> None:
    _, _dataset, gold_path, items = _load_items(tmp_path, (10, 11, 12))
    module = _esc_run_module()
    gold = module.load_gold_artifact(gold_path)

    assert gold.version is None
    assert [item.record.item_id for item in items] == [10, 11, 12]
    for item in items:
        assert item.gold.record_id == item.record.record_id
        assert item.gold.item_id == item.record.item_id
        assert item.gold.event1 == item.record.event1
        assert item.gold.event2 == item.record.event2
        expected = build_laya_request(item.record.sentence, item.record.event1, item.record.event2)
        assert item.request.canonical.sha256 == expected.canonical.sha256
        assert _canonical_json(item.request.canonical.payload) == _canonical_json(
            expected.canonical.payload,
        )
        assert item.request.state == {
            "sentence": item.record.sentence,
            "event1": item.record.event1,
            "event2": item.record.event2,
        }


def test_prepare_esc_items_rejects_missing_or_extra_ids(tmp_path: Path) -> None:
    module, dataset, _, _gold_path = _write_inputs(tmp_path, (10, 11, 12))
    records = _gold_records(dataset)
    missing_path = tmp_path / "missing_gold.jsonl"
    _write_jsonl(missing_path, records[:2])
    with pytest.raises(ValueError) as missing:
        module.prepare_esc_items(dataset, module.load_gold_artifact(missing_path))
    missing_message = str(missing.value)
    assert (
        str(records[2]["record_id"]) in missing_message
        or str(records[2]["item_id"]) in missing_message
    )

    extra_path = tmp_path / "extra_gold.jsonl"
    _write_jsonl(
        extra_path,
        [
            *records,
            {
                "record_id": "esc:extra",
                "item_id": 99,
                "choice": "cause",
                "event1": "extra",
                "event2": "row",
            },
        ],
    )
    with pytest.raises(ValueError) as extra:
        module.prepare_esc_items(dataset, module.load_gold_artifact(extra_path))
    extra_message = str(extra.value)
    assert "esc:extra" in extra_message or "99" in extra_message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("item_id", 999),
        ("event1", "wrong event1"),
        ("event2", "wrong event2"),
    ],
)
def test_prepare_esc_items_rejects_field_mismatches(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    module, dataset, _, gold_path = _write_inputs(tmp_path, (10, 11))
    records = _gold_records(dataset)
    records[0][field] = value
    _write_jsonl(gold_path, records)

    with pytest.raises(ValueError):
        module.prepare_esc_items(dataset, module.load_gold_artifact(gold_path))


def test_prepare_esc_items_rejects_source_sha_mismatch(tmp_path: Path) -> None:
    module, dataset, _, gold_path = _write_inputs(tmp_path, (10, 11))
    gold_dir = tmp_path / "gold-dir"
    gold_dir.mkdir()
    (gold_dir / "gold_choice_reconstructed.jsonl").write_bytes(gold_path.read_bytes())
    (gold_dir / "manifest.json").write_text(
        json.dumps({"source_esc": {"sha256": "0" * 64}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        module.prepare_esc_items(dataset, module.load_gold_artifact(gold_dir))


def test_run_laya_esc_writes_success_artifacts(tmp_path: Path) -> None:
    _module, dataset, gold_path, items = _load_items(
        tmp_path,
        (1, 2),
        choices={2: "caused_by"},
    )
    raw_one = _valid_raw("cause", secret=True)
    adapter = _ScriptedAdapter([_response(raw_one), _response(_valid_raw("caused_by"))])

    result = _run(
        tmp_path,
        items,
        adapter,
        code_metadata={"commit": "deadbeef", "dirty": False},
    )

    run_dir = tmp_path / "runs" / "esc-run"
    assert result.run_id == "esc-run"
    assert result.run_dir == run_dir
    assert result.status == "ok"
    assert result.request_count == 2
    assert len(result.records) == 2
    assert [record.item_id for record in result.records] == [1, 2]

    output_files = {
        path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()
    }
    assert output_files == {
        "predictions.jsonl",
        "raw/response.jsonl",
        "meta.json",
        "metrics.json",
        "metrics.md",
    }

    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert [record["item_id"] for record in predictions] == [1, 2]
    first, second = predictions
    assert first["run_id"] == "esc-run"
    assert first["gold_choice"] == "cause"
    assert first["status"] == "ok"
    assert first["prediction"] == "cause"
    assert first["prediction_valid"] is True
    assert first["probability_valid"] is True
    assert first["probabilities"] == pytest.approx([0.6, 0.3, 0.1])
    assert first["model"] == "fake-laya"
    assert first["adapter_version"] == "laya-adapter-v1"
    assert first["parser_version"] == "eci-parser-v1"
    assert first["attempt"] == 1
    assert first["raw_request_id"] == f"esc-run:{first['record_id']}:1"
    assert first["request_hash"] == items[0].request.canonical.sha256
    assert first["control_status"] is None
    assert second["prediction"] == "caused_by"
    assert second["probabilities"] == pytest.approx([0.1, 0.8, 0.1])

    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert len(raw_entries) == 2
    entry = raw_entries[0]
    assert entry["run_id"] == "esc-run"
    assert entry["item_id"] == 1
    assert entry["record_id"] == first["record_id"]
    assert entry["attempt"] == 1
    assert entry["status"] == "ok"
    assert entry["request_hash"] == items[0].request.canonical.sha256
    assert entry["response_hash"] == first["response_hash"]
    assert entry["response_hash"] == _canonical_sha256(entry["raw_response"])
    assert entry["response_hash"] != _canonical_sha256(raw_one)
    canonical = cast("Mapping[str, object]", entry["canonical"])
    assert canonical["sha256"] == items[0].request.canonical.sha256
    assert _canonical_json(canonical["payload"]) == _canonical_json(
        items[0].request.canonical.payload,
    )
    assert _canonical_json(canonical["options"]) == _canonical_json(
        items[0].request.canonical.options,
    )
    assert entry["state"] == items[0].request.state
    assert entry["questions"] == items[0].request.questions
    input_stats = cast("Mapping[str, object]", entry["input_stats"])
    assert input_stats["sentence_words"] == len(items[0].record.sentence.split())
    assert input_stats["event1_words"] == len(items[0].record.event1.split())
    assert input_stats["event2_words"] == len(items[0].record.event2.split())
    redacted = cast("dict[str, object]", entry["raw_response"])
    assert redacted["apiKey"] == REDACTED
    second_raw = cast("Mapping[str, object]", raw_entries[1]["raw_response"])
    assert "apiKey" not in second_raw
    assert entry["error"] is None
    assert isinstance(entry["latency_ms"], (int, float))
    assert entry["latency_ms"] >= 0

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "esc-run"
    assert meta["status"] == "ok"
    assert meta["stopped"] is False
    assert meta["run_kind"] == "laya-only-exploratory"
    assert meta["gate_status"] == "phase2-open"
    assert meta["limit"] is None
    assert meta["resumed"] is False
    assert meta["item_count"] == 2
    assert meta["request_count"] == 2
    assert meta["status_counts"] == {"ok": 2}
    assert meta["control_status_counts"] == {}
    assert meta["command"] == ["synthetic", "esc"]
    assert meta["code"] == {"commit": "deadbeef", "dirty": False}
    assert meta["runtime_metadata"] == {"device": "cpu"}
    assert meta["settings"] == asdict(LayaSettings.from_env({}))
    assert meta["budget"] == {
        "max_requests": 2,
        "max_retries": 0,
        "max_wall_seconds": 7200.0,
    }
    assert meta["options"] == [
        {"id": choice_id, "letter": letter, "description": description}
        for choice_id, letter, description in zip(
            CHOICE_IDS,
            CHOICE_LETTERS,
            CHOICE_DESCRIPTIONS,
            strict=True,
        )
    ]
    assert meta["prompt_version"] == PROMPT_VERSION
    assert meta["schema_version"] == SCHEMA_VERSION
    assert Path(meta["data"]["path"]).resolve() == dataset.source_path.resolve()
    assert Path(meta["data"]["resolved_path"]).resolve() == dataset.source_path.resolve()
    assert meta["data"]["sha256"] == dataset.source_sha256
    assert meta["data"]["cwd"] == str(Path.cwd())
    assert Path(meta["gold"]["path"]).resolve() == gold_path.resolve()
    assert Path(meta["gold"]["resolved_path"]).resolve() == gold_path.resolve()
    assert meta["gold"]["sha256"] == hashlib.sha256(gold_path.read_bytes()).hexdigest()
    assert meta["gold"]["version"] is None
    assert (
        meta["predictions_sha256"]
        == hashlib.sha256(
            (run_dir / "predictions.jsonl").read_bytes(),
        ).hexdigest()
    )
    assert (
        meta["raw_sha256"]
        == hashlib.sha256(
            (run_dir / "raw/response.jsonl").read_bytes(),
        ).hexdigest()
    )
    assert Path(meta["metrics_path"]).resolve() == (run_dir / "metrics.json").resolve()
    assert Path(meta["metrics_md_path"]).resolve() == (run_dir / "metrics.md").resolve()

    environment = cast("Mapping[str, object]", meta["environment"])
    assert isinstance(environment["python_version"], str)
    assert environment["cwd"] == str(Path.cwd())
    uv_lock_sha256 = environment["uv_lock_sha256"]
    assert uv_lock_sha256 is None or (isinstance(uv_lock_sha256, str) and len(uv_lock_sha256) == 64)
    packages = cast("Mapping[str, object]", environment["packages"])
    assert set(packages) >= ENVIRONMENT_PACKAGES
    assert all(value is None or isinstance(value, str) for value in packages.values())

    started = datetime.fromisoformat(str(meta["started_at_utc"]).replace("Z", "+00:00"))
    finished = datetime.fromisoformat(str(meta["finished_at_utc"]).replace("Z", "+00:00"))
    assert started.tzinfo is not None
    assert finished.tzinfo is not None
    assert finished >= started

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["definition_version"] == "eci-metrics-v1"
    assert metrics["classification"]["N"] == 2
    assert metrics["classification"]["accuracy"] == pytest.approx(1.0)
    probability = metrics["probability"]
    assert probability["N"] == 2
    assert probability["valid"] == 2
    assert probability["fail"] == 0
    assert probability["coverage"] == pytest.approx(1.0)
    assert probability["common_N"] == 2
    assert probability["brier"] == pytest.approx(0.16)
    assert probability["nll"] == pytest.approx(-(math.log(0.6) + math.log(0.8)) / 2)
    assert probability["ece"] == pytest.approx(0.3)
    assert len(probability["ece_bins"]) == 10
    assert metrics["direction_errors"] == {
        "gold_cause_pred_caused_by": 0,
        "gold_caused_by_pred_cause": 0,
        "gold_causal_pred_none": 0,
        "gold_none_pred_causal": 0,
        "invalid_causal": 0,
        "invalid_none": 0,
    }
    assert metrics["status_counts"] == {"ok": 2}
    assert metrics["control_status_counts"] == {}


def test_run_laya_esc_marks_invalid_response_and_missing_probability(
    tmp_path: Path,
) -> None:
    _module, _, _, items = _load_items(tmp_path, (1, 2))
    adapter = _ScriptedAdapter([_response(_invalid_raw()), _response(_missing_raw())])

    result = _run(tmp_path, items, adapter)

    assert result.status == "completed_with_errors"
    invalid, missing = result.records
    assert invalid.status == "invalid_response"
    assert invalid.prediction is None
    assert invalid.prediction_valid is False
    assert invalid.probability_valid is False
    assert invalid.probabilities is None
    assert missing.status == "missing_probability"
    assert missing.prediction == "none"
    assert missing.probability_valid is False
    assert missing.probabilities is None

    meta = json.loads((result.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed_with_errors"
    assert meta["status_counts"] == {"invalid_response": 1, "missing_probability": 1}


def test_run_status_ok_when_only_missing_probability(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([_response(_missing_raw())])

    result = _run(tmp_path, items, adapter)

    assert result.status == "ok"
    assert result.records[0].status == "missing_probability"


def test_run_laya_esc_retries_timeout_and_records_both_attempts(tmp_path: Path) -> None:
    module, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([TimeoutError("first timeout"), _response(_valid_raw())])

    result = _run(
        tmp_path,
        items,
        adapter,
        budget=module.EscRunBudget(max_requests=2, max_retries=1),
    )

    assert len(adapter.requests) == 2
    assert result.status == "ok"
    assert result.request_count == 2
    assert result.records[0].attempt == 2
    assert result.records[0].status == "ok"
    raw_entries = _read_jsonl(result.run_dir / "raw/response.jsonl")
    assert [entry["attempt"] for entry in raw_entries] == [1, 2]
    assert [entry["status"] for entry in raw_entries] == ["api_error", "ok"]


def test_run_laya_esc_does_not_retry_runtime_errors(tmp_path: Path) -> None:
    module, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([RuntimeError("boom"), _response(_valid_raw())])

    result = _run(
        tmp_path,
        items,
        adapter,
        budget=module.EscRunBudget(max_requests=2, max_retries=1),
    )

    assert len(adapter.requests) == 1
    assert result.status == "completed_with_errors"
    record = result.records[0]
    assert record.status == "api_error"
    assert record.control_status is None
    assert record.prediction is None
    assert record.probabilities is None
    assert "boom" in str(record.error).lower()


def test_run_laya_esc_budget_exhaustion_marks_remaining_items(tmp_path: Path) -> None:
    module, _, _, items = _load_items(tmp_path, (1, 2, 3))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])

    result = _run(
        tmp_path,
        items,
        adapter,
        budget=module.EscRunBudget(max_requests=1, max_retries=0),
    )

    assert len(adapter.requests) == 1
    assert result.status == "budget_exhausted"
    assert result.request_count == 1
    assert [record.status for record in result.records] == ["ok", "api_error", "api_error"]
    assert [record.control_status for record in result.records] == [
        None,
        "budget_exhausted",
        "budget_exhausted",
    ]
    for record in result.records[1:]:
        assert record.prediction_valid is False
        assert record.probability_valid is False
        assert record.prediction is None
        assert record.probabilities is None
        assert record.error

    raw_entries = _read_jsonl(result.run_dir / "raw/response.jsonl")
    assert len(raw_entries) == 1
    meta = json.loads((result.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "budget_exhausted"
    assert meta["stopped"] is True
    assert meta["status_counts"] == {"ok": 1, "api_error": 2}
    assert meta["control_status_counts"] == {"budget_exhausted": 2}
    assert meta["request_count"] == 1


def test_run_laya_esc_resume_skips_completed_items_and_preserves_order(
    tmp_path: Path,
) -> None:
    _, _, _, items = _load_items(tmp_path, (1, 2, 3))
    first_adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])
    first = _run(tmp_path, items[:1], first_adapter, run_id="resume-run")

    assert first.status == "ok"
    assert len(first.records) == 1

    second_adapter = _ScriptedAdapter(
        [_response(_valid_raw("cause")), _response(_valid_raw("cause"))],
    )
    second = _run(
        tmp_path,
        items,
        second_adapter,
        run_id="resume-run",
        resume=True,
    )

    assert second.status == "ok"
    assert len(second_adapter.requests) == 2
    assert second.request_count == 3
    assert [record.item_id for record in second.records] == [1, 2, 3]
    predictions = _read_jsonl(second.run_dir / "predictions.jsonl")
    assert [record["item_id"] for record in predictions] == [1, 2, 3]
    record_ids = [record["record_id"] for record in predictions]
    assert len(record_ids) == len(set(record_ids))
    assert all(record["gold_choice"] == "cause" for record in predictions)

    meta = json.loads((second.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["resumed"] is True
    assert meta["item_count"] == 3
    assert meta["request_count"] == 3


def test_run_laya_esc_resume_rejects_duplicate_existing_record(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1, 2))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])
    result = _run(tmp_path, items[:1], adapter, run_id="duplicate-run")

    predictions_path = result.run_dir / "predictions.jsonl"
    line = predictions_path.read_text(encoding="utf-8").splitlines()[0]
    predictions_path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _run(
            tmp_path,
            items,
            _ScriptedAdapter([_response(_valid_raw("cause"))]),
            run_id="duplicate-run",
            resume=True,
        )


def test_run_laya_esc_resume_rejects_gold_mismatch(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1, 2))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])
    result = _run(tmp_path, items[:1], adapter, run_id="gold-mismatch-run")

    predictions_path = result.run_dir / "predictions.jsonl"
    records = _read_jsonl(predictions_path)
    records[0]["gold_choice"] = "none"
    _write_jsonl(predictions_path, records)

    with pytest.raises(ValueError):
        _run(
            tmp_path,
            items,
            _ScriptedAdapter([_response(_valid_raw("cause"))]),
            run_id="gold-mismatch-run",
            resume=True,
        )


def test_run_laya_esc_refuses_existing_run_directory_without_resume(
    tmp_path: Path,
) -> None:
    _, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])
    _run(tmp_path, items, adapter, run_id="existing-run")

    fresh = _ScriptedAdapter([_response(_valid_raw("cause"))])
    with pytest.raises(FileExistsError):
        _run(tmp_path, items, fresh, run_id="existing-run")
    assert fresh.requests == []


def test_run_laya_esc_resume_requires_an_existing_run_directory(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([])

    with pytest.raises(FileNotFoundError):
        _run(tmp_path, items, adapter, run_id="never-created", resume=True)
    assert adapter.requests == []


def test_esc_main_dry_run_does_not_call_factory_or_write_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, dataset, esc_path, gold_path = _write_inputs(tmp_path, (1, 2))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    factory = _ExplodingFactory()

    exit_code = module.esc_main(
        [
            "--esc-data",
            str(esc_path),
            "--gold",
            str(gold_path),
            "--output-dir",
            str(output_dir),
        ],
        adapter_factory=factory,
        runtime_metadata={},
    )

    assert exit_code == 0
    assert factory.calls == 0
    assert list(output_dir.iterdir()) == []
    output = capsys.readouterr().out.lower()
    assert str(len(dataset.records)) in output
    assert "gold" in output
    assert "cause" in output


def test_esc_main_real_path_with_fake_factory_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, dataset, esc_path, gold_path = _write_inputs(tmp_path, (1, 2))
    for key in (
        "ECI_LAYA_CHECKPOINT",
        "ECI_LAYA_REVISION",
        "ECI_LAYA_SDK_VERSION",
        "ECI_LAYA_DEVICE",
        "ECI_LAYA_DOWNLOAD_WEIGHTS",
    ):
        monkeypatch.delenv(key, raising=False)
    adapter = _ScriptedAdapter(
        [_response(_valid_raw("cause")), _response(_valid_raw("cause"))],
    )
    factory = _AdapterFactory(adapter)
    output_dir = tmp_path / "out"

    exit_code = module.esc_main(
        [
            "--authorize-real-laya",
            "--esc-data",
            str(esc_path),
            "--gold",
            str(gold_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "cli-run",
        ],
        adapter_factory=factory,
        runtime_metadata={"mode": "fake"},
    )

    assert exit_code == 0
    assert factory.calls == 1
    assert len(adapter.requests) == 2
    run_dir = output_dir / "cli-run"
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == len(dataset.records)
    assert all(record["status"] == "ok" for record in predictions)
    assert (run_dir / "raw/response.jsonl").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.md").exists()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "cli-run"
    assert meta["status"] == "ok"
    assert meta["runtime_metadata"] == {"mode": "fake"}
    output = capsys.readouterr().out
    assert "cli-run" in output
    assert "metrics" in output.lower()


class _StatusError(Exception):
    status = 500


def test_run_laya_esc_resume_reattempts_budget_exhausted_items(tmp_path: Path) -> None:
    module, _, _, items = _load_items(tmp_path, (1, 2, 3))
    first_adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])
    first = _run(
        tmp_path,
        items,
        first_adapter,
        run_id="resume-budget",
        budget=module.EscRunBudget(max_requests=1),
    )

    assert first.status == "budget_exhausted"
    assert [record.control_status for record in first.records] == [
        None,
        "budget_exhausted",
        "budget_exhausted",
    ]

    second_adapter = _ScriptedAdapter(
        [_response(_valid_raw("cause")), _response(_valid_raw("cause"))],
    )
    second = _run(
        tmp_path,
        items,
        second_adapter,
        run_id="resume-budget",
        resume=True,
        budget=module.EscRunBudget(max_requests=3),
    )

    assert second.status == "ok"
    assert len(second_adapter.requests) == 2
    assert [record.status for record in second.records] == ["ok", "ok", "ok"]
    assert [record.control_status for record in second.records] == [None, None, None]

    predictions = _read_jsonl(second.run_dir / "predictions.jsonl")
    assert [record["item_id"] for record in predictions] == [1, 2, 3]
    record_ids = [record["record_id"] for record in predictions]
    assert len(record_ids) == len(set(record_ids))
    assert all(record["status"] == "ok" for record in predictions)

    meta = json.loads((second.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ok"
    assert meta["control_status_counts"] == {}
    invocations = cast("list[Mapping[str, object]]", meta["invocations"])
    assert len(invocations) == 2
    assert invocations[0]["status"] == "budget_exhausted"
    assert invocations[0]["command"] == ["synthetic", "esc"]
    assert invocations[1]["status"] == "ok"
    assert invocations[1]["command"] == ["synthetic", "esc"]


def test_run_laya_esc_resume_rejects_stale_run_signature(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1, 2))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause")), _response(_valid_raw("cause"))])
    result = _run(tmp_path, items, adapter, run_id="stale-signature")

    meta_path = result.run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["run_signature"] = "stale-signature"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")

    resume_adapter = _ScriptedAdapter([])
    with pytest.raises(ValueError, match="run_signature"):
        _run(
            tmp_path,
            items,
            resume_adapter,
            run_id="stale-signature",
            resume=True,
        )
    assert resume_adapter.requests == []


def test_run_laya_esc_resume_rejects_stale_settings(tmp_path: Path) -> None:
    module, _, _, items = _load_items(tmp_path, (1, 2))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause")), _response(_valid_raw("cause"))])
    _run(tmp_path, items, adapter, run_id="stale-settings")

    resume_adapter = _ScriptedAdapter([])
    stale_settings = LayaSettings.from_env({"ECI_LAYA_CHECKPOINT": "other/checkpoint"})
    with pytest.raises(ValueError, match="run_signature"):
        module.run_laya_esc(
            items=items,
            adapter=resume_adapter,
            output_dir=tmp_path / "runs",
            run_id="stale-settings",
            settings=stale_settings,
            runtime_metadata={},
            budget=module.EscRunBudget(max_requests=2),
            resume=True,
            command=["synthetic", "esc"],
        )
    assert resume_adapter.requests == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "actual-foreign-run"),
        ("request_hash", "0" * 64),
    ],
)
def test_run_laya_esc_resume_rejects_foreign_prediction_lineage(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    _, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])
    result = _run(tmp_path, items, adapter, run_id="foreign-run")

    predictions_path = result.run_dir / "predictions.jsonl"
    records = _read_jsonl(predictions_path)
    records[0][field] = value
    _write_jsonl(predictions_path, records)

    resume_adapter = _ScriptedAdapter([])
    with pytest.raises(ValueError, match=field):
        _run(
            tmp_path,
            items,
            resume_adapter,
            run_id="foreign-run",
            resume=True,
        )
    assert resume_adapter.requests == []


def test_run_laya_esc_resume_recovers_orphan_success_without_adapter_call(
    tmp_path: Path,
) -> None:
    _, _, _, items = _load_items(tmp_path, (1, 2))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause")), _response(_valid_raw("cause"))])
    result = _run(tmp_path, items, adapter, run_id="orphan-success")

    predictions_path = result.run_dir / "predictions.jsonl"
    raw_entries = _read_jsonl(result.run_dir / "raw/response.jsonl")
    records = _read_jsonl(predictions_path)
    _write_jsonl(predictions_path, records[1:])

    resume_adapter = _ScriptedAdapter([])
    resumed = _run(
        tmp_path,
        items,
        resume_adapter,
        run_id="orphan-success",
        resume=True,
    )

    assert resume_adapter.requests == []
    assert resumed.status == "ok"
    assert resumed.request_count == 2
    recovered = resumed.records[0]
    assert recovered.status == "ok"
    assert recovered.prediction == "cause"
    assert recovered.attempt == 1
    assert recovered.response_hash == raw_entries[0]["response_hash"]
    predictions = _read_jsonl(resumed.run_dir / "predictions.jsonl")
    assert [record["item_id"] for record in predictions] == [1, 2]
    assert len({record["record_id"] for record in predictions}) == 2


def test_run_laya_esc_resume_recovers_orphan_api_error(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([RuntimeError("boom")])
    result = _run(tmp_path, items, adapter, run_id="orphan-error")

    assert result.status == "completed_with_errors"
    _write_jsonl(result.run_dir / "predictions.jsonl", [])
    raw_entries = _read_jsonl(result.run_dir / "raw/response.jsonl")

    resume_adapter = _ScriptedAdapter([])
    resumed = _run(
        tmp_path,
        items,
        resume_adapter,
        run_id="orphan-error",
        resume=True,
    )

    assert resume_adapter.requests == []
    assert resumed.status == "completed_with_errors"
    recovered = resumed.records[0]
    assert recovered.status == "api_error"
    assert recovered.control_status is None
    assert recovered.prediction is None
    assert recovered.attempt == 1
    assert recovered.response_hash == raw_entries[0]["response_hash"]
    assert "boom" in str(recovered.error).lower()


def test_run_laya_esc_resume_rejects_malformed_raw_with_line_number(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1,))
    run_dir = tmp_path / "runs" / "malformed-raw"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "response.jsonl").write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    adapter = _ScriptedAdapter([])
    with pytest.raises(ValueError, match=r"line 2"):
        _run(
            tmp_path,
            items,
            adapter,
            run_id="malformed-raw",
            resume=True,
        )
    assert adapter.requests == []


def test_run_laya_esc_resume_rejects_duplicate_raw_attempts(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])
    result = _run(tmp_path, items, adapter, run_id="duplicate-raw")

    raw_path = result.run_dir / "raw/response.jsonl"
    line = raw_path.read_text(encoding="utf-8").splitlines()[0]
    raw_path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    resume_adapter = _ScriptedAdapter([])
    with pytest.raises(ValueError, match="duplicate"):
        _run(
            tmp_path,
            items,
            resume_adapter,
            run_id="duplicate-raw",
            resume=True,
        )
    assert resume_adapter.requests == []


def test_run_laya_esc_resume_rejects_foreign_raw_run_id(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])
    result = _run(tmp_path, items, adapter, run_id="raw-owner")

    raw_path = result.run_dir / "raw/response.jsonl"
    entry = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
    entry["run_id"] = "foreign-run"
    raw_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    resume_adapter = _ScriptedAdapter([])
    with pytest.raises(ValueError, match="run_id"):
        _run(
            tmp_path,
            items,
            resume_adapter,
            run_id="raw-owner",
            resume=True,
        )
    assert resume_adapter.requests == []


def test_run_laya_esc_retry_budget_exhaustion_preserves_failure(tmp_path: Path) -> None:
    module, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([_StatusError("server down")])

    result = _run(
        tmp_path,
        items,
        adapter,
        budget=module.EscRunBudget(max_requests=1, max_retries=1),
    )

    assert result.status == "budget_exhausted"
    record = result.records[0]
    assert record.status == "api_error"
    assert record.control_status == "budget_exhausted"
    assert "server down" in str(record.error)
    assert "retry" in str(record.error).lower()
    assert len(adapter.requests) == 1


def test_run_meta_records_signature_and_source_diff(tmp_path: Path) -> None:
    _, _, _, items = _load_items(tmp_path, (1,))
    adapter = _ScriptedAdapter([_response(_valid_raw("cause"))])

    result = _run(tmp_path, items, adapter, run_id="metadata-run")

    meta = json.loads((result.run_dir / "meta.json").read_text(encoding="utf-8"))
    signature = meta["run_signature"]
    assert isinstance(signature, str)
    assert len(signature) == 64
    code = cast("Mapping[str, object]", meta["code"])
    assert "source_diff_sha256" in code
    diff_sha = code["source_diff_sha256"]
    assert diff_sha is None or (isinstance(diff_sha, str) and len(diff_sha) == 64)
    invocations = cast("list[Mapping[str, object]]", meta["invocations"])
    assert len(invocations) == 1
    assert invocations[0]["status"] == "ok"


def test_esc_main_records_actual_invocation_vector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _, esc_path, gold_path = _write_inputs(tmp_path, (1, 2))
    for key in (
        "ECI_LAYA_CHECKPOINT",
        "ECI_LAYA_REVISION",
        "ECI_LAYA_SDK_VERSION",
        "ECI_LAYA_DEVICE",
        "ECI_LAYA_DOWNLOAD_WEIGHTS",
    ):
        monkeypatch.delenv(key, raising=False)
    adapter = _ScriptedAdapter(
        [_response(_valid_raw("cause")), _response(_valid_raw("cause"))],
    )
    factory = _AdapterFactory(adapter)
    argv = [
        "--authorize-real-laya",
        "--esc-data",
        str(esc_path),
        "--gold",
        str(gold_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "invocation-run",
    ]

    exit_code = module.esc_main(
        argv,
        adapter_factory=factory,
        runtime_metadata={"mode": "fake"},
    )

    assert exit_code == 0
    meta = json.loads(
        (tmp_path / "out" / "invocation-run" / "meta.json").read_text(encoding="utf-8"),
    )
    assert meta["command"] == [
        "python",
        "-m",
        "eci_decision_models.esc_run",
        *argv,
    ]
