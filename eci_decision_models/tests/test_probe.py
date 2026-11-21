"""Public contracts for the synthetic (non-ESC) Laya probe runner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import eci_decision_models
from eci_decision_models.canonical import CHOICE_IDS, PROMPT_VERSION, SCHEMA_VERSION
from eci_decision_models.laya import (
    LayaRequest,
    LayaResponse,
    LayaSettings,
    parse_laya_response,
)
from eci_decision_models.probability import PredictionStatus
from eci_decision_models.probe import (
    ProbeBudget,
    ProbeCase,
    ProbeRunResult,
    default_probe_cases,
    probe_main,
    run_laya_probe,
)

REDACTED = "[REDACTED]"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FIELDS = {
    "run_id",
    "case_id",
    "record_id",
    "kind",
    "expected_choice",
    "model",
    "attempt",
    "raw_request_id",
    "request_hash",
    "response_hash",
    "adapter_version",
    "parser_version",
    "prediction",
    "status",
    "prediction_valid",
    "probability_valid",
    "probabilities",
    "error",
    "latency_ms",
    "control_status",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _executed_package_sha256() -> str:
    package_dir = Path(eci_decision_models.__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(
        package_dir.rglob("*.py"),
        key=lambda path: path.relative_to(package_dir).as_posix(),
    )
    for path in files:
        digest.update(path.relative_to(package_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _case(
    case_id: str,
    kind: str,
    expected_choice: str | None = None,
) -> ProbeCase:
    return ProbeCase(
        case_id=case_id,
        kind=kind,
        sentence=f"Synthetic {kind} sentence.",
        event1=f"{kind} event1",
        event2=f"{kind} event2",
        expected_choice=expected_choice,
    )


def _valid_raw() -> dict[str, object]:
    return {
        "answers": {
            "eci_relation": {
                "choice": "cause",
                "probabilities": {
                    "cause": 0.6000,
                    "caused_by": 0.3000,
                    "none": 0.2000,
                },
            }
        },
        "usage": {
            "input_tokens": 10,
            "state_tokens": 4,
            "output_tokens": 2,
        },
        "nested": {
            "accessToken": "nested-secret-token",
            "input_tokens": 7,
        },
        "apiKey": "camel-secret-key",
        "API-KEY": "hyphen-secret-key",
        "model": "fake-laya",
    }


def _missing_raw() -> dict[str, object]:
    return {
        "answers": {"eci_relation": {"choice": "none"}},
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
    usage = raw.get("usage")
    model = raw.get("model")
    return LayaResponse(
        raw=raw,
        validated=parse_laya_response(raw),
        usage=copy.deepcopy(usage) if isinstance(usage, dict) else {},
        model=model if isinstance(model, str) else None,
    )


class _ScriptedAdapter:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[LayaRequest] = []
        self.responses: list[LayaResponse] = []
        self.raw_before: list[object] = []

    def predict(self, request: LayaRequest) -> LayaResponse:
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("scripted adapter ran out of outcomes")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, LayaResponse)
        self.responses.append(outcome)
        self.raw_before.append(copy.deepcopy(outcome.raw))
        return outcome


def _raw_canonical(entry: Mapping[str, object]) -> Mapping[str, object]:
    canonical = entry.get("canonical")
    if isinstance(canonical, Mapping):
        return canonical
    return {
        "payload": entry.get("canonical_payload"),
        "sha256": entry.get("canonical_sha256"),
        "options": entry.get("canonical_options"),
    }


def _raw_response(entry: Mapping[str, object]) -> Mapping[str, object]:
    value = entry.get("raw_response")
    if isinstance(value, Mapping):
        return value
    value = entry.get("response")
    if isinstance(value, Mapping):
        return value
    raise AssertionError("raw attempt is missing its raw response")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _run_two_cases(
    tmp_path: Path,
    *,
    budget: ProbeBudget | None = None,
    code_metadata: Mapping[str, object] | None = None,
) -> tuple[
    ProbeRunResult,
    _ScriptedAdapter,
    list[ProbeCase],
    Path,
    LayaSettings,
    dict[str, object],
]:
    cases = [
        _case("case-cause", "cause", "cause"),
        _case("case-empty", "empty", None),
    ]
    adapter = _ScriptedAdapter([_response(_valid_raw()), _response(_missing_raw())])
    settings = LayaSettings.from_env({})
    runtime_metadata: dict[str, object] = {
        "revision": settings.revision,
        "model_id": settings.checkpoint,
        "device": "cpu",
        "dtype": "float32",
    }
    run_id = "probe-run"
    result = run_laya_probe(
        cases=cases,
        adapter=adapter,
        output_dir=tmp_path,
        run_id=run_id,
        settings=settings,
        runtime_metadata=runtime_metadata,
        budget=budget or ProbeBudget(),
        code_metadata=code_metadata,
    )
    return result, adapter, cases, tmp_path / run_id, settings, runtime_metadata


def test_default_probe_cases_are_fixed_synthetic_six() -> None:
    cases = list(default_probe_cases())

    assert len(cases) == 6
    assert isinstance(cases[0], ProbeCase)
    assert [case.kind for case in cases] == [
        "cause",
        "caused_by",
        "none",
        "negation",
        "empty",
        "long",
    ]
    assert len({case.case_id for case in cases}) == 6
    assert all(case.expected_choice is None or case.expected_choice in CHOICE_IDS for case in cases)
    assert all(isinstance(case.sentence, str) for case in cases)
    assert all(not hasattr(case, "gold_choice") for case in cases)
    assert list(default_probe_cases()) == cases


def test_probe_budget_defaults() -> None:
    budget = ProbeBudget()

    assert budget.max_requests == 6
    assert budget.max_retries == 0
    assert budget.max_wall_seconds == 600
    assert budget.max_cost == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_requests": -1},
        {"max_retries": -1},
        {"max_wall_seconds": -0.1},
        {"max_cost": -0.01},
        {"max_requests": True},
        {"max_requests": "6"},
        {"max_retries": 1.5},
        {"max_wall_seconds": math.inf},
        {"max_wall_seconds": math.nan},
        {"max_cost": math.inf},
        {"max_cost": math.nan},
    ],
)
def test_probe_budget_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        ProbeBudget(**kwargs)


def test_run_laya_probe_refuses_existing_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "probe-run"
    run_dir.mkdir()
    sentinel = run_dir / "predictions.jsonl"
    sentinel.write_text("do-not-overwrite\n", encoding="utf-8")

    adapter = _ScriptedAdapter([_response(_valid_raw())])
    settings = LayaSettings.from_env({})

    with pytest.raises((FileExistsError, ValueError)):
        run_laya_probe(
            cases=[_case("case-cause", "cause", "cause")],
            adapter=adapter,
            output_dir=tmp_path,
            run_id="probe-run",
            settings=settings,
            runtime_metadata={},
            budget=ProbeBudget(),
        )

    assert sentinel.read_text(encoding="utf-8") == "do-not-overwrite\n"
    assert adapter.requests == []


def test_run_laya_probe_writes_predictions_with_required_contract(tmp_path: Path) -> None:
    result, adapter, cases, run_dir, _, _ = _run_two_cases(tmp_path)

    assert isinstance(result, ProbeRunResult)
    predictions_path = run_dir / "predictions.jsonl"
    assert predictions_path.exists()
    records = _read_jsonl(predictions_path)
    assert [record["case_id"] for record in records] == [case.case_id for case in cases]

    for record in records:
        assert set(record) >= PREDICTION_FIELDS
        assert "gold_choice" not in record
        assert "ESC_rel" not in record
        assert record["run_id"] == "probe-run"
        assert isinstance(record["record_id"], str) and record["record_id"]
        assert record["record_id"] == f"probe:{record['request_hash']}"
        assert record["control_status"] is None
        assert isinstance(record["latency_ms"], (int, float))
        assert record["latency_ms"] >= 0

    cause, empty = records
    assert cause["kind"] == "cause"
    assert cause["expected_choice"] == "cause"
    assert cause["status"] == "ok"
    assert cause["prediction"] == "cause"
    assert cause["prediction_valid"] is True
    assert cause["probability_valid"] is True
    assert cause["probabilities"] == pytest.approx((0.6 / 1.1, 0.3 / 1.1, 0.2 / 1.1))

    assert empty["kind"] == "empty"
    assert empty["expected_choice"] is None
    assert empty["status"] == "missing_probability"
    assert empty["prediction"] == "none"
    assert empty["prediction_valid"] is True
    assert empty["probability_valid"] is False
    assert empty["probabilities"] is None
    assert len(adapter.requests) == 2

    output_files = {
        path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()
    }
    assert output_files == {"predictions.jsonl", "raw/response.jsonl", "meta.json"}


def test_run_laya_probe_raw_attempts_are_redacted_and_hashable(tmp_path: Path) -> None:
    _, adapter, _, run_dir, _, _ = _run_two_cases(tmp_path)

    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert [entry["attempt"] for entry in raw_entries] == [1, 1]
    assert [entry["status"] for entry in raw_entries] == ["ok", "missing_probability"]

    first = raw_entries[0]
    request = adapter.requests[0]
    canonical = _raw_canonical(first)
    assert first["state"] == request.state
    assert first["questions"] == request.questions
    assert _canonical_json(canonical["payload"]) == _canonical_json(request.canonical.payload)
    assert _canonical_json(canonical["options"]) == _canonical_json(request.canonical.options)
    assert canonical["sha256"] == request.canonical.sha256
    assert first["request_hash"] == request.canonical.sha256

    redacted: Any = _raw_response(first)
    assert redacted["apiKey"] == REDACTED
    assert redacted["API-KEY"] == REDACTED
    assert redacted["nested"]["accessToken"] == REDACTED
    assert redacted["nested"]["input_tokens"] == 7
    assert redacted["usage"]["input_tokens"] == 10
    assert redacted["usage"]["state_tokens"] == 4
    assert first["response_hash"] == _canonical_sha256(redacted)
    assert adapter.responses[0].raw == adapter.raw_before[0]

    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert predictions[0]["request_hash"] == first["request_hash"]
    assert predictions[0]["response_hash"] == first["response_hash"]


def test_run_laya_probe_meta_documents_probe(tmp_path: Path) -> None:
    code_metadata = {"commit": "deadbeefcafebabe", "dirty": True}
    _, adapter, cases, run_dir, settings, runtime_metadata = _run_two_cases(
        tmp_path,
        code_metadata=code_metadata,
    )

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    assert meta["run_id"] == "probe-run"
    assert meta["status"] in {"ok", "completed"}
    assert meta["non_esc"] is True
    assert meta["label_source"] == "synthetic_probe"
    assert meta["case_count"] == len(cases)
    assert meta["request_count"] == len(adapter.requests)
    assert meta["status_counts"].get("ok") == 1
    assert meta["status_counts"].get("missing_probability") == 1

    environment = meta["environment"]
    environment_text = json.dumps(environment, sort_keys=True)
    assert platform.python_version() in environment_text
    assert platform.machine() in environment_text
    assert platform.system() in environment_text
    packages = environment["packages"]
    assert isinstance(packages, Mapping)
    assert {
        "laya",
        "torch",
        "transformers",
        "numpy",
        "huggingface-hub",
        "safetensors",
        "eci-decision-models",
    } <= set(packages)
    assert all(value is None or isinstance(value, str) for value in packages.values())
    uv_lock = PROJECT_ROOT / "uv.lock"
    assert uv_lock.exists()
    assert environment["uv_lock_sha256"] == hashlib.sha256(uv_lock.read_bytes()).hexdigest()
    assert environment["sys_executable"] == sys.executable
    assert environment["sys_prefix"] == sys.prefix
    assert environment["executed_package_sha256"] == _executed_package_sha256()
    assert environment["cwd"] == str(Path.cwd())
    assert environment["command"]

    assert meta["settings"]["checkpoint"] == settings.checkpoint
    assert meta["settings"]["revision"] == settings.revision
    assert meta["settings"]["sdk_version"] == settings.sdk_version
    assert meta["settings"]["device"] == settings.device
    assert meta["runtime_metadata"] == runtime_metadata

    assert meta["budget"]["max_requests"] == 6
    assert meta["budget"]["max_retries"] == 0
    assert meta["budget"]["max_wall_seconds"] == 600
    assert meta["budget"]["max_cost"] == 0.0

    assert [option["id"] for option in meta["options"]] == list(CHOICE_IDS)
    assert meta["prompt_version"] == PROMPT_VERSION
    assert meta["schema_version"] == SCHEMA_VERSION

    code = meta.get("code") or meta.get("code_metadata") or meta
    assert code["commit"] == code_metadata["commit"]
    assert code["dirty"] is True

    predictions_bytes = (run_dir / "predictions.jsonl").read_bytes()
    raw_bytes = (run_dir / "raw/response.jsonl").read_bytes()
    assert meta["predictions_sha256"] == hashlib.sha256(predictions_bytes).hexdigest()
    assert meta["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()

    started = datetime.fromisoformat(str(meta["started_at_utc"]).replace("Z", "+00:00"))
    finished = datetime.fromisoformat(str(meta["finished_at_utc"]).replace("Z", "+00:00"))
    assert started.tzinfo is not None
    assert finished.tzinfo is not None
    assert finished >= started

    meta_text = json.dumps(meta, sort_keys=True)
    assert str(tmp_path) not in meta_text
    assert "metrics" not in meta_text


def test_run_laya_probe_meta_falls_back_to_cwd_for_git_and_uv_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = PROJECT_ROOT
    site_packages = tmp_path / "site-packages" / "eci_decision_models"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(
        "eci_decision_models.probe.__file__",
        str(site_packages / "probe.py"),
    )
    monkeypatch.chdir(project_root)

    if not (project_root.parent / ".git").exists():
        pytest.skip("requires git checkout; source archive has no Git metadata")

    expected_commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_uv_lock_sha256 = hashlib.sha256((project_root / "uv.lock").read_bytes()).hexdigest()

    _, _, _, run_dir, _, _ = _run_two_cases(tmp_path)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    assert meta["code"]["commit"] == expected_commit
    assert isinstance(meta["code"]["dirty"], bool)
    assert meta["environment"]["uv_lock_sha256"] == expected_uv_lock_sha256
    assert meta["environment"]["cwd"] == str(project_root)


def test_run_laya_probe_invalid_response_completes_with_errors(tmp_path: Path) -> None:
    cases = [_case("case-invalid", "none", None)]
    adapter = _ScriptedAdapter([_response(_invalid_raw())])
    settings = LayaSettings.from_env({})

    run_laya_probe(
        cases=cases,
        adapter=adapter,
        output_dir=tmp_path,
        run_id="probe-invalid",
        settings=settings,
        runtime_metadata={},
        budget=ProbeBudget(),
    )

    run_dir = tmp_path / "probe-invalid"
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    assert predictions[0]["status"] == "invalid_response"
    assert predictions[0]["prediction"] is None
    assert predictions[0]["probabilities"] is None
    assert meta["status"] == "completed_with_errors"
    assert meta["status_counts"].get("invalid_response") == 1


def test_run_laya_probe_does_not_retry_when_max_retries_is_zero(tmp_path: Path) -> None:
    cases = [_case("case-cause", "cause", "cause")]
    adapter = _ScriptedAdapter([TimeoutError("timeout"), _response(_valid_raw())])
    settings = LayaSettings.from_env({})

    run_laya_probe(
        cases=cases,
        adapter=adapter,
        output_dir=tmp_path,
        run_id="probe-timeout",
        settings=settings,
        runtime_metadata={},
        budget=ProbeBudget(max_retries=0),
    )

    run_dir = tmp_path / "probe-timeout"
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    predictions = _read_jsonl(run_dir / "predictions.jsonl")

    assert len(adapter.requests) == 1
    assert len(raw_entries) == 1
    assert raw_entries[0]["status"] == "api_error"
    assert "timeout" in str(raw_entries[0]["error"]).lower()
    assert predictions[0]["status"] == "api_error"
    assert predictions[0]["prediction"] is None
    assert predictions[0]["probabilities"] is None
    assert predictions[0]["error"]
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status_counts"].get("api_error") == 1
    assert meta["status"] in {"api_error", "completed_with_errors"}


def test_run_laya_probe_retries_timeout_once_when_allowed(tmp_path: Path) -> None:
    cases = [_case("case-cause", "cause", "cause")]
    adapter = _ScriptedAdapter([TimeoutError("first timeout"), _response(_valid_raw())])
    settings = LayaSettings.from_env({})

    run_laya_probe(
        cases=cases,
        adapter=adapter,
        output_dir=tmp_path,
        run_id="probe-retry",
        settings=settings,
        runtime_metadata={},
        budget=ProbeBudget(max_retries=1),
    )

    run_dir = tmp_path / "probe-retry"
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    predictions = _read_jsonl(run_dir / "predictions.jsonl")

    assert len(adapter.requests) == 2
    assert [entry["attempt"] for entry in raw_entries] == [1, 2]
    assert [entry["status"] for entry in raw_entries] == ["api_error", "ok"]
    assert predictions[0]["attempt"] == 2
    assert predictions[0]["status"] == "ok"
    assert predictions[0]["prediction"] == "cause"


def test_run_laya_probe_does_not_retry_runtime_error(tmp_path: Path) -> None:
    cases = [_case("case-cause", "cause", "cause")]
    adapter = _ScriptedAdapter([RuntimeError("boom"), _response(_valid_raw())])
    settings = LayaSettings.from_env({})

    run_laya_probe(
        cases=cases,
        adapter=adapter,
        output_dir=tmp_path,
        run_id="probe-runtime-error",
        settings=settings,
        runtime_metadata={},
        budget=ProbeBudget(max_retries=1),
    )

    run_dir = tmp_path / "probe-runtime-error"
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    predictions = _read_jsonl(run_dir / "predictions.jsonl")

    assert len(adapter.requests) == 1
    assert len(raw_entries) == 1
    assert raw_entries[0]["status"] == "api_error"
    assert "boom" in str(raw_entries[0]["error"]).lower()
    assert predictions[0]["status"] == "api_error"
    assert predictions[0]["prediction"] is None
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status_counts"].get("api_error") == 1
    assert meta["status"] in {"api_error", "completed_with_errors"}


def test_run_laya_probe_budget_counts_retries_and_stops_before_next_case(
    tmp_path: Path,
) -> None:
    cases = [
        _case("case-cause", "cause", "cause"),
        _case("case-none", "none", "none"),
    ]
    adapter = _ScriptedAdapter([TimeoutError("retry me"), _response(_valid_raw())])
    settings = LayaSettings.from_env({})

    run_laya_probe(
        cases=cases,
        adapter=adapter,
        output_dir=tmp_path,
        run_id="probe-budget",
        settings=settings,
        runtime_metadata={},
        budget=ProbeBudget(max_requests=2, max_retries=1),
    )

    run_dir = tmp_path / "probe-budget"
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    assert len(adapter.requests) == 2
    assert len(raw_entries) == 2
    assert [entry["case_id"] for entry in raw_entries] == ["case-cause", "case-cause"]
    assert predictions[0]["case_id"] == "case-cause"
    assert predictions[0]["status"] == "ok"
    assert predictions[1]["case_id"] == "case-none"
    assert predictions[1]["status"] == "api_error"
    assert predictions[1]["control_status"] == "budget_exhausted"
    assert predictions[1]["prediction"] is None
    assert predictions[1]["probabilities"] is None
    assert predictions[1]["error"]
    assert meta["status"] == "budget_exhausted"
    assert meta["status_counts"].get("api_error") == 1
    assert meta["control_status_counts"].get("budget_exhausted") == 1
    assert meta["stopped"] is True
    assert meta["request_count"] == 2


def _run_single_case(
    tmp_path: Path,
    *,
    run_id: str,
    case: ProbeCase,
    outcome: object,
    budget: ProbeBudget | None = None,
) -> list[dict[str, object]]:
    adapter = _ScriptedAdapter([outcome])
    run_laya_probe(
        cases=[case],
        adapter=adapter,
        output_dir=tmp_path,
        run_id=run_id,
        settings=LayaSettings.from_env({}),
        runtime_metadata={},
        budget=budget or ProbeBudget(),
    )
    return _read_jsonl(tmp_path / run_id / "predictions.jsonl")


def test_record_id_is_stable_content_fingerprint_across_run_ids(tmp_path: Path) -> None:
    case = _case("case-cause", "cause", "cause")

    first = _run_single_case(
        tmp_path,
        run_id="probe-run-a",
        case=case,
        outcome=_response(_valid_raw()),
    )[0]
    second = _run_single_case(
        tmp_path,
        run_id="probe-run-b",
        case=case,
        outcome=_response(_valid_raw()),
    )[0]

    assert first["record_id"] == second["record_id"]
    assert first["request_hash"] == second["request_hash"]


def test_record_id_never_contains_run_id_or_position(tmp_path: Path) -> None:
    run_id = "probe-run-alpha"
    record = _run_single_case(
        tmp_path,
        run_id=run_id,
        case=_case("case-cause", "cause", "cause"),
        outcome=_response(_valid_raw()),
    )[0]

    assert record["record_id"] == f"probe:{record['request_hash']}"
    record_id = record["record_id"]
    raw_request_id = record["raw_request_id"]
    assert isinstance(record_id, str)
    assert isinstance(raw_request_id, str)
    assert run_id not in record_id
    assert ":00:" not in record_id
    assert raw_request_id.startswith(f"{run_id}:")


def test_record_id_differs_for_different_content(tmp_path: Path) -> None:
    cases = [
        _case("case-cause", "cause", "cause"),
        _case("case-none", "none", "none"),
    ]
    adapter = _ScriptedAdapter([_response(_valid_raw()), _response(_valid_raw())])
    run_laya_probe(
        cases=cases,
        adapter=adapter,
        output_dir=tmp_path,
        run_id="probe-content",
        settings=LayaSettings.from_env({}),
        runtime_metadata={},
        budget=ProbeBudget(),
    )

    records = _read_jsonl(tmp_path / "probe-content" / "predictions.jsonl")
    assert len(records) == 2
    assert records[0]["record_id"] != records[1]["record_id"]


def test_every_per_record_status_is_in_frozen_enum(tmp_path: Path) -> None:
    frozen = {status.value for status in PredictionStatus}
    case = _case("case-cause", "cause", "cause")

    _run_single_case(
        tmp_path,
        run_id="probe-status-normal",
        case=case,
        outcome=_response(_valid_raw()),
    )
    _run_single_case(
        tmp_path,
        run_id="probe-status-timeout",
        case=case,
        outcome=TimeoutError("forced timeout"),
    )
    budget_cases = [
        _case("case-first", "cause", "cause"),
        _case("case-second", "none", "none"),
    ]
    adapter = _ScriptedAdapter([_response(_valid_raw())])
    run_laya_probe(
        cases=budget_cases,
        adapter=adapter,
        output_dir=tmp_path,
        run_id="probe-status-budget",
        settings=LayaSettings.from_env({}),
        runtime_metadata={},
        budget=ProbeBudget(max_requests=1),
    )

    for run_id in (
        "probe-status-normal",
        "probe-status-timeout",
        "probe-status-budget",
    ):
        records = _read_jsonl(tmp_path / run_id / "predictions.jsonl")
        assert records
        assert all(record["status"] in frozen for record in records)

    budget_records = _read_jsonl(tmp_path / "probe-status-budget" / "predictions.jsonl")
    assert budget_records[1]["control_status"] == "budget_exhausted"


class _FakeFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        raise AssertionError("dry-run must not create an adapter")


def test_probe_main_defaults_to_dry_run_without_calling_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ECI_ESC_DATA", str(tmp_path / "does-not-exist.jsonl"))
    factory = _FakeFactory()

    exit_code = probe_main([], adapter_factory=factory)

    assert exit_code == 0
    assert factory.calls == 0
    output = capsys.readouterr().out
    assert "6" in output
    lowered = output.lower()
    assert "hash" in lowered or "sha256" in lowered
