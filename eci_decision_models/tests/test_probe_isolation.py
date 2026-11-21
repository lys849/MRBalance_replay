"""Public contracts for the isolated, hard-wall-clock Laya probe runner."""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing
import os
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from eci_decision_models.laya import (
    LayaResponse,
    LayaSettings,
    build_laya_request,
    parse_laya_response,
)
from eci_decision_models.probe import (
    ProbeBudget,
    ProbeCase,
    ProbeRecord,
    ProbeRunResult,
    default_probe_cases,
    probe_main,
    run_laya_probe_isolated,
)

TEST_TIME_LIMIT_SECONDS = 15.0


def _success_response() -> LayaResponse:
    raw: dict[str, object] = {
        "answers": {
            "eci_relation": {
                "choice": "cause",
                "probabilities": {"cause": 0.6, "caused_by": 0.3, "none": 0.1},
            }
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "model": "isolated-fake",
    }
    return LayaResponse(
        raw=raw,
        validated=parse_laya_response(raw),
        usage={"input_tokens": 1, "output_tokens": 1},
        model="isolated-fake",
    )


def _invalid_response() -> LayaResponse:
    raw: dict[str, object] = {"answers": "not-a-mapping"}
    return LayaResponse(
        raw=raw,
        validated=parse_laya_response(raw),
        usage={},
        model="isolated-fake",
    )


class _SuccessAdapter:
    def predict(self, request: object) -> LayaResponse:
        return _success_response()


class _SuccessFactory:
    def __call__(self) -> _SuccessAdapter:
        return _SuccessAdapter()


class _HangFirstAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, request: object) -> LayaResponse:
        self.calls += 1
        if self.calls == 1:
            threading.Event().wait()
        return _success_response()


class _HangFirstFactory:
    def __call__(self) -> _HangFirstAdapter:
        return _HangFirstAdapter()


class _SuccessThenHangAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, request: object) -> LayaResponse:
        self.calls += 1
        if self.calls == 2:
            threading.Event().wait()
        return _success_response()


class _SuccessThenHangFactory:
    def __call__(self) -> _SuccessThenHangAdapter:
        return _SuccessThenHangAdapter()


class _HangingFactory:
    def __call__(self) -> object:
        threading.Event().wait()
        raise AssertionError("unreachable")


class _InvalidAdapter:
    def predict(self, request: object) -> LayaResponse:
        return _invalid_response()


class _InvalidFactory:
    def __call__(self) -> _InvalidAdapter:
        return _InvalidAdapter()


class _TimeoutAdapter:
    def predict(self, request: object) -> LayaResponse:
        raise TimeoutError("forced isolated timeout")


class _TimeoutFactory:
    def __call__(self) -> _TimeoutAdapter:
        return _TimeoutAdapter()


class _LoadErrorFactory:
    def __call__(self) -> object:
        raise RuntimeError("load exploded")


class _SuccessThenExitAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, request: object) -> LayaResponse:
        self.calls += 1
        if self.calls == 2:
            os._exit(7)
        return _success_response()


class _SuccessThenExitFactory:
    def __call__(self) -> _SuccessThenExitAdapter:
        return _SuccessThenExitAdapter()


class _WriteAllThenExitFactory:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id

    def __call__(self) -> object:
        cases = list(default_probe_cases())
        raw_path = self.run_dir / "raw" / "response.jsonl"
        predictions_path = self.run_dir / "predictions.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            predictions_path,
            [
                _iso_prediction(case, self.run_id, index).to_dict()
                for index, case in enumerate(cases)
            ],
        )
        _write_jsonl(
            raw_path,
            [_iso_raw_entry(case, self.run_id) for case in cases],
        )
        os._exit(7)


class _WriteRawOnlyThenExitFactory:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id

    def __call__(self) -> object:
        case = default_probe_cases()[0]
        raw_path = self.run_dir / "raw" / "response.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(raw_path, [_iso_raw_entry(case, self.run_id)])
        os._exit(9)


class _WriteAllThenHangFactory:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id

    def __call__(self) -> object:
        cases = list(default_probe_cases())
        raw_path = self.run_dir / "raw" / "response.jsonl"
        predictions_path = self.run_dir / "predictions.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            predictions_path,
            [
                _iso_prediction(case, self.run_id, index).to_dict()
                for index, case in enumerate(cases)
            ],
        )
        _write_jsonl(
            raw_path,
            [_iso_raw_entry(case, self.run_id) for case in cases],
        )
        threading.Event().wait()
        raise AssertionError("unreachable")


class _WriteMixedThenHangFactory:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id

    def __call__(self) -> object:
        cases = list(default_probe_cases())
        raw_path = self.run_dir / "raw" / "response.jsonl"
        predictions_path = self.run_dir / "predictions.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        predictions: list[dict[str, object]] = [_iso_prediction(cases[0], self.run_id, 0).to_dict()]
        predictions.extend(
            _iso_budget_prediction(case, self.run_id, index)
            for index, case in enumerate(cases[1:], start=1)
        )
        raw_entries: list[dict[str, object]] = [_iso_raw_entry(cases[0], self.run_id)]
        raw_entries.extend(_iso_budget_raw_entry(case, self.run_id) for case in cases[1:])
        _write_jsonl(predictions_path, predictions)
        _write_jsonl(raw_path, raw_entries)
        threading.Event().wait()
        raise AssertionError("unreachable")


class _RawWriteThenExitAdapter:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id

    def predict(self, request: object) -> LayaResponse:
        case = default_probe_cases()[0]
        raw_path = self.run_dir / "raw" / "response.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(raw_path, [_iso_raw_entry(case, self.run_id)])
        os._exit(9)


class _RawWriteThenExitFactory:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id

    def __call__(self) -> _RawWriteThenExitAdapter:
        return _RawWriteThenExitAdapter(self.run_dir, self.run_id)


@contextlib.contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"test guard exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _read_meta(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    return cast("dict[str, Any]", payload)


def _assert_meta_hashes(run_dir: Path) -> dict[str, Any]:
    meta = _read_meta(run_dir)
    predictions = (run_dir / "predictions.jsonl").read_bytes()
    raw = (run_dir / "raw/response.jsonl").read_bytes()
    assert meta["predictions_sha256"] == hashlib.sha256(predictions).hexdigest()
    assert meta["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    return meta


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _iso_raw_response() -> dict[str, object]:
    return {
        "answers": {
            "eci_relation": {
                "choice": "cause",
                "probabilities": {"cause": 0.6, "caused_by": 0.3, "none": 0.1},
            }
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "model": "isolated-fake",
    }


def _iso_raw_entry(case: ProbeCase, run_id: str, attempt: int = 1) -> dict[str, object]:
    request = build_laya_request(case.sentence, case.event1, case.event2)
    raw_response = _iso_raw_response()
    return {
        "case_id": case.case_id,
        "attempt": attempt,
        "status": "ok",
        "state": request.state,
        "questions": request.questions,
        "canonical": {
            "payload": request.canonical.payload,
            "options": request.canonical.options,
            "sha256": request.canonical.sha256,
        },
        "request_hash": request.canonical.sha256,
        "response_hash": _canonical_sha256(raw_response),
        "raw_response": raw_response,
        "error": None,
        "latency_ms": 0.0,
    }


def _iso_prediction(case: ProbeCase, run_id: str, index: int, attempt: int = 1) -> ProbeRecord:
    request = build_laya_request(case.sentence, case.event1, case.event2)
    return ProbeRecord(
        run_id=run_id,
        case_id=case.case_id,
        record_id=f"probe:{request.canonical.sha256}",
        kind=case.kind,
        expected_choice=case.expected_choice,
        model="isolated-fake",
        attempt=attempt,
        raw_request_id=f"{run_id}:{case.case_id}:{attempt}",
        request_hash=request.canonical.sha256,
        response_hash=_canonical_sha256(_iso_raw_response()),
        adapter_version="isolated-test-adapter",
        parser_version="isolated-test-parser",
        prediction="cause",
        status="ok",
        prediction_valid=True,
        probability_valid=True,
        probabilities=(0.6, 0.3, 0.1),
        error=None,
        latency_ms=0.0,
    )


def _json_roundtrip(value: object) -> object:
    return json.loads(_canonical_json(value))


def _iso_budget_prediction(case: ProbeCase, run_id: str, index: int) -> dict[str, object]:
    record = _iso_prediction(case, run_id, index).to_dict()
    record.update(
        {
            "prediction": None,
            "status": "api_error",
            "control_status": "budget_exhausted",
            "prediction_valid": False,
            "probability_valid": False,
            "probabilities": None,
            "error": "request budget exhausted before this case ran",
            "response_hash": _canonical_sha256(None),
        }
    )
    return record


def _iso_budget_raw_entry(case: ProbeCase, run_id: str) -> dict[str, object]:
    entry = _iso_raw_entry(case, run_id)
    entry.update(
        {
            "status": "api_error",
            "raw_response": None,
            "error": "request budget exhausted before this case ran",
            "response_hash": _canonical_sha256(None),
        }
    )
    return entry


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(_canonical_json(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


def _run_isolated(
    tmp_path: Path,
    factory: Any,
    *,
    run_id: str,
    budget: ProbeBudget,
    cases: list[Any] | None = None,
) -> tuple[ProbeRunResult, Path, float]:
    selected_cases = cases if cases is not None else list(default_probe_cases())
    settings = LayaSettings.from_env({})
    started = time.monotonic()
    with _time_limit(TEST_TIME_LIMIT_SECONDS):
        result = run_laya_probe_isolated(
            cases=selected_cases,
            adapter_factory=factory,
            output_dir=tmp_path,
            run_id=run_id,
            settings=settings,
            budget=budget,
            code_metadata={"commit": "isolated-test", "dirty": False},
        )
    elapsed = time.monotonic() - started
    return result, tmp_path / run_id, elapsed


def _find_probe_run(output_dir: Path) -> Path:
    if (output_dir / "meta.json").exists():
        return output_dir
    run_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    return run_dirs[0]


def test_isolated_first_case_hang_marks_timeout_and_budget_exhausted(
    tmp_path: Path,
) -> None:
    result, run_dir, elapsed = _run_isolated(
        tmp_path,
        _HangFirstFactory(),
        run_id="iso-hang-first",
        budget=ProbeBudget(max_wall_seconds=1.0),
    )

    assert isinstance(result, ProbeRunResult)
    assert elapsed < 8.0
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == 6
    assert predictions[0]["status"] == "api_error"
    assert "timeout" in str(predictions[0]["error"]).lower()
    assert predictions[0]["prediction"] is None
    assert predictions[0]["probabilities"] is None
    assert [record["status"] for record in predictions[1:]] == ["api_error"] * 5
    assert [record["control_status"] for record in predictions[1:]] == ["budget_exhausted"] * 5

    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert raw_entries
    assert all(entry["case_id"] == predictions[0]["case_id"] for entry in raw_entries)
    assert raw_entries[0]["status"] == "api_error"

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] == "budget_exhausted"
    assert meta["stopped"] is True
    assert meta["status_counts"].get("api_error") == 6
    assert meta["control_status_counts"].get("budget_exhausted") == 5
    assert meta["termination"]["timed_out"] is True
    assert meta["termination"]["exitcode"] is not None
    assert meta["termination"]["exitcode"] != 0


def test_isolated_incremental_flush_keeps_first_success_before_second_hang(
    tmp_path: Path,
) -> None:
    _, run_dir, elapsed = _run_isolated(
        tmp_path,
        _SuccessThenHangFactory(),
        run_id="iso-incremental",
        budget=ProbeBudget(max_wall_seconds=2.0),
    )

    assert elapsed < 8.0
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == 6
    assert predictions[0]["status"] == "ok"
    assert predictions[1]["status"] == "api_error"
    assert "timeout" in str(predictions[1]["error"]).lower()
    assert [record["status"] for record in predictions[2:]] == ["api_error"] * 4
    assert [record["control_status"] for record in predictions[2:]] == ["budget_exhausted"] * 4

    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert len(raw_entries) == 2
    assert [entry["status"] for entry in raw_entries] == ["ok", "api_error"]
    assert raw_entries[0]["case_id"] == predictions[0]["case_id"]
    assert raw_entries[1]["case_id"] == predictions[1]["case_id"]

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] == "budget_exhausted"
    assert meta["status_counts"].get("ok") == 1
    assert meta["status_counts"].get("api_error") == 5
    assert meta["control_status_counts"].get("budget_exhausted") == 4
    assert meta["termination"]["timed_out"] is True
    assert meta["termination"]["exitcode"] is not None
    assert meta["termination"]["exitcode"] != 0


def test_isolated_factory_hang_still_writes_explicit_unrun_records(
    tmp_path: Path,
) -> None:
    _, run_dir, elapsed = _run_isolated(
        tmp_path,
        _HangingFactory(),
        run_id="iso-hang-factory",
        budget=ProbeBudget(max_wall_seconds=1.0),
    )

    assert elapsed < 8.0
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "raw/response.jsonl").exists()
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == 6
    assert [record["status"] for record in predictions] == ["api_error"] * 6
    assert [record["control_status"] for record in predictions] == ["budget_exhausted"] * 6
    assert all(record["prediction"] is None for record in predictions)
    assert all(record["probabilities"] is None for record in predictions)
    assert all(record["error"] for record in predictions)
    assert _read_jsonl(run_dir / "raw/response.jsonl") == []

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] == "budget_exhausted"
    assert meta["stopped"] is True
    assert meta["status_counts"].get("api_error") == 6
    assert meta["control_status_counts"].get("budget_exhausted") == 6
    assert meta["termination"]["timed_out"] is True
    assert meta["termination"]["exitcode"] is not None
    assert meta["termination"]["exitcode"] != 0


def test_isolated_normal_run_succeeds_and_leaves_no_child(tmp_path: Path) -> None:
    result, run_dir, elapsed = _run_isolated(
        tmp_path,
        _SuccessFactory(),
        run_id="iso-success",
        budget=ProbeBudget(max_wall_seconds=30.0),
    )

    assert isinstance(result, ProbeRunResult)
    assert elapsed < 8.0
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == 6
    assert [record["status"] for record in predictions] == ["ok"] * 6
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert len(raw_entries) == 6
    assert [entry["status"] for entry in raw_entries] == ["ok"] * 6

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] in {"ok", "completed"}
    assert meta["status_counts"].get("ok") == 6
    assert meta["termination"]["timed_out"] is False
    assert meta["termination"]["exitcode"] == 0

    for _ in range(20):
        if not multiprocessing.active_children():
            break
        time.sleep(0.05)
    assert multiprocessing.active_children() == []


def test_isolated_factory_load_error_is_fatal_not_budget(tmp_path: Path) -> None:
    _, run_dir, elapsed = _run_isolated(
        tmp_path,
        _LoadErrorFactory(),
        run_id="iso-load-error",
        budget=ProbeBudget(max_wall_seconds=10.0),
    )

    assert elapsed < 8.0
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == 6
    assert all(record["status"] in {"api_error", "failed"} for record in predictions)
    assert all(record["control_status"] is None for record in predictions)
    assert all(record["prediction"] is None for record in predictions)

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] in {"completed_with_errors", "failed"}
    assert "fatal_error" in meta
    assert "load exploded" in json.dumps(meta["fatal_error"]).lower()
    assert meta["control_status_counts"].get("budget_exhausted", 0) == 0
    assert meta["termination"]["timed_out"] is False
    assert meta["termination"]["exitcode"] not in {None, 0}


def test_isolated_native_crash_after_success_is_api_error_not_timeout(
    tmp_path: Path,
) -> None:
    _, run_dir, elapsed = _run_isolated(
        tmp_path,
        _SuccessThenExitFactory(),
        run_id="iso-native-crash",
        budget=ProbeBudget(max_wall_seconds=10.0),
    )

    assert elapsed < 8.0
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == 6
    assert predictions[0]["status"] == "ok"
    assert predictions[1]["status"] == "api_error"
    second_error = str(predictions[1]["error"]).lower()
    assert "exited" in second_error or "child exited" in second_error
    assert "7" in second_error
    assert "timeout" not in second_error
    for record in predictions[2:]:
        assert record["status"] == "api_error"
        assert record["control_status"] == "not_run"
        assert record["prediction"] is None
        assert "child exited" in str(record["error"]).lower()

    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert raw_entries[0]["status"] == "ok"
    assert all("timeout" not in str(entry.get("error", "")).lower() for entry in raw_entries)

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] in {"completed_with_errors", "failed"}
    assert meta["status"] != "budget_exhausted"
    assert meta["control_status_counts"].get("budget_exhausted", 0) == 0
    assert meta["termination"]["timed_out"] is False
    assert meta["termination"]["exitcode"] == 7


def test_isolated_child_writes_all_records_before_exit(tmp_path: Path) -> None:
    run_id = "iso-all-before-meta"
    run_dir = tmp_path / run_id
    _, returned_run_dir, elapsed = _run_isolated(
        tmp_path,
        _WriteAllThenExitFactory(run_dir, run_id),
        run_id=run_id,
        budget=ProbeBudget(max_wall_seconds=10.0),
    )

    assert elapsed < 8.0
    assert returned_run_dir == run_dir
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert len(predictions) == 6
    assert len(raw_entries) == 6
    assert [record["status"] for record in predictions] == ["ok"] * 6
    assert all(record["control_status"] is None for record in predictions)
    assert [record["response_hash"] for record in predictions] == [
        entry["response_hash"] for entry in raw_entries
    ]

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] in {"ok", "completed", "completed_with_errors", "failed"}
    assert meta["status"] != "budget_exhausted"
    assert meta["status_counts"].get("ok") == 6
    assert meta["control_status_counts"].get("budget_exhausted", 0) == 0
    assert meta["request_count"] == 6
    assert meta["termination"]["timed_out"] is False
    assert meta["termination"]["exitcode"] == 7


def test_isolated_raw_only_reconciles_one_prediction_without_duplicate_timeout(
    tmp_path: Path,
) -> None:
    run_id = "iso-raw-only"
    run_dir = tmp_path / run_id
    _, returned_run_dir, elapsed = _run_isolated(
        tmp_path,
        _WriteRawOnlyThenExitFactory(run_dir, run_id),
        run_id=run_id,
        budget=ProbeBudget(max_wall_seconds=10.0),
    )

    assert elapsed < 8.0
    assert returned_run_dir == run_dir
    cases = list(default_probe_cases())
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert len(raw_entries) == 1
    assert raw_entries[0]["case_id"] == cases[0].case_id
    assert raw_entries[0]["attempt"] == 1
    assert len({(entry["case_id"], entry["attempt"]) for entry in raw_entries}) == 1

    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == 1
    assert predictions[0]["case_id"] == raw_entries[0]["case_id"]
    assert predictions[0]["attempt"] == raw_entries[0]["attempt"] == 1
    assert predictions[0]["status"] == "ok"
    assert predictions[0]["response_hash"] == raw_entries[0]["response_hash"]

    meta = _assert_meta_hashes(run_dir)
    assert meta["request_count"] == 1
    assert meta["status"] != "budget_exhausted"
    assert meta["control_status_counts"].get("budget_exhausted", 0) == 0


def test_isolated_hang_after_all_records_keeps_completion_not_budget(
    tmp_path: Path,
) -> None:
    run_id = "iso-hang-after-complete"
    run_dir = tmp_path / run_id
    _, returned_run_dir, elapsed = _run_isolated(
        tmp_path,
        _WriteAllThenHangFactory(run_dir, run_id),
        run_id=run_id,
        budget=ProbeBudget(max_wall_seconds=2.0),
    )

    assert elapsed < 8.0
    assert returned_run_dir == run_dir
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert len(predictions) == 6
    assert len(raw_entries) == 6
    assert [record["status"] for record in predictions] == ["ok"] * 6
    assert all(record["control_status"] is None for record in predictions)

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] in {"ok", "completed"}
    assert meta["stopped"] is False
    assert meta["status_counts"].get("ok") == 6
    assert meta["control_status_counts"].get("budget_exhausted", 0) == 0
    termination = meta["termination"]
    assert termination["timed_out"] is True
    assert termination["exitcode"] is not None
    assert termination["exitcode"] != 0
    after_completion = termination.get(
        "termination_after_completion",
        meta.get("termination_after_completion"),
    )
    assert after_completion is True


def test_isolated_hang_preserves_prewritten_budget_records_and_marks_timeout(
    tmp_path: Path,
) -> None:
    run_id = "iso-precomputed-budget"
    run_dir = tmp_path / run_id
    cases = list(default_probe_cases())
    expected_predictions = [_json_roundtrip(_iso_prediction(cases[0], run_id, 0).to_dict())] + [
        _json_roundtrip(_iso_budget_prediction(case, run_id, index))
        for index, case in enumerate(cases[1:], start=1)
    ]
    expected_raw = [_json_roundtrip(_iso_raw_entry(cases[0], run_id))] + [
        _json_roundtrip(_iso_budget_raw_entry(case, run_id)) for case in cases[1:]
    ]

    _, returned_run_dir, elapsed = _run_isolated(
        tmp_path,
        _WriteMixedThenHangFactory(run_dir, run_id),
        run_id=run_id,
        budget=ProbeBudget(max_wall_seconds=2.0),
    )

    assert elapsed < 8.0
    assert returned_run_dir == run_dir
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert predictions == expected_predictions
    assert raw_entries == expected_raw
    assert [record["status"] for record in predictions] == ["ok"] + ["api_error"] * 5
    assert [record["control_status"] for record in predictions[1:]] == ["budget_exhausted"] * 5
    assert predictions[0]["prediction"] == "cause"
    assert all(record["prediction"] is None for record in predictions[1:])

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] == "budget_exhausted"
    assert meta["stopped"] is True
    assert meta["status_counts"].get("api_error") == 5
    assert meta["control_status_counts"].get("budget_exhausted") == 5
    termination = meta["termination"]
    assert termination["timed_out"] is True
    assert termination["exitcode"] is not None
    assert termination["exitcode"] != 0
    assert termination.get("termination_after_completion") in {None, False}
    assert meta.get("termination_after_completion") in {None, False}


def test_isolated_runtime_raw_then_exit_reconciles_without_budget(
    tmp_path: Path,
) -> None:
    run_id = "iso-runtime-raw-exit"
    run_dir = tmp_path / run_id
    _, returned_run_dir, elapsed = _run_isolated(
        tmp_path,
        _RawWriteThenExitFactory(run_dir, run_id),
        run_id=run_id,
        budget=ProbeBudget(max_wall_seconds=10.0),
    )

    assert elapsed < 8.0
    assert returned_run_dir == run_dir
    cases = list(default_probe_cases())
    raw_entries = _read_jsonl(run_dir / "raw/response.jsonl")
    assert len(raw_entries) == 1
    assert raw_entries[0]["case_id"] == cases[0].case_id
    assert raw_entries[0]["attempt"] == 1
    assert len({(entry["case_id"], entry["attempt"]) for entry in raw_entries}) == 1

    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == 6
    assert predictions[0]["case_id"] == raw_entries[0]["case_id"]
    assert predictions[0]["attempt"] == raw_entries[0]["attempt"] == 1
    assert predictions[0]["status"] == "ok"
    assert predictions[0]["response_hash"] == raw_entries[0]["response_hash"]
    for record in predictions[1:]:
        assert record["status"] == "api_error"
        assert record["control_status"] == "not_run"
        assert record["prediction"] is None
    assert all(record["control_status"] != "budget_exhausted" for record in predictions)

    meta = _assert_meta_hashes(run_dir)
    assert meta["status"] != "budget_exhausted"
    assert meta["control_status_counts"].get("budget_exhausted", 0) == 0
    assert meta["request_count"] >= 1
    assert meta["termination"]["timed_out"] is False
    assert meta["termination"]["exitcode"] == 9


def _run_cli(output_dir: Path, factory: Any, extra_args: list[str]) -> int:
    argv = ["--authorize-real-laya", "--output-dir", str(output_dir), *extra_args]
    with _time_limit(TEST_TIME_LIMIT_SECONDS):
        return probe_main(argv, adapter_factory=factory)


def test_probe_main_returns_zero_for_successful_isolated_run(tmp_path: Path) -> None:
    exit_code = _run_cli(tmp_path, _SuccessFactory(), [])

    assert exit_code == 0
    run_dir = _find_probe_run(tmp_path)
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    meta = _assert_meta_hashes(run_dir)
    assert [record["status"] for record in predictions] == ["ok"] * 6
    assert meta["status"] in {"ok", "completed"}
    assert meta["status_counts"].get("ok") == 6


@pytest.mark.parametrize(
    ("factory", "extra_args", "expected_status", "expected_counts"),
    [
        (
            _InvalidFactory(),
            [],
            "completed_with_errors",
            {"invalid_response": 6},
        ),
        (
            _TimeoutFactory(),
            [],
            None,
            {"api_error": 6},
        ),
        (
            _SuccessFactory(),
            ["--max-requests", "1"],
            "budget_exhausted",
            {"ok": 1, "api_error": 5},
        ),
    ],
    ids=["invalid-response", "api-error", "budget-exhausted"],
)
def test_probe_main_returns_nonzero_for_error_runs(
    tmp_path: Path,
    factory: Any,
    extra_args: list[str],
    expected_status: str | None,
    expected_counts: dict[str, int],
) -> None:
    exit_code = _run_cli(tmp_path, factory, extra_args)

    assert exit_code != 0
    run_dir = _find_probe_run(tmp_path)
    meta = _assert_meta_hashes(run_dir)
    if expected_status is not None:
        assert meta["status"] == expected_status
    else:
        assert meta["status"] in {"api_error", "completed_with_errors"}
    for status, count in expected_counts.items():
        assert meta["status_counts"].get(status) == count
