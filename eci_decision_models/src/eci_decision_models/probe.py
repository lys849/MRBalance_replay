"""Synthetic, non-ESC probe runner for the Laya adapter seam.

The runner only ever uses the fixed synthetic probe cases defined in this
module.  It does not read ESC data, does not treat ``expected_choice`` as an ESC
gold label, and does not create or load a real model unless the CLI caller
explicitly authorizes that path.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

from eci_decision_models.artifacts import redact_secrets, sha256_file
from eci_decision_models.canonical import (
    CHOICE_DESCRIPTIONS,
    CHOICE_IDS,
    CHOICE_LETTERS,
    PROMPT_VERSION,
    SCHEMA_VERSION,
)
from eci_decision_models.laya import (
    LayaAdapter,
    LayaRequest,
    LayaResponse,
    LayaSettings,
    LayaTextClient,
    build_laya_request,
    laya_runtime_metadata,
    load_laya_agent,
    parse_laya_response,
)
from eci_decision_models.probability import PredictionStatus

_REDACTED = "[REDACTED]"
_ADAPTER_VERSION = "laya-adapter-v1"
_PARSER_VERSION = "eci-parser-v1"
_CONTROL_NAME = ".probe_control.json"
_ENVIRONMENT_PACKAGES = (
    "laya",
    "torch",
    "transformers",
    "numpy",
    "huggingface-hub",
    "safetensors",
    "eci-decision-models",
)


class ProbeAdapter(Protocol):
    """Minimal adapter surface used by the probe runner."""

    def predict(self, request: LayaRequest) -> LayaResponse: ...


@dataclass(frozen=True)
class ProbeCase:
    """One fixed synthetic probe input, never an ESC gold row."""

    case_id: str
    kind: str
    sentence: str
    event1: str
    event2: str
    expected_choice: str | None = None


@dataclass(frozen=True)
class ProbeBudget:
    """Hard limits enforced against real adapter calls, retries included."""

    max_requests: int = 6
    max_retries: int = 0
    max_wall_seconds: float = 600
    max_cost: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_requests, bool)
            or not isinstance(self.max_requests, int)
            or self.max_requests < 0
        ):
            raise ValueError("max_requests must be a non-negative integer")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        if (
            isinstance(self.max_wall_seconds, bool)
            or not isinstance(self.max_wall_seconds, (int, float))
            or not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds < 0
        ):
            raise ValueError("max_wall_seconds must be a finite non-negative number")
        if (
            isinstance(self.max_cost, bool)
            or not isinstance(self.max_cost, (int, float))
            or not math.isfinite(self.max_cost)
            or self.max_cost < 0
        ):
            raise ValueError("max_cost must be a finite non-negative number")


@dataclass(frozen=True)
class ProbeRecord:
    """One synthetic probe prediction row, independent of ESC gold records."""

    run_id: str
    case_id: str
    record_id: str
    kind: str
    expected_choice: str | None
    model: str | None
    attempt: int
    raw_request_id: str
    request_hash: str
    response_hash: str
    adapter_version: str
    parser_version: str
    prediction: str | None
    status: str
    prediction_valid: bool
    probability_valid: bool
    probabilities: tuple[float, float, float] | None
    error: str | None
    latency_ms: float
    control_status: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeRunResult:
    """Summary of one completed probe run."""

    run_id: str
    run_dir: Path
    status: str
    case_count: int
    request_count: int
    predictions_path: Path
    raw_path: Path
    meta_path: Path
    records: tuple[ProbeRecord, ...] = ()


@dataclass
class _ExecutionState:
    predictions: list[ProbeRecord]
    raw_entries: list[dict[str, object]]
    request_count: int
    stopped: bool


def default_probe_cases() -> tuple[ProbeCase, ...]:
    """Return the fixed, deterministic set of six English synthetic cases."""
    long_sentence = " ".join(["The long synthetic event sequence continues."] * 40)
    return (
        ProbeCase(
            case_id="probe-cause",
            kind="cause",
            sentence="The storm caused the flood.",
            event1="the storm",
            event2="the flood",
            expected_choice="cause",
        ),
        ProbeCase(
            case_id="probe-caused-by",
            kind="caused_by",
            sentence="The flood was caused by the storm.",
            event1="the flood",
            event2="the storm",
            expected_choice="caused_by",
        ),
        ProbeCase(
            case_id="probe-none",
            kind="none",
            sentence="The cat sat on the mat.",
            event1="the cat",
            event2="the mat",
            expected_choice="none",
        ),
        ProbeCase(
            case_id="probe-negation",
            kind="negation",
            sentence="The storm did not cause the flood.",
            event1="the storm",
            event2="the flood",
            expected_choice=None,
        ),
        ProbeCase(
            case_id="probe-empty",
            kind="empty",
            sentence="",
            event1="",
            event2="",
            expected_choice=None,
        ),
        ProbeCase(
            case_id="probe-long",
            kind="long",
            sentence=long_sentence,
            event1="the first event",
            event2="the final event",
            expected_choice=None,
        ),
    )


def run_laya_probe(
    *,
    cases: Sequence[ProbeCase],
    adapter: ProbeAdapter,
    output_dir: str | Path,
    run_id: str,
    settings: LayaSettings,
    runtime_metadata: Mapping[str, object],
    budget: ProbeBudget,
    code_metadata: Mapping[str, object] | None = None,
) -> ProbeRunResult:
    """Run a non-isolated probe for tests and offline helpers.

    The real CLI path uses ``run_laya_probe_isolated`` so model work cannot
    outlive the parent wall deadline.
    """
    run_dir, raw_dir = _prepare_run_dir(output_dir, run_id)
    predictions_path = run_dir / "predictions.jsonl"
    raw_path = raw_dir / "response.jsonl"
    predictions_path.touch()
    raw_path.touch()

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    state = _execute_cases(
        cases=cases,
        adapter=adapter,
        run_id=run_id,
        predictions_path=predictions_path,
        raw_path=raw_path,
        budget=budget,
        started_clock=started_clock,
        control_path=None,
        control_state=None,
    )
    _, status = _write_meta(
        run_dir=run_dir,
        run_id=run_id,
        case_count=len(cases),
        state=state,
        settings=settings,
        runtime_metadata=runtime_metadata,
        budget=budget,
        code_metadata=code_metadata,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
    )
    return ProbeRunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        case_count=len(cases),
        request_count=state.request_count,
        predictions_path=predictions_path,
        raw_path=raw_path,
        meta_path=run_dir / "meta.json",
        records=tuple(state.predictions),
    )


def run_laya_probe_isolated(
    *,
    cases: Sequence[ProbeCase],
    adapter_factory: Callable[[], ProbeAdapter],
    output_dir: str | Path,
    run_id: str,
    settings: LayaSettings,
    budget: ProbeBudget,
    code_metadata: Mapping[str, object] | None = None,
) -> ProbeRunResult:
    """Run the probe in a spawned child under a parent-enforced wall deadline."""
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"probe run directory already exists: {run_dir}")

    control_path = run_dir / _CONTROL_NAME
    child_code_metadata = dict(code_metadata) if code_metadata is not None else None
    process = multiprocessing.get_context("spawn").Process(
        target=_isolated_child_entry,
        args=(
            tuple(cases),
            adapter_factory,
            str(output_dir),
            run_id,
            settings,
            budget,
            child_code_metadata,
            str(control_path),
        ),
    )
    deadline = time.monotonic() + budget.max_wall_seconds
    process.start()
    timed_out = _monitor_process(process, deadline)
    exitcode = process.exitcode
    process.close()

    meta_path = run_dir / "meta.json"
    if not timed_out and exitcode == 0 and meta_path.exists():
        _cleanup_control(control_path)
        return _result_from_artifacts(run_dir, run_id, len(cases))

    try:
        return _finalize_isolated_run(
            cases=cases,
            run_dir=run_dir,
            run_id=run_id,
            control_path=control_path,
            settings=settings,
            budget=budget,
            code_metadata=code_metadata,
            timed_out=timed_out,
            exitcode=exitcode,
        )
    finally:
        _cleanup_control(control_path)


def probe_main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: Callable[[], ProbeAdapter] | None = None,
) -> int:
    """CLI entry point; dry-run is the default and never touches a model."""
    args = _build_parser().parse_args(None if argv is None else list(argv))
    if not args.authorize_real_laya:
        _print_dry_run()
        return 0

    env: dict[str, object] = dict(os.environ)
    if args.device is not None:
        env["ECI_LAYA_DEVICE"] = str(args.device)
    if args.download_weights:
        env["ECI_LAYA_DOWNLOAD_WEIGHTS"] = "true"
    settings = LayaSettings.from_env(env)

    factory = (
        adapter_factory if adapter_factory is not None else _DefaultLayaAdapterFactory(settings)
    )
    budget = ProbeBudget(
        max_requests=args.max_requests,
        max_retries=args.max_retries,
        max_wall_seconds=args.max_wall_seconds,
    )
    run_id = args.run_id or _default_run_id()
    result = run_laya_probe_isolated(
        cases=default_probe_cases(),
        adapter_factory=factory,
        output_dir=Path(args.output_dir),
        run_id=run_id,
        settings=settings,
        budget=budget,
    )
    if result.status not in {"ok", "completed"}:
        return 1
    return 0 if all(record.status == "ok" for record in result.records) else 1


@dataclass(frozen=True)
class _DefaultLayaAdapterFactory:
    """Pickle-safe real factory; model loading happens inside the child."""

    settings: LayaSettings

    def __call__(self) -> ProbeAdapter:
        agent = load_laya_agent(self.settings)
        return LayaAdapter(cast("LayaTextClient", agent))


def _execute_cases(
    *,
    cases: Sequence[ProbeCase],
    adapter: ProbeAdapter,
    run_id: str,
    predictions_path: Path,
    raw_path: Path,
    budget: ProbeBudget,
    started_clock: float,
    control_path: Path | None,
    control_state: dict[str, object] | None,
) -> _ExecutionState:
    prepared = [
        (case, build_laya_request(case.sentence, case.event1, case.event2)) for case in cases
    ]
    predictions: list[ProbeRecord] = []
    raw_entries: list[dict[str, object]] = []
    request_count = 0
    stopped = False

    def publish(**updates: object) -> None:
        if control_state is not None:
            control_state.update(updates)
        if control_path is not None and control_state is not None:
            _write_control(control_path, control_state)

    def budget_allows_call() -> bool:
        if request_count >= budget.max_requests:
            return False
        elapsed = time.perf_counter() - started_clock
        return elapsed < budget.max_wall_seconds

    for index, (case, request) in enumerate(prepared):
        record_id = _record_id(request)
        if stopped or not budget_allows_call():
            stopped = True
            record = _budget_record(
                run_id=run_id,
                case=case,
                request=request,
            )
            predictions.append(record)
            _append_jsonl(predictions_path, record.to_dict())
            publish(current_index=None)
            continue

        attempt = 1
        while True:
            request_count += 1
            publish(
                current_index=index,
                current_attempt=attempt,
                request_count=request_count,
            )
            attempt_started = time.perf_counter()
            try:
                response = adapter.predict(request)
            except TimeoutError as exc:
                latency_ms = _elapsed_ms(attempt_started)
                raw_entry, prediction_entry = _error_attempt(
                    run_id=run_id,
                    case=case,
                    request=request,
                    record_id=record_id,
                    attempt=attempt,
                    error=str(exc),
                    latency_ms=latency_ms,
                )
                raw_entries.append(raw_entry)
                _append_jsonl(raw_path, raw_entry)

                if attempt <= budget.max_retries:
                    if budget_allows_call():
                        attempt += 1
                        continue
                    stopped = True
                predictions.append(prediction_entry)
                _append_jsonl(predictions_path, prediction_entry.to_dict())
                publish(current_index=None)
                break
            except Exception as exc:
                latency_ms = _elapsed_ms(attempt_started)
                raw_entry, prediction_entry = _error_attempt(
                    run_id=run_id,
                    case=case,
                    request=request,
                    record_id=record_id,
                    attempt=attempt,
                    error=str(exc),
                    latency_ms=latency_ms,
                )
                raw_entries.append(raw_entry)
                predictions.append(prediction_entry)
                _append_jsonl(raw_path, raw_entry)
                _append_jsonl(predictions_path, prediction_entry.to_dict())
                publish(current_index=None)
                break
            else:
                latency_ms = _elapsed_ms(attempt_started)
                redacted_raw = _redacted_response(response)
                response_hash = _canonical_sha256(redacted_raw)
                raw_entry = _raw_entry(
                    case=case,
                    request=request,
                    attempt=attempt,
                    status=response.validated.status.value,
                    response_hash=response_hash,
                    latency_ms=latency_ms,
                    raw_response=redacted_raw,
                    error=response.validated.error,
                )
                prediction_entry = _validated_prediction_entry(
                    run_id=run_id,
                    case=case,
                    request=request,
                    record_id=record_id,
                    attempt=attempt,
                    response=response,
                    response_hash=response_hash,
                    latency_ms=latency_ms,
                )
                raw_entries.append(raw_entry)
                predictions.append(prediction_entry)
                _append_jsonl(raw_path, raw_entry)
                _append_jsonl(predictions_path, prediction_entry.to_dict())
                publish(current_index=None)
                break

    return _ExecutionState(
        predictions=predictions,
        raw_entries=raw_entries,
        request_count=request_count,
        stopped=stopped,
    )


def _isolated_child_entry(
    cases: tuple[ProbeCase, ...],
    adapter_factory: Callable[[], ProbeAdapter],
    output_dir: str,
    run_id: str,
    settings: LayaSettings,
    budget: ProbeBudget,
    code_metadata: dict[str, object] | None,
    control_path_value: str,
) -> None:
    run_dir, raw_dir = _prepare_run_dir(output_dir, run_id)
    predictions_path = run_dir / "predictions.jsonl"
    raw_path = raw_dir / "response.jsonl"
    predictions_path.touch()
    raw_path.touch()

    control_path = Path(control_path_value)
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    control: dict[str, object] = {
        "factory_ready": False,
        "current_index": None,
        "current_attempt": 1,
        "request_count": 0,
        "started_at_utc": started_at.isoformat(),
        "runtime_metadata": {},
    }
    _write_control(control_path, control)

    try:
        adapter = adapter_factory()
        runtime_source = getattr(adapter, "client", adapter)
        control["runtime_metadata"] = laya_runtime_metadata(runtime_source)
        control["factory_ready"] = True
        _write_control(control_path, control)

        state = _execute_cases(
            cases=cases,
            adapter=adapter,
            run_id=run_id,
            predictions_path=predictions_path,
            raw_path=raw_path,
            budget=budget,
            started_clock=started_clock,
            control_path=control_path,
            control_state=control,
        )
        _write_meta(
            run_dir=run_dir,
            run_id=run_id,
            case_count=len(cases),
            state=state,
            settings=settings,
            runtime_metadata=cast(
                "Mapping[str, object]",
                control.get("runtime_metadata") or {},
            ),
            budget=budget,
            code_metadata=code_metadata,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            termination={"timed_out": False, "exitcode": 0},
        )
    except BaseException as exc:
        control["fatal_error"] = _fatal_error_payload(exc)
        with contextlib.suppress(Exception):
            _write_control(control_path, control)
        raise


def _finalize_isolated_run(
    *,
    cases: Sequence[ProbeCase],
    run_dir: Path,
    run_id: str,
    control_path: Path,
    settings: LayaSettings,
    budget: ProbeBudget,
    code_metadata: Mapping[str, object] | None,
    timed_out: bool,
    exitcode: int | None,
) -> ProbeRunResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    raw_path = raw_dir / "response.jsonl"
    predictions_path.touch(exist_ok=True)
    raw_path.touch(exist_ok=True)

    existing_predictions = _read_jsonl_entries(predictions_path)
    existing_raw = _read_jsonl_entries(raw_path)
    _write_jsonl(predictions_path, existing_predictions)
    _write_jsonl(raw_path, existing_raw)

    control = _read_control(control_path)
    factory_ready = control.get("factory_ready") is True
    current_index = _optional_int(control.get("current_index"))
    current_attempt = _optional_int(control.get("current_attempt")) or 1
    request_count = _optional_int(control.get("request_count")) or 0
    started_at = _parse_control_datetime(control.get("started_at_utc"))
    fatal_error = control.get("fatal_error")
    runtime_metadata = control.get("runtime_metadata")
    if not isinstance(runtime_metadata, Mapping):
        runtime_metadata = {}

    case_list = list(cases)
    case_index = {case.case_id: index for index, case in enumerate(case_list)}
    predictions = [_record_from_mapping(entry) for entry in existing_predictions]
    raw_entries = list(existing_raw)
    for reconciled in _reconcile_predictions(case_list, run_id, predictions, raw_entries):
        predictions.append(reconciled)
        _append_jsonl(predictions_path, reconciled.to_dict())

    request_count = max(request_count, len(raw_entries))
    predicted_cases = {record.case_id for record in predictions}
    status_override: str | None = None
    stopped_override = False
    termination_after_completion = False

    def add_prediction(record: ProbeRecord) -> None:
        predictions.append(record)
        predicted_cases.add(record.case_id)
        _append_jsonl(predictions_path, record.to_dict())

    def add_budget(index: int) -> None:
        case = case_list[index]
        if case.case_id in predicted_cases:
            return
        add_prediction(
            _budget_record(
                run_id=run_id,
                case=case,
                request=build_laya_request(case.sentence, case.event1, case.event2),
            ),
        )

    def add_api_error(
        index: int,
        attempt: int,
        error: str,
        *,
        include_raw: bool,
    ) -> None:
        case = case_list[index]
        request = build_laya_request(case.sentence, case.event1, case.event2)
        raw_entry, record = _error_attempt(
            run_id=run_id,
            case=case,
            request=request,
            record_id=_record_id(request),
            attempt=attempt,
            error=error,
            latency_ms=0.0,
        )
        if include_raw:
            raw_entries.append(raw_entry)
            _append_jsonl(raw_path, raw_entry)
        add_prediction(record)

    def add_not_run(index: int, error: str) -> None:
        case = case_list[index]
        if case.case_id in predicted_cases:
            return
        request = build_laya_request(case.sentence, case.event1, case.event2)
        add_prediction(
            _prediction_entry(
                run_id=run_id,
                case=case,
                request=request,
                record_id=_record_id(request),
                attempt=1,
                status=PredictionStatus.API_ERROR.value,
                control_status="not_run",
                prediction=None,
                prediction_valid=False,
                probability_valid=False,
                probabilities=None,
                error=error,
                response_hash=_canonical_sha256(None),
                latency_ms=0.0,
                model=None,
            ),
        )

    if timed_out:
        all_complete = all(case.case_id in predicted_cases for case in case_list)
        if all_complete:
            statuses = {record.status for record in predictions}
            if statuses - {"ok", "missing_probability"}:
                termination_after_completion = False
                stopped_override = True
                if any(record.control_status == "budget_exhausted" for record in predictions):
                    status_override = "budget_exhausted"
                else:
                    status_override = "completed_with_errors"
            else:
                stopped_override = False
                termination_after_completion = True
        else:
            if (
                factory_ready
                and current_index is not None
                and current_index < len(case_list)
                and case_list[current_index].case_id not in predicted_cases
            ):
                add_api_error(
                    current_index,
                    current_attempt,
                    "isolated probe wall-clock timeout",
                    include_raw=True,
                )
            for index in range(len(case_list)):
                add_budget(index)
            status_override = "budget_exhausted"
            stopped_override = True
    else:
        abnormal_exit = exitcode is not None and exitcode != 0
        if not factory_ready:
            if not predictions and not raw_entries:
                message = _child_error_message(exitcode, fatal_error)
                for index in range(len(case_list)):
                    add_api_error(index, 1, message, include_raw=False)
                status_override = "completed_with_errors"
            elif abnormal_exit:
                status_override = "completed_with_errors"
        else:
            child_message = _child_error_message(exitcode, fatal_error)
            if (
                current_index is not None
                and current_index < len(case_list)
                and case_list[current_index].case_id not in predicted_cases
            ):
                add_api_error(
                    current_index,
                    current_attempt,
                    child_message,
                    include_raw=True,
                )
            for index in range(len(case_list)):
                add_not_run(index, child_message)
            if abnormal_exit:
                status_override = "completed_with_errors"

    predictions.sort(
        key=lambda record: (case_index.get(record.case_id, len(case_list)), record.attempt),
    )
    _write_jsonl(predictions_path, [record.to_dict() for record in predictions])

    state = _ExecutionState(
        predictions=predictions,
        raw_entries=raw_entries,
        request_count=request_count,
        stopped=stopped_override,
    )
    _, status = _write_meta(
        run_dir=run_dir,
        run_id=run_id,
        case_count=len(case_list),
        state=state,
        settings=settings,
        runtime_metadata=cast("Mapping[str, object]", runtime_metadata),
        budget=budget,
        code_metadata=code_metadata,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        fatal_error=fatal_error,
        status_override=status_override,
        stopped_override=stopped_override,
        termination={"timed_out": timed_out, "exitcode": exitcode},
        termination_after_completion=termination_after_completion,
    )
    return ProbeRunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        case_count=len(case_list),
        request_count=request_count,
        predictions_path=predictions_path,
        raw_path=raw_path,
        meta_path=run_dir / "meta.json",
        records=tuple(predictions),
    )


def _reconcile_predictions(
    cases: Sequence[ProbeCase],
    run_id: str,
    predictions: Sequence[ProbeRecord],
    raw_entries: Sequence[Mapping[str, object]],
) -> list[ProbeRecord]:
    predicted_cases = {record.case_id for record in predictions}
    raw_by_case: dict[str, list[Mapping[str, object]]] = {}
    for entry in raw_entries:
        case_id = entry.get("case_id")
        if isinstance(case_id, str):
            raw_by_case.setdefault(case_id, []).append(entry)

    reconciled: list[ProbeRecord] = []
    for case in cases:
        if case.case_id in predicted_cases:
            continue
        entries = raw_by_case.get(case.case_id)
        if not entries:
            continue
        entry = max(
            entries,
            key=lambda candidate: _optional_int(candidate.get("attempt")) or 1,
        )
        reconciled.append(
            _prediction_from_raw_entry(
                run_id=run_id,
                case=case,
                entry=entry,
            ),
        )
    return reconciled


def _prediction_from_raw_entry(
    *,
    run_id: str,
    case: ProbeCase,
    entry: Mapping[str, object],
) -> ProbeRecord:
    request = build_laya_request(case.sentence, case.event1, case.event2)
    raw_response = entry.get("raw_response")
    model: str | None = None
    if isinstance(raw_response, Mapping):
        validated = parse_laya_response(raw_response)
        status = validated.status.value
        prediction = validated.choice if validated.prediction_valid else None
        prediction_valid = validated.prediction_valid
        probability_valid = validated.probability_valid
        probabilities = validated.probabilities
        error = validated.error
        model_value = raw_response.get("model")
        if isinstance(model_value, str):
            model = model_value
    else:
        status = str(entry.get("status", PredictionStatus.API_ERROR.value))
        prediction = None
        prediction_valid = False
        probability_valid = False
        probabilities = None
        error_value = entry.get("error")
        error = str(error_value) if error_value is not None else None

    response_hash = entry.get("response_hash")
    if not isinstance(response_hash, str):
        response_hash = _canonical_sha256(raw_response)
    latency_value = entry.get("latency_ms")
    latency_ms = (
        float(latency_value)
        if isinstance(latency_value, (int, float)) and not isinstance(latency_value, bool)
        else 0.0
    )
    return _prediction_entry(
        run_id=run_id,
        case=case,
        request=request,
        record_id=_record_id(request),
        attempt=_optional_int(entry.get("attempt")) or 1,
        status=status,
        prediction=prediction,
        prediction_valid=prediction_valid,
        probability_valid=probability_valid,
        probabilities=probabilities,
        error=error,
        response_hash=response_hash,
        latency_ms=latency_ms,
        model=model,
    )


def _fatal_error_payload(exc: BaseException) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    redacted = redact_secrets(payload)
    return cast("dict[str, object]", redacted) if isinstance(redacted, Mapping) else payload


def _child_error_message(exitcode: int | None, fatal_error: object) -> str:
    if exitcode is None:
        message = "child exited unexpectedly"
    else:
        message = f"child exited with code {exitcode}"
    if fatal_error is not None:
        message = f"{message}: {json.dumps(fatal_error, sort_keys=True, ensure_ascii=False)}"
    return message


def _prepare_run_dir(output_dir: str | Path, run_id: str) -> tuple[Path, Path]:
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"probe run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir()
    return run_dir, raw_dir


def _monitor_process(process: Any, deadline: float) -> bool:
    while process.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join()
            return True
        process.join(timeout=min(0.05, remaining))
    process.join()
    return False


def _result_from_artifacts(run_dir: Path, run_id: str, fallback_case_count: int) -> ProbeRunResult:
    meta_value = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    meta = cast("dict[str, object]", meta_value)
    entries = _read_jsonl_entries(run_dir / "predictions.jsonl")
    return ProbeRunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=str(meta.get("status", "unknown")),
        case_count=_optional_int(meta.get("case_count")) or fallback_case_count,
        request_count=_optional_int(meta.get("request_count")) or 0,
        predictions_path=run_dir / "predictions.jsonl",
        raw_path=run_dir / "raw" / "response.jsonl",
        meta_path=run_dir / "meta.json",
        records=tuple(_record_from_mapping(entry) for entry in entries),
    )


def _write_meta(
    *,
    run_dir: Path,
    run_id: str,
    case_count: int,
    state: _ExecutionState,
    settings: LayaSettings,
    runtime_metadata: Mapping[str, object],
    budget: ProbeBudget,
    code_metadata: Mapping[str, object] | None,
    started_at: datetime,
    finished_at: datetime,
    fatal_error: object | None = None,
    status_override: str | None = None,
    stopped_override: bool | None = None,
    termination: Mapping[str, object] | None = None,
    termination_after_completion: bool = False,
) -> tuple[dict[str, object], str]:
    predictions_path = run_dir / "predictions.jsonl"
    raw_path = run_dir / "raw" / "response.jsonl"
    status = (
        status_override
        if status_override is not None
        else _run_status(
            state.predictions,
            state.stopped,
        )
    )
    stopped = state.stopped if stopped_override is None else stopped_override
    meta: dict[str, object] = {
        "run_id": run_id,
        "status": status,
        "stopped": stopped,
        "non_esc": True,
        "label_source": "synthetic_probe",
        "case_count": case_count,
        "request_count": state.request_count,
        "status_counts": _status_counts(state.predictions),
        "control_status_counts": _control_status_counts(state.predictions),
        "settings": asdict(settings),
        "runtime_metadata": _json_safe_copy(dict(runtime_metadata)),
        "environment": _environment_metadata(),
        "budget": asdict(budget),
        "code": _code_metadata(code_metadata),
        "options": [
            {"id": choice_id, "letter": letter, "description": description}
            for choice_id, letter, description in zip(
                CHOICE_IDS,
                CHOICE_LETTERS,
                CHOICE_DESCRIPTIONS,
                strict=True,
            )
        ],
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "predictions_sha256": sha256_file(predictions_path),
        "raw_sha256": sha256_file(raw_path),
    }
    if fatal_error is not None:
        meta["fatal_error"] = _json_safe_copy(fatal_error)
    termination_value: dict[str, object] = (
        dict(termination) if termination is not None else {"timed_out": False, "exitcode": None}
    )
    if termination_after_completion:
        termination_value["termination_after_completion"] = True
    meta["termination"] = termination_value
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta, status


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the synthetic non-ESC Laya probe")
    parser.add_argument("--authorize-real-laya", action="store_true")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--output-dir", default="artifacts/probes")
    parser.add_argument("--run-id")
    parser.add_argument("--max-requests", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-wall-seconds", type=float, default=600.0)
    return parser


def _print_dry_run() -> None:
    cases = default_probe_cases()
    print(f"Synthetic probe dry run: {len(cases)} cases; no model will be loaded")
    for case in cases:
        request = build_laya_request(case.sentence, case.event1, case.event2)
        print(f"- {case.case_id} ({case.kind}): sha256={request.canonical.sha256}")


def _default_run_id() -> str:
    return "probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _record_id(request: LayaRequest) -> str:
    return f"probe:{request.canonical.sha256}"


def _budget_record(
    *,
    run_id: str,
    case: ProbeCase,
    request: LayaRequest,
) -> ProbeRecord:
    return _prediction_entry(
        run_id=run_id,
        case=case,
        request=request,
        record_id=_record_id(request),
        attempt=1,
        status=PredictionStatus.API_ERROR.value,
        control_status="budget_exhausted",
        prediction=None,
        prediction_valid=False,
        probability_valid=False,
        probabilities=None,
        error="request budget exhausted before this case ran",
        response_hash=_canonical_sha256(None),
        latency_ms=0.0,
        model=None,
    )


def _redacted_response(response: LayaResponse) -> dict[str, object]:
    redacted = redact_secrets(response.raw)
    return cast("dict[str, object]", redacted) if isinstance(redacted, Mapping) else {}


def _validated_prediction_entry(
    *,
    run_id: str,
    case: ProbeCase,
    request: LayaRequest,
    record_id: str,
    attempt: int,
    response: LayaResponse,
    response_hash: str,
    latency_ms: float,
) -> ProbeRecord:
    validated = response.validated
    prediction = validated.choice if validated.prediction_valid else None
    return _prediction_entry(
        run_id=run_id,
        case=case,
        request=request,
        record_id=record_id,
        attempt=attempt,
        status=validated.status.value,
        prediction=prediction,
        prediction_valid=validated.prediction_valid,
        probability_valid=validated.probability_valid,
        probabilities=validated.probabilities,
        error=validated.error,
        response_hash=response_hash,
        latency_ms=latency_ms,
        model=response.model,
    )


def _error_attempt(
    *,
    run_id: str,
    case: ProbeCase,
    request: LayaRequest,
    record_id: str,
    attempt: int,
    error: str,
    latency_ms: float,
) -> tuple[dict[str, object], ProbeRecord]:
    response_hash = _canonical_sha256(None)
    raw_entry = _raw_entry(
        case=case,
        request=request,
        attempt=attempt,
        status=PredictionStatus.API_ERROR.value,
        response_hash=response_hash,
        latency_ms=latency_ms,
        raw_response=None,
        error=error,
    )
    prediction_entry = _prediction_entry(
        run_id=run_id,
        case=case,
        request=request,
        record_id=record_id,
        attempt=attempt,
        status=PredictionStatus.API_ERROR.value,
        prediction=None,
        prediction_valid=False,
        probability_valid=False,
        probabilities=None,
        error=error,
        response_hash=response_hash,
        latency_ms=latency_ms,
        model=None,
    )
    return raw_entry, prediction_entry


def _prediction_entry(
    *,
    run_id: str,
    case: ProbeCase,
    request: LayaRequest,
    record_id: str,
    attempt: int,
    status: str,
    prediction: str | None,
    prediction_valid: bool,
    probability_valid: bool,
    probabilities: tuple[float, float, float] | None,
    error: str | None,
    response_hash: str,
    latency_ms: float,
    model: str | None,
    control_status: str | None = None,
) -> ProbeRecord:
    return ProbeRecord(
        run_id=run_id,
        case_id=case.case_id,
        record_id=record_id,
        kind=case.kind,
        expected_choice=case.expected_choice,
        model=model,
        attempt=attempt,
        raw_request_id=f"{run_id}:{case.case_id}:{attempt}",
        request_hash=request.canonical.sha256,
        response_hash=response_hash,
        adapter_version=_ADAPTER_VERSION,
        parser_version=_PARSER_VERSION,
        prediction=prediction,
        status=status,
        prediction_valid=prediction_valid,
        probability_valid=probability_valid,
        probabilities=probabilities,
        error=error,
        latency_ms=latency_ms,
        control_status=control_status,
    )


def _raw_entry(
    *,
    case: ProbeCase,
    request: LayaRequest,
    attempt: int,
    status: str,
    response_hash: str,
    latency_ms: float,
    raw_response: object,
    error: str | None,
) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "attempt": attempt,
        "status": status,
        "state": _json_safe_copy(request.state),
        "questions": _json_safe_copy(request.questions),
        "canonical": {
            "payload": _json_safe_copy(request.canonical.payload),
            "options": _json_safe_copy(request.canonical.options),
            "sha256": request.canonical.sha256,
        },
        "request_hash": request.canonical.sha256,
        "response_hash": response_hash,
        "raw_response": raw_response,
        "error": error,
        "latency_ms": latency_ms,
    }


def _run_status(predictions: Sequence[ProbeRecord], stopped: bool) -> str:
    if stopped:
        return "budget_exhausted"
    error_statuses = {
        PredictionStatus.API_ERROR.value,
        PredictionStatus.INVALID_RESPONSE.value,
        PredictionStatus.INVALID_PROBABILITY.value,
    }
    if any(entry.status in error_statuses for entry in predictions):
        return "completed_with_errors"
    return "ok"


def _status_counts(predictions: Sequence[ProbeRecord]) -> dict[str, int]:
    frozen_statuses = {status.value for status in PredictionStatus}
    counts: dict[str, int] = {}
    for entry in predictions:
        if entry.status not in frozen_statuses:
            continue
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return counts


def _control_status_counts(predictions: Sequence[ProbeRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in predictions:
        if entry.control_status is None:
            continue
        counts[entry.control_status] = counts.get(entry.control_status, 0) + 1
    return counts


def _environment_metadata() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in _ENVIRONMENT_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    uv_lock = _find_project_file("uv.lock")
    command = " ".join(sys.argv) or "unknown"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "uv_lock_sha256": sha256_file(uv_lock) if uv_lock is not None else None,
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "executed_package_sha256": _executed_package_sha256(),
        "cwd": str(Path.cwd()),
        "command": command,
    }


def _executed_package_sha256() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(
        package_dir.rglob("*.py"),
        key=lambda path: path.relative_to(package_dir).as_posix(),
    )
    for path in files:
        relative = path.relative_to(package_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_safe_copy(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_copy(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe_copy(item) for item in value]
    return str(value)


def _write_jsonl(path: Path, entries: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            for entry in entries
        ),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, entry: Mapping[str, object]) -> None:
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl_entries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            entries.append(cast("dict[str, object]", value))
    return entries


def _record_from_mapping(entry: Mapping[str, object]) -> ProbeRecord:
    data: dict[str, object] = dict(entry)
    data.setdefault("control_status", None)
    probabilities = data.get("probabilities")
    if isinstance(probabilities, list):
        data["probabilities"] = tuple(probabilities)
    return ProbeRecord(**cast("Any", data))


def _write_control(path: Path, state: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_control(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _cleanup_control(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_name(path.name + ".tmp").unlink(missing_ok=True)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _parse_control_datetime(value: object) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _code_metadata(injected: Mapping[str, object] | None) -> dict[str, object]:
    if injected is not None:
        return cast("dict[str, object]", _json_safe_copy(dict(injected)))
    return _git_code_metadata()


def _git_code_metadata() -> dict[str, object]:
    repo_root = _find_git_root()
    if repo_root is None:
        return {"commit": None, "dirty": None}
    commit = _git_output(repo_root, ["rev-parse", "HEAD"])
    status = _git_output(repo_root, ["status", "--porcelain"])
    return {"commit": commit, "dirty": None if status is None else bool(status.strip())}


def _search_roots() -> list[Path]:
    """Try the live module path first, then the working directory tree."""
    roots: list[Path] = []
    seen: set[Path] = set()
    module_root = Path(__file__).resolve().parent
    for candidate in (module_root, *module_root.parents):
        if candidate not in seen:
            seen.add(candidate)
            roots.append(candidate)
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if candidate not in seen:
            seen.add(candidate)
            roots.append(candidate)
    return roots


def _find_project_file(name: str) -> Path | None:
    for root in _search_roots():
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _find_git_root() -> Path | None:
    for candidate in _search_roots():
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_output(repo_root: Path, arguments: Sequence[str]) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


# Keep the constant importable for callers that want to assert redaction output.
__all__ = [
    "REDACTED",
    "ProbeAdapter",
    "ProbeBudget",
    "ProbeCase",
    "ProbeRecord",
    "ProbeRunResult",
    "default_probe_cases",
    "probe_main",
    "run_laya_probe",
    "run_laya_probe_isolated",
]

REDACTED = _REDACTED


if __name__ == "__main__":
    raise SystemExit(probe_main())
