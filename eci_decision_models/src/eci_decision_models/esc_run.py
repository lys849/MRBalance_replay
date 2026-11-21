"""Laya-only ESC evaluation runner.

The runner joins a normalized ESC dataset with a reconstructed gold artifact,
builds canonical Laya requests, and records every attempt as a standardized
``PredictionRecord``.  It is deliberately offline by default: the CLI performs
a dry-run unless real Laya execution is explicitly authorized, and the runner
only talks to the injected adapter seam.

``EscRunBudget.max_wall_seconds`` is a scheduling budget, not a hard process
deadline.  It is only checked between requests (including between retries), so
one slow adapter call can overrun it.  Use external subprocess isolation when a
hard wall-clock bound is required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, cast

from eci_decision_models.artifacts import (
    PredictionRecord,
    atomic_write_text,
    read_predictions_jsonl,
    redact_secrets,
    sha256_file,
    write_predictions_jsonl,
)
from eci_decision_models.canonical import (
    CHOICE_DESCRIPTIONS,
    CHOICE_IDS,
    CHOICE_LETTERS,
    PROMPT_VERSION,
    SCHEMA_VERSION,
)
from eci_decision_models.data import EscDataset, EscRecord, load_esc
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
from eci_decision_models.report import build_single_model_metrics, write_metrics_files

_ADAPTER_VERSION: Final = "laya-adapter-v1"
_PARSER_VERSION: Final = "eci-parser-v1"
_GOLD_FILENAME: Final = "gold_choice_reconstructed.jsonl"
_MANIFEST_FILENAME: Final = "manifest.json"
_ENVIRONMENT_PACKAGES: Final = (
    "laya",
    "torch",
    "transformers",
    "numpy",
    "huggingface-hub",
    "safetensors",
    "eci-decision-models",
)
_ERROR_STATUSES: Final = frozenset(
    {
        PredictionStatus.INVALID_RESPONSE.value,
        PredictionStatus.INVALID_PROBABILITY.value,
        PredictionStatus.API_ERROR.value,
    },
)
_FROZEN_STATUSES: Final = frozenset(status.value for status in PredictionStatus)
_BUDGET_CONTROL_STATUS: Final = "budget_exhausted"


@dataclass(frozen=True)
class GoldChoice:
    """One reconstructed gold decision for an ESC record."""

    record_id: str
    item_id: int
    choice: str
    event1: str
    event2: str


@dataclass(frozen=True)
class GoldArtifact:
    """A reconstructed gold JSONL plus its optional manifest provenance."""

    path: Path
    sha256: str
    version: str | None
    records: Mapping[str, GoldChoice]
    manifest: Mapping[str, object] | None


@dataclass(frozen=True)
class EscItem:
    """One dataset row aligned with its gold choice and canonical request."""

    record: EscRecord
    gold: GoldChoice
    request: LayaRequest
    data_path: Path | None = None
    data_sha256: str | None = None
    gold_path: Path | None = None
    gold_sha256: str | None = None
    gold_version: str | None = None


@dataclass(frozen=True)
class EscRunBudget:
    """Hard request and wall-clock limits for one ESC run."""

    max_requests: int
    max_retries: int = 0
    max_wall_seconds: float = 7200.0

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


@dataclass(frozen=True)
class EscRunResult:
    """The outcome of one ESC run, including all standardized records."""

    run_id: str
    run_dir: Path
    status: str
    records: tuple[PredictionRecord, ...]
    request_count: int
    metrics: dict[str, object]


class EscAdapter(Protocol):
    """Minimal adapter surface used by the ESC runner."""

    def predict(self, request: LayaRequest) -> LayaResponse: ...


def _require_object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return cast("Mapping[str, object]", value)


def _require_str(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _require_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _resolve_gold_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / _GOLD_FILENAME
    return candidate


def _read_manifest(gold_path: Path) -> Mapping[str, object] | None:
    manifest_path = gold_path.parent / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read gold manifest {manifest_path}: {exc}") from exc
    return _require_object(payload, f"gold manifest {manifest_path}")


def _parse_gold_line(raw: object, line_number: int) -> GoldChoice:
    record = _require_object(raw, f"gold line {line_number}")
    record_id = _require_str(record.get("record_id"), f"gold line {line_number} record_id")
    item_id = _require_int(record.get("item_id"), f"gold line {line_number} item_id")
    choice = _require_str(record.get("choice"), f"gold line {line_number} choice")
    event1 = _require_str(record.get("event1"), f"gold line {line_number} event1")
    event2 = _require_str(record.get("event2"), f"gold line {line_number} event2")
    if choice not in CHOICE_IDS:
        raise ValueError(
            f"gold line {line_number} choice {choice!r} must be one of {list(CHOICE_IDS)!r}",
        )
    return GoldChoice(
        record_id=record_id,
        item_id=item_id,
        choice=choice,
        event1=event1,
        event2=event2,
    )


def load_gold_artifact(path: str | Path) -> GoldArtifact:
    """Load a gold JSONL file, or a directory with a JSONL plus manifest."""
    gold_path = _resolve_gold_path(path)
    try:
        text = gold_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read gold artifact {gold_path}: {exc}") from exc

    manifest = _read_manifest(gold_path)
    version: str | None = None
    if manifest is not None:
        raw_version = manifest.get("gold_version")
        if raw_version is not None:
            version = _require_str(raw_version, "manifest gold_version")
        summary = manifest.get("summary")
        if isinstance(summary, Mapping):
            hard_failures = summary.get("hard_failures")
            if hard_failures is not None and hard_failures != 0:
                raise ValueError(
                    f"gold manifest reports {hard_failures!r} hard failures; expected 0",
                )

    records: dict[str, GoldChoice] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"gold line {line_number}: invalid JSON: {exc.msg}") from exc
        choice = _parse_gold_line(raw, line_number)
        if choice.record_id in records:
            raise ValueError(f"gold line {line_number}: duplicate record_id {choice.record_id!r}")
        records[choice.record_id] = choice

    return GoldArtifact(
        path=gold_path,
        sha256=sha256_file(gold_path),
        version=version,
        records=MappingProxyType(records),
        manifest=manifest,
    )


def _format_examples(values: Sequence[str]) -> str:
    examples = list(values[:3])
    return ", ".join(repr(value) for value in examples)


def prepare_esc_items(
    dataset: EscDataset,
    gold: GoldArtifact,
) -> tuple[EscItem, ...]:
    """Align dataset records with gold choices and build canonical requests."""
    dataset_ids = {record.record_id for record in dataset.records}
    gold_ids = set(gold.records)
    missing = [record.record_id for record in dataset.records if record.record_id not in gold_ids]
    extra = sorted(gold_ids - dataset_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(
                f"missing gold for {len(missing)} record(s): {_format_examples(missing)}",
            )
        if extra:
            details.append(f"{len(extra)} extra gold record(s): {_format_examples(extra)}")
        raise ValueError("dataset and gold record_id sets differ; " + "; ".join(details))

    if gold.manifest is not None:
        source_esc = gold.manifest.get("source_esc")
        if isinstance(source_esc, Mapping):
            expected_sha = source_esc.get("sha256")
            if isinstance(expected_sha, str) and expected_sha != dataset.source_sha256:
                raise ValueError(
                    "gold manifest source_esc.sha256 does not match the dataset sha256",
                )

    items: list[EscItem] = []
    for record in dataset.records:
        choice = gold.records[record.record_id]
        if choice.item_id != record.item_id:
            raise ValueError(
                f"record {record.record_id!r}: gold item_id {choice.item_id!r} "
                f"does not match dataset item_id {record.item_id!r}",
            )
        if choice.event1 != record.event1:
            raise ValueError(f"record {record.record_id!r}: gold event1 does not match dataset")
        if choice.event2 != record.event2:
            raise ValueError(f"record {record.record_id!r}: gold event2 does not match dataset")
        request = build_laya_request(record.sentence, record.event1, record.event2)
        items.append(
            EscItem(
                record=record,
                gold=choice,
                request=request,
                data_path=dataset.source_path,
                data_sha256=dataset.source_sha256,
                gold_path=gold.path,
                gold_sha256=gold.sha256,
                gold_version=gold.version,
            ),
        )
    return tuple(items)


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


def _append_jsonl(path: Path, entry: Mapping[str, object]) -> None:
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl_entries(path: Path) -> list[dict[str, object]]:
    """Read JSONL strictly, raising with the file and line on malformed input."""
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number}: expected a JSON object")
        entries.append(cast("dict[str, object]", value))
    return entries


def _load_raw_entries(path: Path) -> list[dict[str, object]]:
    """Load raw attempts and reject duplicate ``(record_id, attempt)`` pairs."""
    entries = _read_jsonl_entries(path)
    seen: set[tuple[str, int]] = set()
    for line_number, entry in enumerate(entries, start=1):
        record_id = _require_str(entry.get("record_id"), f"{path} line {line_number} record_id")
        attempt = _require_int(entry.get("attempt"), f"{path} line {line_number} attempt")
        key = (record_id, attempt)
        if key in seen:
            raise ValueError(
                f"{path} line {line_number}: duplicate (record_id, attempt) {key!r}",
            )
        seen.add(key)
    return entries


def _redacted_raw(response: LayaResponse) -> dict[str, object]:
    redacted = redact_secrets(response.raw)
    if isinstance(redacted, Mapping):
        return cast("dict[str, object]", _json_safe_copy(dict(redacted)))
    return {}


def _raw_entry(
    *,
    run_id: str,
    item: EscItem,
    attempt: int,
    status: str,
    response_hash: str,
    latency_ms: float,
    raw_response: object,
    error: str | None,
) -> dict[str, object]:
    request = item.request
    record = item.record
    return {
        "run_id": run_id,
        "item_id": record.item_id,
        "record_id": record.record_id,
        "attempt": attempt,
        "status": status,
        "request_hash": request.canonical.sha256,
        "response_hash": response_hash,
        "latency_ms": latency_ms,
        "canonical": {
            "payload": _json_safe_copy(request.canonical.payload),
            "options": _json_safe_copy(request.canonical.options),
            "sha256": request.canonical.sha256,
        },
        "state": _json_safe_copy(request.state),
        "questions": _json_safe_copy(request.questions),
        "input_stats": {
            "sentence_words": len(record.sentence.split()),
            "event1_words": len(record.event1.split()),
            "event2_words": len(record.event2.split()),
        },
        "raw_response": raw_response,
        "error": error,
    }


def _prediction_record(
    *,
    run_id: str,
    item: EscItem,
    attempt: int,
    settings: LayaSettings,
    status: str,
    prediction: str | None,
    prediction_valid: bool,
    probability_valid: bool,
    probabilities: tuple[float, float, float] | None,
    error: str | None,
    response_hash: str,
    model: str | None,
    control_status: str | None = None,
) -> PredictionRecord:
    record = item.record
    return PredictionRecord(
        run_id=run_id,
        item_id=record.item_id,
        record_id=record.record_id,
        model=model or settings.checkpoint,
        attempt=attempt,
        raw_request_id=f"{run_id}:{record.record_id}:{attempt}",
        request_hash=item.request.canonical.sha256,
        response_hash=response_hash,
        adapter_version=_ADAPTER_VERSION,
        parser_version=_PARSER_VERSION,
        gold_choice=item.gold.choice,
        prediction=prediction,
        status=status,
        prediction_valid=prediction_valid,
        probability_valid=probability_valid,
        probabilities=probabilities,
        error=error,
        control_status=control_status,
    )


def _budget_record(
    *,
    run_id: str,
    item: EscItem,
    attempt: int,
    settings: LayaSettings,
) -> PredictionRecord:
    return _prediction_record(
        run_id=run_id,
        item=item,
        attempt=attempt,
        settings=settings,
        status=PredictionStatus.API_ERROR.value,
        prediction=None,
        prediction_valid=False,
        probability_valid=False,
        probabilities=None,
        error="request budget exhausted before this item ran",
        response_hash=_canonical_sha256(None),
        model=None,
        control_status="budget_exhausted",
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return False
    return status in (408, 429) or 500 <= status <= 599


def _run_status(records: Sequence[PredictionRecord], stopped: bool) -> str:
    if stopped:
        return "budget_exhausted"
    if any(record.status in _ERROR_STATUSES for record in records):
        return "completed_with_errors"
    return "ok"


def _status_counts(records: Sequence[PredictionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.status not in _FROZEN_STATUSES:
            continue
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def _control_status_counts(records: Sequence[PredictionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.control_status is None:
            continue
        counts[record.control_status] = counts.get(record.control_status, 0) + 1
    return counts


def _search_roots() -> list[Path]:
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
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_output_raw(repo_root: Path, arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_code_metadata() -> dict[str, object]:
    repo_root = _find_git_root()
    if repo_root is None:
        return {"commit": None, "dirty": False, "source_diff_sha256": None}
    commit = _git_output(repo_root, ["rev-parse", "HEAD"])
    status = _git_output(repo_root, ["status", "--porcelain"])
    diff = _git_output_raw(repo_root, ["diff", "HEAD"])
    return {
        "commit": commit,
        "dirty": False if status is None else bool(status.strip()),
        "source_diff_sha256": (
            None if diff is None else hashlib.sha256(diff.encode("utf-8")).hexdigest()
        ),
    }


def _code_metadata(injected: Mapping[str, object] | None) -> dict[str, object]:
    if injected is not None:
        return cast("dict[str, object]", _json_safe_copy(dict(injected)))
    return _git_code_metadata()


def _environment_metadata() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in _ENVIRONMENT_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    uv_lock = _find_project_file("uv.lock")
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "uv_lock_sha256": sha256_file(uv_lock) if uv_lock is not None else None,
        "packages": packages,
    }


def _write_meta(run_dir: Path, meta: Mapping[str, object]) -> None:
    atomic_write_text(
        run_dir / "meta.json",
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _items_meta(items: Sequence[EscItem]) -> tuple[dict[str, object], dict[str, object]]:
    if not items:
        empty_data: dict[str, object] = {
            "path": None,
            "resolved_path": None,
            "sha256": None,
            "cwd": str(Path.cwd()),
        }
        empty_gold: dict[str, object] = {
            "path": None,
            "resolved_path": None,
            "sha256": None,
            "version": None,
        }
        return empty_data, empty_gold
    first = items[0]
    data: dict[str, object] = {
        "path": str(first.data_path) if first.data_path is not None else None,
        "resolved_path": (
            str(Path(first.data_path).resolve()) if first.data_path is not None else None
        ),
        "sha256": first.data_sha256,
        "cwd": str(Path.cwd()),
    }
    gold: dict[str, object] = {
        "path": str(first.gold_path) if first.gold_path is not None else None,
        "resolved_path": (
            str(Path(first.gold_path).resolve()) if first.gold_path is not None else None
        ),
        "sha256": first.gold_sha256,
        "version": first.gold_version,
    }
    return data, gold


def _options_metadata() -> list[dict[str, str]]:
    return [
        {"id": choice_id, "letter": letter, "description": description}
        for choice_id, letter, description in zip(
            CHOICE_IDS,
            CHOICE_LETTERS,
            CHOICE_DESCRIPTIONS,
            strict=True,
        )
    ]


def _run_signature(items: Sequence[EscItem], settings: LayaSettings) -> str:
    data_meta, gold_meta = _items_meta(items)
    payload = {
        "data_sha256": data_meta["sha256"],
        "gold_sha256": gold_meta["sha256"],
        "gold_version": gold_meta["version"],
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "options": _options_metadata(),
        "checkpoint": settings.checkpoint,
        "revision": settings.revision,
        "sdk_version": settings.sdk_version,
        "device": settings.device,
        "adapter_version": _ADAPTER_VERSION,
        "parser_version": _PARSER_VERSION,
    }
    return _canonical_sha256(payload)


def _read_meta(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read run metadata {path}: {exc}") from exc
    return dict(_require_object(payload, f"run metadata {path}"))


def _validate_resume_meta(
    prior_meta: Mapping[str, object],
    *,
    signature: str,
    settings: LayaSettings,
    data_meta: Mapping[str, object],
    gold_meta: Mapping[str, object],
) -> None:
    prior_signature = prior_meta.get("run_signature")
    if "run_signature" in prior_meta:
        if prior_signature != signature:
            raise ValueError(
                "run_signature mismatch: the existing run used different data, gold, "
                "prompt, options, or adapter settings",
            )
        return

    prior_data = prior_meta.get("data")
    if (
        isinstance(prior_data, Mapping)
        and "sha256" in prior_data
        and prior_data["sha256"] != data_meta["sha256"]
    ):
        raise ValueError("data.sha256 mismatch with the existing run")
    prior_gold = prior_meta.get("gold")
    if isinstance(prior_gold, Mapping):
        if "sha256" in prior_gold and prior_gold["sha256"] != gold_meta["sha256"]:
            raise ValueError("gold.sha256 mismatch with the existing run")
        if "version" in prior_gold and prior_gold["version"] != gold_meta["version"]:
            raise ValueError("gold.version mismatch with the existing run")
    if "settings" in prior_meta and prior_meta["settings"] != asdict(settings):
        raise ValueError("settings mismatch with the existing run")
    if "prompt_version" in prior_meta and prior_meta["prompt_version"] != PROMPT_VERSION:
        raise ValueError("prompt_version mismatch with the existing run")
    if "schema_version" in prior_meta and prior_meta["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch with the existing run")
    if "options" in prior_meta and prior_meta["options"] != _options_metadata():
        raise ValueError("options mismatch with the existing run")


def _prior_invocations(prior_meta: Mapping[str, object]) -> list[dict[str, object]]:
    raw = prior_meta.get("invocations")
    if isinstance(raw, list):
        entries: list[dict[str, object]] = []
        for item in raw:
            if isinstance(item, Mapping):
                entries.append(cast("dict[str, object]", _json_safe_copy(dict(item))))
        return entries
    return [
        {
            "command": prior_meta.get("command"),
            "started_at_utc": prior_meta.get("started_at_utc"),
            "finished_at_utc": prior_meta.get("finished_at_utc"),
            "limit": prior_meta.get("limit"),
            "resumed": prior_meta.get("resumed"),
            "status": prior_meta.get("status"),
        },
    ]


def _validate_existing_prediction(
    record: PredictionRecord,
    *,
    item: EscItem,
    run_id: str,
) -> None:
    if record.run_id != run_id:
        raise ValueError(
            f"existing prediction {record.record_id!r} run_id {record.run_id!r} "
            f"does not match current run {run_id!r}",
        )
    if record.item_id != item.record.item_id:
        raise ValueError(
            f"existing prediction {record.record_id!r} item_id {record.item_id!r} "
            f"does not match item {item.record.item_id!r}",
        )
    if record.request_hash != item.request.canonical.sha256:
        raise ValueError(
            f"existing prediction {record.record_id!r} request_hash does not match "
            "the current canonical request",
        )
    if record.gold_choice != item.gold.choice:
        raise ValueError(
            f"existing prediction {record.record_id!r} gold_choice "
            f"{record.gold_choice!r} does not match item gold {item.gold.choice!r}",
        )
    if record.adapter_version != _ADAPTER_VERSION:
        raise ValueError(
            f"existing prediction {record.record_id!r} adapter_version "
            f"{record.adapter_version!r} does not match {_ADAPTER_VERSION!r}",
        )
    if record.parser_version != _PARSER_VERSION:
        raise ValueError(
            f"existing prediction {record.record_id!r} parser_version "
            f"{record.parser_version!r} does not match {_PARSER_VERSION!r}",
        )


def _max_attempt_by_record(entries: Sequence[Mapping[str, object]]) -> dict[str, int]:
    attempts: dict[str, int] = {}
    for entry in entries:
        record_id = cast("str", entry["record_id"])
        attempt = cast("int", entry["attempt"])
        attempts[record_id] = max(attempts.get(record_id, 0), attempt)
    return attempts


def _latest_raw_by_record(
    entries: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    latest: dict[str, Mapping[str, object]] = {}
    latest_attempt: dict[str, int] = {}
    for entry in entries:
        record_id = cast("str", entry["record_id"])
        attempt = cast("int", entry["attempt"])
        if attempt >= latest_attempt.get(record_id, -1):
            latest_attempt[record_id] = attempt
            latest[record_id] = entry
    return latest


def _recover_prediction_from_raw(
    *,
    entry: Mapping[str, object],
    item: EscItem,
    run_id: str,
    settings: LayaSettings,
) -> PredictionRecord:
    """Rebuild a terminal prediction from a persisted raw attempt.

    A retryable ``api_error`` left behind by an interrupted retry is finalized
    as ``api_error`` instead of being retried, so resume never duplicates a
    ``(record_id, attempt)`` raw entry.
    """
    record_id = _require_str(entry.get("record_id"), "raw record_id")
    if record_id != item.record.record_id:
        raise ValueError(
            f"raw entry record_id {record_id!r} does not match item {item.record.record_id!r}",
        )
    item_id = _require_int(entry.get("item_id"), "raw item_id")
    if item_id != item.record.item_id:
        raise ValueError(
            f"raw entry item_id {item_id!r} does not match item {item.record.item_id!r}",
        )
    attempt = _require_int(entry.get("attempt"), "raw attempt")
    status = _require_str(entry.get("status"), "raw status")
    if status not in _FROZEN_STATUSES:
        raise ValueError(f"raw entry status {status!r} is not a frozen status")
    request_hash = _require_str(entry.get("request_hash"), "raw request_hash")
    if request_hash != item.request.canonical.sha256:
        raise ValueError(f"raw entry {record_id!r} request_hash does not match the item")
    response_hash = _require_str(entry.get("response_hash"), "raw response_hash")

    raw_response = entry.get("raw_response")
    plain_response: dict[str, object] | None
    if isinstance(raw_response, Mapping):
        plain_response = cast("dict[str, object]", _json_safe_copy(dict(raw_response)))
        if _canonical_sha256(plain_response) != response_hash:
            raise ValueError(f"raw entry {record_id!r} response_hash does not match its payload")
    else:
        plain_response = None
        if response_hash != _canonical_sha256(None):
            raise ValueError(f"raw entry {record_id!r} is missing its response payload")

    model_value = plain_response.get("model") if plain_response is not None else None
    model = model_value if isinstance(model_value, str) else settings.checkpoint

    if status == PredictionStatus.API_ERROR.value:
        error_value = entry.get("error")
        return _prediction_record(
            run_id=run_id,
            item=item,
            attempt=attempt,
            settings=settings,
            status=status,
            prediction=None,
            prediction_valid=False,
            probability_valid=False,
            probabilities=None,
            error=error_value if isinstance(error_value, str) else None,
            response_hash=response_hash,
            model=model,
        )

    if plain_response is None:
        raise ValueError(f"raw entry {record_id!r} status {status!r} requires a raw response")
    validated = parse_laya_response(plain_response)
    if validated.status.value != status:
        raise ValueError(
            f"raw entry {record_id!r} status {status!r} does not match parsed "
            f"status {validated.status.value!r}",
        )
    return _prediction_record(
        run_id=run_id,
        item=item,
        attempt=attempt,
        settings=settings,
        status=status,
        prediction=validated.choice if validated.prediction_valid else None,
        prediction_valid=validated.prediction_valid,
        probability_valid=validated.probability_valid,
        probabilities=validated.probabilities if validated.probability_valid else None,
        error=validated.error,
        response_hash=response_hash,
        model=model,
    )


def run_laya_esc(
    *,
    items: Sequence[EscItem],
    adapter: EscAdapter,
    output_dir: str | Path,
    run_id: str,
    settings: LayaSettings,
    runtime_metadata: Mapping[str, object],
    budget: EscRunBudget,
    resume: bool = False,
    code_metadata: Mapping[str, object] | None = None,
    zero_policy: str = "error",
    nll_epsilon: float | None = None,
    command: Sequence[str] | None = None,
    limit: int | None = None,
) -> EscRunResult:
    """Run Laya over aligned ESC items, appending every attempt as it finishes.

    ``budget.max_wall_seconds`` is checked between requests only; it is a
    scheduling budget rather than a hard process deadline.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    selected = list(items) if limit is None else list(items)[:limit]

    run_dir = Path(output_dir) / run_id
    if resume:
        if not run_dir.exists():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    else:
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.jsonl"
    raw_path = run_dir / "raw" / "response.jsonl"
    meta_path = run_dir / "meta.json"

    data_meta, gold_meta = _items_meta(selected)
    signature = _run_signature(selected, settings)
    prior_meta = _read_meta(meta_path) if resume else None
    if prior_meta is not None:
        _validate_resume_meta(
            prior_meta,
            signature=signature,
            settings=settings,
            data_meta=data_meta,
            gold_meta=gold_meta,
        )

    existing_predictions: list[PredictionRecord] = []
    if resume and predictions_path.exists():
        existing_predictions = read_predictions_jsonl(predictions_path)
    item_by_id = {item.record.record_id: item for item in selected}
    raw_entries = _load_raw_entries(raw_path)
    for entry in raw_entries:
        entry_run_id = _require_str(entry.get("run_id"), "raw run_id")
        if entry_run_id != run_id:
            raise ValueError(
                f"raw entry {entry.get('record_id')!r} run_id {entry_run_id!r} "
                f"does not match current run {run_id!r}",
            )
    raw_by_key = {
        (cast("str", entry["record_id"]), cast("int", entry["attempt"])): entry
        for entry in raw_entries
    }
    max_attempts = _max_attempt_by_record(raw_entries)

    predictions_by_id: dict[str, PredictionRecord] = {}
    seen_existing: set[str] = set()
    for record in existing_predictions:
        if record.record_id in seen_existing:
            raise ValueError(f"duplicate record_id {record.record_id!r} in existing predictions")
        seen_existing.add(record.record_id)
        item = item_by_id.get(record.record_id)
        if item is None:
            raise ValueError(f"existing prediction {record.record_id!r} does not match any item")
        _validate_existing_prediction(record, item=item, run_id=run_id)
        raw_entry = raw_by_key.get((record.record_id, record.attempt))
        if raw_entry is not None and raw_entry.get("response_hash") != record.response_hash:
            raise ValueError(
                f"existing prediction {record.record_id!r} attempt {record.attempt} "
                "response_hash does not match its raw entry",
            )
        if record.control_status == _BUDGET_CONTROL_STATUS:
            # Budget placeholders are pending work, not terminal predictions.
            continue
        predictions_by_id[record.record_id] = record

    latest_raw = _latest_raw_by_record(raw_entries)
    for item in selected:
        record_id = item.record.record_id
        if record_id in predictions_by_id:
            continue
        latest = latest_raw.get(record_id)
        if latest is not None:
            predictions_by_id[record_id] = _recover_prediction_from_raw(
                entry=latest,
                item=item,
                run_id=run_id,
                settings=settings,
            )

    if resume:
        initial_records = [
            predictions_by_id[item.record.record_id]
            for item in selected
            if item.record.record_id in predictions_by_id
        ]
        # Atomically drop pending placeholders before any new adapter call so a
        # crash can never leave duplicate record_ids behind.
        write_predictions_jsonl(initial_records, predictions_path)
    elif not predictions_path.exists():
        predictions_path.touch()
    if not raw_path.exists():
        raw_path.touch()

    request_count = len(raw_entries)
    stopped = False
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc)

    def budget_allows_call() -> bool:
        if request_count >= budget.max_requests:
            return False
        return time.perf_counter() - started_clock < budget.max_wall_seconds

    def next_attempt(record_id: str) -> int:
        return max_attempts.get(record_id, 0) + 1

    def store(record: PredictionRecord) -> None:
        predictions_by_id[record.record_id] = record
        _append_jsonl(predictions_path, record.to_dict())

    for item in selected:
        record_id = item.record.record_id
        if record_id in predictions_by_id:
            continue
        if stopped:
            store(
                _budget_record(
                    run_id=run_id,
                    item=item,
                    attempt=next_attempt(record_id),
                    settings=settings,
                ),
            )
            continue

        attempt = next_attempt(record_id)
        while True:
            if not budget_allows_call():
                stopped = True
                store(
                    _budget_record(
                        run_id=run_id,
                        item=item,
                        attempt=attempt,
                        settings=settings,
                    ),
                )
                break

            request_count += 1
            attempt_started = time.perf_counter()
            try:
                response = adapter.predict(item.request)
            except Exception as exc:
                latency_ms = max(0.0, (time.perf_counter() - attempt_started) * 1000.0)
                error = str(exc)
                response_hash = _canonical_sha256(None)
                _append_jsonl(
                    raw_path,
                    _raw_entry(
                        run_id=run_id,
                        item=item,
                        attempt=attempt,
                        status=PredictionStatus.API_ERROR.value,
                        response_hash=response_hash,
                        latency_ms=latency_ms,
                        raw_response=None,
                        error=error,
                    ),
                )
                if _is_retryable(exc) and attempt <= budget.max_retries:
                    if budget_allows_call():
                        attempt += 1
                        continue
                    stopped = True
                    budget_error = (
                        f"{error}; retry was not attempted because the request budget is exhausted"
                    )
                    store(
                        _prediction_record(
                            run_id=run_id,
                            item=item,
                            attempt=attempt,
                            settings=settings,
                            status=PredictionStatus.API_ERROR.value,
                            prediction=None,
                            prediction_valid=False,
                            probability_valid=False,
                            probabilities=None,
                            error=budget_error,
                            response_hash=response_hash,
                            model=None,
                            control_status=_BUDGET_CONTROL_STATUS,
                        ),
                    )
                    break
                store(
                    _prediction_record(
                        run_id=run_id,
                        item=item,
                        attempt=attempt,
                        settings=settings,
                        status=PredictionStatus.API_ERROR.value,
                        prediction=None,
                        prediction_valid=False,
                        probability_valid=False,
                        probabilities=None,
                        error=error,
                        response_hash=response_hash,
                        model=None,
                    ),
                )
                break
            else:
                latency_ms = max(0.0, (time.perf_counter() - attempt_started) * 1000.0)
                validated = response.validated
                raw_response = _redacted_raw(response)
                response_hash = _canonical_sha256(raw_response)
                _append_jsonl(
                    raw_path,
                    _raw_entry(
                        run_id=run_id,
                        item=item,
                        attempt=attempt,
                        status=validated.status.value,
                        response_hash=response_hash,
                        latency_ms=latency_ms,
                        raw_response=raw_response,
                        error=validated.error,
                    ),
                )
                store(
                    _prediction_record(
                        run_id=run_id,
                        item=item,
                        attempt=attempt,
                        settings=settings,
                        status=validated.status.value,
                        prediction=validated.choice if validated.prediction_valid else None,
                        prediction_valid=validated.prediction_valid,
                        probability_valid=validated.probability_valid,
                        probabilities=(
                            validated.probabilities if validated.probability_valid else None
                        ),
                        error=validated.error,
                        response_hash=response_hash,
                        model=response.model,
                    ),
                )
                break

    final_records = [predictions_by_id[item.record.record_id] for item in selected]
    write_predictions_jsonl(final_records, predictions_path)

    status = _run_status(final_records, stopped)
    finished_at = datetime.now(timezone.utc)

    metrics = build_single_model_metrics(
        final_records,
        zero_policy=zero_policy,
        nll_epsilon=nll_epsilon,
    )

    command_value = list(command) if command is not None else list(sys.argv)
    invocations = _prior_invocations(prior_meta) if prior_meta is not None else []
    invocations.append(
        {
            "command": command_value,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "limit": limit,
            "resumed": resume,
            "status": status,
        },
    )

    metrics_path = run_dir / "metrics.json"
    metrics_md_path = run_dir / "metrics.md"
    meta: dict[str, object] = {
        "run_id": run_id,
        "status": status,
        "stopped": stopped,
        "run_kind": "laya-only-exploratory",
        "gate_status": "phase2-open",
        "limit": limit,
        "resumed": resume,
        "run_signature": signature,
        "invocations": invocations,
        "item_count": len(selected),
        "request_count": request_count,
        "status_counts": _status_counts(final_records),
        "control_status_counts": _control_status_counts(final_records),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "data": data_meta,
        "gold": gold_meta,
        "settings": asdict(settings),
        "model": settings.checkpoint,
        "runtime_metadata": _json_safe_copy(dict(runtime_metadata)),
        "options": _options_metadata(),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "budget": asdict(budget),
        "command": command_value,
        "code": _code_metadata(code_metadata),
        "environment": _environment_metadata(),
        "predictions_sha256": sha256_file(predictions_path),
        "raw_sha256": sha256_file(raw_path),
        "metrics_path": str(metrics_path),
        "metrics_md_path": str(metrics_md_path),
    }
    written_metrics_path, written_metrics_md_path = write_metrics_files(
        run_dir,
        metrics,
        meta,
    )
    meta["metrics_path"] = str(written_metrics_path)
    meta["metrics_md_path"] = str(written_metrics_md_path)
    _write_meta(run_dir, meta)

    return EscRunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        records=tuple(final_records),
        request_count=request_count,
        metrics=metrics,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Laya-only ESC evaluation")
    parser.add_argument(
        "--esc-data",
        default=os.environ.get(
            "ECI_ESC_DATA",
            "../reference/MRBalance/data/test/ESC_intra_data.json",
        ),
    )
    parser.add_argument("--gold", default="artifacts/gold_choice_reconstructed/v1")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=7200.0,
        help=(
            "Scheduling budget in seconds, checked only between requests; this is "
            "not a hard process deadline. Use external subprocess isolation for a "
            "hard wall-clock limit."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--zero-policy", default="error")
    parser.add_argument("--nll-epsilon", type=float)
    parser.add_argument("--authorize-real-laya", action="store_true")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--run-id")
    return parser


def _print_dry_run(dataset: EscDataset, gold: GoldArtifact, items: Sequence[EscItem]) -> None:
    counts = Counter(item.gold.choice for item in items)
    print(f"ESC dry run: {len(items)} items across {len(dataset.records)} dataset rows")
    print(f"gold version: {gold.version if gold.version is not None else 'N/A'}")
    print("choice counts: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if gold.manifest is not None:
        summary = gold.manifest.get("summary")
        hard_failures = summary.get("hard_failures") if isinstance(summary, Mapping) else None
        print(f"manifest hard failures: {hard_failures if hard_failures is not None else 0}")
    print("no model will be loaded; pass --authorize-real-laya to execute")


def esc_main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: Callable[[], EscAdapter] | None = None,
    runtime_metadata: Mapping[str, object] | None = None,
) -> int:
    """CLI entry point; a dry run is the default and never creates an adapter."""
    args = _build_parser().parse_args(None if argv is None else list(argv))

    try:
        dataset = load_esc(args.esc_data)
        gold = load_gold_artifact(args.gold)
        items = prepare_esc_items(dataset, gold)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    selected = list(items) if args.limit is None else list(items)[: args.limit]

    if not args.authorize_real_laya:
        _print_dry_run(dataset, gold, selected)
        return 0

    if not args.run_id:
        print("error: --run-id is required for a real Laya run", file=sys.stderr)
        return 1

    command_value = (
        list(sys.argv) if argv is None else ["python", "-m", "eci_decision_models.esc_run", *argv]
    )

    env: dict[str, object] = dict(os.environ)
    if args.device is not None:
        env["ECI_LAYA_DEVICE"] = str(args.device)
    if args.download_weights:
        env["ECI_LAYA_DOWNLOAD_WEIGHTS"] = "true"

    try:
        settings = LayaSettings.from_env(env)
        if adapter_factory is not None:
            adapter = adapter_factory()
            resolved_runtime = dict(runtime_metadata) if runtime_metadata is not None else {}
        else:
            agent = load_laya_agent(settings)
            adapter = LayaAdapter(client=cast("LayaTextClient", agent))
            resolved_runtime = (
                dict(runtime_metadata)
                if runtime_metadata is not None
                else laya_runtime_metadata(agent)
            )

        max_requests = args.max_requests if args.max_requests is not None else len(selected)
        budget = EscRunBudget(
            max_requests=max_requests,
            max_retries=args.max_retries,
            max_wall_seconds=args.max_wall_seconds,
        )
        result = run_laya_esc(
            items=items,
            adapter=adapter,
            output_dir=args.output_dir,
            run_id=args.run_id,
            settings=settings,
            runtime_metadata=resolved_runtime,
            budget=budget,
            resume=args.resume,
            zero_policy=args.zero_policy,
            nll_epsilon=args.nll_epsilon,
            command=command_value,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"run directory: {result.run_dir}")
    print(f"status: {result.status}")
    print(f"metrics: {result.run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(esc_main())
