"""Run manifests and reproducible experiment artifacts.

Every completed run writes ``<output_dir>/run_meta.json``.  When an experiment
directory is supplied the same manifest is copied there together with
``metrics.json``, ``metrics.md`` and a deterministic gzip of the response file.
A failed run writes nothing: manifest generation happens only after the run
loop and metric computation have finished successfully.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from . import __version__
from .data import load_jsonl
from .evaluation.metrics import Metrics, MetricsReport, format_report

if TYPE_CHECKING:
    from .runner import RunConfig, RunStats

SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 hex digest of ``path``, or ``None`` if it is missing."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_counts(path: Path) -> tuple[int, int]:
    """Return ``(n_records, n_failed)`` for a runner output file."""
    if not path.exists():
        return 0, 0
    n_records = 0
    n_failed = 0
    for record in load_jsonl(path):
        n_records += 1
        if record.get("error"):
            n_failed += 1
    return n_records, n_failed


def _jsonable(value: Any) -> Any:
    """Recursively convert a dataclass payload into JSON-safe primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Return a JSON-serialisable representation of ``config``."""
    return cast(dict[str, Any], _jsonable(dataclasses.asdict(config)))


def _metric_dict(metrics: Metrics) -> dict[str, float | int]:
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "pos_recall": metrics.pos_recall,
        "neg_recall": metrics.neg_recall,
        "accuracy": metrics.accuracy,
        "n_samples": metrics.n_samples,
    }


def metrics_to_dict(report: MetricsReport) -> dict[str, dict[str, float | int]]:
    """Return metrics in the manifest's ``all/intra/inter`` layout."""
    return {
        "all": _metric_dict(report.overall),
        "intra": _metric_dict(report.intra),
        "inter": _metric_dict(report.inter),
    }


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_info() -> dict[str, str | bool | None]:
    """Return commit, branch and dirty flag; failures become ``None``."""
    commit = _git_value("rev-parse", "HEAD")
    branch = _git_value("rev-parse", "--abbrev-ref", "HEAD")
    status = _git_value("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def _dump_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def deterministic_gzip(source: Path, destination: Path) -> None:
    """Gzip ``source`` to ``destination`` with reproducible header bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open("rb") as source_handle,
        destination.open("wb") as destination_handle,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=destination_handle,
            mtime=0,
            compresslevel=9,
        ) as gzip_handle,
    ):
        while chunk := source_handle.read(1024 * 1024):
            gzip_handle.write(chunk)


def build_manifest(
    *,
    config: RunConfig,
    stats: RunStats,
    report: MetricsReport,
    model: str,
    reasoning_effort: str,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    """Build the run manifest payload without touching the filesystem."""
    output_path = Path(config.output_dir) / "response.jsonl"
    input_path = Path(config.input_path)
    n_output, n_failed = _jsonl_counts(output_path)
    n_input = len(load_jsonl(input_path)) if input_path.exists() else 0

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": config.run_id or Path(config.output_dir).name,
        "created_at": utc_now(),
        "started_at": started_at,
        "finished_at": finished_at,
        "command": list(sys.argv),
        "package_version": __version__,
        "git": git_info(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "config": config_to_dict(config),
        "data": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "n_records": n_input,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "n_records": n_output,
            "n_failed": n_failed,
        },
        "stats": {
            "processed": stats.processed,
            "skipped": stats.skipped,
            "failed": stats.failed,
            "calls": stats.n_calls,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "elapsed_seconds": stats.elapsed_seconds,
        },
        "metrics": metrics_to_dict(report),
    }


def write_manifest(
    *,
    config: RunConfig,
    stats: RunStats,
    report: MetricsReport,
    model: str,
    reasoning_effort: str,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    """Write ``run_meta.json`` and, when configured, the experiment artifacts."""
    manifest = build_manifest(
        config=config,
        stats=stats,
        report=report,
        model=model,
        reasoning_effort=reasoning_effort,
        started_at=started_at,
        finished_at=finished_at,
    )
    payload = _dump_json(manifest)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_meta.json").write_text(payload, encoding="utf-8")

    if config.experiment_dir is not None:
        experiment_dir = Path(config.experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "meta.json").write_text(payload, encoding="utf-8")
        (experiment_dir / "metrics.json").write_text(
            _dump_json(manifest["metrics"]), encoding="utf-8"
        )
        (experiment_dir / "metrics.md").write_text(format_report(report) + "\n", encoding="utf-8")
        deterministic_gzip(
            source=output_dir / "response.jsonl",
            destination=experiment_dir / "response.jsonl.gz",
        )

    return manifest
