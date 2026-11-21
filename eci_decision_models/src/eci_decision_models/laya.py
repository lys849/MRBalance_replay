"""Laya adapter seam for the canonical ECI decision contract.

The module intentionally does not import the real ``laya`` package at import
time.  Offline tests and metadata work must never trigger a download, a model
load, or a network call.  The default loader imports ``laya`` lazily and only
after the caller has explicitly allowed weight downloads.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from eci_decision_models.canonical import (
    CHOICE_DESCRIPTIONS,
    CHOICE_IDS,
    QUESTION_INSTRUCTIONS,
    CanonicalRequest,
    build_canonical_request,
)
from eci_decision_models.probability import (
    PredictionStatus,
    ValidatedPrediction,
    validate_prediction,
)

_DEFAULT_CHECKPOINT = "convaiinnovations/laya-typed-decisions"
_DEFAULT_REVISION = "e929ae5cf69bc34259cd2f95c9e91145b818b1f0"
_DEFAULT_SDK_VERSION = "0.4.17"
_DEFAULT_DEVICE = "auto"

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_DEVICE_PATTERN = re.compile(r"cuda:\d+")
_SIMPLE_DEVICES = frozenset({"auto", "cpu", "cuda", "mps"})


class LayaTextClient(Protocol):
    """Minimal client surface required by the adapter."""

    def system_one(
        self,
        state: dict[str, object],
        questions: dict[str, object],
    ) -> object: ...


@dataclass(frozen=True)
class LayaSettings:
    """Frozen runtime settings with strict environment validation."""

    checkpoint: str = _DEFAULT_CHECKPOINT
    revision: str = _DEFAULT_REVISION
    sdk_version: str = _DEFAULT_SDK_VERSION
    device: str = _DEFAULT_DEVICE
    download_weights: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, object]) -> LayaSettings:
        checkpoint = _read_str(env, "ECI_LAYA_CHECKPOINT", _DEFAULT_CHECKPOINT)
        revision = _read_str(env, "ECI_LAYA_REVISION", _DEFAULT_REVISION)
        sdk_version = _read_str(env, "ECI_LAYA_SDK_VERSION", _DEFAULT_SDK_VERSION)
        device = _read_str(env, "ECI_LAYA_DEVICE", _DEFAULT_DEVICE).lower()
        download_weights = _read_bool(env, "ECI_LAYA_DOWNLOAD_WEIGHTS", False)

        _validate_revision(revision)
        _validate_device(device)
        return cls(
            checkpoint=checkpoint,
            revision=revision,
            sdk_version=sdk_version,
            device=device,
            download_weights=download_weights,
        )


@dataclass(frozen=True)
class LayaRequest:
    """One canonical request plus the plain containers sent to Laya."""

    canonical: CanonicalRequest
    state: dict[str, object]
    questions: dict[str, object]


@dataclass(frozen=True)
class LayaResponse:
    """Adapter output with an independent raw copy and parsed prediction."""

    raw: dict[str, object]
    validated: ValidatedPrediction
    usage: dict[str, object]
    model: str | None


def build_laya_request(sentence: str, event1: str, event2: str) -> LayaRequest:
    """Build a canonical request and fresh plain Laya input containers."""
    canonical = build_canonical_request(sentence, event1, event2)
    state: dict[str, object] = {
        "sentence": sentence,
        "event1": event1,
        "event2": event2,
    }
    criteria: dict[str, str] = {
        choice_id: description
        for choice_id, description in zip(CHOICE_IDS, CHOICE_DESCRIPTIONS, strict=True)
    }
    questions: dict[str, object] = {
        "eci_relation": {
            "type": "choice",
            "instructions": QUESTION_INSTRUCTIONS,
            "criteria": criteria,
        },
    }
    return LayaRequest(canonical=canonical, state=state, questions=questions)


def _invalid_response(message: str) -> ValidatedPrediction:
    return ValidatedPrediction(
        status=PredictionStatus.INVALID_RESPONSE,
        prediction_valid=False,
        probability_valid=False,
        choice=None,
        probabilities=None,
        error=message,
    )


def parse_laya_response(raw: object) -> ValidatedPrediction:
    """Parse Laya's ``system_one`` payload into the shared prediction type."""
    if not isinstance(raw, Mapping):
        return _invalid_response("Laya response must be a mapping")

    answers = raw.get("answers")
    if not isinstance(answers, Mapping):
        return _invalid_response("Laya response is missing a mapping field 'answers'")

    entry = answers.get("eci_relation")
    if not isinstance(entry, Mapping):
        return _invalid_response("Laya response is missing a mapping answer 'eci_relation'")

    choice = cast("str | None", entry.get("choice"))
    probabilities = cast("Mapping[str, float] | None", entry.get("probabilities"))
    return validate_prediction(choice, probabilities)


def _plain_copy(value: object) -> object:
    """Recursively copy mappings/lists into independent plain containers."""
    if isinstance(value, Mapping):
        return {str(key): _plain_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_copy(item) for item in value)
    if isinstance(value, set):
        return {_plain_copy(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_plain_copy(item) for item in value)
    return value


@dataclass(frozen=True)
class LayaAdapter:
    """Thin adapter that keeps Laya's raw response and parsed view separate."""

    client: LayaTextClient

    def predict(self, request: LayaRequest) -> LayaResponse:
        raw_result = self.client.system_one(request.state, request.questions)
        source: Mapping[str, object] = raw_result if isinstance(raw_result, Mapping) else {}
        raw = cast("dict[str, object]", _plain_copy(source))
        usage_source = source.get("usage")
        usage = (
            cast("dict[str, object]", _plain_copy(usage_source))
            if isinstance(usage_source, Mapping)
            else {}
        )
        model_value = source.get("model")
        model = model_value if isinstance(model_value, str) else None
        return LayaResponse(
            raw=raw,
            validated=parse_laya_response(source),
            usage=usage,
            model=model,
        )


def _agent_value(agent: object, name: str) -> object:
    """Read a direct agent attribute, falling back to ``agent.cfg``."""
    if hasattr(agent, name):
        return getattr(agent, name)
    cfg = getattr(agent, "cfg", None)
    if isinstance(cfg, Mapping):
        if name in cfg:
            return cfg[name]
    elif cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return None


def _stringify(value: object) -> object:
    if value is None:
        return None
    return str(value)


def _json_safe_copy(value: object) -> object:
    """Return a plain, JSON-serializable deep copy of a temperature/config value."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_copy(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe_copy(item) for item in value]
    return str(value)


def _inference_dtype(agent: object) -> object:
    dtype = _agent_value(agent, "dtype")
    dtype_for = getattr(agent, "dtype_for", None)
    if not callable(dtype_for):
        return _stringify(dtype)
    try:
        return _stringify(cast("Callable[[int], object]", dtype_for)(1))
    except Exception:
        return _stringify(dtype)


def laya_runtime_metadata(agent: object) -> dict[str, object]:
    """Expose honest, JSON-safe runtime metadata from direct attrs or ``cfg``."""
    return {
        "revision": _json_safe_copy(_agent_value(agent, "revision")),
        "model_id": _json_safe_copy(_agent_value(agent, "model_id")),
        "device": _stringify(_agent_value(agent, "device")),
        "dtype": _stringify(_agent_value(agent, "dtype")),
        "backend": _json_safe_copy(_agent_value(agent, "backend")),
        "amp_enabled": _json_safe_copy(_agent_value(agent, "amp_enabled")),
        "mps_amp_min_rows": _json_safe_copy(_agent_value(agent, "mps_amp_min_rows")),
        "inference_dtype": _inference_dtype(agent),
        "max_len": _json_safe_copy(_agent_value(agent, "max_len")),
        "head_max_len": _json_safe_copy(_agent_value(agent, "head_max_len")),
        "temperature_raw": _json_safe_copy(_agent_value(agent, "temperature_raw")),
        "temperature_by_options_raw": _json_safe_copy(
            _agent_value(agent, "temperature_by_options_raw"),
        ),
        "temperature": _json_safe_copy(_agent_value(agent, "temperature")),
        "temperature_by_options": _json_safe_copy(
            _agent_value(agent, "temperature_by_options"),
        ),
    }


def _default_laya_loader(checkpoint: str, *, revision: str, device: str) -> object:
    """Import and call the real Laya loader only when explicitly requested."""
    laya = importlib.import_module("laya")
    return laya.load(checkpoint, device=device, revision=revision)


def load_laya_agent(
    settings: LayaSettings,
    *,
    loader: Callable[..., object] | None = None,
) -> object:
    """Load a Laya agent, refusing downloads unless the caller opted in."""
    if not settings.download_weights:
        raise RuntimeError(
            "ECI_LAYA_DOWNLOAD_WEIGHTS must be true before loading Laya weights",
        )

    try:
        installed_version = importlib.metadata.version("laya")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Laya SDK is not installed; cannot verify sdk version {settings.sdk_version!r}",
        ) from exc
    if installed_version != settings.sdk_version:
        raise RuntimeError(
            f"Laya SDK version {installed_version!r} does not match "
            f"settings sdk_version {settings.sdk_version!r}",
        )

    selected_loader = loader if loader is not None else _default_laya_loader
    agent = selected_loader(
        settings.checkpoint,
        device=settings.device,
        revision=settings.revision,
    )
    loaded_revision = _agent_value(agent, "revision")
    if loaded_revision != settings.revision:
        raise RuntimeError(
            f"Laya agent revision {loaded_revision!r} does not match "
            f"settings revision {settings.revision!r}",
        )
    return agent


def _read_str(env: Mapping[str, object], key: str, default: str) -> str:
    value = env.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{key} must not be empty")
    return stripped


def _read_bool(env: Mapping[str, object], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be 'true' or 'false'")
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{key} must be 'true' or 'false'")


def _validate_revision(revision: str) -> None:
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("ECI_LAYA_REVISION must be a 40-character lowercase hex SHA")


def _validate_device(device: str) -> None:
    if device in _SIMPLE_DEVICES or _DEVICE_PATTERN.fullmatch(device) is not None:
        return
    raise ValueError(
        f"ECI_LAYA_DEVICE must be one of {sorted(_SIMPLE_DEVICES)!r} or 'cuda:<index>'",
    )
