"""Jev adapter seam for the canonical ECI decision contract.

The module intentionally does not import ``typesafe_sdk`` at import time.  The
default loader imports the SDK lazily, after the caller has supplied an API key,
so offline tests and metadata work never require the optional dependency, a
network call, or an API key.

The API key is read only by :func:`load_jev_client` and is never stored on a
settings or response object, so it cannot leak through ``repr`` or artifacts.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

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

_DEFAULT_BASE_URL = "https://openrouter.ai/api"
_DEFAULT_MODEL = "typesafe/jev-1.14"
_DEFAULT_SDK_VERSION = "0.7.5"
_API_KEY_ENV = "OPENROUTER_API_KEY"

__all__ = [
    "JevAdapter",
    "JevRequest",
    "JevResponse",
    "JevSettings",
    "JevTextClient",
    "build_jev_request",
    "build_sdk_questions",
    "load_jev_client",
    "parse_jev_response",
]


class JevTextClient(Protocol):
    """Minimal client surface required by the adapter."""

    def system_one(
        self,
        *,
        model: str,
        state: dict[str, object],
        questions: dict[str, object],
    ) -> object: ...


@dataclass(frozen=True)
class JevSettings:
    """Frozen runtime settings with strict environment validation."""

    base_url: str = _DEFAULT_BASE_URL
    model: str = _DEFAULT_MODEL
    sdk_version: str = _DEFAULT_SDK_VERSION

    @classmethod
    def from_env(cls, env: Mapping[str, object]) -> JevSettings:
        return cls(
            base_url=_read_str(env, "ECI_JEV_BASE_URL", _DEFAULT_BASE_URL),
            model=_read_str(env, "ECI_JEV_MODEL", _DEFAULT_MODEL),
            sdk_version=_read_str(env, "ECI_JEV_SDK_VERSION", _DEFAULT_SDK_VERSION),
        )


@dataclass(frozen=True)
class JevRequest:
    """One canonical request plus the plain containers sent to Jev."""

    canonical: CanonicalRequest
    state: dict[str, object]
    questions: dict[str, object]


@dataclass(frozen=True)
class JevResponse:
    """Adapter output with an independent raw copy and parsed prediction."""

    raw: dict[str, object]
    validated: ValidatedPrediction
    usage: dict[str, object]
    model: str | None
    request_id: str | None
    provider: str | None
    cost: float | None


def build_jev_request(sentence: str, event1: str, event2: str) -> JevRequest:
    """Build a canonical request and fresh plain Jev input containers."""
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
    return JevRequest(canonical=canonical, state=state, questions=questions)


def build_sdk_questions() -> dict[str, object]:
    """Build typed ``Choice`` questions using a lazy SDK import."""
    typesafe_sdk = cast("Any", importlib.import_module("typesafe_sdk"))
    choice_type = cast("Callable[..., object]", typesafe_sdk.Choice)
    criteria: dict[str, str] = {
        choice_id: description
        for choice_id, description in zip(CHOICE_IDS, CHOICE_DESCRIPTIONS, strict=True)
    }
    return {
        "eci_relation": choice_type(
            instructions=QUESTION_INSTRUCTIONS,
            criteria=criteria,
        ),
    }


def _invalid_response(message: str) -> ValidatedPrediction:
    return ValidatedPrediction(
        status=PredictionStatus.INVALID_RESPONSE,
        prediction_valid=False,
        probability_valid=False,
        choice=None,
        probabilities=None,
        error=message,
    )


def parse_jev_response(raw: object) -> ValidatedPrediction:
    """Parse Jev's ``system_one`` payload into the shared prediction type."""
    if not isinstance(raw, Mapping):
        return _invalid_response("Jev response must be a mapping")

    answers = raw.get("answers")
    if not isinstance(answers, Mapping):
        return _invalid_response("Jev response is missing a mapping field 'answers'")

    entry = answers.get("eci_relation")
    if not isinstance(entry, Mapping):
        return _invalid_response("Jev response is missing a mapping answer 'eci_relation'")

    choice = cast("str | None", entry.get("choice"))
    probabilities = cast("Mapping[str, float] | None", entry.get("probabilities"))
    return validate_prediction(choice, probabilities)


def _plain_copy(value: object) -> object:
    """Recursively copy a value into independent, JSON-safe plain containers."""
    if isinstance(value, Mapping):
        return {str(key): _plain_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_copy(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain_copy(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _sdk_raw_mapping(result: object) -> Mapping[str, object] | None:
    """Read ``raw_http_response.json()`` from a typed SDK response, when present."""
    raw_http = getattr(result, "raw_http_response", None)
    if raw_http is None:
        return None
    json_method = getattr(raw_http, "json", None)
    if not callable(json_method):
        return None
    payload = cast("Callable[[], object]", json_method)()
    if isinstance(payload, Mapping):
        return cast("Mapping[str, object]", payload)
    return None


def _raw_mapping(result: object) -> Mapping[str, object]:
    """Return a raw mapping from a plain mapping or a typed SDK response."""
    if isinstance(result, Mapping):
        return cast("Mapping[str, object]", result)
    extracted = _sdk_raw_mapping(result)
    return extracted if extracted is not None else {}


def _string_field(source: Mapping[str, object], key: str) -> str | None:
    value = source.get(key)
    return value if isinstance(value, str) else None


def _cost_field(usage: Mapping[str, object]) -> float | None:
    value = usage.get("cost")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class JevAdapter:
    """Thin adapter that keeps Jev's raw response and parsed view separate."""

    client: JevTextClient
    model: str = _DEFAULT_MODEL

    def predict(self, request: JevRequest) -> JevResponse:
        raw_result = self.client.system_one(
            model=self.model,
            state=request.state,
            questions=build_sdk_questions(),
        )
        source = _raw_mapping(raw_result)
        raw = cast("dict[str, object]", _plain_copy(source))
        usage_source = source.get("usage")
        usage = (
            cast("dict[str, object]", _plain_copy(usage_source))
            if isinstance(usage_source, Mapping)
            else {}
        )
        return JevResponse(
            raw=raw,
            validated=parse_jev_response(source),
            usage=usage,
            model=_string_field(source, "model"),
            request_id=_string_field(source, "id"),
            provider=_string_field(source, "provider"),
            cost=_cost_field(usage),
        )


def _filter_supported_kwargs(
    factory: Callable[..., object],
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Drop kwargs the target callable does not document as parameters.

    The confirmed constructor signature accepts ``api_key``, ``base_url`` and
    ``compression``.  Retry and timeout values are deliberately left at the
    SDK's documented defaults because :class:`JevSettings` does not configure
    them; no values are guessed.
    """
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in parameters}


def load_jev_client(
    settings: JevSettings,
    *,
    environ: Mapping[str, object] | None = None,
    client_factory: Callable[..., object] | None = None,
) -> object:
    """Verify the SDK version and construct a client with the caller's key."""
    env = environ if environ is not None else os.environ
    api_key_value = env.get(_API_KEY_ENV)
    if not isinstance(api_key_value, str) or not api_key_value.strip():
        raise RuntimeError(f"{_API_KEY_ENV} must be set to a non-empty value")
    api_key = api_key_value.strip()

    try:
        installed_version = importlib.metadata.version("typesafe-sdk")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"TypeSafe SDK is not installed; cannot verify sdk version {settings.sdk_version!r}",
        ) from exc
    if installed_version != settings.sdk_version:
        raise RuntimeError(
            f"TypeSafe SDK version {installed_version!r} does not match "
            f"settings sdk_version {settings.sdk_version!r}",
        )

    if client_factory is None:
        typesafe_sdk = cast("Any", importlib.import_module("typesafe_sdk"))
        factory = cast("Callable[..., object]", typesafe_sdk.TypeSafeClient)
    else:
        factory = client_factory

    kwargs = _filter_supported_kwargs(
        factory,
        {
            "api_key": api_key,
            "base_url": settings.base_url,
            "compression": False,
        },
    )
    return factory(**kwargs)


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
