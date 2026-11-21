"""Public contracts for the Jev adapter seam, exercised with fakes only."""

from __future__ import annotations

import copy
import importlib.metadata
import sys
import types
from dataclasses import dataclass
from typing import Any, cast

import pytest

from eci_decision_models.canonical import (
    CHOICE_DESCRIPTIONS,
    CHOICE_IDS,
    QUESTION_INSTRUCTIONS,
    build_canonical_request,
)
from eci_decision_models.jev import (
    JevAdapter,
    JevRequest,
    JevResponse,
    JevSettings,
    build_jev_request,
    build_sdk_questions,
    load_jev_client,
    parse_jev_response,
)
from eci_decision_models.laya import build_laya_request
from eci_decision_models.probability import PredictionStatus, ValidatedPrediction

DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "typesafe/jev-1.14"
DEFAULT_SDK_VERSION = "0.7.5"
_MISSING = object()


def _raw_answer(
    *,
    choice: str = "cause",
    probabilities: object = _MISSING,
) -> dict[str, object]:
    entry: dict[str, object] = {"choice": choice}
    if probabilities is not _MISSING:
        entry["probabilities"] = probabilities
    return {"answers": {"eci_relation": entry}}


def _patch_typesafe_version(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    original = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "typesafe-sdk":
            return version
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)


@dataclass(frozen=True)
class _FakeChoice:
    instructions: str
    criteria: dict[str, str]


class _FakeTypeSafeClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.fixture
def fake_typesafe_sdk(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("typesafe_sdk")
    module_any = cast("Any", module)
    module_any.Choice = _FakeChoice
    module_any.TypeSafeClient = _FakeTypeSafeClient
    monkeypatch.setitem(sys.modules, "typesafe_sdk", module)
    return module


def test_jev_settings_defaults() -> None:
    settings = JevSettings.from_env({})

    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.model == DEFAULT_MODEL
    assert settings.sdk_version == DEFAULT_SDK_VERSION


def test_jev_settings_env_overrides() -> None:
    settings = JevSettings.from_env(
        {
            "ECI_JEV_BASE_URL": "https://example.test/api",
            "ECI_JEV_MODEL": "typesafe/jev-test",
            "ECI_JEV_SDK_VERSION": "9.9.9",
        }
    )

    assert settings.base_url == "https://example.test/api"
    assert settings.model == "typesafe/jev-test"
    assert settings.sdk_version == "9.9.9"


@pytest.mark.parametrize(
    "key",
    [
        "ECI_JEV_BASE_URL",
        "ECI_JEV_MODEL",
        "ECI_JEV_SDK_VERSION",
    ],
)
@pytest.mark.parametrize("value", ["", "   ", 123])
def test_jev_settings_rejects_invalid_values(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        JevSettings.from_env({key: value})


def test_jev_settings_never_holds_or_reveals_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-key")
    settings = JevSettings.from_env({})

    assert not hasattr(settings, "api_key")
    assert "super-secret-key" not in repr(settings)
    assert "super-secret-key" not in str(settings)


@pytest.mark.parametrize(
    ("sentence", "event1", "event2"),
    [
        ("The storm destroyed the house.", "the storm", "the house was destroyed"),
        ("She smiled because he apologized.", "she smiled", "he apologized"),
        ("", "event one", "event two"),
    ],
)
def test_build_jev_request_matches_laya_canonical(
    sentence: str,
    event1: str,
    event2: str,
) -> None:
    jev_request = build_jev_request(sentence, event1, event2)
    laya_request = build_laya_request(sentence, event1, event2)
    expected = build_canonical_request(sentence, event1, event2)

    assert isinstance(jev_request, JevRequest)
    assert jev_request.canonical.sha256 == laya_request.canonical.sha256
    assert jev_request.canonical.sha256 == expected.sha256
    assert jev_request.canonical.payload == expected.payload
    assert jev_request.state == laya_request.state
    assert isinstance(jev_request.state, dict)

    jev_question: Any = jev_request.questions["eci_relation"]
    laya_question: Any = laya_request.questions["eci_relation"]
    assert jev_question["type"] == "choice"
    assert jev_question["instructions"] == QUESTION_INSTRUCTIONS
    assert list(jev_question["criteria"].items()) == list(
        zip(CHOICE_IDS, CHOICE_DESCRIPTIONS, strict=True)
    )
    assert jev_question["criteria"] == laya_question["criteria"]


def test_parse_jev_response_normalizes_complete_probabilities() -> None:
    raw = _raw_answer(
        choice="cause",
        probabilities={"cause": 5.0, "caused_by": 3.0, "none": 2.0},
    )

    parsed = parse_jev_response(raw)

    assert isinstance(parsed, ValidatedPrediction)
    assert parsed.status is PredictionStatus.OK
    assert parsed.prediction_valid is True
    assert parsed.probability_valid is True
    assert parsed.choice == "cause"
    assert parsed.probabilities == pytest.approx((0.5, 0.3, 0.2))
    assert parsed.error is None


@pytest.mark.parametrize(
    "probabilities",
    [
        None,
        {"cause": 0.5, "caused_by": 0.3},
        {},
    ],
)
def test_parse_jev_response_missing_probabilities_is_missing_probability(
    probabilities: object,
) -> None:
    parsed = parse_jev_response(_raw_answer(probabilities=probabilities))

    assert parsed.status is PredictionStatus.MISSING_PROBABILITY
    assert parsed.prediction_valid is True
    assert parsed.probability_valid is False
    assert parsed.probabilities is None


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5, 0.3, 0.2],
        {"cause": -0.1, "caused_by": 0.6, "none": 0.5},
        {"cause": 0.0, "caused_by": 0.0, "none": 0.0},
        {"cause": "high", "caused_by": 0.2, "none": 0.1},
    ],
)
def test_parse_jev_response_bad_probabilities_is_invalid_probability(
    probabilities: object,
) -> None:
    parsed = parse_jev_response(_raw_answer(probabilities=probabilities))

    assert parsed.status is PredictionStatus.INVALID_PROBABILITY
    assert parsed.prediction_valid is True
    assert parsed.probability_valid is False
    assert parsed.probabilities is None


def test_parse_jev_response_argmax_mismatch_is_invalid_probability() -> None:
    parsed = parse_jev_response(
        _raw_answer(
            choice="cause",
            probabilities={"cause": 0.1, "caused_by": 0.2, "none": 0.7},
        )
    )

    assert parsed.status is PredictionStatus.INVALID_PROBABILITY
    assert parsed.error is not None
    assert "argmax" in parsed.error


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-mapping",
        {},
        {"answers": {}},
        {"answers": []},
        {"answers": {"eci_relation": "cause"}},
        {"answers": {"eci_relation": None}},
        {"answers": {"eci_relation": {}}},
        {
            "answers": {
                "eci_relation": {
                    "probabilities": {"cause": 0.5, "caused_by": 0.3, "none": 0.2},
                },
            },
        },
    ],
)
def test_parse_jev_response_missing_answer_or_choice_is_invalid_response(
    raw: object,
) -> None:
    parsed = parse_jev_response(raw)

    assert parsed.status is PredictionStatus.INVALID_RESPONSE
    assert parsed.prediction_valid is False
    assert parsed.probability_valid is False
    assert parsed.probabilities is None
    assert isinstance(parsed.error, str)
    assert parsed.error


def test_build_sdk_questions_uses_choice_with_canonical_criteria(
    fake_typesafe_sdk: types.ModuleType,
) -> None:
    questions = build_sdk_questions()

    assert fake_typesafe_sdk.__name__ == "typesafe_sdk"
    choice = questions["eci_relation"]
    assert isinstance(choice, _FakeChoice)
    assert choice.instructions == QUESTION_INSTRUCTIONS
    assert list(choice.criteria.items()) == list(zip(CHOICE_IDS, CHOICE_DESCRIPTIONS, strict=True))


class _FakeClient:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def system_one(
        self,
        *,
        model: str,
        state: dict[str, object],
        questions: dict[str, object],
    ) -> object:
        self.calls.append(
            {
                "model": model,
                "state": copy.deepcopy(state),
                "questions": copy.deepcopy(questions),
            }
        )
        return self.result


class _FakeRawHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def json(self) -> dict[str, object]:
        return self.payload


class _FakeSdkResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    @property
    def raw_http_response(self) -> _FakeRawHttpResponse:
        return _FakeRawHttpResponse(self.payload)


def test_jev_adapter_parses_raw_mapping_and_returns_independent_copy(
    fake_typesafe_sdk: types.ModuleType,
) -> None:
    client_raw: Any = {
        "id": "req_123",
        "model": DEFAULT_MODEL,
        "provider": "openrouter",
        "answers": {
            "eci_relation": {
                "type": "choice",
                "choice": "cause",
                "confidence": 0.6,
                "probabilities": {"cause": 0.5, "caused_by": 0.3, "none": 0.2},
            }
        },
        "usage": {"input_tokens": 11, "output_tokens": 7, "cost": 0.0012},
    }
    client = _FakeClient(client_raw)
    request = build_jev_request("sentence", "event1", "event2")

    response = JevAdapter(client, model=DEFAULT_MODEL).predict(request)

    assert fake_typesafe_sdk.__name__ == "typesafe_sdk"
    assert isinstance(response, JevResponse)
    assert client.calls[0]["model"] == DEFAULT_MODEL
    assert client.calls[0]["state"] == request.state
    call_questions = client.calls[0]["questions"]
    assert isinstance(call_questions, dict)
    call_choice = call_questions["eci_relation"]
    assert isinstance(call_choice, _FakeChoice)
    assert call_choice.instructions == QUESTION_INSTRUCTIONS
    assert call_choice.criteria == dict(zip(CHOICE_IDS, CHOICE_DESCRIPTIONS, strict=True))

    assert isinstance(response.validated, ValidatedPrediction)
    assert response.validated.status is PredictionStatus.OK
    assert response.validated.probabilities == pytest.approx((0.5, 0.3, 0.2))
    assert response.raw == client_raw
    assert type(response.raw) is dict
    assert response.usage == client_raw["usage"]
    assert type(response.usage) is dict
    assert response.model == DEFAULT_MODEL
    assert response.request_id == "req_123"
    assert response.provider == "openrouter"
    assert response.cost == pytest.approx(0.0012)

    assert response.raw is not client_raw
    response_raw: Any = response.raw
    response_usage: Any = response.usage
    response_raw["answers"]["eci_relation"]["choice"] = "none"
    response_raw["usage"]["input_tokens"] = 999
    response_usage["cost"] = 999

    assert client_raw["answers"]["eci_relation"]["choice"] == "cause"
    assert client_raw["usage"]["input_tokens"] == 11
    assert client_raw["usage"]["cost"] == 0.0012


def test_jev_adapter_reads_typed_sdk_raw_http_response(
    fake_typesafe_sdk: types.ModuleType,
) -> None:
    payload: dict[str, object] = {
        "id": "req_456",
        "model": "typesafe/jev-1.14",
        "provider": "openrouter",
        "answers": {
            "eci_relation": {
                "choice": "caused_by",
                "probabilities": {"cause": 0.2, "caused_by": 0.7, "none": 0.1},
            }
        },
        "usage": {"input_tokens": 3, "output_tokens": 4, "cost": 0.0005},
    }
    response = JevAdapter(_FakeClient(_FakeSdkResponse(payload))).predict(
        build_jev_request("sentence", "event1", "event2")
    )

    assert fake_typesafe_sdk.__name__ == "typesafe_sdk"
    assert response.validated.status is PredictionStatus.OK
    assert response.validated.choice == "caused_by"
    assert response.raw == payload
    assert response.request_id == "req_456"
    assert response.provider == "openrouter"
    assert response.cost == pytest.approx(0.0005)


class _RecordingFactory:
    def __init__(self) -> None:
        self.result = object()
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.result


@pytest.mark.parametrize("api_key", ["", "   "])
def test_load_jev_client_requires_non_empty_api_key(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str,
) -> None:
    _patch_typesafe_version(monkeypatch, DEFAULT_SDK_VERSION)

    with pytest.raises(RuntimeError) as excinfo:
        load_jev_client(
            JevSettings.from_env({}),
            environ={"OPENROUTER_API_KEY": api_key},
            client_factory=_RecordingFactory(),
        )

    assert "OPENROUTER_API_KEY" in str(excinfo.value)


def test_load_jev_client_rejects_sdk_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_typesafe_version(monkeypatch, "0.7.4")

    with pytest.raises(RuntimeError) as excinfo:
        load_jev_client(
            JevSettings.from_env({}),
            environ={"OPENROUTER_API_KEY": "test-key"},
            client_factory=_RecordingFactory(),
        )

    message = str(excinfo.value)
    assert "0.7.4" in message
    assert DEFAULT_SDK_VERSION in message


def test_load_jev_client_passes_documented_constructor_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_typesafe_version(monkeypatch, DEFAULT_SDK_VERSION)
    factory = _RecordingFactory()
    settings = JevSettings.from_env(
        {
            "ECI_JEV_BASE_URL": "https://example.test/api",
            "ECI_JEV_MODEL": "typesafe/jev-test",
        }
    )

    client = load_jev_client(
        settings,
        environ={"OPENROUTER_API_KEY": "  test-key  "},
        client_factory=factory,
    )

    assert client is factory.result
    assert factory.calls == [
        {
            "api_key": "test-key",
            "base_url": "https://example.test/api",
            "compression": False,
        }
    ]


def test_load_jev_client_does_not_guess_retry_or_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_typesafe_version(monkeypatch, DEFAULT_SDK_VERSION)

    class _SignatureFactory:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def __call__(
            self,
            *,
            api_key: str,
            base_url: str,
            compression: bool,
            retry: object = None,
            timeout: float = 30.0,
        ) -> object:
            self.kwargs = {
                "api_key": api_key,
                "base_url": base_url,
                "compression": compression,
            }
            return object()

    factory = _SignatureFactory()

    load_jev_client(
        JevSettings.from_env({}),
        environ={"OPENROUTER_API_KEY": "test-key"},
        client_factory=factory,
    )

    assert factory.kwargs == {
        "api_key": "test-key",
        "base_url": DEFAULT_BASE_URL,
        "compression": False,
    }


def test_load_jev_client_uses_sdk_factory_without_network(
    monkeypatch: pytest.MonkeyPatch,
    fake_typesafe_sdk: types.ModuleType,
) -> None:
    _patch_typesafe_version(monkeypatch, DEFAULT_SDK_VERSION)

    client = load_jev_client(
        JevSettings.from_env({}),
        environ={"OPENROUTER_API_KEY": "test-key"},
    )

    assert fake_typesafe_sdk.__name__ == "typesafe_sdk"
    assert isinstance(client, _FakeTypeSafeClient)
    assert client.kwargs == {
        "api_key": "test-key",
        "base_url": DEFAULT_BASE_URL,
        "compression": False,
    }
