"""Public contracts for the Laya adapter seam, exercised with fakes only."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from types import SimpleNamespace
from typing import Any

import pytest

from eci_decision_models.canonical import (
    CHOICE_DESCRIPTIONS,
    CHOICE_IDS,
    QUESTION_INSTRUCTIONS,
    build_canonical_request,
)
from eci_decision_models.laya import (
    LayaAdapter,
    LayaRequest,
    LayaResponse,
    LayaSettings,
    build_laya_request,
    laya_runtime_metadata,
    load_laya_agent,
    parse_laya_response,
)
from eci_decision_models.probability import PredictionStatus, ValidatedPrediction

PINNED_REVISION = "e929ae5cf69bc34259cd2f95c9e91145b818b1f0"
PINNED_CHECKPOINT = "convaiinnovations/laya-typed-decisions"
VALID_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _settings_with_download() -> LayaSettings:
    return LayaSettings.from_env(
        {
            "ECI_LAYA_DOWNLOAD_WEIGHTS": "true",
            "ECI_LAYA_DEVICE": "cpu",
        }
    )


def _patch_laya_version(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    original = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "laya":
            return version
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)


def _raw_answer(
    *,
    choice: str = "cause",
    probabilities: object | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {"choice": choice}
    if probabilities is not None:
        entry["probabilities"] = probabilities
    return {"answers": {"eci_relation": entry}}


def test_laya_settings_defaults() -> None:
    settings = LayaSettings.from_env({})

    assert settings.checkpoint == PINNED_CHECKPOINT
    assert settings.revision == PINNED_REVISION
    assert settings.sdk_version == "0.4.17"
    assert settings.device == "auto"
    assert settings.download_weights is False


def test_laya_settings_env_overrides() -> None:
    settings = LayaSettings.from_env(
        {
            "ECI_LAYA_CHECKPOINT": "org/custom-laya",
            "ECI_LAYA_REVISION": VALID_REVISION,
            "ECI_LAYA_DEVICE": "cpu",
            "ECI_LAYA_DOWNLOAD_WEIGHTS": "true",
        }
    )

    assert settings.checkpoint == "org/custom-laya"
    assert settings.revision == VALID_REVISION
    assert settings.device == "cpu"
    assert settings.download_weights is True


@pytest.mark.parametrize(
    "revision",
    [
        "",
        "abc123",
        "e929ae5cf69bc34259cd2f95c9e91145b818b1f",
        "e929ae5cf69bc34259cd2f95c9e91145b818b1f00",
        "z" * 40,
    ],
)
def test_laya_settings_rejects_invalid_revision(revision: str) -> None:
    with pytest.raises(ValueError):
        LayaSettings.from_env({"ECI_LAYA_REVISION": revision})


@pytest.mark.parametrize("device", ["tpu", "npu", "quantum"])
def test_laya_settings_rejects_invalid_device(device: str) -> None:
    with pytest.raises(ValueError):
        LayaSettings.from_env({"ECI_LAYA_DEVICE": device})


@pytest.mark.parametrize("value", ["maybe", "", "2", "treu"])
def test_laya_settings_rejects_invalid_download_weights_boolean(value: str) -> None:
    with pytest.raises(ValueError):
        LayaSettings.from_env({"ECI_LAYA_DOWNLOAD_WEIGHTS": value})


def test_laya_settings_accepts_false_download_weights_override() -> None:
    settings = LayaSettings.from_env({"ECI_LAYA_DOWNLOAD_WEIGHTS": "false"})

    assert settings.download_weights is False


def test_build_laya_request_has_canonical_state_and_questions() -> None:
    request = build_laya_request(
        "The storm destroyed the house.",
        "the storm",
        "the house was destroyed",
    )

    assert isinstance(request, LayaRequest)
    expected_canonical = build_canonical_request(
        "The storm destroyed the house.",
        "the storm",
        "the house was destroyed",
    )
    assert request.canonical.sha256 == expected_canonical.sha256
    assert _canonical_json(request.canonical.payload) == _canonical_json(expected_canonical.payload)
    state: Any = request.state
    questions: Any = request.questions
    assert state == {
        "sentence": "The storm destroyed the house.",
        "event1": "the storm",
        "event2": "the house was destroyed",
    }
    assert isinstance(state, dict)

    assert questions["eci_relation"]["type"] == "choice"
    assert questions["eci_relation"]["instructions"] == QUESTION_INSTRUCTIONS
    criteria = questions["eci_relation"]["criteria"]
    assert list(criteria.items()) == list(zip(CHOICE_IDS, CHOICE_DESCRIPTIONS, strict=True))


def test_laya_request_state_and_questions_are_mutable_copies() -> None:
    request = build_laya_request(
        "The storm destroyed the house.",
        "the storm",
        "the house was destroyed",
    )
    payload_before = _canonical_json(request.canonical.payload)
    options_before = _canonical_json(request.canonical.options)
    sha_before = request.canonical.sha256
    state: Any = request.state
    questions: Any = request.questions

    state["sentence"] = "tampered"
    questions["eci_relation"]["criteria"][CHOICE_IDS[0]] = "tampered"

    assert _canonical_json(request.canonical.payload) == payload_before
    assert _canonical_json(request.canonical.options) == options_before
    assert request.canonical.sha256 == sha_before
    assert _canonical_sha256(request.canonical.payload) == request.canonical.sha256


def test_parse_laya_response_normalizes_complete_probabilities() -> None:
    raw = _raw_answer(
        choice="cause",
        probabilities={"cause": 5.0, "caused_by": 3.0, "none": 2.0},
    )

    parsed = parse_laya_response(raw)

    assert isinstance(parsed, ValidatedPrediction)
    assert parsed.status is PredictionStatus.OK
    assert parsed.prediction_valid is True
    assert parsed.probability_valid is True
    assert parsed.choice == "cause"
    assert parsed.probabilities == pytest.approx((0.5, 0.3, 0.2))
    assert parsed.error is None


def test_parse_laya_response_missing_probabilities_is_missing_probability() -> None:
    parsed = parse_laya_response(_raw_answer(choice="cause"))

    assert parsed.status is PredictionStatus.MISSING_PROBABILITY
    assert parsed.prediction_valid is True
    assert parsed.probability_valid is False
    assert parsed.choice == "cause"
    assert parsed.probabilities is None


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"answers": {}},
        {"answers": []},
        {"answers": {"eci_relation": "cause"}},
        {"answers": {"eci_relation": None}},
    ],
)
def test_parse_laya_response_missing_or_nonmapping_answer_is_invalid_response(
    raw: dict[str, object],
) -> None:
    parsed = parse_laya_response(raw)

    assert parsed.status is PredictionStatus.INVALID_RESPONSE
    assert parsed.prediction_valid is False
    assert parsed.probability_valid is False
    assert parsed.probabilities is None
    assert isinstance(parsed.error, str)
    assert parsed.error


def test_parse_laya_response_ignores_confidence_and_action_probability() -> None:
    raw = {
        "answers": {
            "eci_relation": {
                "choice": "cause",
                "confidence": 0.99,
                "action": {"act_probability": 0.99},
            }
        }
    }

    parsed = parse_laya_response(raw)

    assert parsed.status is PredictionStatus.MISSING_PROBABILITY
    assert parsed.probabilities is None


class _FakeClient:
    def __init__(self, raw: dict[str, object]) -> None:
        self.raw = raw
        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    def system_one(
        self,
        state: dict[str, object],
        questions: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((copy.deepcopy(state), copy.deepcopy(questions)))
        return self.raw


def test_laya_adapter_calls_client_and_returns_independent_copies() -> None:
    client_raw: Any = {
        "answers": {
            "eci_relation": {
                "choice": "cause",
                "probabilities": {"cause": 0.6, "caused_by": 0.3, "none": 0.1},
            }
        },
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "details": {"total_tokens": 18},
        },
        "model": "fake-laya",
    }
    client = _FakeClient(client_raw)
    request = build_laya_request("sentence", "event1", "event2")

    response = LayaAdapter(client).predict(request)

    assert isinstance(response, LayaResponse)
    assert client.calls == [(request.state, request.questions)]
    assert response.raw == client_raw
    assert type(response.raw) is dict
    assert isinstance(response.validated, ValidatedPrediction)
    assert response.validated.status is PredictionStatus.OK
    assert response.usage == client_raw["usage"]
    assert type(response.usage) is dict
    assert response.raw is not client.raw
    assert response.usage is not client.raw["usage"]

    response_raw: Any = response.raw
    response_usage: Any = response.usage
    response_raw["answers"]["eci_relation"]["choice"] = "none"
    response_raw["usage"]["input_tokens"] = 999
    response_usage["output_tokens"] = 999
    response_usage["details"]["total_tokens"] = 999

    assert client_raw["answers"]["eci_relation"]["choice"] == "cause"
    assert client_raw["usage"]["input_tokens"] == 11
    assert client_raw["usage"]["output_tokens"] == 7
    assert client_raw["usage"]["details"]["total_tokens"] == 18


class _FakeLoader:
    def __init__(self, agent: object) -> None:
        self.agent = agent
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return self.agent


class _FakeAgent:
    def __init__(
        self,
        revision: str = PINNED_REVISION,
        *,
        model_id: str = PINNED_CHECKPOINT,
        device: str = "cpu",
        dtype: str = "float32",
        backend: str = "mps",
        amp_enabled: bool = True,
        mps_amp_min_rows: int = 64,
        max_len: int = 1024,
        head_max_len: int = 256,
        temperature_raw: object = "0.7",
        temperature_by_options_raw: object = None,
        temperature: object = 0.7,
        temperature_by_options: object = None,
    ) -> None:
        self.revision = revision
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.backend = backend
        self.amp_enabled = amp_enabled
        self.mps_amp_min_rows = mps_amp_min_rows
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.temperature_raw = temperature_raw
        self.temperature_by_options_raw = (
            {"cause": "0.7", "caused_by": "0.7", "none": "0.7"}
            if temperature_by_options_raw is None
            else temperature_by_options_raw
        )
        self.temperature = temperature
        self.temperature_by_options = (
            {"cause": 0.7, "caused_by": 0.7, "none": 0.7}
            if temperature_by_options is None
            else temperature_by_options
        )

    def dtype_for(self, batch_size: int) -> str:
        return "float16" if batch_size == 1 else "float32"


def _call_has_value(
    call: tuple[tuple[object, ...], dict[str, object]],
    expected: object,
) -> bool:
    args, kwargs = call
    return expected in args or expected in kwargs.values()


def test_load_laya_agent_uses_pinned_loader_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_laya_version(monkeypatch, "0.4.17")
    settings = _settings_with_download()
    agent = _FakeAgent()
    loader = _FakeLoader(agent)

    loaded = load_laya_agent(settings, loader=loader)

    assert loaded is agent
    assert len(loader.calls) == 1
    call = loader.calls[0]
    assert _call_has_value(call, settings.checkpoint)
    assert _call_has_value(call, settings.revision)
    assert _call_has_value(call, settings.device)
    assert "local_files_only" not in call[1]


def test_load_laya_agent_requires_download_weights_before_calling_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_laya_version(monkeypatch, "0.4.17")
    settings = LayaSettings.from_env({})
    loader = _FakeLoader(_FakeAgent())

    with pytest.raises(Exception) as excinfo:
        load_laya_agent(settings, loader=loader)

    assert loader.calls == []
    message = str(excinfo.value).lower()
    assert "download" in message or "weight" in message


def test_load_laya_agent_rejects_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_laya_version(monkeypatch, "0.4.17")
    settings = _settings_with_download()
    loader = _FakeLoader(_FakeAgent(revision=VALID_REVISION))

    with pytest.raises(Exception) as excinfo:
        load_laya_agent(settings, loader=loader)

    assert str(excinfo.value)


def test_load_laya_agent_rejects_sdk_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_laya_version(monkeypatch, "0.4.16")
    settings = _settings_with_download()
    loader = _FakeLoader(_FakeAgent())

    with pytest.raises(RuntimeError) as excinfo:
        load_laya_agent(settings, loader=loader)

    message = str(excinfo.value).lower()
    assert "sdk" in message or "version" in message
    assert "0.4.16" in message or "0.4.17" in message


def test_laya_runtime_metadata_is_json_serializable_and_honest() -> None:
    agent = _FakeAgent()

    metadata = laya_runtime_metadata(agent)

    assert metadata["revision"] == PINNED_REVISION
    assert metadata["model_id"] == PINNED_CHECKPOINT
    assert metadata["device"] == "cpu"
    assert metadata["dtype"] == "float32"
    assert metadata["backend"] == "mps"
    assert metadata["amp_enabled"] is True
    assert metadata["mps_amp_min_rows"] == 64
    assert metadata["inference_dtype"] == agent.dtype_for(1)
    assert metadata["inference_dtype"] == "float16"
    assert metadata["inference_dtype"] != metadata["dtype"]
    assert metadata["max_len"] == 1024
    assert metadata["head_max_len"] == 256
    assert metadata["temperature_raw"] == "0.7"
    assert metadata["temperature_by_options_raw"] == agent.temperature_by_options_raw
    assert metadata["temperature"] == 0.7
    assert metadata["temperature_by_options"] == agent.temperature_by_options
    assert "confidence" not in metadata
    assert "probabilities" not in metadata

    serialized = json.dumps(metadata, sort_keys=True)
    assert json.loads(serialized) == metadata


class _CfgOnlyAgent:
    def __init__(self, cfg: object) -> None:
        self.cfg = cfg
        self.dtype = "float32"

    def dtype_for(self, batch_size: int) -> str:
        return "float16"


@pytest.mark.parametrize(
    "cfg",
    [
        {"max_len": 1024, "head_max_len": 256},
        SimpleNamespace(max_len=1024, head_max_len=256),
    ],
    ids=["mapping", "object"],
)
def test_laya_runtime_metadata_reads_lengths_from_cfg(cfg: object) -> None:
    agent = _CfgOnlyAgent(cfg)

    metadata = laya_runtime_metadata(agent)

    assert not hasattr(agent, "max_len")
    assert not hasattr(agent, "head_max_len")
    assert metadata["max_len"] == 1024
    assert metadata["head_max_len"] == 256
