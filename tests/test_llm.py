"""Tests for the OpenAI-compatible client using injected fakes only."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, cast

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from mrbalance_replay.config import Settings
from mrbalance_replay.llm import GenerationResult, LLMClient, OpenAIClient

_BASE_SETTINGS = Settings(openai_api_key="test-key", max_retries=3)


def _settings(**changes: Any) -> Settings:
    return replace(_BASE_SETTINGS, **changes)


@dataclass
class _FakeMessage:
    content: str | None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage | None = None


def _response(content: str | None = "answer", usage: _FakeUsage | None = None) -> _FakeResponse:
    return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=content))], usage=usage)


class _FakeCompletions:
    def __init__(self, outcomes: list[_FakeResponse | BaseException]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._outcomes = outcomes

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, *outcomes: _FakeResponse | BaseException) -> None:
        self.completions = _FakeCompletions(list(outcomes))
        self.chat = _FakeChat(self.completions)


_REQUEST = httpx.Request("POST", "https://example.test/v1/chat/completions")


def _status_error(cls: Any, status: int, message: str) -> BaseException:
    response = httpx.Response(status, request=_REQUEST)
    return cast(BaseException, cls(message, response=response, body=None))


def _rate_limit() -> BaseException:
    return _status_error(RateLimitError, 429, "rate limited")


def _internal_server_error() -> BaseException:
    return _status_error(InternalServerError, 500, "server error")


def _timeout() -> BaseException:
    return APITimeoutError(request=cast(Any, _REQUEST))


def _connection_error() -> BaseException:
    return APIConnectionError(request=cast(Any, _REQUEST))


def _authentication_error() -> BaseException:
    return _status_error(AuthenticationError, 401, "bad key")


def _permission_denied() -> BaseException:
    return _status_error(PermissionDeniedError, 403, "forbidden")


def _bad_request() -> BaseException:
    return _status_error(BadRequestError, 400, "invalid request")


def test_client_satisfies_protocol() -> None:
    client: LLMClient = OpenAIClient(_settings(), _FakeClient(_response()))
    assert isinstance(client, OpenAIClient)


def test_request_parameters_match_official_defaults() -> None:
    fake = _FakeClient(_response())
    client = OpenAIClient(_settings(), fake)
    messages = [{"role": "user", "content": "hello"}]

    result = client.generate(messages)

    assert result == GenerationResult(content="answer")
    assert len(fake.completions.calls) == 1
    call = fake.completions.calls[0]
    assert call["messages"] == messages
    assert call["model"] == "deepseek-flash"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 1024
    assert call["n"] == 1
    assert call["reasoning_effort"] == "none"


def test_model_override_is_forwarded() -> None:
    fake = _FakeClient(_response())
    client = OpenAIClient(_settings(), fake)

    client.generate([{"role": "user", "content": "hello"}], model="other-model")

    assert fake.completions.calls[0]["model"] == "other-model"


def test_empty_reasoning_effort_is_omitted() -> None:
    fake = _FakeClient(_response())
    client = OpenAIClient(_settings(reasoning_effort=""), fake)

    client.generate([{"role": "user", "content": "hello"}])

    assert "reasoning_effort" not in fake.completions.calls[0]


def test_usage_is_mapped_and_missing_usage_defaults_to_zero() -> None:
    fake = _FakeClient(_response(usage=_FakeUsage(prompt_tokens=7, completion_tokens=3)))
    client = OpenAIClient(_settings(), fake)

    result = client.generate([{"role": "user", "content": "hello"}])

    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3

    fake_without_usage = _FakeClient(_response(content=None))
    result_without_usage = OpenAIClient(_settings(), fake_without_usage).generate(
        [{"role": "user", "content": "hello"}]
    )
    assert result_without_usage == GenerationResult(content="")


@pytest.mark.parametrize(
    "error_factory",
    [_rate_limit, _internal_server_error, _timeout, _connection_error],
)
def test_transient_errors_are_retried(monkeypatch: pytest.MonkeyPatch, error_factory: Any) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    fake = _FakeClient(error_factory(), _response("recovered"))
    client = OpenAIClient(_settings(max_retries=3), fake)

    result = client.generate([{"role": "user", "content": "hello"}])

    assert result.content == "recovered"
    assert len(fake.completions.calls) == 2


@pytest.mark.parametrize(
    "error_factory",
    [_authentication_error, _permission_denied, _bad_request],
)
def test_fatal_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch, error_factory: Any) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    error = error_factory()
    fake = _FakeClient(error, _response())
    client = OpenAIClient(_settings(max_retries=3), fake)

    with pytest.raises(type(error)) as exc_info:
        client.generate([{"role": "user", "content": "hello"}])

    assert exc_info.value is error
    assert len(fake.completions.calls) == 1


def test_retry_exhaustion_raises_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    fake = _FakeClient(_rate_limit(), _rate_limit(), _rate_limit(), _rate_limit())
    client = OpenAIClient(_settings(max_retries=2), fake)

    with pytest.raises(RateLimitError):
        client.generate([{"role": "user", "content": "hello"}])

    assert len(fake.completions.calls) == 2


def test_type_error_downgrades_reasoning_effort_and_caches_it() -> None:
    error = TypeError("Completions.create() got an unexpected keyword argument 'reasoning_effort'")
    fake = _FakeClient(error, _response("first"), _response("second"))
    client = OpenAIClient(_settings(), fake)

    assert client.generate([{"role": "user", "content": "hello"}]).content == "first"
    assert len(fake.completions.calls) == 2
    assert "reasoning_effort" in fake.completions.calls[0]
    assert "reasoning_effort" not in fake.completions.calls[1]

    assert client.generate([{"role": "user", "content": "hello"}]).content == "second"
    assert len(fake.completions.calls) == 3
    assert "reasoning_effort" not in fake.completions.calls[2]


def test_bad_request_downgrades_reasoning_effort_and_caches_it() -> None:
    error = _status_error(
        BadRequestError,
        400,
        "Unrecognized request argument supplied: reasoning_effort",
    )
    fake = _FakeClient(error, _response("first"), _response("second"))
    client = OpenAIClient(_settings(), fake)

    assert client.generate([{"role": "user", "content": "hello"}]).content == "first"
    assert "reasoning_effort" in fake.completions.calls[0]
    assert "reasoning_effort" not in fake.completions.calls[1]

    assert client.generate([{"role": "user", "content": "hello"}]).content == "second"
    assert "reasoning_effort" not in fake.completions.calls[2]
