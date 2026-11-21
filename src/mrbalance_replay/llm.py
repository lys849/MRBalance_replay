"""OpenAI-compatible LLM client used by the replay pipeline.

The official implementation decorates ``generate_answer`` with
``backoff.on_exception(..., Exception, max_tries=20)``.  That retries every
failure, including authentication and invalid-request errors, twenty times.
This module intentionally deviates from that behaviour:

* Only transient failures (rate limits, timeouts, connection errors, server
  errors and malformed JSON payloads) are retried with exponential backoff.
* Fatal client errors (authentication, permission and bad-request errors) are
  raised immediately.

The per-request parameters (``temperature=0.2``, ``max_tokens=1024``, ``n=1``)
match the official client.  ``reasoning_effort`` is a DeepSeek V4 extension; if
the installed SDK or the configured endpoint rejects it, the parameter is
dropped, the downgrade is cached, and the request is retried once.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Protocol

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

_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    JSONDecodeError,
)

_FATAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
)


@dataclass(frozen=True)
class GenerationResult:
    """Normalized response from a chat-completion request."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    """Minimal interface required by the debate pipeline."""

    def generate(
        self, messages: Sequence[dict[str, str]], model: str | None = None
    ) -> GenerationResult: ...


def _reasoning_effort_rejected(exc: Exception) -> bool:
    """Return whether an error looks like a rejected ``reasoning_effort``."""

    text = str(exc).lower()
    return (
        "reasoning_effort" in text
        or "reasoning effort" in text
        or "unknown parameter" in text
        or "unrecognized request argument" in text
    )


class OpenAIClient:
    """Thin wrapper around the OpenAI Python client.

    Parameters
    ----------
    settings:
        Resolved runtime configuration.
    client:
        Optional pre-built OpenAI client (or a test double).  Injecting a client
        avoids importing or constructing the real SDK client.
    """

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._reasoning_effort_unsupported = False
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        self._client: Any = client

    def generate(
        self, messages: Sequence[dict[str, str]], model: str | None = None
    ) -> GenerationResult:
        """Generate one completion, retrying transient failures only."""

        response = self._request_with_retries(messages, model or self._settings.model)
        return self._to_result(response)

    def _request_with_retries(self, messages: Sequence[dict[str, str]], model: str) -> Any:
        attempts = max(1, self._settings.max_retries)
        for attempt in range(attempts):
            try:
                return self._create(messages, model)
            except _FATAL_EXCEPTIONS:
                raise
            except _RETRYABLE_EXCEPTIONS:
                if attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable retry loop exit")  # pragma: no cover

    def _create(self, messages: Sequence[dict[str, str]], model: str) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "n": 1,
        }
        effort = self._settings.reasoning_effort
        if not effort or self._reasoning_effort_unsupported:
            return self._client.chat.completions.create(**kwargs)

        kwargs["reasoning_effort"] = effort
        try:
            return self._client.chat.completions.create(**kwargs)
        except TypeError as exc:
            # Some SDK versions do not accept the keyword at all.
            if "reasoning_effort" not in str(exc).lower():
                raise
            return self._retry_without_reasoning_effort(kwargs)
        except BadRequestError as exc:
            # Some endpoints accept the keyword at the HTTP layer but reject it.
            if not _reasoning_effort_rejected(exc):
                raise
            return self._retry_without_reasoning_effort(kwargs)

    def _retry_without_reasoning_effort(self, kwargs: dict[str, Any]) -> Any:
        self._reasoning_effort_unsupported = True
        kwargs.pop("reasoning_effort", None)
        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _to_result(response: Any) -> GenerationResult:
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return GenerationResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
