"""Runtime configuration for the MRBalance replay pipeline.

Credentials are read from environment variables.  ``load_settings`` may also
populate missing variables from a ``.env`` file, but an already-exported
environment variable always wins.  Importing this module never performs
network I/O and never constructs an API client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_RETRIES = 20


class SettingsError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Resolved configuration shared by the pipeline and the LLM client."""

    openai_api_key: str
    openai_base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_retries: int = DEFAULT_MAX_RETRIES


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load settings from the environment, optionally seeding from ``.env``.

    ``python-dotenv`` is called with ``override=False`` so values that are
    already present in the process environment take precedence over the file.
    A missing ``OPENAI_API_KEY`` raises :class:`SettingsError` with actionable
    guidance instead of failing later during the first API request.
    """

    load_dotenv(dotenv_path=env_file, override=False)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SettingsError(
            "OPENAI_API_KEY is not set. Export it in the environment or add it to "
            "your .env file (see .env.example)."
        )

    return Settings(
        openai_api_key=api_key,
        openai_base_url=_env_or_default("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        model=_env_or_default("MRBALANCE_MODEL", DEFAULT_MODEL),
        reasoning_effort=_env_or_default("MRBALANCE_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
    )
