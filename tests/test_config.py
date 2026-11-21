"""Tests for environment-backed settings."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mrbalance_replay.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE,
    SettingsError,
    load_settings,
)

_SETTING_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "MRBALANCE_MODEL",
    "MRBALANCE_REASONING_EFFORT",
)


def _clear_setting_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SETTING_VARS:
        monkeypatch.delenv(name, raising=False)


def test_missing_api_key_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_setting_vars(monkeypatch)

    with pytest.raises(SettingsError, match="OPENAI_API_KEY"):
        load_settings(tmp_path / "missing.env")


def test_defaults_are_applied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_setting_vars(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = load_settings(tmp_path / "missing.env")

    assert settings.openai_api_key == "test-key"
    assert settings.openai_base_url == DEFAULT_BASE_URL
    assert settings.model == DEFAULT_MODEL
    assert settings.reasoning_effort == DEFAULT_REASONING_EFFORT
    assert settings.temperature == DEFAULT_TEMPERATURE
    assert settings.max_tokens == DEFAULT_MAX_TOKENS
    assert settings.max_retries == DEFAULT_MAX_RETRIES


def test_environment_overrides_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=from-dotenv\nMRBALANCE_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    monkeypatch.setenv("MRBALANCE_MODEL", "environment-model")

    settings = load_settings(env_file)

    assert settings.openai_api_key == "from-environment"
    assert settings.model == "environment-model"


def test_dotenv_seeds_missing_variables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_setting_vars(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    previous = os.environ.pop("OPENAI_API_KEY", None)
    try:
        settings = load_settings(env_file)
        assert settings.openai_api_key == "from-dotenv"
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        if previous is not None:
            os.environ["OPENAI_API_KEY"] = previous


def test_custom_endpoint_model_and_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_setting_vars(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MRBALANCE_MODEL", "custom-model")
    monkeypatch.setenv("MRBALANCE_REASONING_EFFORT", "high")

    settings = load_settings(tmp_path / "missing.env")

    assert settings.openai_base_url == "https://example.test/v1"
    assert settings.model == "custom-model"
    assert settings.reasoning_effort == "high"
