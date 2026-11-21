"""Smoke tests for the project skeleton."""

import mrbalance_replay


def test_package_exposes_version() -> None:
    """The package should expose a non-empty version string."""
    assert isinstance(mrbalance_replay.__version__, str)
    assert mrbalance_replay.__version__
