"""Evaluation helpers and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mrbalance_replay.evaluation.metrics import (
        Metrics,
        MetricsReport,
        compute_metrics,
        format_report,
    )

__all__ = ["Metrics", "MetricsReport", "compute_metrics", "format_report"]


def __getattr__(name: str) -> object:
    """Import metrics lazily so ``python -m`` does not double-import it."""
    if name in __all__:
        from mrbalance_replay.evaluation import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
