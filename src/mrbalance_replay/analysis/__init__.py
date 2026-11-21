"""Offline analyses over stored experiment outputs."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .label_modes import ModeComparison, compare_label_modes, format_comparison

__all__ = ["ModeComparison", "compare_label_modes", "format_comparison"]


def __getattr__(name: str) -> object:
    """Import lazily so ``python -m ...label_modes`` does not double-load."""

    if name in __all__:
        from . import label_modes

        return getattr(label_modes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
