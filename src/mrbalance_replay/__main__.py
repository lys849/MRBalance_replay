"""Module entry point: ``python -m mrbalance_replay`` runs the replay pipeline."""

from __future__ import annotations

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
