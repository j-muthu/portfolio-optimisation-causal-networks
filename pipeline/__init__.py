"""Top-level package for the thesis pipeline."""

from __future__ import annotations

from pipeline.closed_loop import ClosedLoopResult, run_closed_loop

__all__ = ["ClosedLoopResult", "run_closed_loop"]
