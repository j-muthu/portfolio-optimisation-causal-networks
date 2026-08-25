"""Causal-HSP pipeline for the S&P 100 thesis.

Re-exports the legacy asset-only DYNOTEARS/VARLiNGAM API for back-compat.
"""

from __future__ import annotations

from pipeline.closed_loop import ClosedLoopResult, run_closed_loop
from pipeline.data import Dataset, build_dataset
from pipeline.discovery import (
    DynotearsWindow,
    RollingDynotearsResult,
    RollingVarLingamResult,
    VarLingamWindow,
    run_dynotears_window,
    run_rolling_dynotears,
    run_rolling_varlingam,
    run_varlingam_window,
)
from pipeline.discovery.diagnostics import (
    analyse_rolling,
    causal_order_drift,
    compare_rolling,
    detect_regime_changes,
    sector_flow,
)

__all__ = [
    "Dataset",
    "build_dataset",
    "ClosedLoopResult",
    "run_closed_loop",
    "run_rolling_dynotears",
    "run_dynotears_window",
    "run_rolling_varlingam",
    "run_varlingam_window",
    "DynotearsWindow",
    "RollingDynotearsResult",
    "VarLingamWindow",
    "RollingVarLingamResult",
    "analyse_rolling",
    "detect_regime_changes",
    "sector_flow",
    "causal_order_drift",
    "compare_rolling",
]
