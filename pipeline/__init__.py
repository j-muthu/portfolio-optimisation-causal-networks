"""Causal-HSP pipeline for the S&P 100 thesis.

Top-level structure (see ``Causal Factor Discovery Pipeline.md`` and
``Closed-Loop Causal-HSP Portfolio.md`` for the architectural plan):

* :mod:`pipeline.data` -- universe, asset prices, driver pool, alignment
* :mod:`pipeline.discovery` -- rolling DYNOTEARS / VARLiNGAM with prior-knowledge masks
* :mod:`pipeline.factor_selection` -- greedy driver selection from causal output
* :mod:`pipeline.sensitivities` -- per-asset FFNN + AAD sensitivity vectors
* :mod:`pipeline.portfolio` -- HRP / HSP / Causal-HSP variants and benchmarks
* :mod:`pipeline.feedback` -- utility EMA + lookahead-safe storage for V2
* :mod:`pipeline.evaluation` -- metrics, regime breakdowns, bootstrap CIs

The legacy asset-only DYNOTEARS/VARLiNGAM API (``Dataset``, ``build_dataset``,
``run_rolling_dynotears``, ``run_rolling_varlingam``) is re-exported here for
back-compat while the new Stage 1 / Stage 2 modules come online.
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
