"""Causal-discovery wrappers.

* ``dynotears`` -- rolling DYNOTEARS with native ``tabu_edges`` enforcement of
  the asset → driver mask.
* ``varlingam`` -- rolling VARLiNGAM with optional ridge/LASSO Stage 1, native
  ``prior_knowledge`` for B₀ and post-fit projection for the lagged blocks.
* ``diagnostics`` -- per-window stationarity flags, fit-quality summaries, and
  network-density time series (reused by ``evaluation.regime``).
"""

from __future__ import annotations

from pipeline.discovery.dynotears import (
    DynotearsWindow,
    RollingDynotearsResult,
    run_dynotears_window,
    run_rolling_dynotears,
)
from pipeline.discovery.varlingam import (
    RollingVarLingamResult,
    VarLingamWindow,
    run_rolling_varlingam,
    run_varlingam_window,
)

__all__ = [
    "DynotearsWindow",
    "RollingDynotearsResult",
    "run_dynotears_window",
    "run_rolling_dynotears",
    "VarLingamWindow",
    "RollingVarLingamResult",
    "run_varlingam_window",
    "run_rolling_varlingam",
]
