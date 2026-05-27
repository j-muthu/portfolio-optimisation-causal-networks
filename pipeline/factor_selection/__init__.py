"""Stage 1 driver-selection.

* :mod:`pipeline.factor_selection.prune` -- Stage A: pool-down by outgoing
  causal-edge score.
* :mod:`pipeline.factor_selection.k_calibration` -- one-off Kneedle +
  permutation-null K selection on the burn-in window.
* :mod:`pipeline.factor_selection.greedy` -- Stage B: conditional greedy
  refinement on the pool.
* :mod:`pipeline.factor_selection.selector` -- top-level α-blend of causal
  evidence and lagged driver utility, then Stage A + Stage B.
"""

from __future__ import annotations

from pipeline.factor_selection.correlation_selector import (
    CorrelationSelectionResult,
    cumulative_correlation_score,
    select_top_k_corr,
)
from pipeline.factor_selection.greedy import StageBResult, greedy_select
from pipeline.factor_selection.k_calibration import KCalibration, calibrate_K, kneedle
from pipeline.factor_selection.prune import StageAResult, prune_to_pool, stage_a_score
from pipeline.factor_selection.selector import SelectionResult, select_drivers

__all__ = [
    "StageAResult",
    "stage_a_score",
    "prune_to_pool",
    "KCalibration",
    "calibrate_K",
    "kneedle",
    "StageBResult",
    "greedy_select",
    "SelectionResult",
    "select_drivers",
    # V0 baseline (cumulative-correlation selection)
    "CorrelationSelectionResult",
    "cumulative_correlation_score",
    "select_top_k_corr",
]
