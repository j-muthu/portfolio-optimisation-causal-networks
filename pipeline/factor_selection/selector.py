"""Top-level selector: combines Stage A pruning, Stage B greedy refinement, and
the closed-loop driver-utility blend.

This is the function Stage 1's pipeline calls once per rebalance date:

    selected_drivers[t] = select(window_t, U_lookup=..., K=..., α=..., γ=...)

The α-blend (per ``Closed-Loop Causal-HSP Portfolio.md:124``):

    score_d[t] = α · z(causal_score_d[t]) + (1-α) · z(U_d[t-1])

where ``z(·)`` is a z-score normalisation across the candidate pool so that
the causal and utility scales are commensurate. The blended score replaces
the raw Stage A score for the purpose of pruning to the pool that Stage B
then refines on.

Burn-in: for the first ``burn_in_rebalances`` rebalances the selector forces
``α=1`` (no feedback). After that it uses the configured ``α``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from pipeline.factor_selection.greedy import StageBResult, greedy_select
from pipeline.factor_selection.prune import StageAResult, prune_to_pool, stage_a_score

logger = logging.getLogger(__name__)


# ============================================================================
# Helper: z-score across a pool
# ============================================================================
def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma < 1e-12:
        return pd.Series(np.zeros_like(series.values), index=series.index, name=series.name)
    return (series - mu) / sigma


# ============================================================================
# Top-level selector
# ============================================================================
@dataclass
class SelectionResult:
    """What Stage 1 persists per rebalance date.

    Attributes
    ----------
    rebalance_date:
        The rebalance timestamp ``t``.
    selected:
        Ordered list of selected driver names (length ``≤ K``; order is the
        Stage B addition order).
    pool:
        Stage A pool that Stage B refined on.
    K:
        Target selected count (may exceed ``len(selected)`` if Stage B
        stopped early on ε or pool exhaustion).
    stage_a:
        Full :class:`StageAResult` for diagnostics.
    stage_b:
        Full :class:`StageBResult` for diagnostics.
    blended_scores:
        The α·causal + (1-α)·utility scores used at pool-construction time.
    alpha_effective:
        The actual α applied (forced to 1.0 during burn-in).
    utility_lookup_timestamp:
        The timestamp of the utility row consulted (may be ``None`` during
        burn-in). Forms part of the lookahead audit trail.
    """

    rebalance_date: pd.Timestamp
    selected: list[str]
    pool: list[str]
    K: int
    stage_a: StageAResult
    stage_b: StageBResult
    blended_scores: pd.Series
    alpha_effective: float
    utility_lookup_timestamp: pd.Timestamp | None = None
    metadata: dict = field(default_factory=dict)


def select_drivers(
    rebalance_date: pd.Timestamp | str,
    discovery_window: Any,
    driver_window: pd.DataFrame,
    asset_window: pd.DataFrame,
    K: int,
    alpha: float = 0.6,
    utility_lookup: Callable[[pd.Timestamp], tuple[pd.Series, pd.Timestamp | None]]
        | None = None,
    rebalance_index: int = 0,
    burn_in_rebalances: int = 6,
    method: str = "dynotears",
    target_fraction: float = 0.10,
    pool_multiplier: int = 2,
    lags: int = 1,
    epsilon: float | None = None,
    ridge_alpha: float = 1.0,
) -> SelectionResult:
    """End-to-end selection for one rebalance date.

    Parameters
    ----------
    rebalance_date:
        Timestamp at which selection is happening. Used as the lookahead
        reference when reading utility state.
    discovery_window:
        Per-window discovery output (e.g. ``JointDynotearsWindow``).
    driver_window, asset_window:
        Per-window driver/asset frames (already z-scored — Stage B's
        auxiliary model expects normalised inputs).
    K:
        Selected count target.
    alpha:
        Mix between causal evidence (α=1) and historical utility (α=0).
        Sensitivity sweep in the closed-loop plan: {0.4, 0.6, 0.8, 1.0}.
        Forced to 1.0 during burn-in.
    utility_lookup:
        Optional callable ``t -> (U_series, lookup_timestamp)`` from
        :mod:`pipeline.feedback.storage`. Returns the driver utility vector
        valid at ``t`` (i.e. derived only from realised periods strictly
        before ``t``). If ``None`` (or burn-in), pure causal selection.
    rebalance_index:
        0-based index of this rebalance in the backtest. Used to force
        α=1 during the first ``burn_in_rebalances`` rebalances.
    burn_in_rebalances:
        Default 6 (≈ 6 months of monthly rebalances).
    method, target_fraction, pool_multiplier:
        Pass-through to Stage A.
    lags, epsilon, ridge_alpha:
        Pass-through to Stage B.
    """
    t = pd.Timestamp(rebalance_date)
    # Stage A score on the causal evidence alone.
    stage_a = stage_a_score(discovery_window, method=method, target_fraction=target_fraction)
    causal_scores = stage_a.scores

    # Utility lookup (skip during burn-in).
    burn_in = rebalance_index < burn_in_rebalances
    alpha_eff = 1.0 if burn_in else float(alpha)
    utility_series: pd.Series = pd.Series(0.0, index=causal_scores.index)
    lookup_ts: pd.Timestamp | None = None
    if not burn_in and utility_lookup is not None:
        utility_series, lookup_ts = utility_lookup(t)
        # Re-index to the candidate set; missing drivers get U = 0.
        utility_series = utility_series.reindex(causal_scores.index).fillna(0.0)

    # Blend.
    if alpha_eff >= 1.0:
        blended = causal_scores.copy()
    else:
        z_causal = _zscore(causal_scores)
        z_util = _zscore(utility_series)
        blended = alpha_eff * z_causal + (1.0 - alpha_eff) * z_util
        blended.name = "blended_score"

    # Pool: top-2K by blended score, restricted to drivers with strictly
    # positive causal evidence (we never select a driver that has *no*
    # causal signal at this window, even if its utility is high).
    nonzero = causal_scores[causal_scores > 0].index
    blended_pool_sorted = blended.loc[nonzero].sort_values(ascending=False)
    target_pool = pool_multiplier * K
    pool = blended_pool_sorted.head(target_pool).index.tolist()

    # Stage B refinement on the pool.
    stage_b = greedy_select(
        driver_window=driver_window,
        asset_window=asset_window,
        pool=pool,
        K=K,
        lags=lags,
        epsilon=epsilon,
        alpha=ridge_alpha,
    )

    return SelectionResult(
        rebalance_date=t,
        selected=stage_b.selected,
        pool=pool,
        K=K,
        stage_a=stage_a,
        stage_b=stage_b,
        blended_scores=blended,
        alpha_effective=alpha_eff,
        utility_lookup_timestamp=lookup_ts,
        metadata={
            "rebalance_index": rebalance_index,
            "burn_in_active": burn_in,
            "method": method,
            "alpha_configured": alpha,
            "lags": lags,
            "epsilon": epsilon,
        },
    )


__all__ = ["SelectionResult", "select_drivers"]
