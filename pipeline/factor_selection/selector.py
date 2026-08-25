"""Top-level selector, called once per rebalance date: Stage A pruning,
Stage B greedy refinement, and the closed-loop utility blend.

Blend: score = alpha * z(causal) + (1 - alpha) * z(utility), z-scored across
the pool so the scales are commensurate. During burn-in alpha is forced to 1.
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


# Helper: z-score across a pool
def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma < 1e-12:
        return pd.Series(np.zeros_like(series.values), index=series.index, name=series.name)
    return (series - mu) / sigma


# Top-level selector
@dataclass
class SelectionResult:
    """What Stage 1 persists per rebalance date.

    ``selected`` is in Stage B addition order and may be shorter than ``K``
    if Stage B stopped early. ``alpha_effective`` is the alpha actually
    applied (1.0 during burn-in); ``utility_lookup_timestamp`` is part of
    the lookahead audit trail.
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

    ``driver_window`` / ``asset_window`` must already be z-scored. ``alpha``
    mixes causal evidence (1) with historical utility (0) and is forced to 1
    for the first ``burn_in_rebalances`` rebalances. ``utility_lookup`` maps
    ``t`` to ``(U_series, lookup_timestamp)`` derived only from periods
    strictly before ``t``; ``None`` means pure causal selection.
    """
    t = pd.Timestamp(rebalance_date)
    stage_a = stage_a_score(discovery_window, method=method, target_fraction=target_fraction)
    causal_scores = stage_a.scores

    # Utility lookup (skip during burn-in).
    burn_in = rebalance_index < burn_in_rebalances
    alpha_eff = 1.0 if burn_in else float(alpha)
    utility_series: pd.Series = pd.Series(0.0, index=causal_scores.index)
    lookup_ts: pd.Timestamp | None = None
    if not burn_in and utility_lookup is not None:
        utility_series, lookup_ts = utility_lookup(t)
        # Missing drivers get U = 0.
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
    # positive causal evidence; high utility alone never selects a driver.
    nonzero = causal_scores[causal_scores > 0].index
    blended_pool_sorted = blended.loc[nonzero].sort_values(ascending=False)
    target_pool = pool_multiplier * K
    pool = blended_pool_sorted.head(target_pool).index.tolist()

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
