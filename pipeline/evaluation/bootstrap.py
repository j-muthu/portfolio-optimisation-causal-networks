"""Politis-Romano stationary block bootstrap for Sharpe-difference CIs.

Standard bootstrap (iid resampling) destroys the temporal autocorrelation in
return series and underestimates the variance of statistics like the Sharpe
ratio. The stationary block bootstrap (Politis & Romano 1994) preserves the
local time-series structure by resampling *blocks* of geometrically-distributed
length, then computes the statistic on each resample.

Used in ``Closed-Loop Causal-HSP Portfolio.md`` §Metrics to put a 95 % CI on
``ΔSharpe`` (strategy vs V0 baseline) and test the null ``ΔSharpe = 0`` at the
5 % level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.evaluation.metrics import annualised_sharpe

logger = logging.getLogger(__name__)


# ============================================================================
# Block bootstrap
# ============================================================================
def stationary_block_indices(
    n: int, mean_block_length: float, rng: np.random.Generator,
) -> np.ndarray:
    """Generate ``n`` indices via Politis-Romano stationary bootstrap.

    Each block starts at a uniformly random index and has geometrically
    distributed length with mean ``mean_block_length``. Blocks wrap around
    the end of the series so the resample is exactly length ``n``.
    """
    if mean_block_length <= 1.0:
        return rng.integers(0, n, size=n)
    p = 1.0 / mean_block_length
    indices = np.empty(n, dtype=int)
    i = 0
    while i < n:
        block_start = int(rng.integers(0, n))
        block_len = 1 + int(rng.geometric(p))
        block_len = min(block_len, n - i)
        for k in range(block_len):
            indices[i + k] = (block_start + k) % n
        i += block_len
    return indices


def bootstrap_statistic(
    series: pd.Series,
    statistic,
    n_resamples: int = 1000,
    mean_block_length: float = 21.0,
    seed: int = 42,
) -> np.ndarray:
    """Resample ``series`` via stationary block bootstrap; apply ``statistic``.

    Returns the array of ``n_resamples`` bootstrap statistic values. Default
    block length 21 ≈ one trading month, typical for monthly-cycle equity
    return analyses.
    """
    rng = np.random.default_rng(seed)
    arr = series.dropna().to_numpy()
    n = len(arr)
    out = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = stationary_block_indices(n, mean_block_length, rng)
        out[b] = statistic(pd.Series(arr[idx]))
    return out


# ============================================================================
# Sharpe-difference CI
# ============================================================================
@dataclass
class SharpeDiffCI:
    """Bootstrap-derived ``ΔSharpe = Sharpe(A) - Sharpe(B)`` and its 95 % CI."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    p_value_two_sided: float
    n_resamples: int


def sharpe_difference_ci(
    returns_a: pd.Series,
    returns_b: pd.Series,
    n_resamples: int = 1000,
    mean_block_length: float = 21.0,
    confidence: float = 0.95,
    seed: int = 42,
    periods_per_year: int = 252,
) -> SharpeDiffCI:
    """95 % stationary-block-bootstrap CI on ``Sharpe(a) - Sharpe(b)``.

    The two return series must share the same index; missing values are
    dropped pairwise. The bootstrap resamples the *joint* (a, b) panel so
    correlations between the strategies are preserved.
    """
    df = pd.concat([returns_a.rename("a"), returns_b.rename("b")], axis=1).dropna()
    arr = df.to_numpy()
    n = len(arr)
    rng = np.random.default_rng(seed)

    def diff(panel):
        return annualised_sharpe(pd.Series(panel[:, 0]), periods_per_year) \
            - annualised_sharpe(pd.Series(panel[:, 1]), periods_per_year)

    point = diff(arr)
    diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = stationary_block_indices(n, mean_block_length, rng)
        diffs[b] = diff(arr[idx])
    alpha = (1 - confidence) / 2
    lo, hi = float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha))
    # Two-sided p-value: fraction of bootstrap diffs as extreme as 0 in the
    # opposite direction of the point estimate.
    if point >= 0:
        p = float(np.mean(diffs <= 0)) * 2
    else:
        p = float(np.mean(diffs >= 0)) * 2
    p = min(p, 1.0)
    return SharpeDiffCI(
        point_estimate=point, ci_lower=lo, ci_upper=hi,
        p_value_two_sided=p, n_resamples=n_resamples,
    )


__all__ = [
    "stationary_block_indices",
    "bootstrap_statistic",
    "SharpeDiffCI",
    "sharpe_difference_ci",
]
