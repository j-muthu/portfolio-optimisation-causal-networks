"""Politis-Romano (1994) stationary block bootstrap for confidence intervals
on Sharpe differences.

I use block resampling because iid resampling destroys the autocorrelation
in returns and underestimates the variance of the Sharpe ratio. Block
resampling keeps the local time-series structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.evaluation.metrics import annualised_sharpe

logger = logging.getLogger(__name__)


# Block bootstrap
def stationary_block_indices(
    n: int, mean_block_length: float, rng: np.random.Generator,
) -> np.ndarray:
    """Return n resample indices. Blocks start uniformly at random, have a
    geometric length with the given mean and wrap around the end of the
    series."""
    if mean_block_length <= 1.0:
        return rng.integers(0, n, size=n)
    p = 1.0 / mean_block_length
    indices = np.empty(n, dtype=int)
    i = 0
    while i < n:
        block_start = int(rng.integers(0, n))
        block_len = int(rng.geometric(p))
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
    """Return the statistic over n_resamples block resamples. The default
    block length of 21 is roughly 1 trading month."""
    rng = np.random.default_rng(seed)
    arr = series.dropna().to_numpy()
    n = len(arr)
    out = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = stationary_block_indices(n, mean_block_length, rng)
        out[b] = statistic(pd.Series(arr[idx]))
    return out


# Sharpe-difference CI
@dataclass
class SharpeDiffCI:
    """Bootstrap result for Sharpe(A) - Sharpe(B) with its confidence
    interval and two-sided p-value."""

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
    """Return a stationary block bootstrap confidence interval on
    Sharpe(a) - Sharpe(b). I resample the joint (a, b) panel so that the
    correlation between the 2 strategies is preserved. Missing values are
    dropped pairwise."""
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
    # The two-sided p-value is the fraction of bootstrap differences on the
    # other side of 0 from the point estimate.
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
