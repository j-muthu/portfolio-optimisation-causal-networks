"""Cumulative-correlation driver selection (Rodriguez-Dominguez 2023), which
is the V0 baseline. Each driver is scored by its summed absolute correlation
with the asset block over the given lags, and I take the top K. No causal
inference is involved. I ported this from the HSP notebook's
cumulative-correlation block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Score
def cumulative_correlation_score(
    driver_window: pd.DataFrame,
    asset_window: pd.DataFrame,
    lags: Sequence[int] = (0, 1),
) -> pd.Series:
    """Return the sum of |corr(driver_{t-lag}, asset_t)| over lags and assets
    for each driver.

    ``(0, 1)`` is the paper's default lag set. The result is a Series
    indexed by driver name. Pairs with too little overlap contribute zero.
    """
    scores: dict[str, float] = {}
    for d in driver_window.columns:
        total = 0.0
        for lag in lags:
            if lag == 0:
                d_series = driver_window[d]
            else:
                d_series = driver_window[d].shift(lag)
            for a in asset_window.columns:
                common = d_series.dropna().index.intersection(
                    asset_window[a].dropna().index
                )
                if len(common) < 10:
                    continue
                rho = d_series.loc[common].corr(asset_window[a].loc[common])
                if not np.isnan(rho):
                    total += abs(float(rho))
        scores[d] = total
    return pd.Series(scores, name="cumcorr_score")


# Top-K selection
@dataclass
class CorrelationSelectionResult:
    """Output of :func:`select_top_k_corr`. It matches
    :class:`SelectionResult` closely enough for downstream code."""

    rebalance_date: pd.Timestamp
    selected: list[str]
    scores: pd.Series  # all candidates with their score
    K: int
    lags: tuple[int, ...]

    @property
    def stage_b(self):  # matches the SelectionResult interface
        return None


def select_top_k_corr(
    driver_window: pd.DataFrame,
    asset_window: pd.DataFrame,
    K: int,
    rebalance_date: pd.Timestamp | str | None = None,
    lags: Sequence[int] = (0, 1),
) -> CorrelationSelectionResult:
    """Return the top ``K`` drivers by cumulative correlation (the V0
    selector).

    The signature matches :func:`select_drivers` so that Stage 1 can call
    either selector in the same way.
    """
    scores = cumulative_correlation_score(driver_window, asset_window, lags=lags)
    sorted_desc = scores.sort_values(ascending=False)
    selected = sorted_desc.head(K).index.tolist()
    rdate = pd.Timestamp(rebalance_date) if rebalance_date is not None \
        else pd.Timestamp(driver_window.index[-1])
    logger.info(
        "cum-corr select [t=%s, K=%d, lags=%s]: %s",
        rdate.date(), K, tuple(lags), selected,
    )
    return CorrelationSelectionResult(
        rebalance_date=rdate,
        selected=selected,
        scores=scores,
        K=K,
        lags=tuple(lags),
    )


__all__ = [
    "cumulative_correlation_score",
    "CorrelationSelectionResult",
    "select_top_k_corr",
]
