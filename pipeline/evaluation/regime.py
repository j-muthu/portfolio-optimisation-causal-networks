"""Regime classification for conditional-performance analysis: NBER
recessions (FRED USREC), VIX quintiles, and causal-network density quintiles.
Each returns boolean masks aligned to the daily return index.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# NBER recessions
def nber_recession_dates(daily_index: pd.DatetimeIndex) -> pd.Series:
    """True on trading days inside an NBER recession (FRED USREC, monthly,
    forward-filled to trading days)."""
    from pipeline.data.drivers import fetch_fred_series

    usrec = fetch_fred_series("USREC")
    daily = usrec.reindex(daily_index, method="ffill").fillna(0).astype(int)
    return (daily == 1).rename("nber_recession")


# VIX quintile vol regimes
def vix_regime_masks(
    vix: pd.Series, quantile_low: float = 0.20, quantile_high: float = 0.80,
) -> dict[str, pd.Series]:
    """Top vs bottom VIX-quintile masks; the middle 60% is in neither.
    Thresholds use the full sample, which is fine post-hoc but would be
    lookahead if used for trading decisions."""
    vix = vix.dropna()
    lo = vix.quantile(quantile_low)
    hi = vix.quantile(quantile_high)
    return {
        "low_vol": (vix <= lo).rename("low_vol"),
        "high_vol": (vix >= hi).rename("high_vol"),
    }


# Causal-network density regimes
def network_density_regimes(
    density_series: pd.Series,
    quantile_low: float = 0.20,
    quantile_high: float = 0.80,
) -> dict[str, pd.Series]:
    """Quintile masks on a network-density series. Reindex per-window density
    to the daily calendar (ffill from window ends) before calling."""
    s = density_series.dropna()
    lo = s.quantile(quantile_low)
    hi = s.quantile(quantile_high)
    return {
        "low_density": (s <= lo).rename("low_density"),
        "high_density": (s >= hi).rename("high_density"),
    }


# Regime-conditional aggregation
def regime_conditional_summary(
    returns: pd.Series,
    masks: dict[str, pd.Series],
    summary_fn,
) -> pd.DataFrame:
    """Apply summary_fn(returns_subset) -> dict per regime mask, plus an
    unconditional "all" row."""
    rows = {"all": summary_fn(returns)}
    for name, mask in masks.items():
        sub = returns.where(mask).dropna()
        rows[name] = summary_fn(sub) if not sub.empty else {}
    return pd.DataFrame(rows).T


__all__ = [
    "nber_recession_dates",
    "vix_regime_masks",
    "network_density_regimes",
    "regime_conditional_summary",
]
