"""Regime classification for conditional-performance analysis.

I use NBER recessions (FRED USREC), VIX quintiles and causal-network density
quintiles. Each function returns boolean masks aligned to the daily return
index.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# NBER recessions
def nber_recession_dates(daily_index: pd.DatetimeIndex) -> pd.Series:
    """Return True on trading days inside an NBER recession. I take the
    monthly FRED USREC series and forward-fill it to trading days."""
    from pipeline.data.drivers import fetch_fred_series

    usrec = fetch_fred_series("USREC")
    daily = usrec.reindex(daily_index, method="ffill").fillna(0).astype(int)
    return (daily == 1).rename("nber_recession")


# VIX quintile vol regimes
def vix_regime_masks(
    vix: pd.Series, quantile_low: float = 0.20, quantile_high: float = 0.80,
) -> dict[str, pd.Series]:
    """Return the top and bottom VIX-quintile masks. The middle 60% is in
    neither. The thresholds use the full sample, which is fine for post-hoc
    analysis but would be lookahead in a trading decision."""
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
    """Return quintile masks on a network-density series. Reindex the
    per-window density to the daily calendar (forward-fill from window ends)
    before calling this."""
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
    """Apply summary_fn(returns_subset) -> dict to each regime mask and add an
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
