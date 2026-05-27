"""Regime classification for the conditional-performance analysis.

Three regime definitions per ``Closed-Loop Causal-HSP Portfolio.md`` §Metrics:

1. **NBER recession dates** (binary). Pulled from FRED's USREC series.
2. **VIX-based vol regimes**. Top vs bottom quintile of VIX over the full
   backtest sample.
3. **Causal-network density regimes**. Density of the V2 discovery W block
   over time, split into quintiles. Reuses
   :func:`pipeline.discovery.diagnostics.graph_density`.

All three return a ``pd.Series[bool]`` aligned to the backtest's daily
return index, suitable for ``returns[mask]``-style conditional aggregation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# NBER recessions
# ============================================================================
def nber_recession_dates(daily_index: pd.DatetimeIndex) -> pd.Series:
    """Boolean series: ``True`` on trading days inside an NBER recession.

    Uses FRED's ``USREC`` indicator (monthly, 1 = recession, 0 = expansion),
    forward-filled to trading days.
    """
    from pipeline.data.drivers import fetch_fred_series

    usrec = fetch_fred_series("USREC")
    # USREC is monthly; reindex to daily and forward-fill.
    daily = usrec.reindex(daily_index, method="ffill").fillna(0).astype(int)
    return (daily == 1).rename("nber_recession")


# ============================================================================
# VIX quintile vol regimes
# ============================================================================
def vix_regime_masks(
    vix: pd.Series, quantile_low: float = 0.20, quantile_high: float = 0.80,
) -> dict[str, pd.Series]:
    """Top vs bottom VIX-quintile masks.

    Returns ``{"low_vol": mask, "high_vol": mask}``. Both masks are aligned
    to ``vix.index``; the middle 60% is excluded from both. The quintile
    thresholds are computed over the *full sample*, which is acceptable here
    because the masks are used post-hoc for regime-conditional aggregation
    (not for forward-looking trading decisions).
    """
    vix = vix.dropna()
    lo = vix.quantile(quantile_low)
    hi = vix.quantile(quantile_high)
    return {
        "low_vol": (vix <= lo).rename("low_vol"),
        "high_vol": (vix >= hi).rename("high_vol"),
    }


# ============================================================================
# Causal-network density regimes
# ============================================================================
def network_density_regimes(
    density_series: pd.Series,
    quantile_low: float = 0.20,
    quantile_high: float = 0.80,
) -> dict[str, pd.Series]:
    """Quintile masks on a causal-network density time series.

    The density series is typically per-window — pass it through here only
    after reindexing to the daily backtest calendar (forward-fill from
    window-end dates).
    """
    s = density_series.dropna()
    lo = s.quantile(quantile_low)
    hi = s.quantile(quantile_high)
    return {
        "low_density": (s <= lo).rename("low_density"),
        "high_density": (s >= hi).rename("high_density"),
    }


# ============================================================================
# Regime-conditional aggregation
# ============================================================================
def regime_conditional_summary(
    returns: pd.Series,
    masks: dict[str, pd.Series],
    summary_fn,
) -> pd.DataFrame:
    """Apply ``summary_fn`` to ``returns`` restricted to each regime mask.

    ``summary_fn(returns_subset) -> dict``. The result is one row per regime
    name plus an "all" row covering the unconditional sample.
    """
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
