"""Cumulative-correlation driver selection — Rodriguez-Dominguez 2023 (HSP).

This is the **V0 baseline** driver-selection method. For each candidate
driver ``d``, score it by summed absolute correlation with the asset block:

    score_d = Σ_{lag ∈ lags} Σ_{asset i} | corr(d_{t-lag}, asset_i_t) |

Take the top ``K``. No causal inference involved — just classical
correlation. The contrast with our :func:`pipeline.factor_selection.select_drivers`
(causal greedy + utility blend) is exactly the experimental contribution of
the thesis.

Ported from the inline cum-corr block in
``thesis/HSP/Full_Project_Notebook.ipynb`` ("Correlation window for drivers
selection" → "Rank the previous results and select the top n=N_Drivers
largest value names for the lag 0, 1 and both cases"). The notebook handles
three lag configurations: lag-0 only, lag-1 only, or both summed; we expose
the same via the ``lags`` tuple parameter.

Entry points
------------
* :func:`cumulative_correlation_score` -- per-driver score over a window.
* :func:`select_top_k_corr` -- top-K names; mirrors the signature of
  :func:`pipeline.factor_selection.select_drivers` so the V0 path in
  ``stage1_pipeline.py`` can swap selection methods with a one-line change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Score
# ============================================================================
def cumulative_correlation_score(
    driver_window: pd.DataFrame,
    asset_window: pd.DataFrame,
    lags: Sequence[int] = (0, 1),
) -> pd.Series:
    """``Σ_{lag} Σ_{asset} |corr(driver_{t-lag}, asset_t)|`` per driver.

    Parameters
    ----------
    driver_window:
        Driver panel for the selection window (one column per candidate).
    asset_window:
        Asset panel, same index as ``driver_window``.
    lags:
        Lags to sum over. ``(0,)`` is contemporaneous-only; ``(0, 1)`` is
        the paper's default; ``(1,)`` is lagged-only (closest analogue to
        our Stage A causal score which excludes contemporaneous edges).

    Returns
    -------
    ``pd.Series`` indexed by driver name, sorted ascending by name (caller
    sorts by value to pick the top-K). NaN-safe: pairs with insufficient
    overlap contribute zero.
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


# ============================================================================
# Top-K selection
# ============================================================================
@dataclass
class CorrelationSelectionResult:
    """Output of :func:`select_top_k_corr` — mirrors :class:`SelectionResult`
    just enough for downstream code to consume."""

    rebalance_date: pd.Timestamp
    selected: list[str]
    scores: pd.Series  # all candidates with their score
    K: int
    lags: tuple[int, ...]

    @property
    def stage_b(self):  # interface-compat with SelectionResult
        return None


def select_top_k_corr(
    driver_window: pd.DataFrame,
    asset_window: pd.DataFrame,
    K: int,
    rebalance_date: pd.Timestamp | str | None = None,
    lags: Sequence[int] = (0, 1),
) -> CorrelationSelectionResult:
    """Top-``K`` drivers by cumulative correlation. Vanilla-HSP (V0) selector.

    The signature mirrors :func:`pipeline.factor_selection.select_drivers`'s
    contract so ``stage1_pipeline.run_stage1`` can route through either
    selector with the same call shape (causal vs cum-corr).
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
