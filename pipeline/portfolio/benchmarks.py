"""Benchmark portfolios: 1/N, min-variance, MVO, cap-weighted.

All return a name-indexed pd.Series of weights summing to 1, matching the
HRP/HSP signature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight(asset_names: list[str]) -> pd.Series:
    """``w_i = 1/N`` for every asset."""
    n = len(asset_names)
    if n == 0:
        raise ValueError("empty asset list")
    return pd.Series([1.0 / n] * n, index=asset_names, name="weight")


def _shrunk_covariance(returns: pd.DataFrame) -> np.ndarray:
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(returns.dropna().to_numpy()).covariance_


def min_variance(
    returns_window: pd.DataFrame,
    asset_names: list[str] | None = None,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.Series:
    """Long-only minimum-variance via cvxpy with Ledoit-Wolf shrunk covariance."""
    # source code available at: https://github.com/cvxpy/cvxpy
    import cvxpy as cp

    asset_names = asset_names or list(returns_window.columns)
    R = returns_window[asset_names].dropna()
    Sigma = _shrunk_covariance(R)
    n = len(asset_names)
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), constraints)
    prob.solve(solver="CLARABEL")
    if w.value is None:
        raise RuntimeError(f"min_variance solver returned status: {prob.status}")
    return pd.Series(np.asarray(w.value), index=asset_names, name="weight")


def mean_variance(
    returns_window: pd.DataFrame,
    asset_names: list[str] | None = None,
    risk_aversion: float = 3.0,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.Series:
    """Markowitz MVO ``min_w  γ wᵀΣw - μᵀw`` with shrunk covariance + sample mean."""
    # source code available at: https://github.com/cvxpy/cvxpy
    import cvxpy as cp

    asset_names = asset_names or list(returns_window.columns)
    R = returns_window[asset_names].dropna()
    mu = R.mean().to_numpy()
    Sigma = _shrunk_covariance(R)
    n = len(asset_names)
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)
    obj = risk_aversion * cp.quad_form(w, cp.psd_wrap(Sigma)) - mu @ w
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver="CLARABEL")
    if w.value is None:
        raise RuntimeError(f"mean_variance solver returned status: {prob.status}")
    return pd.Series(np.asarray(w.value), index=asset_names, name="weight")


def cap_weighted(
    asset_names: list[str],
    prices_at_rebalance: pd.Series,
    shares_outstanding: pd.Series,
) -> pd.Series:
    """Market-cap weights from ``price × shares_outstanding``.

    Uses current shares-outstanding (yfinance) as an approximation until
    CRSP historical shares are available.
    """
    caps = (
        prices_at_rebalance.reindex(asset_names)
        * shares_outstanding.reindex(asset_names)
    ).dropna()
    if caps.empty:
        raise ValueError("no usable market-cap data for the supplied assets")
    weights = caps / caps.sum()
    return weights.reindex(asset_names).fillna(0.0).rename("weight")


__all__ = [
    "equal_weight",
    "min_variance",
    "mean_variance",
    "cap_weighted",
]
