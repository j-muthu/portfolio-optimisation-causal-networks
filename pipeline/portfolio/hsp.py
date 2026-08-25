"""Hierarchical Sensitivity Parity (Rodriguez-Dominguez 2023): HRP with the
distance matrix replaced by Euclidean distance in sensitivity space.

V0 / V1 / V2 differ only in how the input ``S`` was selected upstream.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.portfolio._old_v123 import nearest_psd
from pipeline.portfolio.hrp import hrp_weights
from pipeline.sensitivities.sensitivity_matrix import distance_from_S

logger = logging.getLogger(__name__)


# Covariance helpers
def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Plain sample covariance of asset returns over the supplied window."""
    return returns.cov()


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance (recommended at N ≈ T)."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(returns.dropna().to_numpy())
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


def defactored_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Single-factor residual covariance: regress each asset on the equal-weight
    cross-sectional mean and return the residual covariance (same ddof as
    ``DataFrame.cov``). The direction-free de-factoring control.
    """
    rets = returns.dropna()
    X = rets.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    m = X.mean(axis=1)
    beta = (X.T @ m) / max(float(m @ m), 1e-300)
    resid = X - np.outer(m, beta)
    cov = (resid.T @ resid) / (len(rets) - 1)
    return pd.DataFrame(cov, index=rets.columns, columns=rets.columns)


# HSP weights
def hsp_weights_from_S(
    S: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
    use_ledoit_wolf: bool = False,
    psd_project_distance: bool = False,
) -> pd.Series:
    """HSP weights from an ``(N, K)`` sensitivity matrix + a return window.

    Rows of ``S`` must align with ``asset_names``. ``psd_project_distance``
    projects the distance to nearest-PSD before clustering (rarely needed;
    kept as a safeguard).
    """
    if S.shape[0] != len(asset_names):
        raise ValueError(
            f"S has {S.shape[0]} rows but {len(asset_names)} asset_names"
        )
    returns = returns_window[asset_names].dropna()
    D = distance_from_S(S, asset_names)
    if psd_project_distance:
        D = pd.DataFrame(
            nearest_psd(D.to_numpy()), index=D.index, columns=D.columns
        )
    cov_fn = ledoit_wolf_covariance if use_ledoit_wolf else sample_covariance
    cov = cov_fn(returns)
    # Align cov to D's index ordering (defensive).
    cov = cov.loc[D.index, D.columns]
    return hrp_weights(D, cov, linkage_method=linkage_method)


def hsp_weights_from_window(
    window,  # SensitivityWindow
    returns_window: pd.DataFrame,
    **kwargs,
) -> pd.Series:
    """Convenience wrapper that pulls ``S`` / ``asset_names`` from a SensitivityWindow."""
    return hsp_weights_from_S(
        S=window.S,
        asset_names=window.asset_names,
        returns_window=returns_window,
        **kwargs,
    )


__all__ = [
    "sample_covariance",
    "ledoit_wolf_covariance",
    "defactored_covariance",
    "hsp_weights_from_S",
    "hsp_weights_from_window",
]
