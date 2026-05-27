"""Hierarchical Sensitivity Parity (Rodriguez-Dominguez 2023) — HRP with the
distance matrix replaced by the Euclidean distance in *sensitivity space*.

Two flavours are exposed:

* :func:`hsp_weights_from_S` — directly takes the per-window sensitivity
  matrix ``S`` from :func:`pipeline.sensitivities.fit_sensitivities_window`
  and a return panel for the sample covariance.
* :func:`hsp_weights_from_window` — convenience wrapper that takes a
  ``SensitivityWindow`` and the corresponding asset-return frame.

V0 (vanilla HSP) uses cumulative-correlation driver selection; the input
``S`` here is correlation-derived. V1 / V2 (Causal-HSP) use causal-discovery-
based driver selection but the *clustering math* is identical — only the
upstream selection differs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.portfolio._old_v123 import nearest_psd
from pipeline.portfolio.hrp import hrp_weights
from pipeline.sensitivities.sensitivity_matrix import distance_from_S

logger = logging.getLogger(__name__)


# ============================================================================
# Sample covariance helpers
# ============================================================================
def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Plain sample covariance of asset returns over the supplied window."""
    return returns.cov()


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance — recommended at N ≈ T (S&P 100 + 504d)."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(returns.dropna().to_numpy())
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


# ============================================================================
# HSP weights
# ============================================================================
def hsp_weights_from_S(
    S: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
    use_ledoit_wolf: bool = False,
    psd_project_distance: bool = False,
) -> pd.Series:
    """Compute HSP weights from a sensitivity matrix + a return window.

    Parameters
    ----------
    S:
        ``(N, K)`` per-asset sensitivity vectors. Rows must align with
        ``asset_names``.
    asset_names:
        N asset names; also used to subset and order ``returns_window``.
    returns_window:
        Daily returns over the same window as the discovery / FFNN fit;
        used to estimate the allocation covariance.
    linkage_method:
        SciPy linkage. Default ``"single"`` per López de Prado / HSP.
    use_ledoit_wolf:
        Shrink the covariance via Ledoit-Wolf — robust at N ≈ T.
    psd_project_distance:
        If ``True``, project the sensitivity-space distance matrix to its
        nearest PSD before clustering. The Euclidean distance is symmetric
        and non-negative by construction so PSD-projection is rarely needed,
        but the plan calls for it as a safeguard.
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
    "hsp_weights_from_S",
    "hsp_weights_from_window",
]
