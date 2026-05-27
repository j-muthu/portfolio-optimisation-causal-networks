"""Sensitivity → distance + covariance utilities for the HSP clustering step.

Given a per-window sensitivity matrix ``S ∈ ℝ^{N × K}`` from
:mod:`pipeline.sensitivities.ffnn`, this module produces:

* ``distance_from_S`` — the Rodriguez-Dominguez sensitivity-space distance,
  ``D[i, j] = ||s_i - s_j||_2``. Symmetric by construction, zero-diagonal,
  non-negative. The plan calls for a nearest-PSD projection downstream
  before HRP-style clustering — that lives in
  :mod:`pipeline.portfolio._old_v123.nearest_psd`, kept for reuse.
* ``correlation_from_S`` — alternative formulation: cosine-similarity of
  sensitivity vectors mapped to ``d = sqrt(0.5 (1 - corr))`` per the
  Lopez-de-Prado HRP distance form. Useful as a robustness comparison.

Both forms return a ``pd.DataFrame`` with asset names on both axes so HRP /
HSP can key by ticker rather than positional index.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Distance forms
# ============================================================================
def distance_from_S(
    S: np.ndarray, asset_names: Sequence[str]
) -> pd.DataFrame:
    """``D[i, j] = ||s_i - s_j||_2`` Euclidean distance in sensitivity space.

    Returns a symmetric, zero-diagonal, non-negative DataFrame indexed by
    asset name on both axes. No PSD guarantee — the matrix is a distance,
    not a similarity, and PSD-projection should happen at the consumer
    (Lopez-de-Prado HRP doesn't require PSD distance, only PSD covariance).
    """
    if S.ndim != 2:
        raise ValueError(f"S must be 2-d (N × K); got shape {S.shape}")
    if S.shape[0] != len(asset_names):
        raise ValueError(
            f"S has {S.shape[0]} rows but {len(asset_names)} asset_names supplied"
        )
    diff = S[:, None, :] - S[None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    D = (D + D.T) / 2.0  # enforce exact symmetry against float-roundoff
    np.fill_diagonal(D, 0.0)
    return pd.DataFrame(D, index=list(asset_names), columns=list(asset_names))


def correlation_from_S(
    S: np.ndarray, asset_names: Sequence[str]
) -> pd.DataFrame:
    """Cosine correlation of sensitivity vectors, ``corr_{ij} = s_i · s_j / (||s_i|| ||s_j||)``.

    Useful when the sensitivity vector magnitudes are very heterogeneous and
    you want clustering driven by *direction* of sensitivity rather than
    magnitude.
    """
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1e-12)
    Z = S / norms
    corr = Z @ Z.T
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=list(asset_names), columns=list(asset_names))


def lopez_de_prado_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix to the HRP distance form ``sqrt(0.5(1-corr))``."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy()), 0.0, None))
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


# ============================================================================
# Quick diagnostics (used in the K-appropriateness verification step)
# ============================================================================
def distance_concentration(D: pd.DataFrame) -> float:
    """``σ(D_ij) / E[D_ij]`` for the off-diagonal entries.

    Plan §K diagnostics: tracking this over the K sensitivity-sweep range
    quantifies the curse of dimensionality (low value ⇒ everything looks
    equidistant ⇒ K too high). Computed on the upper triangle only.
    """
    arr = D.to_numpy()
    iu = np.triu_indices_from(arr, k=1)
    vals = arr[iu]
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=0))
    if mu < 1e-12:
        return 0.0
    return sigma / mu


def effective_dimensionality(S: np.ndarray, var_explained: float = 0.95) -> int:
    """Smallest ``q`` such that top-``q`` PCs of ``S`` explain ``var_explained`` of variance.

    Plan §K diagnostics: if effective_dim ≪ K, K is too high and most
    sensitivity-space coordinates are redundant.
    """
    if S.shape[0] < 2:
        return S.shape[1]
    centred = S - S.mean(axis=0, keepdims=True)
    # SVD-based PCA; handles non-square cleanly.
    _, sv, _ = np.linalg.svd(centred, full_matrices=False)
    var = (sv ** 2)
    total = var.sum()
    if total < 1e-12:
        return S.shape[1]
    cum = np.cumsum(var / total)
    return int(np.searchsorted(cum, var_explained) + 1)


__all__ = [
    "distance_from_S",
    "correlation_from_S",
    "lopez_de_prado_distance",
    "distance_concentration",
    "effective_dimensionality",
]
