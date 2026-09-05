"""Legacy HRP integration (Lopez de Prado): turns causal graphs into portfolio
weights via causal distances and/or the SVAR-implied structural covariance.
Kept for its reusable helpers (nearest_psd, symmetrise, causal_embedding_distance).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Matrix utilities
def symmetrise(matrix: np.ndarray) -> np.ndarray:
    """Symmetrise a directed causal matrix: ``(|M| + |M^T|) / 2``."""
    abs_m = np.abs(matrix)
    return 0.5 * (abs_m + abs_m.T)


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project onto the nearest PSD matrix by clipping negative eigenvalues."""
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    vals_clipped = np.clip(vals, 0.0, None)
    return (vecs * vals_clipped) @ vecs.T


# Distance matrices
def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Standard HRP correlation distance ``sqrt(0.5 * (1 - corr))``."""
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return dist


def symmetrise_distance(matrix: np.ndarray) -> np.ndarray:
    """Symmetrise to a similarity, scale to [0, 1], return ``1 - similarity``."""
    sim = symmetrise(matrix)
    off_diag = sim.copy()
    np.fill_diagonal(off_diag, 0.0)
    peak = off_diag.max()
    if peak > 0:
        sim = sim / peak
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return np.clip(dist, 0.0, None)


def causal_embedding_distance(matrix: np.ndarray) -> np.ndarray:
    """Embed each asset as ``e_i = [M[i, :], M[:, i]]`` and take Euclidean
    distances. Keeps more of the causal signature than symmetrising.
    """
    out_edges = matrix
    in_edges = matrix.T
    embedding = np.concatenate([out_edges, in_edges], axis=1)  # (d, 2d)
    diff = embedding[:, None, :] - embedding[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 0.0)
    return dist


# Hierarchical Risk Parity
def _quasi_diagonal_order(linkage: np.ndarray) -> list[int]:
    """Return the leaf order that quasi-diagonalises the linkage tree."""
    linkage = linkage.astype(int)
    n_items = linkage[-1, 3]
    order = pd.Series([linkage[-1, 0], linkage[-1, 1]])
    while order.max() >= n_items:
        order.index = range(0, 2 * order.shape[0], 2)
        clusters = order[order >= n_items]
        i = clusters.index
        j = clusters.values - n_items
        order[i] = linkage[j, 0]
        expanded = pd.Series(linkage[j, 1], index=i + 1)
        order = pd.concat([order, expanded]).sort_index()
        order.index = range(order.shape[0])
    return order.tolist()


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    """Inverse-variance-weighted variance of a sub-cluster."""
    sub = cov[np.ix_(items, items)]
    inv_diag = 1.0 / np.diag(sub)
    weights = inv_diag / inv_diag.sum()
    return float(weights @ sub @ weights)


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Allocate weights by HRP recursive bisection over the quasi-diagonal order."""
    weights = np.ones(len(order))
    clusters = [order]
    while clusters:
        clusters = [
            half
            for cluster in clusters
            for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
            if len(cluster) > 1
        ]
        for k in range(0, len(clusters), 2):
            left, right = clusters[k], clusters[k + 1]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            for idx in left:
                weights[idx] *= alpha
            for idx in right:
                weights[idx] *= 1.0 - alpha
    return weights


def hrp_weights(
    cov: np.ndarray,
    dist: np.ndarray,
    tickers: list[str] | None = None,
    linkage_method: str = "single",
) -> pd.Series:
    """HRP weights: ``dist`` drives clustering, ``cov`` drives allocation.

    Returns a Series of weights summing to 1.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    d = cov.shape[0]
    condensed = squareform(dist, checks=False)
    tree = linkage(condensed, method=linkage_method)
    order = _quasi_diagonal_order(tree)
    weights = _recursive_bisection(cov, order)
    weights = weights / weights.sum()
    return pd.Series(weights, index=tickers if tickers is not None else range(d))


# Comparison driver
def compare_hrp(
    returns: pd.DataFrame,
    causal_matrix: np.ndarray,
    distance: str = "embedding",
    linkage_method: str = "single",
) -> pd.DataFrame:
    """HRP weights for the correlation baseline and the causal variant.

    ``distance`` is "embedding" or "symmetrise". Returns a DataFrame with
    columns ``correlation_hrp`` and ``causal_hrp``.
    """
    tickers = list(returns.columns)
    cov = returns.cov().to_numpy()
    corr = returns.corr().to_numpy()

    corr_dist = correlation_distance(corr)
    if distance == "embedding":
        causal_dist = causal_embedding_distance(causal_matrix)
    elif distance == "symmetrise":
        causal_dist = symmetrise_distance(causal_matrix)
    else:
        raise ValueError(f"unknown distance: {distance!r}")

    baseline = hrp_weights(cov, corr_dist, tickers, linkage_method)
    causal = hrp_weights(cov, causal_dist, tickers, linkage_method)
    return pd.DataFrame({"correlation_hrp": baseline, "causal_hrp": causal})
