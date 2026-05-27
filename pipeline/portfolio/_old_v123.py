"""Step 3 (later phase) -- portfolio integration via Hierarchical Risk Parity.

This module turns the causal graphs from Plan A / Plan B into portfolio weights
and lets them be compared against the correlation baseline.

HRP (Lopez de Prado) has two stages, and the causal structure can be injected
into either:

1. **Clustering** -- needs a *distance* matrix (symmetric, non-negative, zero
   diagonal): which assets are similar.  Hierarchical clustering cannot run on
   an asymmetric matrix, so the directed causal matrix must be symmetrised.
2. **Allocation** -- recursive bisection needs a *covariance* matrix: how risky
   each sub-cluster is.

The integration variants follow the project's HRP notes:

* **v1** -- causal distance for clustering, sample covariance for allocation.
* **v2** -- causal distance for clustering, the SVAR-implied *structural*
  covariance for allocation.
* **v3** -- a convex blend of causal and correlation information at *both*
  stages, swept over a mixing coefficient.

Two ways to derive a distance from a directed causal matrix ``M`` (``i -> j``):

* :func:`symmetrise_distance` -- the plan's ``(|M| + |M^T|) / 2`` similarity,
  mapped to a distance.
* :func:`causal_embedding_distance` -- the notes' preferred route: embed each
  asset as ``[outgoing edges, incoming edges]`` and take Euclidean distances.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Matrix utilities
# ============================================================================
def symmetrise(matrix: np.ndarray) -> np.ndarray:
    """Symmetrise a directed causal matrix: ``(|M| + |M^T|) / 2``.

    Clustering is non-directional (assets are in the same cluster or not), so
    losing edge direction here is acceptable -- see the project's HRP notes.
    """
    abs_m = np.abs(matrix)
    return 0.5 * (abs_m + abs_m.T)


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the nearest positive-semidefinite one.

    Negative eigenvalues are clipped to zero and the matrix reconstructed.  The
    structural covariance and some causal similarity matrices are not PSD by
    construction; clustering/allocation need them to be.
    """
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    vals_clipped = np.clip(vals, 0.0, None)
    return (vecs * vals_clipped) @ vecs.T


# ============================================================================
# Distance matrices
# ============================================================================
def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Standard HRP correlation distance ``sqrt(0.5 * (1 - corr))``."""
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return dist


def symmetrise_distance(matrix: np.ndarray) -> np.ndarray:
    """Distance from a causal matrix via the plan's symmetrise-and-normalise (v1).

    The directed matrix is symmetrised to a similarity, scaled to ``[0, 1]`` by
    its maximum off-diagonal entry, and turned into a distance ``1 - similarity``
    (zero on the diagonal).  Higher causal coupling => smaller distance.
    """
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
    """Distance from a causal matrix via in/out edge embeddings (notes' choice).

    Each asset ``i`` is embedded as ``e_i = [M[i, :], M[:, i]]`` -- its outgoing
    and incoming causal edges concatenated -- and distances are Euclidean
    ``D[i, j] = ||e_i - e_j||_2``.  This keeps more of the causal signature
    than symmetrising and is, per the notes, cleaner than graph-edit or
    shortest-path distances.
    """
    out_edges = matrix
    in_edges = matrix.T
    embedding = np.concatenate([out_edges, in_edges], axis=1)  # (d, 2d)
    diff = embedding[:, None, :] - embedding[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 0.0)
    return dist


def blend_distance(
    causal_dist: np.ndarray, corr_dist: np.ndarray, alpha: float
) -> np.ndarray:
    """Convex blend ``alpha * causal + (1 - alpha) * correlation`` distance (v3)."""
    return alpha * causal_dist + (1.0 - alpha) * corr_dist


# ============================================================================
# Covariance matrices
# ============================================================================
def structural_covariance(
    matrix: np.ndarray, residual_cov: np.ndarray | None = None
) -> np.ndarray:
    """SVAR-implied contemporaneous covariance of returns (v2).

    Both DYNOTEARS and VARLiNGAM (in this package's ``i -> j`` convention) imply
    the structural equation ``(I - M^T) x = e``.  The reduced-form covariance is

        Sigma_causal = (I - M^T)^{-1}  Sigma_e  (I - M^T)^{-T}

    Parameters
    ----------
    matrix:
        Contemporaneous causal matrix ``W`` (DYNOTEARS) or ``B0`` (VARLiNGAM).
    residual_cov:
        Covariance of the structural residuals ``Sigma_e``.  Defaults to the
        identity -- appropriate when returns were standardised upstream and a
        better estimate (e.g. ``VARLiNGAM.residuals_``) is unavailable.

    The result is projected to the nearest PSD matrix before being returned.
    """
    d = matrix.shape[0]
    if residual_cov is None:
        residual_cov = np.eye(d)
    inv = np.linalg.inv(np.eye(d) - matrix.T)
    cov = inv @ residual_cov @ inv.T
    return nearest_psd(cov)


def blend_covariance(
    causal_cov: np.ndarray, sample_cov: np.ndarray, alpha: float
) -> np.ndarray:
    """Convex blend ``alpha * causal + (1 - alpha) * sample`` covariance (v3)."""
    return alpha * causal_cov + (1.0 - alpha) * sample_cov


# ============================================================================
# Hierarchical Risk Parity
# ============================================================================
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
    """Hierarchical Risk Parity portfolio weights.

    Parameters
    ----------
    cov:
        Covariance matrix used in the **allocation** stage (recursive bisection).
    dist:
        Distance matrix used in the **clustering** stage.  Inject a causal
        distance here for the v1/v3 variants.
    tickers:
        Optional asset names for the returned Series index.
    linkage_method:
        SciPy linkage method.  HRP's default is ``"single"``; the notes suggest
        trying others, since single linkage is outlier-sensitive.

    Returns
    -------
    Series of weights summing to 1.
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


# ============================================================================
# Comparison driver
# ============================================================================
def compare_hrp(
    returns: pd.DataFrame,
    causal_matrix: np.ndarray,
    distance: str = "embedding",
    linkage_method: str = "single",
) -> pd.DataFrame:
    """Build HRP weights for the correlation baseline and the v1 causal variant.

    Parameters
    ----------
    returns:
        Asset returns over the window, columns = assets (the allocation cov and
        the correlation baseline are estimated from this).
    causal_matrix:
        Contemporaneous causal matrix (``W`` or ``B0``) aligned to ``returns``'
        columns.
    distance:
        ``"embedding"`` (notes' choice) or ``"symmetrise"`` (the plan's
        ``(|M|+|M^T|)/2``) for deriving the causal distance.

    Returns
    -------
    DataFrame with one weight column per method: ``correlation_hrp`` and
    ``causal_hrp``.
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
