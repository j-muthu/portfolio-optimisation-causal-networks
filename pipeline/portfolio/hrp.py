"""Hierarchical Risk Parity (López de Prado 2016) with pluggable distance
and covariance.

The distance drives clustering, the covariance drives allocation; the two
are decoupled so variants can swap either in.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Quasi-diagonalisation + recursive bisection
def quasi_diagonal_order(linkage: np.ndarray, n_items: int) -> list[int]:
    """Leaf order that quasi-diagonalises the linkage tree (López de Prado 2016)."""
    linkage = linkage.astype(int)
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
    """Variance of the *inverse-variance-weighted* portfolio on ``items``."""
    sub = cov[np.ix_(items, items)]
    inv_diag = 1.0 / np.maximum(np.diag(sub), 1e-12)
    weights = inv_diag / inv_diag.sum()
    return float(weights @ sub @ weights)


def recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Allocate by HRP recursive bisection over the quasi-diagonal ordering."""
    weights = np.ones(len(order))
    clusters = [order]
    while clusters:
        clusters = [
            half
            for cluster in clusters
            for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2:])
            if len(cluster) > 1
        ]
        for k in range(0, len(clusters), 2):
            left, right = clusters[k], clusters[k + 1]
            v_left = _cluster_variance(cov, left)
            v_right = _cluster_variance(cov, right)
            alpha = 1.0 - v_left / max(v_left + v_right, 1e-18)
            for idx in left:
                weights[idx] *= alpha
            for idx in right:
                weights[idx] *= 1.0 - alpha
    return weights


# Top-level HRP
def hrp_weights(
    distance: pd.DataFrame,
    covariance: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """HRP weights from a (distance, covariance) pair sharing the same
    asset index. Returns a name-indexed Series summing to 1.
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    if list(distance.index) != list(covariance.index):
        raise ValueError("distance and covariance must share the same asset ordering")
    assets = list(distance.index)
    N = len(assets)

    dist_arr = distance.to_numpy()
    # squareform expects exact symmetry.
    dist_arr = (dist_arr + dist_arr.T) / 2.0
    np.fill_diagonal(dist_arr, 0.0)
    condensed = squareform(dist_arr, checks=False)
    tree = scipy_linkage(condensed, method=linkage_method)
    order = quasi_diagonal_order(tree, n_items=N)
    weights = recursive_bisection(covariance.to_numpy(), order)
    weights = weights / weights.sum()
    return pd.Series(weights, index=assets, name="weight")


# HERC (Raffinot 2018), full-recursion variant
def herc_weights(
    distance: pd.DataFrame,
    covariance: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Hierarchical Equal Risk Contribution weights (Raffinot 2018),
    full-recursion variant.

    Same clustering stage as :func:`hrp_weights`, but allocation walks the
    dendrogram's actual topology top-down, splitting each node's budget by
    inverse cluster variance. The only change from HRP is how the tree is read.
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    if list(distance.index) != list(covariance.index):
        raise ValueError("distance and covariance must share the same asset ordering")
    assets = list(distance.index)
    N = len(assets)

    dist_arr = distance.to_numpy()
    dist_arr = (dist_arr + dist_arr.T) / 2.0
    np.fill_diagonal(dist_arr, 0.0)
    condensed = squareform(dist_arr, checks=False)
    tree = scipy_linkage(condensed, method=linkage_method).astype(int)

    cov = covariance.to_numpy()

    # leaves(node): leaf ids < N; internal node N+j has children tree[j, 0:2].
    def leaves(node: int) -> list[int]:
        if node < N:
            return [node]
        j = node - N
        return leaves(int(tree[j, 0])) + leaves(int(tree[j, 1]))

    weights = np.zeros(N)

    def descend(node: int, budget: float) -> None:
        if node < N:
            weights[node] = budget
            return
        j = node - N
        left, right = int(tree[j, 0]), int(tree[j, 1])
        v_left = _cluster_variance(cov, leaves(left))
        v_right = _cluster_variance(cov, leaves(right))
        alpha = 1.0 - v_left / max(v_left + v_right, 1e-18)
        descend(left, budget * alpha)
        descend(right, budget * (1.0 - alpha))

    descend(N + tree.shape[0] - 1, 1.0)   # root
    weights = weights / weights.sum()
    return pd.Series(weights, index=assets, name="weight")


__all__ = [
    "quasi_diagonal_order",
    "recursive_bisection",
    "hrp_weights",
    "herc_weights",
]
