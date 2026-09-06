"""Causal-ordered bisection: DAG utilities and the D2 and D2s allocators
(Phase II).

I replace HRP's clustering step with a deterministic topological order of the
DAG and then run the existing ``recursive_bisection`` on that order. Inputs
that are not DAGs (GRANGER) go through greedy feedback-arc removal first.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.discovery.asset_graph import AssetGraphWindow, is_dag_matrix
from pipeline.portfolio.directed import (
    structural_covariance_v2,
    total_effect_matrix,
)
from pipeline.portfolio.hrp import recursive_bisection

logger = logging.getLogger(__name__)


class CyclicGraphError(ValueError):
    """Raised when a topological order is requested for a cyclic graph."""


# Feedback-arc removal (non-DAG fallback, GRANGER only)
def remove_feedback_arcs(M: np.ndarray) -> tuple[np.ndarray, int]:
    """Greedily zero the edges with the smallest |M| until the graph is acyclic.

    Return the cleaned matrix and the number of edges dropped.
    """
    M = M.copy()
    dropped = 0
    while not is_dag_matrix(M):
        # The nodes that a Kahn pass cannot eliminate form the cyclic core.
        adj = (M != 0.0).astype(np.int64)
        np.fill_diagonal(adj, 0)
        in_deg = adj.sum(axis=0)
        alive = np.ones(M.shape[0], dtype=bool)
        changed = True
        while changed:
            changed = False
            for i in np.flatnonzero(alive):
                if in_deg[i] == 0:
                    alive[i] = False
                    in_deg -= adj[i]
                    adj[i, :] = 0
                    changed = True
        core = np.flatnonzero(alive)
        sub = np.abs(M[np.ix_(core, core)])
        sub[sub == 0.0] = np.inf
        r, c = np.unravel_index(int(np.argmin(sub)), sub.shape)
        M[core[r], core[c]] = 0.0
        dropped += 1
    if dropped:
        logger.info("remove_feedback_arcs: dropped %d edge(s) to reach a DAG", dropped)
    return M, dropped


# Deterministic topological order
def topological_order(
    M: np.ndarray,
    asset_names: list[str] | None = None,
) -> list[int]:
    """Return the Kahn topological order of ``M`` (edges run i to j), upstream
    first.

    Ties are broken by total downstream influence ``sum_r |B[r, i]|``
    descending, then by asset name ascending, so the order is fully
    deterministic. Raise :class:`CyclicGraphError` on a cyclic graph.
    """
    N = M.shape[0]
    if not is_dag_matrix(M):
        raise CyclicGraphError("graph is cyclic; run remove_feedback_arcs first")
    names = asset_names if asset_names is not None else [str(i) for i in range(N)]
    influence = np.abs(total_effect_matrix(M, is_dag=True)).sum(axis=0)

    adj = (M != 0.0).astype(np.int64)
    np.fill_diagonal(adj, 0)
    in_deg = adj.sum(axis=0).astype(np.int64)
    remaining = set(range(N))
    order: list[int] = []
    while remaining:
        candidates = [i for i in remaining if in_deg[i] == 0]
        # Pick deterministically: largest influence first, then name ascending.
        pick = min(candidates, key=lambda i: (-influence[i], names[i]))
        order.append(pick)
        remaining.discard(pick)
        children = np.flatnonzero(adj[pick])
        adj[pick, children] = 0
        in_deg[children] -= 1
    return order


# D2 / D2s: causal-ordered bisection
def d2_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    covariance: str = "sample",
) -> pd.Series:
    """Run recursive bisection over the topological order, with no clustering
    step.

    ``covariance="sample"`` is D2, where edge direction enters only through
    the ordering. ``"structural"`` is D2s, where it also enters the
    allocation covariance.
    """
    M = graph.M
    if not graph.is_dag:
        M, dropped = remove_feedback_arcs(M)
        logger.info(
            "d2_weights (%s): non-DAG input, %d feedback arc(s) removed",
            graph.end_date, dropped,
        )
    order = topological_order(M, graph.asset_names)

    if covariance == "structural":
        cov = structural_covariance_v2(graph)
    elif covariance == "sample":
        from pipeline.portfolio.directed import _sample_cov

        cov = _sample_cov(graph, returns_window)
        # recursive_bisection indexes by position, so align to the graph's order.
        cov = cov.loc[graph.asset_names, graph.asset_names]
    else:
        raise ValueError(f"covariance must be 'sample' or 'structural', got {covariance!r}")

    weights = recursive_bisection(cov.to_numpy(), order)
    weights = weights / weights.sum()
    return pd.Series(weights, index=graph.asset_names, name="weight")


# Diagnostics (E3/E6 inputs)
def dag_diagnostics(graph: AssetGraphWindow) -> dict:
    """Return the edge density, DAG depth (longest path) and the numbers of
    roots and leaves for 1 window."""
    M = graph.M
    N = graph.n_assets
    adj = M != 0.0
    np.fill_diagonal(adj, False)
    n_edges = int(adj.sum())
    density = n_edges / max(N * (N - 1), 1)
    out_deg = adj.sum(axis=1)
    in_deg = adj.sum(axis=0)
    depth = -1
    if graph.is_dag:
        order = topological_order(M, graph.asset_names)
        longest = np.zeros(N)
        for i in order:
            for j in np.flatnonzero(adj[i]):
                longest[j] = max(longest[j], longest[i] + 1)
        depth = int(longest.max())
    return {
        "end_date": graph.end_date,
        "n_assets": N,
        "n_edges": n_edges,
        "density": density,
        "is_dag": graph.is_dag,
        "dag_depth": depth,
        "n_roots": int((in_deg == 0).sum()),
        "n_leaves": int((out_deg == 0).sum()),
    }


def order_stability(orders: list[list[int]]) -> pd.Series:
    """Return Kendall's tau between the topological orders of consecutive
    windows (E6)."""
    from scipy.stats import kendalltau

    taus = []
    for a, b in zip(orders[:-1], orders[1:]):
        if len(a) != len(b):
            taus.append(np.nan)
            continue
        rank_a = np.argsort(a)
        rank_b = np.argsort(b)
        taus.append(kendalltau(rank_a, rank_b).statistic)
    return pd.Series(taus, name="kendall_tau")


__all__ = [
    "CyclicGraphError",
    "d2_weights",
    "dag_diagnostics",
    "order_stability",
    "remove_feedback_arcs",
    "topological_order",
]
