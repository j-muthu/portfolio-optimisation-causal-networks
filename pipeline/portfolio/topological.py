"""Causal-ordered bisection: DAG utilities + the D2/D2s allocators (Phase II).

HRP's dendrogram exists only to produce a quasi-diagonalising leaf order for
recursive bisection. A DAG carries an ordering for free: sort assets
upstream → downstream (a topological order), then run the *existing*
``recursive_bisection`` on that order. This replaces the clustering step
entirely — the supervisor's "current portfolio optimisers cluster on
symmetric matrices" critique — at near-zero implementation cost, and it is
the purest test of whether edge *direction* carries allocation value: the
order is meaningless for a symmetrised matrix.

Determinism: Kahn's algorithm with a fixed tie-break — among the available
zero-in-degree nodes pick the one with the largest total downstream
influence ``Σᵣ |B[r, i]|`` (the column sum of the total-effect matrix:
how much a unit shock at ``i`` moves the whole system), ties broken by
asset name ascending. Non-DAG inputs (the GRANGER comparator) go through a
greedy minimum-|edge| feedback-arc removal first, with the dropped count
logged and recorded.
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


# ============================================================================
# Feedback-arc removal (non-DAG fallback, GRANGER only)
# ============================================================================
def remove_feedback_arcs(M: np.ndarray) -> tuple[np.ndarray, int]:
    """Zero the smallest-|M| edges until the graph is acyclic.

    Greedy: while a Kahn pass gets stuck, delete the minimum-magnitude edge
    among the still-cyclic remainder and retry. Returns the cleaned matrix
    and the number of edges dropped.
    """
    M = M.copy()
    dropped = 0
    while not is_dag_matrix(M):
        # Nodes not eliminated by a Kahn pass form the cyclic core.
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


# ============================================================================
# Deterministic topological order
# ============================================================================
def topological_order(
    M: np.ndarray,
    asset_names: list[str] | None = None,
) -> list[int]:
    """Kahn topological order of ``M`` (i → j), upstream first.

    Tie-break among available roots: total downstream influence
    ``Σᵣ |B[r, i]|`` descending, then asset name ascending — fully
    deterministic under any permutation of the input labelling. Raises
    :class:`CyclicGraphError` on a cyclic graph (callers pre-clean GRANGER
    graphs with :func:`remove_feedback_arcs`).
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
        # Deterministic pick: max influence, then name ascending.
        pick = min(candidates, key=lambda i: (-influence[i], names[i]))
        order.append(pick)
        remaining.discard(pick)
        children = np.flatnonzero(adj[pick])
        adj[pick, children] = 0
        in_deg[children] -= 1
    return order


# ============================================================================
# D2 / D2s — causal-ordered bisection
# ============================================================================
def d2_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    covariance: str = "sample",
) -> pd.Series:
    """Recursive bisection over the topological order (no clustering step).

    ``covariance="sample"`` is D2 (direction enters only the ordering);
    ``"structural"`` is D2s (ordering *and* allocation covariance).
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
        # Align to the graph's asset order (dropna() keeps column order, but
        # be explicit — recursive_bisection indexes positionally).
        cov = cov.loc[graph.asset_names, graph.asset_names]
    else:
        raise ValueError(f"covariance must be 'sample' or 'structural', got {covariance!r}")

    weights = recursive_bisection(cov.to_numpy(), order)
    weights = weights / weights.sum()
    return pd.Series(weights, index=graph.asset_names, name="weight")


# ============================================================================
# Diagnostics (E3/E6 inputs)
# ============================================================================
def dag_diagnostics(graph: AssetGraphWindow) -> dict:
    """Edge density, DAG depth (longest path), roots/leaves — per window."""
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
    """Kendall's τ between consecutive-window topological orders (E6)."""
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
