"""Unit tests for causal-ordered bisection (topological.py).

Property-tests the topological order, the deterministic tie-break, the
cycle branch (raise + feedback-arc fallback), and the DAG diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.discovery.asset_graph import AssetGraphWindow
from pipeline.portfolio.topological import (
    CyclicGraphError,
    d2_weights,
    dag_diagnostics,
    remove_feedback_arcs,
    topological_order,
)


def _random_dag(N, rng, density=0.3):
    order = rng.permutation(N)
    M = np.zeros((N, N))
    for a in range(N):
        for b in range(a + 1, N):
            if rng.random() < density:
                M[order[a], order[b]] = rng.uniform(0.1, 1.0)
    return M


def _graph(M, names=None, is_dag=True):
    N = M.shape[0]
    names = names or [f"A{i:02d}" for i in range(N)]
    return AssetGraphWindow(
        end_date=pd.Timestamp("2020-06-30"),
        asset_names=names,
        M=np.asarray(M, dtype=float),
        zscore_std=np.ones(N),
        resid_var_z=np.ones(N),
        method="dynotears",
        tau=0.0,
        is_dag=is_dag,
    )


# topological_order
def test_order_respects_all_edges_on_random_dags():
    rng = np.random.default_rng(101)
    for _ in range(100):
        N = int(rng.integers(3, 15))
        M = _random_dag(N, rng)
        order = topological_order(M)
        posn = {node: k for k, node in enumerate(order)}
        for i, j in zip(*np.nonzero(M)):
            assert posn[i] < posn[j], "edge i→j must place i before j"


def test_order_deterministic_under_relabelling():
    rng = np.random.default_rng(55)
    N = 10
    M = _random_dag(N, rng)
    names = [f"A{i:02d}" for i in range(N)]
    base = [names[i] for i in topological_order(M, names)]

    perm = rng.permutation(N)
    M_perm = M[np.ix_(perm, perm)]
    names_perm = [names[i] for i in perm]
    relabelled = [names_perm[i] for i in topological_order(M_perm, names_perm)]
    assert relabelled == base


def test_cycle_raises():
    M = np.zeros((3, 3))
    M[0, 1], M[1, 2], M[2, 0] = 0.5, 0.5, 0.5
    with pytest.raises(CyclicGraphError):
        topological_order(M)


# remove_feedback_arcs
def test_feedback_arc_removal_drops_minimum_edge_on_3cycle():
    M = np.zeros((3, 3))
    M[0, 1], M[1, 2], M[2, 0] = 0.9, 0.5, 0.1  # weakest closes the cycle
    cleaned, dropped = remove_feedback_arcs(M)
    assert dropped == 1
    assert cleaned[2, 0] == 0.0
    assert cleaned[0, 1] == 0.9 and cleaned[1, 2] == 0.5
    topological_order(cleaned)  # now sortable


# d2_weights
def test_d2_valid_weights_and_both_covariances():
    rng = np.random.default_rng(77)
    g = _graph(_random_dag(10, rng))
    idx = pd.bdate_range("2019-01-02", periods=260)
    rets = pd.DataFrame(
        rng.standard_normal((260, 10)) * 0.01, index=idx, columns=g.asset_names
    )
    for cov_kind in ("sample", "structural"):
        w = d2_weights(g, rets, covariance=cov_kind)
        assert list(w.index) == g.asset_names
        assert np.all(w.to_numpy() >= -1e-12)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_d2_handles_non_dag_via_feedback_arc_fallback():
    M = np.zeros((4, 4))
    M[0, 1], M[1, 2], M[2, 0], M[0, 3] = 0.6, 0.5, 0.05, 0.4
    g = _graph(M, is_dag=False)
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-02", periods=200)
    rets = pd.DataFrame(
        rng.standard_normal((200, 4)) * 0.01, index=idx, columns=g.asset_names
    )
    w = d2_weights(g, rets)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


# dag_diagnostics
def test_dag_depth_on_a_chain():
    N = 6
    M = np.zeros((N, N))
    for i in range(N - 1):
        M[i, i + 1] = 0.5
    diag = dag_diagnostics(_graph(M))
    assert diag["dag_depth"] == N - 1
    assert diag["n_roots"] == 1 and diag["n_leaves"] == 1
    assert diag["n_edges"] == N - 1
