"""Integration test: the ablation isolates edge direction.

Controls (D0, D0s, D0lw, D0df) must be transpose-invariant; the
direction-aware allocators must respond to edge reversal. A
graph-sensitivity check guards against allocators ignoring the graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.discovery.asset_graph import AssetGraphWindow
from pipeline.portfolio.directed import dispatch_allocator

N = 8
NAMES = [f"A{i}" for i in range(N)]


def _chainlike_dag(seed=13):
    """Asymmetric DAG with real depth; the node sequence is seed-permuted,
    so different seeds give structurally different graphs."""
    rng = np.random.default_rng(seed)
    seq = rng.permutation(N)
    M = np.zeros((N, N))
    for a in range(N - 1):
        M[seq[a], seq[a + 1]] = rng.uniform(0.4, 0.9)  # backbone chain
    M[seq[0], seq[3]], M[seq[1], seq[5]], M[seq[2], seq[7]] = 0.5, 0.35, 0.45
    return M


def _graph(M, is_dag=True):
    return AssetGraphWindow(
        end_date=pd.Timestamp("2020-06-30"),
        asset_names=NAMES,
        M=np.asarray(M, dtype=float),
        zscore_std=np.linspace(0.8, 1.4, N),
        resid_var_z=np.linspace(0.5, 1.5, N),
        method="dynotears",
        tau=0.0,
        is_dag=is_dag,
    )


def _returns(seed=3, T=260):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=T)
    return pd.DataFrame(
        rng.standard_normal((T, N)) * 0.01, index=idx, columns=NAMES
    )


def test_direction_is_live_d0_vs_d2_differ_on_same_graph():
    g = _graph(_chainlike_dag())
    rets = _returns()
    w0 = dispatch_allocator("D0", g, rets)
    w2 = dispatch_allocator("D2", g, rets)
    assert not np.allclose(w0.to_numpy(), w2.to_numpy())


@pytest.mark.parametrize("name", ["D0", "D0s", "D0lw", "D0df"])
def test_controls_are_transpose_invariant(name):
    M = _chainlike_dag()
    rets = _returns()
    w_fwd = dispatch_allocator(name, _graph(M), rets)
    w_rev = dispatch_allocator(name, _graph(M.T), rets)
    np.testing.assert_allclose(w_fwd.to_numpy(), w_rev.to_numpy(), atol=1e-12)


@pytest.mark.parametrize("name", ["D1", "D2s", "D3", "D4"])
def test_treatments_respond_to_edge_reversal(name):
    M = _chainlike_dag()
    rets = _returns()
    w_fwd = dispatch_allocator(name, _graph(M), rets)
    w_rev = dispatch_allocator(name, _graph(M.T), rets)
    assert not np.allclose(w_fwd.to_numpy(), w_rev.to_numpy(), atol=1e-10), (
        f"{name} must not be blind to edge direction"
    )


def test_d2_responds_to_topological_structure():
    """D2 skips the reversal test (recursive bisection is mirror-invariant),
    but a star vs a chain must move its weights."""
    rets = _returns()
    chain = _chainlike_dag(seed=13)
    star = np.zeros((N, N))
    for j in range(N):
        if j != 3:
            star[3, j] = 0.6
    w_chain = dispatch_allocator("D2", _graph(chain), rets)
    w_star = dispatch_allocator("D2", _graph(star), rets)
    assert not np.allclose(w_chain.to_numpy(), w_star.to_numpy(), atol=1e-10)


@pytest.mark.parametrize("name", ["D0", "D1", "D2", "D3", "D4"])
def test_graph_sensitivity_leak_canary(name):
    """Different graphs must yield different weights."""
    rets = _returns()
    w_a = dispatch_allocator(name, _graph(_chainlike_dag(seed=13)), rets)
    w_b = dispatch_allocator(name, _graph(_chainlike_dag(seed=99)), rets)
    assert not np.allclose(w_a.to_numpy(), w_b.to_numpy(), atol=1e-10)
