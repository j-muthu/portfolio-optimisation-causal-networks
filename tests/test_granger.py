"""Unit tests for the ridge-VAR(1) GRANGER comparator (granger.py).

Verifies support recovery on a simulated VAR(1) with known sparse
coefficients, density matching, the zero driver blocks, and the interface
parity with the joint discovery windows that the Phase-II chokepoint
consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.discovery.asset_graph import asset_graph_from_discovery
from pipeline.discovery.granger import run_granger_joint_window

DRIVERS = ["D0", "D1"]
ASSETS = [f"A{i}" for i in range(8)]
COLUMNS = DRIVERS + ASSETS


def _simulated_var1(T=1500, seed=21):
    """VAR(1) asset panel with a known sparse lag matrix (i → j)."""
    rng = np.random.default_rng(seed)
    N = len(ASSETS)
    true = np.zeros((N, N))
    edges = [(0, 1, 0.45), (0, 2, -0.4), (3, 4, 0.5), (5, 6, 0.4), (2, 7, -0.35)]
    for i, j, v in edges:
        true[i, j] = v
    X = np.zeros((T, N))
    for t in range(1, T):
        X[t] = X[t - 1] @ true + rng.standard_normal(N)
    idx = pd.bdate_range("2015-01-02", periods=T)
    frame = pd.DataFrame(X, index=idx, columns=ASSETS)
    frame[DRIVERS] = rng.standard_normal((T, 2))
    return frame[COLUMNS], true, edges


def test_ridge_recovers_planted_support_at_matched_density():
    jw, true, edges = _simulated_var1()
    target = len(edges) / (len(ASSETS) * (len(ASSETS) - 1))
    win = run_granger_joint_window(
        jw, DRIVERS, ASSETS, target_density=target,
    )
    A = win.asset_to_asset_block(0)
    found = set(zip(*np.nonzero(A)))
    planted = {(i, j) for i, j, _ in edges}
    precision = len(found & planted) / max(len(found), 1)
    assert precision > 0.8, f"precision {precision:.2f} on an easy SNR"


def test_density_matching_hits_target():
    jw, _, _ = _simulated_var1(seed=4)
    N = len(ASSETS)
    target = 0.25
    win = run_granger_joint_window(jw, DRIVERS, ASSETS, target_density=target)
    assert win.achieved_density == pytest.approx(target, abs=1.5 / (N * (N - 1)))


def test_driver_blocks_are_zero_and_stats_full_length():
    jw, _, _ = _simulated_var1(seed=6)
    win = run_granger_joint_window(jw, DRIVERS, ASSETS, target_density=0.2)
    assert np.all(win.driver_to_asset_block(0) == 0.0)
    assert np.all(win.asset_to_driver_block(0) == 0.0)
    assert win.zscore_mean.shape == (len(COLUMNS),)
    assert win.zscore_std.shape == (len(COLUMNS),)
    # Magnitudes only, zero diagonal.
    A = win.asset_to_asset_block(0)
    assert np.all(A >= 0.0)
    assert np.all(np.diag(A) == 0.0)


def test_chokepoint_consumes_granger_window():
    jw, _, _ = _simulated_var1(seed=8)
    win = run_granger_joint_window(jw, DRIVERS, ASSETS, target_density=0.2)
    g = asset_graph_from_discovery(win, jw, method="granger_ridge")
    assert g.asset_names == ASSETS
    np.testing.assert_array_equal(g.M, win.asset_to_asset_block(0))
    assert g.resid_var_z is not None and g.resid_var_z.shape == (len(ASSETS),)
    # is_dag recorded honestly (may be either on simulated data).
    assert g.is_dag == win.is_dag
