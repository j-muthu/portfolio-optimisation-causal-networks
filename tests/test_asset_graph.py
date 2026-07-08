"""Unit tests for the Phase-II asset-graph chokepoint (asset_graph.py).

Round-trips fabricated DYNOTEARS/VARLiNGAM joint windows through
``asset_graph_from_discovery`` and pins down the invariants the fixed-graph
ablation rests on: M equals the asset–asset block after τ-thresholding, the
asset ordering matches the panel, universe slicing drops rows and columns
together, residual variances are hand-verifiable, and the VARLiNGAM ``B0``
``i → j`` convention is consumed without a second transpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.discovery.asset_graph import (
    asset_graph_from_discovery,
    is_dag_matrix,
)
from pipeline.discovery.dynotears import JointDynotearsWindow
from pipeline.discovery.varlingam import JointVarLingamWindow

DRIVERS = ["D0", "D1"]
ASSETS = ["AAA", "BBB", "CCC", "DDD"]
COLUMNS = DRIVERS + ASSETS


def _joint_window(T=120, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=T)
    return pd.DataFrame(
        rng.standard_normal((T, len(COLUMNS))), index=idx, columns=COLUMNS
    )


def _asset_block():
    A = np.zeros((4, 4))
    A[0, 1], A[1, 2], A[0, 3] = 0.5, -0.3, 0.02  # AAA→BBB, BBB→CCC (neg), AAA→DDD (small)
    return A


def _dyno_window(jw):
    d = len(COLUMNS)
    W = np.zeros((d, d))
    aidx = np.arange(2, 6)
    W[np.ix_(aidx, aidx)] = _asset_block()
    W[0, 2] = 0.4  # a driver→asset edge, outside the asset block
    mean = jw.mean(axis=0)
    std = jw.std(axis=0, ddof=0)
    return JointDynotearsWindow(
        index=0, start_row=0, end_row=len(jw),
        start_date=jw.index[0], end_date=jw.index[-1],
        columns=COLUMNS, driver_columns=DRIVERS, asset_columns=ASSETS,
        driver_idx=np.arange(0, 2), asset_idx=aidx,
        W=W, A=[np.zeros((d, d))], p=1,
        lambda_w=0.05, lambda_a=0.05, converged=True,
        acyclic_edges_removed=0,
        zscore_mean=mean.to_numpy(), zscore_std=std.to_numpy(),
        fit_loss=0.0, tabu_enforced=True,
    )


# ============================================================================
# Round-trip + τ + ordering
# ============================================================================
def test_roundtrip_m_equals_asset_block():
    jw = _joint_window()
    disc = _dyno_window(jw)
    g = asset_graph_from_discovery(disc, jw, method="dynotears")
    np.testing.assert_array_equal(g.M, _asset_block())
    assert g.asset_names == ASSETS
    assert g.is_dag
    # Driver→asset edges must never leak into the asset graph.
    assert g.M.shape == (4, 4)


def test_tau_threshold_zeroes_small_edges():
    jw = _joint_window()
    g = asset_graph_from_discovery(_dyno_window(jw), jw, method="dynotears", tau=0.1)
    expected = _asset_block()
    expected[0, 3] = 0.0  # |0.02| < 0.1
    np.testing.assert_array_equal(g.M, expected)
    assert g.meta["n_edges"] == 2


def test_universe_slicing_drops_rows_and_columns_together():
    jw = _joint_window()
    g = asset_graph_from_discovery(
        _dyno_window(jw), jw, method="dynotears", universe=["DDD", "AAA", "BBB"],
    )
    # Caller's order preserved; CCC gone from names, M, stats alike.
    assert g.asset_names == ["DDD", "AAA", "BBB"]
    A = _asset_block()
    expected = A[np.ix_([3, 0, 1], [3, 0, 1])]
    np.testing.assert_array_equal(g.M, expected)
    assert g.zscore_std.shape == (3,)
    assert g.resid_var_z.shape == (3,)


def test_universe_too_small_raises():
    jw = _joint_window()
    with pytest.raises(ValueError, match="≥2 assets"):
        asset_graph_from_discovery(
            _dyno_window(jw), jw, method="dynotears", universe=["AAA"],
        )


# ============================================================================
# Residual variances
# ============================================================================
def test_resid_var_is_unit_when_m_zero():
    jw = _joint_window()
    disc = _dyno_window(jw)
    disc.W[:] = 0.0
    g = asset_graph_from_discovery(disc, jw, method="dynotears")
    # X_z built with the stored (ddof=0) stats has exactly unit variance,
    # and with M=0 the residuals are X_z itself.
    np.testing.assert_allclose(g.resid_var_z, np.ones(4), rtol=1e-10)


def test_resid_var_matches_hand_computation():
    jw = _joint_window(seed=42)
    disc = _dyno_window(jw)
    g = asset_graph_from_discovery(disc, jw, method="dynotears")
    X = jw[ASSETS].to_numpy()
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    X_z = (X - mean) / std
    E = X_z @ (np.eye(4) - _asset_block())
    np.testing.assert_allclose(g.resid_var_z, E.var(axis=0, ddof=0), rtol=1e-10)


def test_no_fit_window_gives_none_residuals():
    jw = _joint_window()
    g = asset_graph_from_discovery(_dyno_window(jw), None, method="dynotears")
    assert g.resid_var_z is None


# ============================================================================
# VARLiNGAM convention (B0 already i → j — no second transpose)
# ============================================================================
def test_varlingam_asset_block_convention():
    jw = _joint_window()
    d = len(COLUMNS)
    B0 = np.zeros((d, d))
    aidx = np.arange(2, 6)
    B0[np.ix_(aidx, aidx)] = _asset_block()
    mean = jw.mean(axis=0)
    std = jw.std(axis=0, ddof=0)
    disc = JointVarLingamWindow(
        index=0, start_row=0, end_row=len(jw),
        start_date=jw.index[0], end_date=jw.index[-1],
        columns=COLUMNS, driver_columns=DRIVERS, asset_columns=ASSETS,
        driver_idx=np.arange(0, 2), asset_idx=aidx,
        B0=B0, B_lags=[np.zeros((d, d))], causal_order=list(range(d)),
        selected_lags=1,
        zscore_mean=mean.to_numpy(), zscore_std=std.to_numpy(),
    )
    # The new accessor mirrors dynotears — same block, no transpose.
    np.testing.assert_array_equal(disc.asset_to_asset_block(0), _asset_block())
    g = asset_graph_from_discovery(disc, jw, method="varlingam")
    np.testing.assert_array_equal(g.M, _asset_block())
    # An asymmetric block must stay asymmetric (a transpose bug would flip it).
    assert g.M[0, 1] == 0.5 and g.M[1, 0] == 0.0


# ============================================================================
# is_dag_matrix
# ============================================================================
def test_is_dag_matrix_detects_cycles():
    M = np.zeros((3, 3))
    M[0, 1], M[1, 2] = 1.0, 1.0
    assert is_dag_matrix(M)
    M[2, 0] = 1.0
    assert not is_dag_matrix(M)
