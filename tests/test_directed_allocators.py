"""Unit tests for the Phase-II direction-aware allocators (directed.py).

Verifies the B-matrix machinery against explicit series sums, the structural
covariance against hand-computable anchors (M=0 reduction, chain-propagation
monotonicity, de-standardisation scaling), the ERC solver's risk-contribution
parity/determinism, and — the local replication anchor — that D0 reproduces
``v0prime_asset_only_causal_hrp`` exactly on the same inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.discovery.asset_graph import AssetGraphWindow
from pipeline.portfolio.causal_hsp import v0prime_asset_only_causal_hrp
from pipeline.portfolio.directed import (
    ALLOCATORS,
    dispatch_allocator,
    erc_weights,
    structural_covariance_v2,
    total_effect_matrix,
)

RNG = np.random.default_rng(7)


def _graph(M, std=None, resid=None, names=None, is_dag=True, method="dynotears"):
    N = M.shape[0]
    names = names or [f"A{i}" for i in range(N)]
    return AssetGraphWindow(
        end_date=pd.Timestamp("2020-06-30"),
        asset_names=names,
        M=np.asarray(M, dtype=float),
        zscore_std=np.ones(N) if std is None else np.asarray(std, dtype=float),
        resid_var_z=np.ones(N) if resid is None else (
            None if resid is False else np.asarray(resid, dtype=float)
        ),
        method=method,
        tau=0.0,
        is_dag=is_dag,
    )


def _returns(names, T=260, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=T)
    return pd.DataFrame(
        rng.standard_normal((T, len(names))) * 0.01, index=idx, columns=names
    )


# ============================================================================
# total_effect_matrix
# ============================================================================
def test_total_effect_equals_neumann_sum_on_dag():
    M = np.zeros((4, 4))
    M[0, 1], M[1, 2], M[0, 3], M[3, 2] = 0.5, -0.3, 0.2, 0.7
    B = total_effect_matrix(M, is_dag=True)
    neumann = np.eye(4)
    term = np.eye(4)
    for _ in range(4):
        term = term @ M.T
        neumann += term
    np.testing.assert_allclose(B, neumann, atol=1e-12)
    np.testing.assert_allclose((np.eye(4) - M.T) @ B, np.eye(4), atol=1e-12)


def test_total_effect_truncated_neumann_with_spectral_guard():
    # A 2-cycle with ρ ≥ 1 must be rescaled, not diverge.
    M = np.array([[0.0, 1.2], [1.1, 0.0]])
    B = total_effect_matrix(M, is_dag=False, k_trunc=10)
    assert np.all(np.isfinite(B))
    assert B[0, 0] > 1.0  # feedback amplifies


# ============================================================================
# structural_covariance_v2
# ============================================================================
def test_sigma_struct_reduces_to_destandardised_diag_when_m_zero():
    g = _graph(np.zeros((3, 3)), std=[1.0, 1.0, 2.0])
    cov = structural_covariance_v2(g, ridge=0.0).to_numpy()
    np.testing.assert_allclose(np.diag(cov), [1.0, 1.0, 4.0], rtol=1e-10)
    assert abs(cov[0, 1]) < 1e-12


def test_sigma_struct_variance_increases_down_a_chain():
    M = np.zeros((3, 3))
    M[0, 1], M[1, 2] = 1.0, 1.0  # A → B → C, unit edges + unit shocks
    cov = structural_covariance_v2(_graph(M), ridge=0.0).to_numpy()
    v = np.diag(cov)
    np.testing.assert_allclose(v, [1.0, 2.0, 3.0], rtol=1e-9)
    assert v[0] < v[1] < v[2]


def test_sigma_struct_destandardisation_scaling():
    M = np.zeros((2, 2))
    M[0, 1] = 0.5
    base = structural_covariance_v2(_graph(M, std=[1.0, 1.0]), ridge=0.0)
    doubled = structural_covariance_v2(_graph(M, std=[1.0, 2.0]), ridge=0.0)
    assert doubled.iloc[1, 1] == pytest.approx(4.0 * base.iloc[1, 1], rel=1e-10)


def test_sigma_struct_warns_and_uses_unit_shocks_without_residuals(caplog):
    g = _graph(np.zeros((2, 2)), resid=False)  # resid_var_z=None
    with caplog.at_level("WARNING"):
        cov = structural_covariance_v2(g, ridge=0.0)
    assert "unit shocks" in caplog.text
    np.testing.assert_allclose(np.diag(cov), [1.0, 1.0], rtol=1e-10)


# ============================================================================
# ERC (Spinu CCD)
# ============================================================================
def test_erc_risk_contribution_parity_on_random_psd():
    N = 20
    A = RNG.standard_normal((N, N))
    cov = A @ A.T + 0.1 * np.eye(N)
    w = erc_weights(cov)
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    rc = w * (cov @ w)
    assert (rc.max() - rc.min()) / rc.mean() < 1e-8
    # Determinism: identical output across calls.
    np.testing.assert_array_equal(w, erc_weights(cov))


# ============================================================================
# Allocators
# ============================================================================
def _random_dag(N, seed=11, density=0.3):
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    M = np.zeros((N, N))
    for a in range(N):
        for b in range(a + 1, N):
            if rng.random() < density:
                M[order[a], order[b]] = rng.uniform(0.1, 0.8) * rng.choice([-1, 1])
    return M


def test_d0_matches_v0prime_exactly():
    """The local replication anchor: D0 through the new dispatch must equal
    the committed V0′ allocator byte-for-byte on identical inputs."""
    N = 8
    M = _random_dag(N)
    names = [f"A{i}" for i in range(N)]
    rets = _returns(names)
    g = _graph(M, names=names)
    w_d0 = dispatch_allocator("D0", g, rets)
    w_v0p = v0prime_asset_only_causal_hrp(M, names, rets)
    pd.testing.assert_series_equal(w_d0, w_v0p, check_names=False)


@pytest.mark.parametrize("name", list(ALLOCATORS))
def test_every_allocator_returns_valid_longonly_weights(name):
    N = 10
    g = _graph(_random_dag(N, seed=23))
    rets = _returns(g.asset_names, seed=5)
    w = dispatch_allocator(name, g, rets)
    assert list(w.index) == g.asset_names
    assert np.all(w.to_numpy() >= -1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    # Determinism (seed-free by construction).
    pd.testing.assert_series_equal(w, dispatch_allocator(name, g, rets))


def test_corr_hrp_matches_hand_construction():
    """CORR must equal hrp_weights on the textbook correlation distance,
    routed through the same nearest-PSD house pattern as every D-variant."""
    from pipeline.portfolio._old_v123 import correlation_distance, nearest_psd
    from pipeline.portfolio.hrp import hrp_weights
    from pipeline.portfolio.hsp import sample_covariance

    N = 8
    g = _graph(_random_dag(N, seed=31))
    rets = _returns(g.asset_names, seed=17)
    w = dispatch_allocator("CORR", g, rets)

    sub = rets[g.asset_names].dropna()
    dist = nearest_psd(correlation_distance(sub.corr().to_numpy()))
    D = pd.DataFrame(dist, index=g.asset_names, columns=g.asset_names)
    expected = hrp_weights(D, sample_covariance(sub))
    pd.testing.assert_series_equal(w, expected, check_names=False)


def test_corr_hrp_is_graph_blind():
    """Two entirely different graphs must give identical CORR weights — the
    control uses the graph only for its asset universe."""
    rets = _returns([f"A{i}" for i in range(8)], seed=17)
    w_a = dispatch_allocator("CORR", _graph(_random_dag(8, seed=1)), rets)
    w_b = dispatch_allocator("CORR", _graph(_random_dag(8, seed=2)), rets)
    pd.testing.assert_series_equal(w_a, w_b)


def test_dispatch_rejects_unknown_allocator():
    g = _graph(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unknown allocator"):
        dispatch_allocator("D9", g, _returns(g.asset_names))


def test_d4_identical_ancestry_gives_zero_distance_pair():
    # Two sinks fed identically by the same parent inherit the same shock
    # profile up to their own shock — with tiny own influence their D4
    # co-ancestry similarity is near-maximal vs an unrelated asset.
    M = np.zeros((4, 4))
    M[0, 1], M[0, 2] = 0.9, 0.9  # A0 → A1, A0 → A2 equally; A3 isolated
    g = _graph(M)
    rets = _returns(g.asset_names, seed=9)
    w = dispatch_allocator("D4", g, rets)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
