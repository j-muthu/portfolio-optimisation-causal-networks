"""Unit tests for K calibration (runtime + multiple-comparisons fix).

t1: BH-FDR recovers planted signal; t2: the max_iter cap doesn't shift the
null score distribution; t3: n_jobs parallelisation is deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.factor_selection.k_calibration import (
    benjamini_hochberg_K_perm,
    permutation_null_threshold,
)


# t1: BH-FDR recovers planted signal where max-of-d fails
def test_h1_bh_zscore_recovers_signal_at_thesis_scale():
    """BH-FDR with z-score p-values flags 5 planted signals at B=100, d=35;
    MC-mode BH fails closed at the same scale."""
    rng = np.random.default_rng(seed=0)
    d = 35
    n_signal = 5
    B = 100  # thesis-realistic; underpowering only matters at this scale

    # Null: every driver's score ~ N(0, 1).
    null_per_driver = rng.standard_normal(size=(B, d))

    # First n_signal drivers at score 4, rest noise.
    real_scores = np.concatenate([
        np.full(n_signal, 4.0),
        rng.standard_normal(d - n_signal),
    ])

    K_z, _, mask_z = benjamini_hochberg_K_perm(
        real_scores, null_per_driver, alpha=0.05, method="zscore",
    )
    assert mask_z[:n_signal].all(), (
        f"BH-FDR (z-score) missed planted signals; mask[:5] = {mask_z[:n_signal]}"
    )
    n_fp = int(mask_z[n_signal:].sum())
    assert n_fp <= 3, (
        f"BH-FDR (z-score) flagged {n_fp} false positives (expected ≤ 3 at α=0.05)"
    )
    assert K_z == n_signal + n_fp

    # MC p-floor 1/101 ~ 0.0099 > BH rank-1 threshold 0.05/35 ~ 0.0014,
    # so MC-BH cannot flag anything regardless of signal strength.
    K_mc, _, mask_mc = benjamini_hochberg_K_perm(
        real_scores, null_per_driver, alpha=0.05, method="mc",
    )
    assert K_mc == 0, (
        f"MC-mode BH expected to fail-closed at B={B}, d={d}, α=0.05 "
        f"(underpowered); got K_mc={K_mc}. If this passes, MC mode's "
        f"discreteness floor has changed — check the +1 adjustments."
    )

    # No "z-score > legacy" assertion here: that comparison is empirical
    # (Phase H.6, real data).


# t2: max_iter cap doesn't shift the score distribution
def test_h2_permuted_max_iter_cap_preserves_distribution():
    """Capping max_iter at 5 vs 50 on shuffled-driver fits leaves the score
    distribution within a small KS-stat."""
    from pipeline.data.alignment import build_joint_matrix
    from pipeline.discovery.dynotears import run_dynotears_joint_window
    from pipeline.factor_selection.prune import stage_a_score

    rng = np.random.default_rng(seed=42)
    T, d_drivers, d_assets = 150, 6, 4
    cal = pd.bdate_range("2020-01-02", periods=T)

    drivers = pd.DataFrame(
        rng.standard_normal((T, d_drivers)),
        index=cal, columns=[f"D{i}" for i in range(d_drivers)],
    )
    assets = pd.DataFrame(
        rng.standard_normal((T, d_assets)),
        index=cal, columns=[f"A{i}" for i in range(d_assets)],
    )
    joint = build_joint_matrix(drivers, assets, calendar=cal, drop_na="any")

    def fit_score(seed: int, max_iter: int) -> np.ndarray:
        """Fit DYNOTEARS on a shuffled-driver window, return Stage A scores."""
        local_rng = np.random.default_rng(seed)
        shuffled = joint.frame.copy()
        for c in joint.driver_columns:
            shuffled[c] = shuffled[c].to_numpy()[local_rng.permutation(T)]
        disc = run_dynotears_joint_window(
            shuffled, joint.driver_columns, joint.asset_columns,
            p=1, max_iter=max_iter, w_threshold=0.01,
        )
        return stage_a_score(disc).scores.to_numpy()

    B = 8  # tiny for test speed
    seeds = list(range(100, 100 + B))
    scores_low_cap = np.array([fit_score(s, max_iter=5) for s in seeds])
    scores_hi_cap = np.array([fit_score(s, max_iter=50) for s in seeds])

    flat_low = scores_low_cap.flatten()
    flat_hi = scores_hi_cap.flatten()

    from scipy import stats
    ks_stat, _p = stats.ks_2samp(flat_low, flat_hi)
    assert ks_stat < 0.25, (
        f"max_iter cap shifted the permuted score distribution: KS={ks_stat:.3f}. "
        f"Either the cap is too aggressive or the distributions differ "
        f"materially. Investigate before relying on the runtime fix."
    )


# t3: n_jobs parallelisation is deterministic
def test_h3_n_jobs_determinism():
    """Same seed: n_jobs=1 and n_jobs=2 give identical null matrices
    (the seed pool is drawn up-front, so execution order doesn't matter)."""
    d = 12
    B = 20

    def fake_fit(seed: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal(d)

    _, _, null_serial = permutation_null_threshold(
        fake_fit, n_permutations=B, rng=np.random.default_rng(99), n_jobs=1,
    )

    try:
        _, _, null_parallel = permutation_null_threshold(
            fake_fit, n_permutations=B, rng=np.random.default_rng(99), n_jobs=2,
        )
    except ImportError:
        pytest.skip("joblib not installed")

    np.testing.assert_array_equal(null_serial, null_parallel)
