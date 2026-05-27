"""Unit tests for Phase H — K calibration runtime + multiple-comparisons fix.

Three tests covering the three pieces of Phase H:

* **t1**: ``benjamini_hochberg_K_perm`` recovers the planted-signal count on
  synthetic data with 5 high-scoring drivers and 30 noise drivers. The
  legacy max-of-d statistic finds 0 signal — that's the bug we're fixing.
* **t2**: capping DYNOTEARS ``max_iter`` on permuted (shuffled-driver) fits
  leaves the null score distribution stable to within a small KS-stat —
  confirming the runtime fix doesn't shift the noise floor.
* **t3**: ``permutation_null_threshold`` with ``n_jobs=2`` produces an
  identical null matrix to ``n_jobs=1`` given the same RNG seed —
  parallelisation is deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.factor_selection.k_calibration import (
    benjamini_hochberg_K_perm,
    permutation_null_threshold,
)


# ============================================================================
# t1 — BH-FDR recovers planted signal where max-of-d fails
# ============================================================================
def test_h1_bh_zscore_recovers_signal_at_thesis_scale():
    """5 real drivers at z ≈ 4 + 30 noise drivers; B=100 (thesis B), d=35.

    BH-FDR with z-score p-values should flag the 5 planted signals.
    The legacy "max-of-d" statistic and MC-mode BH should both fail
    because of the small-B / large-d underpowering — that's the point of
    switching to the parametric path.
    """
    rng = np.random.default_rng(seed=0)
    d = 35
    n_signal = 5
    B = 100  # thesis-realistic; the underpowering claim only matters at this scale

    # Null: every driver's score ~ N(0, 1).
    null_per_driver = rng.standard_normal(size=(B, d))

    # Real: first n_signal drivers at score = 4 (well above noise), rest noise.
    real_scores = np.concatenate([
        np.full(n_signal, 4.0),
        rng.standard_normal(d - n_signal),
    ])

    # BH-FDR via parametric z-score (default).
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

    # MC mode at B=100, d=35 is underpowered: MC p-floor = 1/101 ≈ 0.0099,
    # BH threshold for rank 1 at α=0.05 is 0.05/35 ≈ 0.0014. So MC-BH
    # cannot flag anything regardless of signal strength.
    K_mc, _, mask_mc = benjamini_hochberg_K_perm(
        real_scores, null_per_driver, alpha=0.05, method="mc",
    )
    assert K_mc == 0, (
        f"MC-mode BH expected to fail-closed at B={B}, d={d}, α=0.05 "
        f"(underpowered); got K_mc={K_mc}. If this passes, MC mode's "
        f"discreteness floor has changed — check the +1 adjustments."
    )

    # The z-score path is the operational test. We do not assert
    # "z-score > legacy" in a synthetic — whether legacy keeps up
    # depends sensitively on signal magnitude relative to the null
    # max-of-d 95th percentile. The empirical comparison happens in
    # Phase H.6 (re-running G.5's burn-in calibration on real data,
    # where the legacy method scored K_perm = 0 due to weak signal).


# ============================================================================
# t2 — max_iter cap doesn't shift the score distribution
# ============================================================================
def test_h2_permuted_max_iter_cap_preserves_distribution():
    """Tiny DYNOTEARS smoke test: capping ``max_iter`` to 5 vs 50 on a
    shuffled-driver window produces score distributions within a small
    KS-stat. Permuted fits don't converge regardless, so the cap loses
    no statistical signal.

    Uses a small fixture to keep test time under ~10 seconds.
    """
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
        """Fit DYNOTEARS on a shuffled-driver window and return Stage A scores."""
        local_rng = np.random.default_rng(seed)
        shuffled = joint.frame.copy()
        for c in joint.driver_columns:
            shuffled[c] = shuffled[c].to_numpy()[local_rng.permutation(T)]
        disc = run_dynotears_joint_window(
            shuffled, joint.driver_columns, joint.asset_columns,
            p=1, max_iter=max_iter, w_threshold=0.01,
        )
        return stage_a_score(disc).scores.to_numpy()

    # Run B=8 permuted fits at each cap level (tiny B for test speed).
    B = 8
    seeds = list(range(100, 100 + B))
    scores_low_cap = np.array([fit_score(s, max_iter=5) for s in seeds])
    scores_hi_cap = np.array([fit_score(s, max_iter=50) for s in seeds])

    # Flatten and compare the two score distributions.
    flat_low = scores_low_cap.flatten()
    flat_hi = scores_hi_cap.flatten()

    # KS statistic — purely descriptive, just want them similar.
    from scipy import stats
    ks_stat, _p = stats.ks_2samp(flat_low, flat_hi)
    assert ks_stat < 0.25, (
        f"max_iter cap shifted the permuted score distribution: KS={ks_stat:.3f}. "
        f"Either the cap is too aggressive or the distributions differ "
        f"materially. Investigate before relying on the runtime fix."
    )


# ============================================================================
# t3 — n_jobs parallelisation is deterministic
# ============================================================================
def test_h3_n_jobs_determinism():
    """With the same rng_seed, ``n_jobs=1`` and ``n_jobs=2`` produce
    identical null_per_driver matrices. The pool of seeds is drawn up-front
    in the orchestrator, so each permutation is independent and gets the
    same seed regardless of execution order."""
    d = 12
    B = 20

    # Closure: take a seed, return deterministic per-driver scores.
    def fake_fit(seed: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal(d)

    # n_jobs=1 (serial)
    _, _, null_serial = permutation_null_threshold(
        fake_fit, n_permutations=B, rng=np.random.default_rng(99), n_jobs=1,
    )

    # n_jobs=2 (joblib)
    try:
        _, _, null_parallel = permutation_null_threshold(
            fake_fit, n_permutations=B, rng=np.random.default_rng(99), n_jobs=2,
        )
    except ImportError:
        pytest.skip("joblib not installed")

    # Same RNG seed → same up-front seed pool → same fits → identical results.
    np.testing.assert_array_equal(null_serial, null_parallel)
