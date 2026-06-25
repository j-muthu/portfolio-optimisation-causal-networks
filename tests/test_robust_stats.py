"""Unit tests for the robust-stats battery (PSR / DSR + the report script).

These exercise the *math* on synthetic data — the real run reads the gitignored
``closed_loop.pkl`` bundles, so the bundle-reading path is not covered here.

* **PSR**: a probability in (0,1); monotone in sample length ``T`` and in the
  Sharpe estimate; pinned to a regression value on a fixed RNG draw.
* **DSR**: collapses to ``PSR(SR*=0)`` for a single trial; is ``<= PSR(0)`` once
  more than one trial is deflated against; the expected-max benchmark is positive
  for ``N>1`` and zero for ``N<=1``.
* **robust_stats** pure functions: the PSR/DSR table has the right shape and a
  self-against-self ``psr_vs_baseline`` of 0.5; the measurement-problem helper
  returns the reward SNR and the 21-day Sharpe SE; the hand-rolled Reality Check
  returns a valid p-value; and (if ``arch`` is installed) SPA/MCS run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from scripts import robust_stats as rs


# ============================================================================
# PSR
# ============================================================================
def test_psr_is_probability_and_pinned():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    psr = probabilistic_sharpe_ratio(r, 0.0)
    assert 0.0 < psr < 1.0
    # Regression pin (same draw) — guards against formula drift.
    assert psr == pytest.approx(0.675943, abs=1e-6)


def test_psr_monotone_in_T_and_in_sharpe():
    rng = np.random.default_rng(0)
    short = probabilistic_sharpe_ratio(pd.Series(rng.normal(5e-4, 1e-2, 400)), 0.0)
    long = probabilistic_sharpe_ratio(pd.Series(rng.normal(5e-4, 1e-2, 8000)), 0.0)
    assert long > short  # more data -> more confident a positive Sharpe is real
    lo = probabilistic_sharpe_ratio(pd.Series(rng.normal(2e-4, 1e-2, 3000)), 0.0)
    hi = probabilistic_sharpe_ratio(pd.Series(rng.normal(9e-4, 1e-2, 3000)), 0.0)
    assert hi > lo  # higher Sharpe -> higher PSR


# ============================================================================
# DSR
# ============================================================================
def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(0.01, 1) == 0.0      # single trial: no deflation
    assert expected_max_sharpe(0.0, 50) == 0.0      # no cross-trial variance
    assert expected_max_sharpe(0.01, 50) > expected_max_sharpe(0.01, 5) > 0.0


def test_dsr_collapses_to_psr_for_single_trial_and_deflates_for_many():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    psr0 = probabilistic_sharpe_ratio(r, 0.0)
    pp = float(r.mean() / r.std(ddof=0))
    # one trial == its own per-period Sharpe -> SR* = 0 -> DSR == PSR(0)
    assert deflated_sharpe_ratio(r, [pp]) == pytest.approx(psr0, abs=1e-9)
    # many trials -> SR* > 0 -> DSR strictly below PSR(0)
    trials = list(np.linspace(pp * 0.8, pp * 1.2, 41))
    assert deflated_sharpe_ratio(r, trials) <= psr0


# ============================================================================
# robust_stats pure functions
# ============================================================================
def _synthetic_matrix(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    names = ["V0_w252", "V0prime_w252", "V1-DYNOTEARS_w252",
             "V2-DYNOTEARS_w252", "V1-VARLiNGAM_w252", "V2-VARLiNGAM_w252"]
    mus = [3e-4, 6e-4, 5e-4, 5e-4, 4e-4, 4e-4]
    return pd.DataFrame({n: rng.normal(mu, 1e-2, 1500) for n, mu in zip(names, mus)},
                        index=idx)


def test_psr_dsr_table_shape_and_self_baseline():
    df = _synthetic_matrix()
    tbl = rs.psr_dsr_table(df, baseline="V0_w252")
    assert len(tbl) == df.shape[1]
    assert {"config", "sharpe_ann", "psr_vs_zero", "psr_vs_baseline", "dsr",
            "n_trials"}.issubset(tbl.columns)
    # PSR of the baseline against the baseline's own Sharpe is Phi(0) = 0.5.
    base = tbl.loc[tbl.config == "V0_w252", "psr_vs_baseline"].iloc[0]
    assert base == pytest.approx(0.5, abs=1e-6)
    assert (tbl.n_trials == df.shape[1]).all()


def test_measurement_problem_reports_snr_and_window_se():
    rng = np.random.default_rng(3)
    reward = pd.Series(rng.normal(0.1, 0.22, 215))
    mp = rs.measurement_problem(reward, holding_days=21)
    assert mp["n_rebalances"] == 215
    assert mp["reward_snr"] == pytest.approx(abs(mp["reward_mean"]) / mp["reward_std"])
    assert mp["sharpe_se_window"] == pytest.approx(np.sqrt(1 / 21), abs=1e-6)


def test_handrolled_reality_check_is_a_pvalue():
    df = _synthetic_matrix()
    causal = [c for c in df.columns if not c.startswith("V0_")]
    p = rs._handrolled_reality_check(df, "V0_w252", causal, block_size=21,
                                     reps=200, seed=42)
    assert 0.0 < p <= 1.0


def test_spa_and_mcs_run_when_arch_present():
    pytest.importorskip("arch")
    df = _synthetic_matrix()
    causal = [c for c in df.columns if not c.startswith("V0_")]
    spa = rs.run_spa(df, "V0_w252", causal, reps=200)
    assert spa["engine"] == "arch"
    assert 0.0 <= spa["rc_lower"] <= 1.0
    included, _ = rs.run_mcs(df, list(df.columns), reps=200)
    assert isinstance(included, list) and len(included) >= 1
    assert set(included).issubset(set(df.columns))
