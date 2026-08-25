"""Integration tests for ``pipeline.closed_loop.run_closed_loop``.

t1: the loop genuinely feeds back; t2: alpha=1 degenerates to the V1
open-loop path; t3: the leak canary actually leaks. Tiny fixture
(4 assets, 6 drivers, 8 rebalances) so the full cycle runs in seconds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Fixture
@pytest.fixture(scope="module")
def synthetic_fixture(tmp_path_factory):
    """Small synthetic joint panel with two planted-signal drivers."""
    rng = np.random.default_rng(seed=11)
    T = 280  # trading days
    asset_cols = [f"A{i}" for i in range(4)]
    driver_cols = [f"d_planted_{i}" for i in range(2)] + [f"d_noise_{i}" for i in range(4)]
    n_drivers = len(driver_cols)

    cal = pd.bdate_range("2020-01-02", periods=T)

    # Two planted drivers share a factor with the assets; the other four are
    # noise. Signal is strong enough for DYNOTEARS to find reliably.
    shared = rng.standard_normal(T) * 1.0
    planted = np.stack([
        0.8 * shared + 0.4 * rng.standard_normal(T),
        0.7 * shared + 0.5 * rng.standard_normal(T),
    ], axis=1)
    noise = rng.standard_normal((T, 4)) * 0.5
    drivers = np.hstack([planted, noise])

    # Assets driven by the shared factor + idiosyncratic noise.
    asset_betas = rng.uniform(0.3, 0.8, size=4)
    assets = np.outer(shared, asset_betas) + 0.5 * rng.standard_normal((T, 4))

    drivers_df = pd.DataFrame(drivers, index=cal, columns=driver_cols)
    assets_df = pd.DataFrame(assets, index=cal, columns=asset_cols)
    joint = pd.concat([drivers_df, assets_df], axis=1)

    # Levels -> returns, scaled to realistic daily magnitudes.
    asset_returns = assets_df.diff().fillna(0.0) * 0.01

    rebalance_dates = pd.DatetimeIndex(
        [cal[120 + 20 * i] for i in range(8)]  # ~monthly spacing
    )

    def universe_at(t):
        return list(asset_cols)

    return {
        "joint_frame": joint,
        "asset_returns": asset_returns,
        "driver_columns": driver_cols,
        "asset_columns": asset_cols,
        "rebalance_dates": rebalance_dates,
        "universe_at": universe_at,
        "tmp_dir": tmp_path_factory.mktemp("closed_loop"),
    }


def _common_kwargs(tmp_dir: Path) -> dict:
    """Shared kwargs that keep per-test runtime small."""
    return dict(
        K=3,
        window_size=100,
        lookback_days=60,
        holding_days=21,
        transaction_cost_bps=0.0,
        discovery_kwargs={
            "p": 1, "lambda_w": 0.05, "lambda_a": 0.05, "w_threshold": 0.01,
        },
        sensitivities_kwargs={
            "depths": (1,),
            "widths": (16,),
            "epochs": 20,
            "seed": 42,
            "use_cache": False,
        },
        output_dir=tmp_dir,
    )


# t1: closed loop genuinely feeds back
def test_t1_closed_loop_feeds_back(synthetic_fixture, caplog):
    """After burn-in, the selector sees a populated utility lookup that respects the lookahead gap."""
    caplog.set_level(logging.WARNING)
    from pipeline.closed_loop import run_closed_loop

    fix = synthetic_fixture
    burn_in = 2

    result = run_closed_loop(
        joint_frame=fix["joint_frame"],
        asset_returns=fix["asset_returns"],
        rebalance_dates=fix["rebalance_dates"],
        universe_at=fix["universe_at"],
        driver_columns=fix["driver_columns"],
        asset_columns=fix["asset_columns"],
        selector_kwargs={"alpha": 0.6, "burn_in_rebalances": burn_in},
        gamma_ema=0.3,
        tag="t1",
        **_common_kwargs(fix["tmp_dir"] / "t1"),
    )

    n = len(fix["rebalance_dates"])
    assert len(result.stage1_cache) == n
    assert len(result.backtest.rebalances) == n

    # Burn-in: alpha forced to 1, no lookup.
    for i in range(burn_in):
        sel = result.stage1_cache[fix["rebalance_dates"][i]].selection
        assert sel.metadata["burn_in_active"] is True, f"rebalance {i} should be in burn-in"
        assert sel.alpha_effective == 1.0
        assert sel.utility_lookup_timestamp is None

    # Post-burn-in: at least one rebalance must have a populated lookup
    # (early ones may find no eligible row yet).
    post_lookups = [
        result.stage1_cache[fix["rebalance_dates"][i]].selection.utility_lookup_timestamp
        for i in range(burn_in, n)
    ]
    populated = [ts for ts in post_lookups if ts is not None]
    assert len(populated) > 0, (
        "Closed loop is not actually closed: no post-burn-in rebalance saw a "
        "populated U lookup. Per-rebalance interleaving is broken."
    )

    # Belt and braces: every populated lookup respects the 21-day gap.
    for i in range(burn_in, n):
        sel = result.stage1_cache[fix["rebalance_dates"][i]].selection
        if sel.utility_lookup_timestamp is None:
            continue
        gap = (fix["rebalance_dates"][i] - sel.utility_lookup_timestamp).days
        assert gap >= 21, (
            f"rebalance {i}: lookup ts {sel.utility_lookup_timestamp.date()} "
            f"only {gap} days before t={fix['rebalance_dates'][i].date()} "
            f"(must be ≥ 21)"
        )


# t2: alpha=1 degenerates to V1 open-loop
def test_t2_alpha_one_matches_v1_openloop(synthetic_fixture):
    """alpha=1 closed-loop weights match a direct V1 open-loop run to 1e-8."""
    from pipeline.closed_loop import run_closed_loop
    from pipeline.stage1_pipeline import run_stage1
    from pipeline.stage2_pipeline import run_stage2

    fix = synthetic_fixture
    kw = _common_kwargs(fix["tmp_dir"] / "t2_closed")
    kw_open = _common_kwargs(fix["tmp_dir"] / "t2_open")

    # Closed-loop with alpha=1: utility read but never weighted in.
    cl = run_closed_loop(
        joint_frame=fix["joint_frame"],
        asset_returns=fix["asset_returns"],
        rebalance_dates=fix["rebalance_dates"],
        universe_at=fix["universe_at"],
        driver_columns=fix["driver_columns"],
        asset_columns=fix["asset_columns"],
        selector_kwargs={"alpha": 1.0, "burn_in_rebalances": 0},
        gamma_ema=0.3,
        tag="t2_closed",
        **kw,
    )

    s1 = run_stage1(
        joint_frame=fix["joint_frame"],
        driver_columns=fix["driver_columns"],
        asset_columns=fix["asset_columns"],
        rebalance_dates=fix["rebalance_dates"],
        window_size=kw_open["window_size"],
        K=kw_open["K"],
        discovery_kwargs=kw_open["discovery_kwargs"],
        sensitivities_kwargs=kw_open["sensitivities_kwargs"],
        selector_kwargs={"alpha": 1.0, "burn_in_rebalances": 0},
        tag="t2_open",
        output_dir=kw_open["output_dir"],
    )
    s2 = run_stage2(
        stage1=s1,
        asset_returns=fix["asset_returns"],
        universe_at=fix["universe_at"],
        variants=["V1"],
        linkage_method="single",
        lookback_days=kw_open["lookback_days"],
        holding_days=kw_open["holding_days"],
        transaction_cost_bps=kw_open["transaction_cost_bps"],
        gamma_ema=0.3,
        bootstrap_resamples=0,  # skip bootstrap for speed
        tag="t2_open",
        output_dir=kw_open["output_dir"],
    )

    v1_recs = s2.variants["V1"].backtest.rebalances
    cl_recs = cl.backtest.rebalances
    assert len(v1_recs) == len(cl_recs)

    for cl_rec, v1_rec in zip(cl_recs, v1_recs):
        assert cl_rec.rebalance_date == v1_rec.rebalance_date
        # Pad to the union universe before comparing.
        union = sorted(set(cl_rec.weights.index) | set(v1_rec.weights.index))
        w_cl = cl_rec.weights.reindex(union).fillna(0.0).to_numpy()
        w_v1 = v1_rec.weights.reindex(union).fillna(0.0).to_numpy()
        max_diff = float(np.max(np.abs(w_cl - w_v1)))
        assert max_diff < 1e-8, (
            f"α=1 degeneracy broken at {cl_rec.rebalance_date.date()}: "
            f"closed-loop and V1 weights differ by {max_diff:.2e}"
        )


# t3: leak canary actually fires
def test_t3_leak_canary_fires(synthetic_fixture):
    """The leaky lookup returns future U rows the safe lookup hides.

    Verified at the lookup layer, not the weights layer: on a small
    fixture the downstream selection can be sticky enough that weights
    coincide even when the lookups differ.
    """
    from pipeline.closed_loop import run_closed_loop
    from pipeline.feedback import UtilityStore
    from pipeline.feedback.leak_canary import leaky_lookup, make_leaky_lookup

    fix = synthetic_fixture

    # Normal closed-loop pass to populate a UtilityStore.
    store = UtilityStore.load_or_empty(fix["tmp_dir"] / "t3_safe" / "u.parquet")
    run_closed_loop(
        joint_frame=fix["joint_frame"],
        asset_returns=fix["asset_returns"],
        rebalance_dates=fix["rebalance_dates"],
        universe_at=fix["universe_at"],
        driver_columns=fix["driver_columns"],
        asset_columns=fix["asset_columns"],
        selector_kwargs={"alpha": 0.2, "burn_in_rebalances": 1},
        gamma_ema=0.5,
        utility_store=store,
        tag="t3_safe",
        **_common_kwargs(fix["tmp_dir"] / "t3_safe"),
    )
    assert not store.frame.empty, "fixture didn't produce any U rows"

    # Probe both lookups at every rebalance; at least one must differ.
    n_rows_seen_diff = 0
    leak_examples = []
    for t in fix["rebalance_dates"]:
        u_safe, ts_safe = store.lookup_utility(t, require_strict=False)
        u_leaky, ts_leaky = leaky_lookup(store, t, peek_ahead_days=21)
        if ts_safe != ts_leaky:
            n_rows_seen_diff += 1
            leak_examples.append((t.date(), ts_safe, ts_leaky))
            # The leaky row must sit strictly past the strict-guard cutoff.
            cutoff = t - pd.Timedelta(days=21)
            assert ts_leaky is not None and ts_leaky > cutoff, (
                f"leaky lookup at {t.date()} returned {ts_leaky} which is "
                f"not actually past the lookahead cutoff {cutoff.date()}"
            )

    assert n_rows_seen_diff >= 1, (
        "Leak canary returned the same row as the lookahead-safe lookup at "
        "every rebalance. Either the canary isn't peeking ahead at all, or "
        "no rebalance had a future row available to leak."
    )

    logging.getLogger(__name__).info(
        "Leak canary fired on %d/%d rebalances; sample leaks: %s",
        n_rows_seen_diff, len(fix["rebalance_dates"]), leak_examples[:3],
    )


# F.2: selection_method / discovery_method switches
def test_f2_v0_correlation_skips_discovery(synthetic_fixture):
    """V0 path skips discovery entirely and still recovers the planted drivers."""
    from pipeline.closed_loop import run_closed_loop
    from pipeline.factor_selection.correlation_selector import (
        CorrelationSelectionResult,
    )

    fix = synthetic_fixture
    result = run_closed_loop(
        joint_frame=fix["joint_frame"],
        asset_returns=fix["asset_returns"],
        rebalance_dates=fix["rebalance_dates"],
        universe_at=fix["universe_at"],
        driver_columns=fix["driver_columns"],
        asset_columns=fix["asset_columns"],
        selection_method="correlation",
        selector_kwargs={},  # cum-corr doesn't take alpha/burn_in
        gamma_ema=0.3,
        tag="t_f2_v0",
        **{k: v for k, v in _common_kwargs(fix["tmp_dir"] / "f2_v0").items()
           if k != "discovery_kwargs"},
        discovery_kwargs={},  # ignored on V0 but passed for API stability
    )

    for t in fix["rebalance_dates"]:
        s1 = result.stage1_cache[t]
        assert s1.discovery is None, f"V0 must skip discovery at {t.date()}"
        assert isinstance(s1.selection, CorrelationSelectionResult)
        assert s1.selection.selected, f"V0 must select drivers at {t.date()}"
        assert s1.sensitivities.S.shape == (
            len(fix["asset_columns"]), len(s1.selection.selected)
        )

    # Cum-corr should rank the two planted drivers top.
    sel0 = result.stage1_cache[fix["rebalance_dates"][0]].selection
    top2 = sel0.scores.sort_values(ascending=False).index[:2].tolist()
    planted = {"d_planted_0", "d_planted_1"}
    assert set(top2) == planted, (
        f"V0 cum-corr top-2 should be the planted drivers; got {top2}"
    )


def test_f2_varlingam_discovery_runs(synthetic_fixture):
    """VARLiNGAM path produces JointVarLingamWindow at every rebalance and routes Stage A's varlingam branch."""
    from pipeline.closed_loop import run_closed_loop
    from pipeline.discovery.varlingam import JointVarLingamWindow

    fix = synthetic_fixture

    common = {k: v for k, v in _common_kwargs(fix["tmp_dir"] / "f2_var").items()
              if k != "discovery_kwargs"}

    result = run_closed_loop(
        joint_frame=fix["joint_frame"],
        asset_returns=fix["asset_returns"],
        rebalance_dates=fix["rebalance_dates"],
        universe_at=fix["universe_at"],
        driver_columns=fix["driver_columns"],
        asset_columns=fix["asset_columns"],
        selection_method="causal_greedy",
        discovery_method="varlingam",
        discovery_kwargs={"lags": 1, "criterion": None, "prune": True},
        selector_kwargs={"alpha": 1.0, "burn_in_rebalances": 0},
        gamma_ema=0.3,
        tag="t_f2_var",
        **common,
    )

    for t in fix["rebalance_dates"]:
        s1 = result.stage1_cache[t]
        assert isinstance(s1.discovery, JointVarLingamWindow), (
            f"VARLiNGAM path must produce JointVarLingamWindow; got "
            f"{type(s1.discovery).__name__} at {t.date()}"
        )
        assert s1.selection.metadata.get("method") == "varlingam"


def test_f2_v0_and_v1_select_differently(synthetic_fixture):
    """V0 and V1 pick at least one different driver across the run (the switch is real)."""
    from pipeline.closed_loop import run_closed_loop

    fix = synthetic_fixture
    common = _common_kwargs(fix["tmp_dir"] / "f2_v0v1")

    r_v0 = run_closed_loop(
        joint_frame=fix["joint_frame"], asset_returns=fix["asset_returns"],
        rebalance_dates=fix["rebalance_dates"], universe_at=fix["universe_at"],
        driver_columns=fix["driver_columns"], asset_columns=fix["asset_columns"],
        selection_method="correlation",
        selector_kwargs={},
        tag="t_f2_v0_b",
        **{**common, "output_dir": fix["tmp_dir"] / "f2_v0_b"},
    )
    r_v1 = run_closed_loop(
        joint_frame=fix["joint_frame"], asset_returns=fix["asset_returns"],
        rebalance_dates=fix["rebalance_dates"], universe_at=fix["universe_at"],
        driver_columns=fix["driver_columns"], asset_columns=fix["asset_columns"],
        selection_method="causal_greedy", discovery_method="dynotears",
        selector_kwargs={"alpha": 1.0, "burn_in_rebalances": 0},
        tag="t_f2_v1_b",
        **{**common, "output_dir": fix["tmp_dir"] / "f2_v1_b"},
    )

    differing = 0
    for t in fix["rebalance_dates"]:
        v0_sel = set(r_v0.stage1_cache[t].selection.selected)
        v1_sel = set(r_v1.stage1_cache[t].selection.selected)
        if v0_sel != v1_sel:
            differing += 1
    assert differing >= 1, (
        "V0 (cum-corr) and V1 (causal-greedy) picked identical drivers at "
        "every rebalance — the switch is not genuinely changing selection. "
        "Either the fixture is degenerate or one of the paths is silently "
        "using the other's selector."
    )
