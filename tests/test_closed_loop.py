"""Integration tests for ``pipeline.closed_loop.run_closed_loop``.

t1 checks that the loop really feeds back. t2 checks that alpha=1 reduces
to the V1 open-loop path. t3 checks that the leaky lookup actually leaks. I
use a tiny fixture (4 assets, 6 drivers and 8 rebalances) so that the full
cycle runs in seconds.
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
    """Build a small synthetic joint panel with 2 drivers that carry a
    planted signal."""
    rng = np.random.default_rng(seed=11)
    T = 280  # trading days
    asset_cols = [f"A{i}" for i in range(4)]
    driver_cols = [f"d_planted_{i}" for i in range(2)] + [f"d_noise_{i}" for i in range(4)]
    n_drivers = len(driver_cols)

    cal = pd.bdate_range("2020-01-02", periods=T)

    # The 2 planted drivers share a factor with the assets and the other 4
    # are noise. The signal is strong enough for DYNOTEARS to find reliably.
    shared = rng.standard_normal(T) * 1.0
    planted = np.stack([
        0.8 * shared + 0.4 * rng.standard_normal(T),
        0.7 * shared + 0.5 * rng.standard_normal(T),
    ], axis=1)
    noise = rng.standard_normal((T, 4)) * 0.5
    drivers = np.hstack([planted, noise])

    # The assets are driven by the shared factor plus idiosyncratic noise.
    asset_betas = rng.uniform(0.3, 0.8, size=4)
    assets = np.outer(shared, asset_betas) + 0.5 * rng.standard_normal((T, 4))

    drivers_df = pd.DataFrame(drivers, index=cal, columns=driver_cols)
    assets_df = pd.DataFrame(assets, index=cal, columns=asset_cols)
    joint = pd.concat([drivers_df, assets_df], axis=1)

    # Convert levels to returns, scaled to realistic daily magnitudes.
    asset_returns = assets_df.diff().fillna(0.0) * 0.01

    rebalance_dates = pd.DatetimeIndex(
        [cal[120 + 20 * i] for i in range(8)]  # roughly monthly spacing
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
    """Return the shared kwargs that keep the runtime of each test small."""
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


# t1: the closed loop really feeds back
def test_t1_closed_loop_feeds_back(synthetic_fixture, caplog):
    """After burn-in, the selector sees a populated utility lookup that
    respects the lookahead gap."""
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

    # During burn-in, alpha is forced to 1 and there is no lookup.
    for i in range(burn_in):
        sel = result.stage1_cache[fix["rebalance_dates"][i]].selection
        assert sel.metadata["burn_in_active"] is True, f"rebalance {i} should be in burn-in"
        assert sel.alpha_effective == 1.0
        assert sel.utility_lookup_timestamp is None

    # After burn-in, at least 1 rebalance must have a populated lookup. The
    # early ones may find no eligible row yet.
    post_lookups = [
        result.stage1_cache[fix["rebalance_dates"][i]].selection.utility_lookup_timestamp
        for i in range(burn_in, n)
    ]
    populated = [ts for ts in post_lookups if ts is not None]
    assert len(populated) > 0, (
        "Closed loop is not actually closed: no post-burn-in rebalance saw a "
        "populated U lookup. Per-rebalance interleaving is broken."
    )

    # Also, every populated lookup must respect the 21-day gap.
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


# F.2: the selection_method and discovery_method switches
def test_f2_v0_correlation_skips_discovery(synthetic_fixture):
    """The V0 path skips discovery entirely and still recovers the planted
    drivers."""
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
        selector_kwargs={},  # cumulative correlation does not take alpha or burn_in
        gamma_ema=0.3,
        tag="t_f2_v0",
        **{k: v for k, v in _common_kwargs(fix["tmp_dir"] / "f2_v0").items()
           if k != "discovery_kwargs"},
        discovery_kwargs={},  # ignored on V0 but passed to keep the API the same
    )

    for t in fix["rebalance_dates"]:
        s1 = result.stage1_cache[t]
        assert s1.discovery is None, f"V0 must skip discovery at {t.date()}"
        assert isinstance(s1.selection, CorrelationSelectionResult)
        assert s1.selection.selected, f"V0 must select drivers at {t.date()}"
        assert s1.sensitivities.S.shape == (
            len(fix["asset_columns"]), len(s1.selection.selected)
        )

    # Cumulative correlation should rank the 2 planted drivers top.
    sel0 = result.stage1_cache[fix["rebalance_dates"][0]].selection
    top2 = sel0.scores.sort_values(ascending=False).index[:2].tolist()
    planted = {"d_planted_0", "d_planted_1"}
    assert set(top2) == planted, (
        f"V0 cum-corr top-2 should be the planted drivers; got {top2}"
    )


def test_f2_varlingam_discovery_runs(synthetic_fixture):
    """The VARLiNGAM path produces a JointVarLingamWindow at every rebalance
    and uses Stage A's varlingam branch."""
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
    """V0 and V1 pick at least 1 different driver across the run, so the
    switch is real."""
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
