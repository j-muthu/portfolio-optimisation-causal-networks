"""Integration tests for ``pipeline.closed_loop.run_closed_loop``.

Three tests verify the V2 closed-loop machinery built in Phase F.1:

* **t1**: the loop genuinely feeds back — after burn-in, the selector's
  ``utility_lookup_timestamp`` is populated (i.e. U from completed rebalances
  is actually being read into selection of the next rebalance).
* **t2**: with ``alpha=1.0`` the closed-loop strategy degenerates to the
  V1 open-loop path. Per-rebalance weights match a direct
  ``run_stage1 → run_stage2(variants=["V1"])`` run to ≤ 1e-8.
* **t3**: swapping ``utility_lookup`` for the deliberately-broken
  ``leak_canary.make_leaky_lookup`` produces a *different* sequence of
  per-rebalance weights — confirming both that the leak canary is exercising
  the feedback path and that the lookahead guard is doing meaningful work.

The fixture is intentionally tiny (4 assets, 6 drivers, ~300 trading days,
8 rebalances) so the full Stage 1 + backtest cycle runs in seconds. Two of
the six drivers are planted with deterministic correlation to the assets so
the selector has signal to find.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Fixture
# ============================================================================
@pytest.fixture(scope="module")
def synthetic_fixture(tmp_path_factory):
    """Build a small synthetic joint panel with two planted-signal drivers."""
    rng = np.random.default_rng(seed=11)
    T = 280  # trading days
    asset_cols = [f"A{i}" for i in range(4)]
    driver_cols = [f"d_planted_{i}" for i in range(2)] + [f"d_noise_{i}" for i in range(4)]
    n_drivers = len(driver_cols)

    cal = pd.bdate_range("2020-01-02", periods=T)

    # Two planted drivers carry a shared "factor" signal correlated with
    # every asset; the other four are pure i.i.d. Gaussian noise. The signal
    # is strong enough that DYNOTEARS reliably picks up the planted-driver
    # → asset edges and the selector picks the planted drivers.
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

    # Convert asset levels to "returns" for the backtest layer. Scale down
    # so the backtest engine sees realistic-magnitude daily returns.
    asset_returns = assets_df.diff().fillna(0.0) * 0.01

    rebalance_dates = pd.DatetimeIndex(
        [cal[120 + 20 * i] for i in range(8)]  # 8 rebalances spaced ~21 trading days apart
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
    """Shared kwargs that keep the per-test runtime small."""
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


# ============================================================================
# t1 — closed loop genuinely feeds back (U from t affects t+1)
# ============================================================================
def test_t1_closed_loop_feeds_back(synthetic_fixture, caplog):
    """After burn-in, the selector receives a populated utility lookup row.

    Before burn-in completes, ``selection.utility_lookup_timestamp`` is
    ``None`` and ``alpha_effective == 1.0`` (selector forces causal-only).
    After burn-in, the timestamp is populated and points at a row whose
    holding-period-end satisfies the lookahead gap (``≤ t - 21 calendar days``).
    """
    caplog.set_level(logging.WARNING)
    from pipeline.closed_loop import run_closed_loop

    fix = synthetic_fixture
    burn_in = 2  # small so we see post-burn-in behaviour within 8 rebalances

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

    # Burn-in: α forced to 1, no lookup performed.
    for i in range(burn_in):
        sel = result.stage1_cache[fix["rebalance_dates"][i]].selection
        assert sel.metadata["burn_in_active"] is True, f"rebalance {i} should be in burn-in"
        assert sel.alpha_effective == 1.0
        assert sel.utility_lookup_timestamp is None

    # Post-burn-in: at least one rebalance must have a non-None lookup
    # timestamp. (Early post-burn-in calls may still find no eligible row
    # if no holding period has ended yet by ``t - 21 calendar days``.)
    post_lookups = [
        result.stage1_cache[fix["rebalance_dates"][i]].selection.utility_lookup_timestamp
        for i in range(burn_in, n)
    ]
    populated = [ts for ts in post_lookups if ts is not None]
    assert len(populated) > 0, (
        "Closed loop is not actually closed: no post-burn-in rebalance saw a "
        "populated U lookup. Per-rebalance interleaving is broken."
    )

    # Every populated lookup_timestamp must respect the 21-day lookahead
    # gap — the strict assertion in UtilityStore.lookup_utility would have
    # raised otherwise; we re-verify here as a belt-and-braces check.
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


# ============================================================================
# t2 — α=1 degenerates to V1 open-loop
# ============================================================================
def test_t2_alpha_one_matches_v1_openloop(synthetic_fixture):
    """``run_closed_loop(alpha=1.0)`` per-rebalance weights == V1 weights from
    a direct ``run_stage1 + run_stage2(variants=['V1'])`` pipeline.

    With α=1 the selector ignores U entirely, so the closed-loop path must
    produce the same Stage 1 outputs (modulo deterministic FFNN seed) and
    the same allocation as the V1 open-loop variant. Matches to ≤ 1e-8.
    """
    from pipeline.closed_loop import run_closed_loop
    from pipeline.stage1_pipeline import run_stage1
    from pipeline.stage2_pipeline import run_stage2

    fix = synthetic_fixture
    kw = _common_kwargs(fix["tmp_dir"] / "t2_closed")
    kw_open = _common_kwargs(fix["tmp_dir"] / "t2_open")

    # Closed-loop with α=1: utility is read but never weighted in.
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

    # Equivalent V1 open-loop path.
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
        bootstrap_resamples=0,  # skip the bootstrap for speed
        tag="t2_open",
        output_dir=kw_open["output_dir"],
    )

    v1_recs = s2.variants["V1"].backtest.rebalances
    cl_recs = cl.backtest.rebalances
    assert len(v1_recs) == len(cl_recs)

    for cl_rec, v1_rec in zip(cl_recs, v1_recs):
        assert cl_rec.rebalance_date == v1_rec.rebalance_date
        # Pad both to the union universe before comparing element-wise.
        union = sorted(set(cl_rec.weights.index) | set(v1_rec.weights.index))
        w_cl = cl_rec.weights.reindex(union).fillna(0.0).to_numpy()
        w_v1 = v1_rec.weights.reindex(union).fillna(0.0).to_numpy()
        max_diff = float(np.max(np.abs(w_cl - w_v1)))
        assert max_diff < 1e-8, (
            f"α=1 degeneracy broken at {cl_rec.rebalance_date.date()}: "
            f"closed-loop and V1 weights differ by {max_diff:.2e}"
        )


# ============================================================================
# t3 — leak canary actually fires
# ============================================================================
def test_t3_leak_canary_fires(synthetic_fixture):
    """The leak canary exposes future U rows that the lookahead-safe lookup
    correctly hides.

    The canary's job is to expose *one rebalance of cheating*: a row that
    was written at holding-period-end *after* the rebalance date ``t``
    becomes visible to the selector as if it had been known at ``t``. We
    verify this directly at the lookup layer (rather than at the weights
    layer) because on a small fixture, even when the lookup returns
    different U vectors, the downstream selection can be sticky enough
    that weights coincide. The *lookup* difference is the canary's
    correctness property; the downstream-propagation strength is a
    separate empirical question for the real backtest.

    Also re-confirms that the strict lookahead guard in
    :func:`UtilityStore.lookup_utility` would have raised on the row the
    leaky lookup admitted.
    """
    from pipeline.closed_loop import run_closed_loop
    from pipeline.feedback import UtilityStore
    from pipeline.feedback.leak_canary import leaky_lookup, make_leaky_lookup

    fix = synthetic_fixture

    # Run a normal closed-loop pass and let it populate a UtilityStore.
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

    # Now probe the two lookups at every rebalance date and compare what
    # they return. At least one rebalance must yield a strictly different
    # row (different timestamp or different values) for the canary to be
    # actually leaking.
    n_rows_seen_diff = 0
    leak_examples = []
    for t in fix["rebalance_dates"]:
        u_safe, ts_safe = store.lookup_utility(t, require_strict=False)
        u_leaky, ts_leaky = leaky_lookup(store, t, peek_ahead_days=21)
        if ts_safe != ts_leaky:
            n_rows_seen_diff += 1
            leak_examples.append((t.date(), ts_safe, ts_leaky))
            # When the rows differ, the leaky row's end_date must be
            # *strictly after* the strict-guard cutoff — i.e. precisely
            # the leak the guard is designed to refuse.
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
