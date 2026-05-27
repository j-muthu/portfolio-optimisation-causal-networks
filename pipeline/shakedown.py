"""First real-data shakedown — Phase G of the build-out plan.

Composes every existing piece (data layer + Stage 1 + closed loop) into a
single function that produces the first end-to-end real-data backtest. The
goal is to validate that the whole stack composes on production-scale data
**before** investing in the full ablation matrix (Phase F.2). It also
surfaces real-data idiosyncrasies (NaN gaps in FRED macro series, asset
churn at top-N-by-mcap boundaries, FFNN runtime at d ≈ 100+) which
synthetic-fixture tests cannot.

End-to-end flow
---------------
1. **Universe** — :func:`pipeline.data.universe.top_n_by_mcap_at` at three
   snapshot dates (start, middle, end) and union. Fixed for the run.
2. **Asset prices** — :func:`pipeline.data.assets.fetch_prices` via the
   ``wrds → yfinance`` cascade. CRSP coverage means delisted names are
   honoured.
3. **Drivers** — :func:`pipeline.data.drivers.build_driver_pool` over the
   default ~35-series pool (FRED + Yahoo).
4. **Joint matrix** — :func:`pipeline.data.alignment.build_joint_matrix`
   on the NYSE trading-day calendar; rows with any NaN dropped.
5. **K calibration** *(optional)* — :func:`pipeline.factor_selection.calibrate_K`
   on the burn-in window. If skipped, ``K_default`` is used.
6. **Closed-loop backtest** — :func:`pipeline.closed_loop.run_closed_loop`
   with α=0.6, γ=0.3 (the primary V2 setting from the plan).

Default config is deliberately compact (one calendar year of rebalances,
top-30 assets) to keep the shakedown under ~30 min. Scale up the
``start/end/n_assets`` parameters once the smoke run is clean.

Entry point: :func:`run_shakedown`.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT
from pipeline.closed_loop import ClosedLoopResult, run_closed_loop
from pipeline.data import alignment
from pipeline.data.assets import fetch_prices, fetch_shares_outstanding
from pipeline.data.drivers import DRIVER_CATALOGUE, build_driver_pool
from pipeline.data.universe import (
    fetch_fja05680,
    membership_at,
    top_n_by_mcap_at,
)
from pipeline.discovery.dynotears import run_dynotears_joint_window
from pipeline.factor_selection.k_calibration import KCalibration, calibrate_K
from pipeline.factor_selection.prune import stage_a_score

logger = logging.getLogger(__name__)

RESULTS_ROOT = THESIS_ROOT / "results"


# ============================================================================
# Result container
# ============================================================================
@dataclass
class ShakedownResult:
    """Wraps the closed-loop output + the data-layer artefacts so a notebook
    can introspect every stage."""

    closed_loop: ClosedLoopResult
    joint_matrix: alignment.JointMatrix
    asset_returns: pd.DataFrame
    universe: list[str]
    rebalance_dates: pd.DatetimeIndex
    k_calibration: KCalibration | None = None
    timings: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)


# ============================================================================
# Universe builder — union of top-N at a few snapshot dates
# ============================================================================
def _build_universe(
    snapshot_dates: list[pd.Timestamp],
    n_per_snapshot: int,
    use_cache: bool = True,
) -> list[str]:
    """Union of ``top_n_by_mcap_at`` across a handful of snapshot dates.

    Holding the universe fixed (rather than re-selecting at each rebalance)
    keeps the joint matrix columns stable — required for DYNOTEARS to fit
    consistent ``W``/``A`` matrices across windows. The union of multiple
    snapshots captures churn (e.g. additions like TSLA / NVDA) at the cost
    of occasionally including a name that wasn't in the top-N on a given
    rebalance.
    """
    history = fetch_fja05680(use_cache=use_cache)
    union: set[str] = set()
    for ts in snapshot_dates:
        members = sorted(membership_at(ts, history=history, use_cache=use_cache))
        # We need ALL members' prices to compute the mcap-at-date.
        # Pull a thin window of prices and current shares.
        panel = fetch_prices(members, ts - pd.Timedelta(days=10), ts, use_cache=use_cache)
        if not panel.resolved:
            logger.warning("No member prices resolved for snapshot %s", ts.date())
            continue
        shares = fetch_shares_outstanding(panel.resolved, as_of=ts, use_cache=use_cache)
        tickers = top_n_by_mcap_at(
            ts, n=n_per_snapshot,
            prices=panel.prices, shares_outstanding=shares,
            history=history, use_cache=use_cache,
        )
        logger.info("Top-%d at %s: %s ...", n_per_snapshot, ts.date(), tickers[:5])
        union.update(tickers)
    return sorted(union)


# ============================================================================
# Top-level shakedown driver
# ============================================================================
def run_shakedown(
    start: str = "2018-01-02",
    end: str = "2020-12-31",
    backtest_start: str = "2020-01-02",
    n_assets: int = 30,
    universe_override: list[str] | None = None,
    K_default: int = 10,
    use_k_calibration: bool = True,
    k_calibration_B: int = 50,
    k_calibration_n_jobs: int = 1,
    rebalance_step_days: int = 21,
    window_size: int = 252,
    lookback_days: int = 252,
    holding_days: int = 21,
    transaction_cost_bps: float = 5.0,
    gamma_ema: float = 0.3,
    alpha: float = 0.6,
    burn_in_rebalances: int = 3,
    linkage_method: str = "single",
    discovery_kwargs: dict | None = None,
    sensitivities_kwargs: dict | None = None,
    driver_specs: list | None = None,
    tag: str = "shakedown_2020",
    output_dir: Path | None = None,
    use_cache: bool = True,
) -> ShakedownResult:
    """End-to-end real-data smoke run.

    Default config (compact, ~30 min on Apple Silicon):

    * 2018-01 → 2020-12 data window, backtest 2020-01 → 2020-12 (12 monthly
      rebalances, with the first 2 years available as Stage 1 lookback).
    * Top-30 by CRSP market cap from the S&P 500, fixed for the run.
    * Full ~35-series driver pool from FRED + Yahoo.
    * K calibrated on the burn-in (first window) with B=50 permutations,
      then frozen.
    * V2 closed loop: α=0.6 (favour causal), γ=0.3 (moderate U memory),
      burn-in 3 rebalances.
    """
    output_dir = Path(output_dir) if output_dir else RESULTS_ROOT / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    config = dict(
        start=start, end=end, backtest_start=backtest_start, n_assets=n_assets,
        K_default=K_default, use_k_calibration=use_k_calibration,
        k_calibration_B=k_calibration_B,
        rebalance_step_days=rebalance_step_days,
        window_size=window_size, lookback_days=lookback_days,
        holding_days=holding_days, transaction_cost_bps=transaction_cost_bps,
        gamma_ema=gamma_ema, alpha=alpha,
        burn_in_rebalances=burn_in_rebalances,
        linkage_method=linkage_method,
        tag=tag,
    )

    # ------------------------------------------------------------------
    # 1. Universe
    # ------------------------------------------------------------------
    t0 = time.time()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if universe_override is not None:
        universe = sorted(set(universe_override))
        logger.info("Using user-provided universe: %d tickers", len(universe))
    else:
        # Default path: top-N by CRSP mcap at three snapshots, union. This is
        # accurate but slow on first run (~500 WRDS queries per snapshot).
        # For smoke runs / fast iteration, pass ``universe_override`` instead.
        midpoint = start_ts + (end_ts - start_ts) / 2
        snapshots = [start_ts, midpoint, end_ts]
        logger.info("Building universe from snapshots: %s", [d.date() for d in snapshots])
        universe = _build_universe(snapshots, n_per_snapshot=n_assets, use_cache=use_cache)
    timings["universe_build_s"] = time.time() - t0
    logger.info("Universe size: %d unique tickers", len(universe))

    # ------------------------------------------------------------------
    # 2. Asset prices — full history for the entire universe
    # ------------------------------------------------------------------
    t0 = time.time()
    asset_panel = fetch_prices(universe, start_ts, end_ts, use_cache=use_cache)
    if not asset_panel.resolved:
        raise RuntimeError("No asset prices resolved — check WRDS connectivity")
    asset_prices = asset_panel.prices
    asset_returns = np.log(asset_prices / asset_prices.shift(1)).iloc[1:]
    timings["assets_fetch_s"] = time.time() - t0
    logger.info(
        "Asset prices: %d resolved (wrds=%d, yf=%d, missing=%d), shape=%s",
        len(asset_panel.resolved),
        sum(1 for s in asset_panel.sources.values() if s == "wrds"),
        sum(1 for s in asset_panel.sources.values() if s == "yfinance"),
        len(asset_panel.missing), asset_returns.shape,
    )

    # ------------------------------------------------------------------
    # 3. Drivers — full ~35-series pool
    # ------------------------------------------------------------------
    t0 = time.time()
    nyse_cal = alignment.trading_calendar(start_ts, end_ts)
    pool = build_driver_pool(
        start_ts, end_ts,
        daily_index=nyse_cal,
        specs=driver_specs or DRIVER_CATALOGUE,
        use_cache=use_cache,
    )
    timings["drivers_build_s"] = time.time() - t0
    logger.info(
        "Driver pool: %d retained (%d dropped), %d trading days",
        pool.n_series, len(pool.dropped), len(pool.frame),
    )
    if pool.dropped:
        logger.info("Dropped drivers: %s", list(pool.dropped.keys()))

    # ------------------------------------------------------------------
    # 4. Joint matrix [D | A]
    # ------------------------------------------------------------------
    t0 = time.time()
    joint = alignment.build_joint_matrix(
        drivers=pool.frame, assets=asset_returns, calendar=nyse_cal,
        drop_na="drivers_only",
    )
    timings["joint_build_s"] = time.time() - t0
    n_partial = 0
    if joint.asset_eligibility is not None:
        n_partial = int((~joint.asset_eligibility).any(axis=0).sum())
    logger.info(
        "Joint matrix: shape=%s (%d drivers + %d assets), %d rows dropped for driver NaN; "
        "%d/%d assets have partial coverage (per-asset eligibility mask populated)",
        joint.frame.shape, len(joint.driver_columns), len(joint.asset_columns),
        joint.rows_dropped, n_partial, len(joint.asset_columns),
    )
    if joint.asset_eligibility is not None:
        for a in joint.asset_columns:
            elig = joint.asset_eligibility[a]
            if not elig.all():
                first = elig[elig].index.min()
                logger.info("  %s: %d/%d eligible (first valid date: %s)",
                            a, int(elig.sum()), len(elig),
                            first.date() if first is not pd.NaT else "never")
    if joint.n < window_size + 21 * 6:
        logger.warning(
            "Joint matrix has only %d rows; backtest may be too short. Consider "
            "extending the [start, end] window or reducing window_size.", joint.n,
        )

    # ------------------------------------------------------------------
    # 5. Rebalance dates
    # ------------------------------------------------------------------
    backtest_start_ts = pd.Timestamp(backtest_start)
    cal = joint.frame.index
    first_bt_pos = int(cal.searchsorted(backtest_start_ts, side="left"))
    last_safe_pos = len(cal) - holding_days - 1
    rebalance_dates = pd.DatetimeIndex(
        cal[first_bt_pos:last_safe_pos:rebalance_step_days]
    )
    if len(rebalance_dates) == 0:
        raise RuntimeError(
            f"No rebalance dates: backtest_start={backtest_start} not within joint "
            f"calendar [{cal[0].date()}..{cal[-1].date()}]"
        )
    logger.info(
        "Backtest: %d rebalances %s..%s (step=%d trading days, holding=%d)",
        len(rebalance_dates), rebalance_dates[0].date(),
        rebalance_dates[-1].date(), rebalance_step_days, holding_days,
    )

    # ------------------------------------------------------------------
    # 6. K calibration on the burn-in window (optional)
    # ------------------------------------------------------------------
    cal_result: KCalibration | None = None
    K = K_default
    if use_k_calibration:
        t0 = time.time()
        end_pos = int(cal.searchsorted(rebalance_dates[0], side="right"))
        start_pos = max(0, end_pos - window_size)
        burnin_window = joint.frame.iloc[start_pos:end_pos]
        logger.info(
            "K calibration on burn-in window %s..%s (n=%d), B=%d, n_jobs=%d",
            burnin_window.index[0].date(), burnin_window.index[-1].date(),
            len(burnin_window), k_calibration_B, k_calibration_n_jobs,
        )

        # Build the per-permutation fit-and-score closure.
        disc_kwargs = dict(discovery_kwargs or {})
        rng_for_shuffle = np.random.default_rng(0)

        def _fit_permuted_scores(seed: int) -> np.ndarray:
            local_rng = np.random.default_rng(seed)
            shuffled = burnin_window.copy()
            n = len(shuffled)
            for d in joint.driver_columns:
                shuffled[d] = shuffled[d].to_numpy()[local_rng.permutation(n)]
            try:
                disc = run_dynotears_joint_window(
                    shuffled, joint.driver_columns, joint.asset_columns,
                    **disc_kwargs,
                )
            except Exception as exc:
                logger.debug("Permutation fit failed (seed=%d): %s", seed, exc)
                return np.zeros(len(joint.driver_columns))
            return stage_a_score(disc).scores.to_numpy()

        # Real-fit needs to happen once for K calibration.
        real_disc = run_dynotears_joint_window(
            burnin_window, joint.driver_columns, joint.asset_columns, **disc_kwargs,
        )
        cal_result = calibrate_K(
            real_window=real_disc,
            fit_permuted_score_fn=_fit_permuted_scores,
            method="dynotears",
            n_permutations=k_calibration_B,
            quantile=0.95,
        )
        K = cal_result.K
        timings["k_calibration_s"] = time.time() - t0
        cal_result.save(output_dir / "k_calibration.json")
        logger.info(
            "K calibration done in %.1fs: K_elbow=%d, K_perm=%d, chosen K=%d",
            timings["k_calibration_s"], cal_result.K_elbow, cal_result.K_perm, K,
        )

    # ------------------------------------------------------------------
    # 7. Closed-loop V2 backtest
    # ------------------------------------------------------------------
    t0 = time.time()
    # Restrict asset_returns to the joint matrix's assets (some may have been
    # dropped during the joint NaN purge).
    asset_returns_used = asset_returns[joint.asset_columns]

    # Eligibility-aware universe_at: at rebalance t with lookback W trading
    # days, an asset enters only if it has real data across the entire
    # lookback window (strict full-observability rule). This excludes
    # late-inception names from windows where they'd otherwise inject
    # pre-inception zero-fills into the sample covariance.
    def universe_at(t: pd.Timestamp) -> list[str]:
        if joint.asset_eligibility is None:
            return list(joint.asset_columns)
        cal = joint.frame.index
        end_pos = int(cal.searchsorted(t, side="right"))
        start_pos = max(0, end_pos - lookback_days)
        eligible = joint.assets_eligible_in_window(cal[start_pos], cal[end_pos - 1])
        if len(eligible) < len(joint.asset_columns):
            logger.debug(
                "universe_at(%s): %d/%d eligible (excluded: %s)",
                pd.Timestamp(t).date(), len(eligible), len(joint.asset_columns),
                [a for a in joint.asset_columns if a not in eligible],
            )
        return eligible

    result = run_closed_loop(
        joint_frame=joint.frame,
        asset_returns=asset_returns_used,
        rebalance_dates=rebalance_dates,
        universe_at=universe_at,
        driver_columns=joint.driver_columns,
        asset_columns=joint.asset_columns,
        K=K,
        linkage_method=linkage_method,
        window_size=window_size,
        lookback_days=lookback_days,
        holding_days=holding_days,
        transaction_cost_bps=transaction_cost_bps,
        gamma_ema=gamma_ema,
        discovery_kwargs=discovery_kwargs or {},
        selector_kwargs={"alpha": alpha, "burn_in_rebalances": burn_in_rebalances},
        sensitivities_kwargs=sensitivities_kwargs or {},
        asset_eligibility=joint.asset_eligibility,
        tag=tag,
        output_dir=output_dir,
    )
    timings["closed_loop_s"] = time.time() - t0
    logger.info("Closed loop done in %.1fs (%d rebalances, K=%d)",
                timings["closed_loop_s"], len(rebalance_dates), K)

    config["K_used"] = K
    config["timings_s"] = timings

    # Persist a small results bundle for the notebook.
    bundle = {
        "config": config,
        "timings": timings,
        "universe": universe,
        "joint_columns": list(joint.frame.columns),
        "driver_columns": joint.driver_columns,
        "asset_columns": joint.asset_columns,
        "k_calibration": cal_result.to_dict() if cal_result else None,
        "backtest_summary": result.backtest.to_frame(),
        "selected_drivers_per_rebalance": {
            str(d.date()): r.selection.selected
            for d, r in result.stage1_cache.items()
        },
        "nav_gross": result.backtest.nav_gross,
        "nav_net": result.backtest.nav_net,
    }
    with (output_dir / "shakedown_bundle.pkl").open("wb") as fh:
        pickle.dump(bundle, fh)
    logger.info("Shakedown bundle saved to %s", output_dir / "shakedown_bundle.pkl")

    return ShakedownResult(
        closed_loop=result,
        joint_matrix=joint,
        asset_returns=asset_returns_used,
        universe=universe,
        rebalance_dates=rebalance_dates,
        k_calibration=cal_result,
        timings=timings,
        config=config,
    )


# ============================================================================
# CLI
# ============================================================================
def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Causal-HSP V2 shakedown run.")
    p.add_argument("--start", default="2018-01-02")
    p.add_argument("--end", default="2020-12-31")
    p.add_argument("--backtest-start", default="2020-01-02")
    p.add_argument("--n-assets", type=int, default=30)
    p.add_argument(
        "--universe-override",
        default=None,
        help="Comma-separated ticker list; bypasses the top-N-by-mcap universe build "
        "(use for fast smoke runs).",
    )
    p.add_argument("--K-default", type=int, default=10)
    p.add_argument("--skip-k-calibration", action="store_true")
    p.add_argument("--k-calibration-B", type=int, default=50)
    p.add_argument("--k-calibration-n-jobs", type=int, default=1)
    p.add_argument("--window-size", type=int, default=252)
    p.add_argument("--tag", default="shakedown_2020")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    override = None
    if args.universe_override:
        override = [t.strip() for t in args.universe_override.split(",") if t.strip()]
    res = run_shakedown(
        start=args.start, end=args.end, backtest_start=args.backtest_start,
        n_assets=args.n_assets,
        universe_override=override,
        K_default=args.K_default,
        use_k_calibration=not args.skip_k_calibration,
        k_calibration_B=args.k_calibration_B,
        k_calibration_n_jobs=args.k_calibration_n_jobs,
        window_size=args.window_size,
        tag=args.tag, use_cache=not args.no_cache,
    )

    print()
    print("=" * 70)
    print(f"Shakedown complete: {args.tag}")
    print("=" * 70)
    print(f"  universe: {len(res.universe)} tickers")
    print(f"  joint matrix: {res.joint_matrix.frame.shape}")
    print(f"  K used: {res.config['K_used']}")
    print(f"  rebalances: {len(res.rebalance_dates)}")
    print(f"  timings (s):")
    for k, v in res.timings.items():
        print(f"    {k:24s} {v:>8.1f}")
    print(f"  final NAV (gross/net): {res.closed_loop.backtest.nav_gross.iloc[-1]:.4f} / "
          f"{res.closed_loop.backtest.nav_net.iloc[-1]:.4f}")
    summary = res.closed_loop.summary()
    print()
    print(summary.tail(6).to_string())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())


__all__ = ["ShakedownResult", "run_shakedown"]
