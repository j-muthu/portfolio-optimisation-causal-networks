"""Phase-II step 2 — graph-extraction dry-run + cache-hit gate.

Rebuilds the exact Phase-I joint panel (same universe file, driver drop set,
calendar and window slicing as ``run_shakedown``), computes the discovery-
cache key for every rebalance window, and:

1. **Asserts the cache-hit rate is 100%** for the requested (method, window)
   combos. A miss means the rebuilt joint frame diverges from the one the
   Phase-I fits were keyed on — STOP and reconcile before any backtest, or
   the E1 grid would silently refit 4-minute graphs and decouple Phase II
   from the Phase-I bundles.
2. Loads each cached fit, extracts the asset–asset graph through the
   Phase-II chokepoint, and writes per-window DAG diagnostics
   (density, depth, roots/leaves) → ``results/phase_ii_dag_diagnostics.csv``.

Zero WRDS calls — all data comes from cache/ (same guarantee as Phase I).

Usage
-----
    python -m scripts.extract_asset_graphs                  # all 4 combos
    python -m scripts.extract_asset_graphs --window 252 --method dynotears
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT
from pipeline.data import alignment
from pipeline.data.assets import fetch_prices
from pipeline.data.drivers import DRIVER_CATALOGUE, build_driver_pool
from pipeline.discovery.asset_graph import asset_graph_from_discovery
from pipeline.discovery.cache import CACHE_DIR, discovery_cache_key
from pipeline.portfolio.topological import dag_diagnostics
from scripts.run_phase_ii import (
    BACKTEST_START,
    DATA_END,
    DATA_START,
    DROP_DRIVERS,
    UNIVERSE_FILE,
)

logger = logging.getLogger("extract_asset_graphs")

RESULTS = THESIS_ROOT / "results"


def build_phase_i_inputs():
    """Reproduce run_shakedown's data-prep (steps 2–5) for the Phase-I config."""
    universe = sorted(set(UNIVERSE_FILE.read_text().strip().split(",")))
    driver_specs = [s for s in DRIVER_CATALOGUE if s.name not in DROP_DRIVERS]
    start_ts, end_ts = pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)

    asset_panel = fetch_prices(universe, start_ts, end_ts, use_cache=True)
    asset_prices = asset_panel.prices
    asset_returns = np.log(asset_prices / asset_prices.shift(1)).iloc[1:]

    nyse_cal = alignment.trading_calendar(start_ts, end_ts)
    pool = build_driver_pool(
        start_ts, end_ts, daily_index=nyse_cal, specs=driver_specs, use_cache=True,
    )
    joint = alignment.build_joint_matrix(
        drivers=pool.frame, assets=asset_returns, calendar=nyse_cal,
        drop_na="drivers_only",
    )

    cal = joint.frame.index
    holding_days, step = 21, 21
    first_bt = int(cal.searchsorted(pd.Timestamp(BACKTEST_START), side="left"))
    last_safe = len(cal) - holding_days - 1
    rebalance_dates = pd.DatetimeIndex(cal[first_bt:last_safe:step])
    return joint, rebalance_dates


def check_combo(joint, rebalance_dates, method: str, window: int) -> pd.DataFrame:
    """Cache-hit gate + DAG diagnostics for one (method, window) combo."""
    discovery_kwargs = {"prune": False} if method == "varlingam" else {}
    cal = joint.frame.index
    dcols, acols = list(joint.driver_columns), list(joint.asset_columns)

    hits, misses, rows = 0, [], []
    skipped_lookback = 0
    for t in rebalance_dates:
        end_pos = cal.searchsorted(t, side="right")
        start_pos = max(0, end_pos - window)
        if end_pos - start_pos < window:
            skipped_lookback += 1
            continue
        jw = joint.frame.iloc[start_pos:end_pos]
        key = discovery_cache_key(jw, dcols, acols, method, discovery_kwargs)
        path = CACHE_DIR / f"{key}.pkl"
        if not path.exists():
            misses.append((pd.Timestamp(t).date(), key))
            continue
        hits += 1
        import pickle

        with path.open("rb") as fh:
            disc = pickle.load(fh)
        g = asset_graph_from_discovery(disc, jw, method=method)
        d = dag_diagnostics(g)
        d.update({"rebalance": pd.Timestamp(t), "method": method, "window": window})
        rows.append(d)

    total = hits + len(misses)
    print(
        f"[{method} w{window}] rebalances={len(rebalance_dates)} "
        f"(insufficient lookback: {skipped_lookback}) | cache: {hits}/{total} hits"
    )
    if misses:
        print(f"  FIRST MISSES: {misses[:5]}")
        raise SystemExit(
            f"CACHE-HIT GATE FAILED for {method} w{window}: {len(misses)} misses. "
            "Reconcile keys before running any Phase-II backtest."
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["dynotears", "varlingam"], default=None,
                   help="default: both")
    p.add_argument("--window", type=int, choices=[189, 252, 378, 504], default=None,
                   help="default: both")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    methods = [args.method] if args.method else ["dynotears", "varlingam"]
    windows = [args.window] if args.window else [252, 504]

    joint, rebalance_dates = build_phase_i_inputs()
    print(
        f"Joint panel: {joint.frame.shape} ({len(joint.driver_columns)} drivers "
        f"+ {len(joint.asset_columns)} assets), {len(rebalance_dates)} rebalances"
    )

    frames = []
    for method in methods:
        for window in windows:
            frames.append(check_combo(joint, rebalance_dates, method, window))

    diag = pd.concat(frames, ignore_index=True)
    out = RESULTS / "phase_ii_dag_diagnostics.csv"
    diag.to_csv(out, index=False)
    print(f"\nDAG diagnostics → {out}")
    summary = diag.groupby(["method", "window"]).agg(
        n_windows=("rebalance", "count"),
        mean_density=("density", "mean"),
        mean_depth=("dag_depth", "mean"),
        max_depth=("dag_depth", "max"),
        mean_edges=("n_edges", "mean"),
        all_dags=("is_dag", "all"),
    )
    print(summary.to_string())
    print("\nCACHE-HIT GATE PASSED for all requested combos.")


if __name__ == "__main__":
    main()
