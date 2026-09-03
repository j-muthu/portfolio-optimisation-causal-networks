"""J1: directional-prior verification.

Refits a sample of windows with and without the asset->driver mask and
compares: how much the driver->asset block shifts, how much asset->driver
mass the prior suppresses, and whether Stage-A top-K selection changes.
Uses the exact Phase I data build. Cache-only, no WRDS.

Run:  python -m scripts.verify_directional_prior
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.data import alignment
from pipeline.data.assets import fetch_prices
from pipeline.data.drivers import DRIVER_CATALOGUE, build_driver_pool
from pipeline.discovery.dynotears import run_dynotears_joint_window
from pipeline.factor_selection.prune import stage_a_score

# Reuse Phase I's config so windows line up with the headline run.
from scripts.run_phase_i import (
    DATA_END,
    DATA_START,
    DROP_DRIVERS,
    UNIVERSE_FILE,
)

log = logging.getLogger("verify_prior")

# Sample dates: one calm year plus the major stress regimes.
WINDOW = 252
SAMPLE_DATES = [
    "2008-10-01",  # GFC core
    "2011-09-01",  # Euro crisis
    "2014-06-02",  # calm bull market
    "2018-12-03",  # 2018Q4 selloff
    "2020-03-02",  # COVID crash
    "2022-06-01",  # rate-hike drawdown
]
# DYNOTEARS hyperparameters, matching the Phase I defaults.
DISC_KWARGS = dict(p=1, lambda_w=0.05, lambda_a=0.05, w_threshold=0.01, max_iter=100)
TOP_K = 17  # the calibrated Phase I K (kept for the figure's headline series)
TOP_KS = (5, 10, 15, 20, 25)  # sweep, so the overlap does not rest on one cut-off


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Build the joint matrix exactly as Phase I does.
    universe = UNIVERSE_FILE.read_text().strip().split(",")
    driver_specs = [s for s in DRIVER_CATALOGUE if s.name not in DROP_DRIVERS]
    start_ts, end_ts = pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)

    panel = fetch_prices(universe, start_ts, end_ts, use_cache=True)
    asset_returns = np.log(panel.prices / panel.prices.shift(1)).iloc[1:]
    nyse_cal = alignment.trading_calendar(start_ts, end_ts)
    pool = build_driver_pool(start_ts, end_ts, daily_index=nyse_cal,
                             specs=driver_specs, use_cache=True)
    joint = alignment.build_joint_matrix(
        drivers=pool.frame, assets=asset_returns, calendar=nyse_cal,
        drop_na="drivers_only",
    )
    frame = joint.frame
    drivers = list(joint.driver_columns)
    assets = list(joint.asset_columns)
    cal = pd.DatetimeIndex(frame.index)
    log.info("Joint matrix: %s (%d drivers + %d assets)", frame.shape, len(drivers), len(assets))

    rows = []
    for date in SAMPLE_DATES:
        t = pd.Timestamp(date)
        end_pos = int(cal.searchsorted(t, side="right"))
        start_pos = end_pos - WINDOW
        if start_pos < 0:
            log.warning("Skipping %s: insufficient lookback", date)
            continue
        window = frame.iloc[start_pos:end_pos]

        # Fit twice: with the asset->driver mask (production) and without it.
        masked = run_dynotears_joint_window(
            window, driver_columns=drivers, asset_columns=assets,
            enforce_tabu=True, **DISC_KWARGS,
        )
        free = run_dynotears_joint_window(
            window, driver_columns=drivers, asset_columns=assets,
            enforce_tabu=False, **DISC_KWARGS,
        )

        # (1) Change in the driver->asset block (lag 0 + lag 1 stacked).
        da_masked = np.concatenate(
            [masked.driver_to_asset_block(l).ravel() for l in range(masked.p + 1)]
        )
        da_free = np.concatenate(
            [free.driver_to_asset_block(l).ravel() for l in range(free.p + 1)]
        )
        da_l1 = float(np.abs(da_masked).sum())
        da_delta_l1 = float(np.abs(da_masked - da_free).sum())
        da_rel = da_delta_l1 / da_l1 if da_l1 > 0 else float("nan")

        # (2) Asset->driver mass the prior suppresses (only present in `free`).
        ad_free = np.concatenate(
            [free.asset_to_driver_block(l).ravel() for l in range(free.p + 1)]
        )
        ad_mass = float(np.abs(ad_free).sum())
        ad_edges = int(np.count_nonzero(ad_free))
        # As a fraction of total edge mass in the unconstrained fit.
        total_free_mass = ad_mass + float(np.abs(da_free).sum())
        ad_frac = ad_mass / total_free_mass if total_free_mass > 0 else float("nan")

        # (3) Does the prior change which drivers Stage A ranks at the top?
        rank_masked = stage_a_score(masked, method="dynotears").scores.sort_values(
            ascending=False).index.tolist()
        rank_free = stage_a_score(free, method="dynotears").scores.sort_values(
            ascending=False).index.tolist()
        topk_jac = _jaccard(rank_masked[:TOP_K], rank_free[:TOP_K])
        jac_by_k = {k: _jaccard(rank_masked[:k], rank_free[:k]) for k in TOP_KS}

        rows.append({
            "window_end": t.date(),
            "da_block_L1": round(da_l1, 3),
            "da_rel_change": round(da_rel, 4),
            "ad_suppressed_mass": round(ad_mass, 3),
            "ad_edges": ad_edges,
            "ad_frac_of_total": round(ad_frac, 4),
            "topK_jaccard": round(topk_jac, 3),
            **{f"jaccard_k{k}": round(jac_by_k[k], 3) for k in TOP_KS},
        })
        log.info("%s: da_rel_change=%.3f, ad_mass=%.3f (%d edges, %.1f%% of total), topK_jac=%.3f",
                 t.date(), da_rel, ad_mass, ad_edges, 100 * ad_frac, topk_jac)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("J1 — directional-prior impact across sampled windows (DYNOTEARS, d=%d)" % frame.shape[1])
    print("=" * 90)
    print(df.to_string(index=False))
    print("\nColumn guide:")
    print("  da_block_L1       : L1 mass of the driver->asset block (the block we use), masked fit")
    print("  da_rel_change     : ||masked - free||_1 / ||masked||_1 on driver->asset block")
    print("                      (how much the legitimate structure shifts when the prior is removed)")
    print("  ad_suppressed_mass: L1 mass of asset->driver edges that appear WITHOUT the prior")
    print("  ad_frac_of_total  : that suppressed mass as a fraction of total edge mass (unconstrained)")
    print("  topK_jaccard      : overlap of top-%d Stage-A drivers, masked vs free" % TOP_K)
    print("  jaccard_k<K>      : the same overlap at K in %s" % (TOP_KS,))
    print()
    print("Interpretation: large ad_frac + low da_rel_change => the prior cheaply removes a real")
    print("chunk of (spurious) asset->driver mass without distorting the driver->asset block it")
    print("protects — i.e. the prior earns its place. ad_frac ~ 0 => data already respects the")
    print("directional hypothesis and the prior is near-free.")

    out = UNIVERSE_FILE.parent.parent / "results" / "directional_prior_verification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
