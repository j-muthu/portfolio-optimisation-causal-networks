"""J5 — reduced-scope NTS-NOTEARS probe (non-linear discovery).

A full NTS-NOTEARS backtest is compute-prohibitive (~10-25 min/window at d≈130 →
~50-90 h per variant over 215 rebalances). This probe instead fits NTS-NOTEARS
and DYNOTEARS on a *handful of regime windows* at a *reduced universe* and asks:
does non-linear discovery agree with the linear (DYNOTEARS) driver->asset
structure the thesis uses? Reports, per window:

  * NTS asset->driver block max|.|  (prior sanity — should be ~0)
  * top-K driver-set Jaccard (NTS vs DYNOTEARS Stage-A pools)
  * Spearman rank-correlation of the two methods' Stage-A driver scores
  * wall-clock per fit (substantiates the feasibility verdict)

Run:  python -m scripts.probe_nts_notears
Output: results/j5_nts_probe.csv
"""
from __future__ import annotations

import logging
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.data import alignment
from pipeline.data.assets import fetch_prices
from pipeline.data.drivers import DRIVER_CATALOGUE, build_driver_pool
from pipeline.discovery.dynotears import run_dynotears_joint_window
from pipeline.discovery.nts_notears import run_nts_notears_joint_window
from pipeline.factor_selection.prune import stage_a_score

logging.basicConfig(level=logging.WARNING)
REPO = pathlib.Path(__file__).resolve().parent.parent
DROP = {"hyg_lqd_logret", "vvix"}
N_ASSETS = 25          # reduced universe (probe, not full backtest)
WINDOW = 504
TOPK = 10
WINDOWS = {"2008-10 GFC": "2008-10-01", "2014-06 calm": "2014-06-02",
           "2020-03 COVID": "2020-03-02", "2022-06 hike": "2022-06-01"}


def build_frame():
    univ = REPO.joinpath("scripts/phase_i_universe.txt").read_text().strip().split(",")[:N_ASSETS]
    specs = [s for s in DRIVER_CATALOGUE if s.name not in DROP]
    panel = fetch_prices(univ, "2005-01-03", "2024-12-31", use_cache=True)
    rets = np.log(panel.prices / panel.prices.shift(1)).iloc[1:]
    cal = alignment.trading_calendar("2005-01-03", "2024-12-31")
    pool = build_driver_pool("2005-01-03", "2024-12-31", daily_index=cal, specs=specs, use_cache=True)
    j = alignment.build_joint_matrix(drivers=pool.frame, assets=rets, calendar=cal, drop_na="drivers_only")
    return j


def main():
    j = build_frame()
    frame, dcols, acols = j.frame, list(j.driver_columns), list(j.asset_columns)
    cal = pd.DatetimeIndex(frame.index)
    print(f"probe universe: d={len(dcols)+len(acols)} ({len(dcols)} drivers, {len(acols)} assets)")
    rows = []
    for name, dt in WINDOWS.items():
        end = cal.searchsorted(pd.Timestamp(dt), side="right")
        win = frame.iloc[max(0, end - WINDOW):end]
        if len(win) < WINDOW:
            print(f"  {name}: insufficient lookback, skipping"); continue

        t0 = time.time()
        dyn = run_dynotears_joint_window(win, dcols, acols, p=1)
        t_dyn = time.time() - t0
        t0 = time.time()
        nts = run_nts_notears_joint_window(win, dcols, acols, p=1, hidden=8, max_iter=20, w_threshold=0.1)
        t_nts = time.time() - t0

        s_dyn = stage_a_score(dyn, method="dynotears").scores
        s_nts = stage_a_score(nts, method="dynotears").scores
        s_dyn, s_nts = s_dyn.reindex(dcols).fillna(0), s_nts.reindex(dcols).fillna(0)
        top_dyn = set(s_dyn.sort_values(ascending=False).head(TOPK).index)
        top_nts = set(s_nts.sort_values(ascending=False).head(TOPK).index)
        jac = len(top_dyn & top_nts) / len(top_dyn | top_nts) if (top_dyn | top_nts) else float("nan")
        rho = spearmanr(s_dyn.values, s_nts.values).correlation
        a2d = float(np.abs(nts.asset_to_driver_block(1)).max())

        rows.append({"window": name, "nts_asset2driver_max": round(a2d, 5),
                     f"top{TOPK}_jaccard": round(jac, 3), "spearman_scores": round(float(rho), 3),
                     "dyn_fit_s": round(t_dyn, 1), "nts_fit_s": round(t_nts, 1)})
        print(f"  {name}: Jaccard={jac:.3f} Spearman={rho:.3f} "
              f"NTS a2d_max={a2d:.4f} | t_dyn={t_dyn:.1f}s t_nts={t_nts:.1f}s")

    df = pd.DataFrame(rows)
    out = REPO / "results" / "j5_nts_probe.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    if not df.empty:
        print(f"\nmean top{TOPK} Jaccard={df[f'top{TOPK}_jaccard'].mean():.3f}  "
              f"mean Spearman={df['spearman_scores'].mean():.3f}  "
              f"mean NTS fit={df['nts_fit_s'].mean():.0f}s vs DYNOTEARS {df['dyn_fit_s'].mean():.0f}s")
        full = df['nts_fit_s'].mean() * 215 / 3600
        print(f"→ extrapolated full backtest (215 rebalances) at this reduced d: ~{full:.0f}h/variant "
              f"(d=130 would be several× this) — full path is future work.")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
