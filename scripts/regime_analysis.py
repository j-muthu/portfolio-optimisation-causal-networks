"""J6 — regime-conditional performance analysis (zero new compute).

Reads the persisted Phase I backtest bundles and tabulates, per
(variant × discovery-method × window × regime), the risk metrics that the
interim report's §4.1 promises:

* annualised Sharpe, max drawdown, annualised return / volatility — computed
  on the daily net-return series restricted to each regime;
* one-way annualised turnover within each regime — computed from the
  per-rebalance weights whose rebalance date falls inside the regime.

Regime definitions (per the report's methodology):
  - NBER recession (FRED USREC), and its complement (expansion);
  - VIX top-quintile (high vol) vs bottom-quintile (low vol).
(Network-density regimes need per-window discovery W matrices, which were not
persisted in the bundles — see the plan's J6 note; omitted here.)

Self-check: the "all" row of each variant's table must reproduce that
variant's full-sample Sharpe (already in the headline matrix) to <= 1e-9.

Run:  python -m scripts.regime_analysis
Outputs: results/regime_analysis/{daily_metrics,turnover,excess_sharpe_named}.csv
"""

from __future__ import annotations

import logging
import pathlib
import pickle

import numpy as np
import pandas as pd

from pipeline.data.drivers import fetch_yahoo_series
from pipeline.evaluation.metrics import (
    annualised_sharpe,
    max_drawdown,
    one_way_annualised_turnover,
    performance_summary,
)
from pipeline.evaluation.regime import (
    nber_recession_dates,
    regime_conditional_summary,
    vix_regime_masks,
)

log = logging.getLogger("regime_analysis")

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = RESULTS / "regime_analysis"

# (variant label, discovery method, result-dir tag stem). V0 is the shared
# cum-corr baseline (discovery-agnostic), so it appears once per window.
VARIANTS = [
    ("V0",          "—",        "phase_i_v0_w{w}"),
    ("V0prime",     "dynotears","phase_i_v0prime_w{w}"),
    ("V1-DYNOTEARS","dynotears","phase_i_v1_w{w}"),
    ("V2-DYNOTEARS","dynotears","phase_i_v2_w{w}"),
    ("V1-VARLiNGAM","varlingam","phase_i_v1_varlingam_w{w}"),
    ("V2-VARLiNGAM","varlingam","phase_i_v2_varlingam_w{w}"),
    # Phase-II direction-aware allocators (fixed-graph ablation). DYNO-D0 is
    # byte-identical to V0prime, so only the VARLiNGAM D0 row is added.
    ("VARL-D0",     "varlingam","phase_ii_varlingam_D0_w{w}"),
    ("DYNO-D0s",    "dynotears","phase_ii_dynotears_D0s_w{w}"),
    ("DYNO-D1",     "dynotears","phase_ii_dynotears_D1_w{w}"),
    ("DYNO-D2",     "dynotears","phase_ii_dynotears_D2_w{w}"),
    ("DYNO-D2s",    "dynotears","phase_ii_dynotears_D2s_w{w}"),
    ("DYNO-D3",     "dynotears","phase_ii_dynotears_D3_w{w}"),
    ("DYNO-D4",     "dynotears","phase_ii_dynotears_D4_w{w}"),
    ("VARL-D0s",    "varlingam","phase_ii_varlingam_D0s_w{w}"),
    ("VARL-D1",     "varlingam","phase_ii_varlingam_D1_w{w}"),
    ("VARL-D2",     "varlingam","phase_ii_varlingam_D2_w{w}"),
    ("VARL-D2s",    "varlingam","phase_ii_varlingam_D2s_w{w}"),
    ("VARL-D3",     "varlingam","phase_ii_varlingam_D3_w{w}"),
    ("VARL-D4",     "varlingam","phase_ii_varlingam_D4_w{w}"),
]
WINDOWS = [252, 504]

# Named hand-picked stress windows (to de-hardcode plot_interim_results.py's
# REGIME_EXCESS_SHARPE fixture with computed per-rebalance excess-Sharpe).
NAMED_WINDOWS = {
    "GFC 2007-09":   ("2007-07-01", "2009-06-30"),
    "2018Q4 selloff":("2018-10-01", "2018-12-31"),
    "COVID 2020":    ("2020-02-01", "2020-04-30"),
    "2022 rate-hike":("2022-01-01", "2022-12-31"),
    "Bull 2013-18":  ("2013-01-01", "2018-09-30"),
}


def _load(tag: str):
    path = RESULTS / tag / "closed_loop.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)["backtest"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    # One VIX series spanning the whole backtest, shared across all bundles.
    vix = fetch_yahoo_series("^VIX", pd.Timestamp("2006-06-01"),
                             pd.Timestamp("2025-01-01"), use_cache=True)

    daily_rows, turn_rows, named_rows = [], [], []

    for w in WINDOWS:
        for label, method, stem in VARIANTS:
            tag = stem.format(w=w)
            bt = _load(tag)
            if bt is None:
                log.warning("missing bundle: %s — skipping", tag)
                continue
            nav = bt.nav_net
            rets = nav.pct_change().dropna()

            # --- regime masks aligned to the daily return index ---
            nber = nber_recession_dates(rets.index)
            masks = {
                "nber_recession": nber,
                "nber_expansion": ~nber,
                **vix_regime_masks(vix.reindex(rets.index, method="ffill")),
            }

            # --- (a) daily return-based metrics per regime ---
            summ = regime_conditional_summary(
                rets, masks, summary_fn=lambda r: performance_summary(r)
            )
            # self-check: 'all' Sharpe must match a direct full-sample compute
            chk = annualised_sharpe(rets)
            assert abs(summ.loc["all", "annualised_sharpe"] - chk) < 1e-9, \
                f"{tag}: regime 'all' Sharpe {summ.loc['all','annualised_sharpe']} != {chk}"
            for regime, row in summ.iterrows():
                daily_rows.append({
                    "window": w, "variant": label, "method": method, "regime": regime,
                    "sharpe": round(row["annualised_sharpe"], 4),
                    "ann_return": round(row["annualised_return"], 4),
                    "ann_vol": round(row["annualised_volatility"], 4),
                    "max_drawdown": round(row["max_drawdown"], 4),
                    "n_days": int(masks[regime].reindex(rets.index).fillna(False).sum())
                              if regime != "all" else len(rets),
                })

            # --- (b) per-regime turnover from rebalances in-regime ---
            recs = bt.rebalances
            rdates = pd.DatetimeIndex([r.rebalance_date for r in recs])
            for regime, mask in {"all": pd.Series(True, index=rets.index), **masks}.items():
                m = mask.reindex(rdates, method="ffill").fillna(False).to_numpy()
                w_hist = [recs[i].weights for i in range(len(recs)) if m[i]]
                if len(w_hist) >= 2:
                    turn_rows.append({
                        "window": w, "variant": label, "method": method, "regime": regime,
                        "turnover_1way_annual": round(
                            one_way_annualised_turnover(w_hist, rebalances_per_year=12), 4),
                        "n_rebalances": len(w_hist),
                    })

            # --- (c) named-window mean excess-Sharpe-vs-1/N (de-hardcode Fig b) ---
            #     uses per-rebalance holding_reward (= excess Sharpe vs 1/N).
            if w == 252:  # Fig bars(b) is the 252-window cut
                hr = pd.Series([r.holding_reward for r in recs], index=rdates)
                for name, (s, e) in NAMED_WINDOWS.items():
                    sub = hr.loc[s:e]
                    named_rows.append({
                        "variant": label, "method": method, "named_regime": name,
                        "mean_excess_sharpe": round(float(sub.mean()), 4) if len(sub) else np.nan,
                        "n_rebalances": int(len(sub)),
                    })

    daily_df = pd.DataFrame(daily_rows)
    turn_df = pd.DataFrame(turn_rows)
    named_df = pd.DataFrame(named_rows)

    daily_df.to_csv(OUT / "daily_metrics.csv", index=False)
    turn_df.to_csv(OUT / "turnover.csv", index=False)
    named_df.to_csv(OUT / "excess_sharpe_named.csv", index=False)

    # --- console summary: the headline regime view (Sharpe by variant × regime) ---
    print("\n" + "=" * 88)
    print("J6 — regime-conditional annualised Sharpe (net), by variant × regime")
    print("=" * 88)
    for w in WINDOWS:
        sub = daily_df[daily_df.window == w]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="variant", columns="regime", values="sharpe")
        # order columns sensibly
        cols = [c for c in ["all","nber_recession","nber_expansion","high_vol","low_vol"]
                if c in piv.columns]
        print(f"\n--- window {w} ---")
        print(piv[cols].to_string())

    print("\n=== max drawdown by variant × regime (window 252) ===")
    sub = daily_df[daily_df.window == 252]
    piv = sub.pivot_table(index="variant", columns="regime", values="max_drawdown")
    cols = [c for c in ["all","nber_recession","high_vol","low_vol"] if c in piv.columns]
    print(piv[cols].to_string())

    print(f"\nsaved → {OUT}/ (daily_metrics.csv, turnover.csv, excess_sharpe_named.csv)")


if __name__ == "__main__":
    main()
