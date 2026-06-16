"""Corrected versions of the two interim-report figures (frozen-EEM).

Same format/style as ``plot_interim_results.py`` (which reproduces the *submitted*
interim figures with the original jittery-EEM numbers), but driven entirely by
the frozen-EEM bundles + ``results/regime_analysis/excess_sharpe_named.csv`` —
so the numbers match the corrected FINDINGS. Written to a SEPARATE directory,
``interim_report/figures_corrected/``; the submitted ``interim_report/figures/``
are deliberately left untouched.

What changed vs the submitted figures: V1 w504 Sharpe 0.382→0.372 (EEM artefact),
and V2 ≡ V1 everywhere (the closed loop is inert — the old 2018Q4 "V2 edge" was
a jittery-EEM artefact and is gone, so the V2 bars now overlap V1).

Run:  python -m scripts.plot_interim_corrected
"""
from __future__ import annotations

import pathlib
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.evaluation.metrics import annualised_sharpe

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIG = REPO / "interim_report" / "figures_corrected"

COLOURS = {"V0": "#999999", "V1": "#0072B2", "V2": "#D55E00"}
VARIANTS = ["V0", "V1", "V2"]
LABELS = {"V0": "V0  vanilla HSP (cum-corr)", "V1": "V1  Causal-HSP (open loop)",
          "V2": "V2  Causal-HSP (closed loop)"}
# named-regime → CSV variant row (DYNOTEARS scope, matching the interim report)
CSV_VAR = {"V0": "V0", "V1": "V1-DYNOTEARS", "V2": "V2-DYNOTEARS"}
REGIME_ORDER = ["GFC 2007-09", "2018Q4 selloff", "COVID 2020", "2022 rate-hike", "Bull 2013-18"]
REGIME_LBL = ["GFC\n2007-09", "2018Q4\nselloff", "COVID\n2020", "2022\nrate-hike", "Bull\n2013-18"]


def _nav(v, w):
    nav = pd.Series(pickle.load((RESULTS / f"phase_i_{v.lower()}_w{w}" / "closed_loop.pkl").open("rb"))["backtest"].nav_net)
    nav.index = pd.to_datetime(nav.index)
    return nav.sort_index()


def plot_nav_curve(window=252):
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_box_aspect(1)
    for v in VARIANTS:
        nav = _nav(v, window)
        ax.plot(nav.index, nav.values, color=COLOURS[v],
                lw=2.2 if v == "V2" else 1.6, ls="--" if v == "V2" else "-",
                label=LABELS[v] + (" — ≡ V1" if v == "V2" else ""))
    ax.set_title(f"Cumulative net NAV, 2007–2024 (lookback {window} days) — frozen-EEM")
    ax.set_xlabel("Year")
    ax.set_ylabel("NAV (start = 1.0, net of 5 bps costs)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, "nav_curve.png")


def plot_sharpe_and_regime():
    df = pd.read_csv(RESULTS / "regime_analysis" / "excess_sharpe_named.csv")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # (a) Sharpe by variant × window, from frozen bundles.
    windows = [252, 504]
    x = np.arange(len(VARIANTS)); bw = 0.38
    for j, win in enumerate(windows):
        vals = [annualised_sharpe(_nav(v, win).pct_change().dropna()) for v in VARIANTS]
        axL.bar(x + (j - 0.5) * bw, vals, bw, label=f"{win}-day window",
                color="#009E73" if win == 252 else "#CC79A7")
        for xi, val in zip(x + (j - 0.5) * bw, vals):
            axL.text(xi, val + 0.0015, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels(VARIANTS)
    axL.set_ylim(0.34, 0.395)
    axL.set_ylabel("Annualised Sharpe (net)")
    axL.set_title("(a) Sharpe by variant and lookback window")
    axL.legend(frameon=False, fontsize=9, loc="upper right")
    axL.grid(True, axis="y", alpha=0.3)

    # (b) named-regime mean monthly excess-Sharpe vs 1/N (window 252), from CSV.
    xr = np.arange(len(REGIME_ORDER)); bw2 = 0.26
    for k, v in enumerate(VARIANTS):
        sub = df[df.variant == CSV_VAR[v]].set_index("named_regime")["mean_excess_sharpe"]
        vals = [sub.get(r, np.nan) for r in REGIME_ORDER]
        axR.bar(xr + (k - 1) * bw2, vals, bw2, label=v, color=COLOURS[v])
    axR.axhline(0.0, color="k", lw=0.8)
    axR.set_xticks(xr); axR.set_xticklabels(REGIME_LBL, fontsize=8)
    axR.set_ylabel("Mean monthly excess-Sharpe vs 1/N")
    axR.set_title("(b) Regime-conditional performance (252-day; V2 ≡ V1)")
    axR.legend(frameon=False, fontsize=9, loc="upper right")
    axR.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "sharpe_and_regime.png")


def _save(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("Plotting CORRECTED interim figures (frozen-EEM) …")
    plot_nav_curve(252)
    plot_sharpe_and_regime()
    print("Done.")


if __name__ == "__main__":
    main()
