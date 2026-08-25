"""Plot the two Phase I figures for the interim report.

Bar-chart numbers are hard-coded from Tables 3.1/3.2 so the figures match
the text; recomputed Sharpes are printed as a cross-check.

Run:  python -m scripts.plot_interim_results
"""

from __future__ import annotations

import pathlib
import pickle

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing pipeline makes the pickled classes resolvable by pickle.load.
from pipeline.evaluation.metrics import annualised_sharpe

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "interim_report" / "figures"

# Okabe-Ito colour-blind-safe palette.
COLOURS = {"V0": "#999999", "V1": "#0072B2", "V2": "#D55E00"}
VARIANTS = ["V0", "V1", "V2"]
LABELS = {
    "V0": "V0  vanilla HSP (cum-corr)",
    "V1": "V1  Causal-HSP (open loop)",
    "V2": "V2  Causal-HSP (closed loop)",
}

# Phase I summary numbers (net of 5 bps), identical to Tables 3.1/3.2.
SHARPE = {  # net annualised Sharpe by window
    252: {"V0": 0.371, "V1": 0.382, "V2": 0.382},
    504: {"V0": 0.370, "V1": 0.382, "V2": 0.373},
}
# Regime-conditional mean monthly excess-Sharpe vs 1/N, window 252.
REGIMES = ["GFC\n2007-09", "2018Q4\nselloff", "COVID\n2020", "2022\nrate-hike", "Bull\n2013-18"]
REGIME_EXCESS_SHARPE = {
    "V0": [0.160, 0.355, 0.287, 0.235, -0.128],
    "V1": [0.178, 0.424, 0.343, 0.216, -0.137],
    "V2": [0.189, 0.544, 0.290, 0.171, -0.141],
}


def load_nav_net(variant: str, window: int) -> pd.Series:
    """Read the net NAV series from a Phase I closed-loop bundle."""
    path = RESULTS_DIR / f"phase_i_{variant.lower()}_w{window}" / "closed_loop.pkl"
    with path.open("rb") as fh:
        bundle = pickle.load(fh)
    nav = bundle["backtest"].nav_net
    nav = pd.Series(nav)
    nav.index = pd.to_datetime(nav.index)
    return nav.sort_index()


def plot_nav_curve(window: int = 252) -> None:
    """Figure 1: cumulative NAV (net) for V0/V1/V2 over the full sample."""
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_box_aspect(1)  # square plot box
    for v in VARIANTS:
        nav = load_nav_net(v, window)
        ax.plot(nav.index, nav.values, color=COLOURS[v], lw=1.6, label=LABELS[v])
        sharpe_check = annualised_sharpe(nav.pct_change().dropna())
        print(f"  {v} w{window}: final NAV {nav.iloc[-1]:.3f}, "
              f"Sharpe(recomputed) {sharpe_check:.3f}  [table: {SHARPE[window][v]:.3f}]")
    ax.set_title(f"Cumulative net NAV, 2007-2024 (lookback {window} days)")
    ax.set_xlabel("Year")
    ax.set_ylabel("NAV (start = 1.0, net of 5 bps costs)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "nav_curve.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


def plot_sharpe_and_regime() -> None:
    """Figure 2: Sharpe by variant x window, and regime-conditional excess-Sharpe."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # Left: Sharpe by variant, grouped by window.
    windows = [252, 504]
    x = np.arange(len(VARIANTS))
    bw = 0.38
    for j, win in enumerate(windows):
        vals = [SHARPE[win][v] for v in VARIANTS]
        bars = axL.bar(x + (j - 0.5) * bw, vals, bw,
                       label=f"{win}-day window",
                       color="#009E73" if win == 252 else "#CC79A7")
        for xi, val in zip(x + (j - 0.5) * bw, vals):
            axL.text(xi, val + 0.0015, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    axL.set_xticks(x)
    axL.set_xticklabels(VARIANTS)
    axL.set_ylim(0.34, 0.395)
    axL.set_ylabel("Annualised Sharpe (net)")
    axL.set_title("(a) Sharpe by variant and lookback window")
    axL.legend(frameon=False, fontsize=9, loc="upper right")
    axL.grid(True, axis="y", alpha=0.3)

    # Right: regime-conditional excess-Sharpe (window 252).
    xr = np.arange(len(REGIMES))
    bw2 = 0.26
    for k, v in enumerate(VARIANTS):
        axR.bar(xr + (k - 1) * bw2, REGIME_EXCESS_SHARPE[v], bw2,
                label=v, color=COLOURS[v])
    axR.axhline(0.0, color="k", lw=0.8)
    axR.set_xticks(xr)
    axR.set_xticklabels(REGIMES, fontsize=8)
    axR.set_ylabel("Mean monthly excess-Sharpe vs 1/N")
    axR.set_title("(b) Regime-conditional performance (252-day window)")
    axR.legend(frameon=False, fontsize=9, loc="upper right")
    axR.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "sharpe_and_regime.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Plotting interim results ...")
    plot_nav_curve(window=252)
    plot_sharpe_and_regime()
    print("Done.")


if __name__ == "__main__":
    main()
