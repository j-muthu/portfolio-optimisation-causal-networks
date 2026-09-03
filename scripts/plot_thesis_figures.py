"""Final-report figure set (frozen-EEM), regenerated from committed artefacts.

Saved to results/figures/; the submitted interim_report/figures/ are left
untouched. Figures 8-9 need results/robust_stats.csv, so run
scripts.robust_stats first.

Run:  python -m scripts.plot_thesis_figures
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
FIG = RESULTS / "figures"

# Okabe-Ito colour-blind-safe palette (consistent with plot_interim_results.py).
C = {
    "V0": "#999999",
    "V0prime": "#009E73",
    "V1-DYNOTEARS": "#0072B2",
    "V1-VARLiNGAM": "#E69F00",
    "DYNOTEARS": "#0072B2", "VARLiNGAM": "#E69F00", "NTS": "#CC79A7",
}
LABEL = {
    "V0": "hsp-baseline  HSP as published",
    "V0prime": "skeleton-hrp  asset-only Causal-HRP",
    "V1-DYNOTEARS": "causal-hsp  (DYNOTEARS)",
    "V1-VARLiNGAM": "causal-hsp  (VARLiNGAM)",
}
# bundle tag per (variant, window)
TAG = {
    "V0": "phase_i_v0_w{w}", "V0prime": "phase_i_v0prime_w{w}",
    "V1-DYNOTEARS": "phase_i_v1_w{w}", "V1-VARLiNGAM": "phase_i_v1_varlingam_w{w}",
}
VARIANTS = ["V0", "V0prime", "V1-DYNOTEARS", "V1-VARLiNGAM"]


def _nav(variant: str, window: int) -> pd.Series | None:
    p = RESULTS / TAG[variant].format(w=window) / "closed_loop.pkl"
    if not p.exists():
        return None
    nav = pd.Series(pickle.load(p.open("rb"))["backtest"].nav_net)
    nav.index = pd.to_datetime(nav.index)
    return nav.sort_index()


def _csv(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


# 1. NAV curves
def fig_nav_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, w in zip(axes, (252, 504)):
        for v in VARIANTS:
            nav = _nav(v, w)
            if nav is None:
                continue
            ax.plot(nav.index, nav.values, color=C[v], lw=1.5, label=LABEL[v])
        ax.set_title(f"Lookback {w} days")
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cumulative net NAV (start = 1.0)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Cumulative net NAV, 2007–2024 "
                 "(causal-hsp-feedback ≡ causal-hsp, omitted)", fontsize=12)
    fig.tight_layout()
    _save(fig, "nav_curves.png")


# 2. Sharpe matrix
def fig_sharpe_matrix():
    windows = [252, 504]
    sharpe = {v: {w: (annualised_sharpe(_nav(v, w).pct_change().dropna())
                      if _nav(v, w) is not None else np.nan) for w in windows}
              for v in VARIANTS}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(VARIANTS)); bw = 0.38
    for j, w in enumerate(windows):
        vals = [sharpe[v][w] for v in VARIANTS]
        col = "#009E73" if w == 252 else "#CC79A7"
        ax.bar(x + (j - 0.5) * bw, vals, bw, label=f"{w}-day window", color=col)
        for xi, val in zip(x + (j - 0.5) * bw, vals):
            if not np.isnan(val):
                ax.text(xi, val + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["hsp-baseline", "skeleton-hrp", "causal-hsp\n(DYNO)",
                        "causal-hsp\n(VAR)"], fontsize=8)
    ax.set_ylim(0.34, 0.415)
    ax.set_ylabel("Annualised Sharpe (net)")
    ax.set_title("Net Sharpe by variant and lookback window (frozen-EEM)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "sharpe_matrix.png")


# 3. K-sensitivity (J4a)
def fig_k_sensitivity():
    df = _csv("j4a_k_sensitivity.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, w in zip(axes, (252, 504)):
        sub = df[df.window == w].sort_values("K")
        ax.plot(sub.K, sub.sharpe_V0, "o-", color=C["V0"], label="hsp-baseline")
        ax.plot(sub.K, sub.sharpe_V1, "s-", color=C["V1-DYNOTEARS"],
                label="causal-hsp (DYNOTEARS)")
        ax.axvline(17, color="k", ls=":", lw=0.9, alpha=0.6)
        ax.text(17, ax.get_ylim()[0], " K=17 (Kneedle)", fontsize=7.5, va="bottom", rotation=90)
        ax.set_title(f"Lookback {w} days")
        ax.set_xlabel("K (drivers selected)")
        ax.set_xticks(sub.K)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Annualised Sharpe (net)")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("K-sensitivity: causal-hsp's edge over hsp-baseline "
                 "is not robust to K", fontsize=12)
    fig.tight_layout()
    _save(fig, "k_sensitivity.png")


# 4. Feedback grid (J4b)
def fig_feedback_grid():
    df = _csv("j4b_alpha_gamma.csv")
    if df is None:
        return
    piv = df.pivot(index="alpha", columns="gamma", values="sharpe_V2")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("γ (utility EMA decay)"); ax.set_ylabel("α (causal/utility blend)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center",
                    color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="causal-hsp-feedback net Sharpe")
    ax.set_title("causal-hsp-feedback Sharpe across the α/γ feedback grid:\n"
                 "closed loop is inert (≡ causal-hsp = 0.381 everywhere)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "feedback_grid.png")


# 5. Regime-conditional excess over V0 (w252)
def fig_regime_excess():
    df = _csv("regime_analysis/daily_metrics.csv")
    if df is None:
        return
    df = df[df.window == 252]
    regimes = ["nber_recession", "nber_expansion", "high_vol", "low_vol"]
    rlabel = ["Recession", "Expansion", "High-vol", "Low-vol"]
    variants = ["V0prime", "V1-DYNOTEARS", "V1-VARLiNGAM"]
    base = {r: df[(df.variant == "V0") & (df.regime == r)].sharpe.iloc[0] for r in regimes}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(regimes)); bw = 0.26
    for k, v in enumerate(variants):
        exc = [df[(df.variant == v) & (df.regime == r)].sharpe.iloc[0] - base[r] for r in regimes]
        ax.bar(x + (k - 1) * bw, exc, bw, label=LABEL[v].split("  ")[0] + " " + v.split("-")[-1]
               if "-" in v else LABEL[v].split("  ")[0], color=C[v])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(rlabel)
    ax.set_ylabel("Regime Sharpe − hsp-baseline (net)")
    ax.set_title("Regime-conditional edge over hsp-baseline, window 252\n"
                 "(causal variants win in every regime; skeleton-hrp most)",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "regime_excess.png")


# 6. Directional prior (J1)
# Regime label per sampled window (matches SAMPLE_DATES in verify_directional_prior).
PRIOR_REGIME = {
    "2008-10": "GFC", "2011-09": "Euro crisis", "2014-06": "Calm bull",
    "2018-12": "2018Q4 selloff", "2020-03": "COVID crash", "2022-06": "Rate hikes",
}
# One-hue ordinal ramp for the top-K cut-offs (light -> dark as K grows).
PRIOR_K_SERIES = [
    (10, "jaccard_k10", "#E8904A"), (15, "jaccard_k15", "#C25400"),
    (20, "jaccard_k20", "#6E3000"),
]


def fig_directional_prior():
    df = _csv("directional_prior_verification.csv")
    if df is None:
        return
    dates = [str(w)[:7] for w in df.window_end]
    labels = [f"{PRIOR_REGIME.get(d, d)}\n{d}" for d in dates]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.bar(x, df.ad_frac_of_total * 100, width=0.55, color="#0072B2", alpha=0.6,
           zorder=2, label="asset→driver edge mass if prior removed (% of total, left axis)")
    ax.set_ylabel("Implausible asset→driver mass (%)")
    ax.set_xlabel("Regime")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    # Left/right tick positions coincide (0..50 step 10 <-> 0..1 step 0.2),
    # so one set of gridlines serves both axes.
    ax.set_ylim(0, 55); ax.set_yticks(np.arange(0, 51, 10))
    ax.grid(True, axis="y", alpha=0.5, zorder=0); ax.set_axisbelow(True)
    ax2 = ax.twinx()
    for k, col, colour in PRIOR_K_SERIES:
        if col not in df:
            continue
        ax2.plot(x, df[col], marker="o", ms=5, lw=1.6, color=colour, zorder=3,
                 label=f"top-{k} IoU")
    ax2.set_ylabel("Top-K driver intersection over union")
    ax2.set_ylim(0, 1.1); ax2.set_yticks(np.arange(0, 1.01, 0.2))
    # Two legend rows below the plot: the bar series, then the K series in order.
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8,
               loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.045))
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h2, [l2[0] + " (right axis)"] + l2[1:], frameon=False, fontsize=8,
               loc="lower center", ncol=len(h2), bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.6, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, "directional_prior.png")


# 7. NTS-NOTEARS probe (J5)
def fig_nts_probe():
    df = _csv("j5_nts_probe.csv")
    if df is None:
        return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    wins = [str(w).split()[0] for w in df.window]
    x = np.arange(len(df)); bw = 0.38
    axL.bar(x - bw / 2, df.top10_jaccard, bw, color="#0072B2", label="top-10 driver Jaccard")
    axL.bar(x + bw / 2, df.spearman_scores, bw, color="#CC79A7", label="Spearman (Stage-A scores)")
    axL.axhline(0, color="k", lw=0.8)
    axL.set_xticks(x); axL.set_xticklabels(wins, fontsize=8)
    axL.set_title("(a) NTS-NOTEARS vs DYNOTEARS agreement")
    axL.set_ylabel("agreement")
    axL.legend(frameon=False, fontsize=8, loc="upper left")
    axL.grid(True, axis="y", alpha=0.3)

    axR.bar(x - bw / 2, df.dyn_fit_s, bw, color="#0072B2", label="DYNOTEARS")
    axR.bar(x + bw / 2, df.nts_fit_s, bw, color="#CC79A7", label="NTS-NOTEARS")
    axR.set_yscale("log")
    axR.set_xticks(x); axR.set_xticklabels(wins, fontsize=8)
    axR.set_ylabel("fit time per window (s, log)")
    axR.set_title("(b) per-fit cost (NTS ≈ 10× DYNOTEARS at d=58)")
    axR.legend(frameon=False, fontsize=8, loc="upper right")
    axR.grid(True, axis="y", alpha=0.3, which="both")
    fig.suptitle("J5 — non-linear-discovery probe (reduced d=58)", fontsize=12)
    fig.tight_layout()
    _save(fig, "nts_probe.png")


# 8. Returns distribution (fat tails motivate PSR/DSR)
def fig_returns_distribution():
    from scipy import stats
    pooled = []
    for v in VARIANTS:
        nav = _nav(v, 252)
        if nav is not None:
            pooled.append(nav.pct_change().dropna().to_numpy())
    if not pooled:
        return
    r = np.concatenate(pooled)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(r, bins=200, density=True, color="#0072B2", alpha=0.55,
            label="daily net returns (pooled)")
    xs = np.linspace(r.min(), r.max(), 500)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std(ddof=0)), color="#D55E00",
            lw=1.8, label="fitted normal")
    ax.set_yscale("log")  # log-y exposes the tails
    ax.set_xlabel("daily net return")
    ax.set_ylabel("density (log)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, "returns_distribution.png")


# 9. DSR / MCS adjudication
def fig_dsr_mcs():
    df = _csv("robust_stats.csv")
    if df is None:
        return
    order = ["V0_w252", "V0prime_w252", "V1-DYNOTEARS_w252", "V1-VARLiNGAM_w252"]
    df = df[df.config.isin(order)].set_index("config").reindex(order).dropna(how="all")
    if df.empty:
        return
    labels = ["hsp-baseline", "skeleton-hrp", "causal-hsp\n(DYNO)",
              "causal-hsp\n(VAR)"][:len(df)]
    x = np.arange(len(df)); bw = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - bw / 2, df.psr_vs_zero, bw, color="#56B4E9", label="PSR (vs 0)")
    ax.bar(x + bw / 2, df.dsr, bw, color="#0072B2", label="DSR (deflated, N trials)")
    ax.axhline(0.95, color="k", ls=":", lw=0.9, alpha=0.6)
    ax.text(len(df) - 0.5, 0.952, "0.95", fontsize=7.5, va="bottom", ha="right")
    for xi, (_, row) in zip(x, df.iterrows()):
        ax.text(xi, 0.02, f"Sharpe {row.sharpe_ann:.3f}", ha="center", va="bottom",
                fontsize=8, rotation=90, color="white")
        if str(row.get("in_mcs90", "")).lower() == "true":
            ax.text(xi, max(row.psr_vs_zero, row.dsr) + 0.01, "★ MCS",
                    ha="center", va="bottom", fontsize=8, color="#009E73")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("probability")
    ax.set_title("Distribution- and multiplicity-aware Sharpe (window 252)\n"
                 "★ = in the 90% Model Confidence Set", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "dsr_mcs.png")


def _save(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("Generating thesis figures (frozen-EEM) …")
    for fn in (fig_nav_curves, fig_sharpe_matrix, fig_k_sensitivity, fig_feedback_grid,
               fig_regime_excess, fig_directional_prior, fig_nts_probe,
               fig_returns_distribution, fig_dsr_mcs):
        try:
            fn()
        except Exception as exc:  # one figure failing must not kill the rest
            print(f"  [skip] {fn.__name__}: {exc}")
    print("Done.")


if __name__ == "__main__":
    main()
