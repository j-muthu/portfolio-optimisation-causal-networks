"""Phase-II figure set (F1-F7): oriented-allocator results.

Regenerated entirely from committed artefacts so the figures always match
FINDINGS.md. Saved to results/figures/. Window sets come from the matrix
CSV, so the same code renders two-window and four-window grids.

Run order: collate_phase_ii, regime_analysis, then this script.
Run:  python -m scripts.plot_phase_ii_figures
"""
from __future__ import annotations

import pathlib
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIG = RESULTS / "figures"

# Okabe-Ito (consistent with plot_thesis_figures.py).
C = {
    "V0": "#999999",
    "D0": "#009E73",
    "D1": "#0072B2",
    "D2s": "#D55E00",
    "V1": "#E69F00",
    "w189": "#56B4E9",
    "w252": "#0072B2",
    "w378": "#CC79A7",
    "w504": "#D55E00",
}
# The reported family. D3/D4 were run but fall outside the crossing.
ALLOC_ORDER = ["D0", "D0s", "D1", "D2", "D2s"]

# Report display names; CSV tags stay D0/D0s/D1/D2/D2s. Mirrors the \corrhrp,
# \skelhrp, ... macros in final_report/main.tex; keep the two in step.
DISPLAY = {"D0": "skeleton-hrp", "D0s": "undirected-hrp", "D1": "semcov-hrp",
           "D2": "topo-hrp", "D2s": "topo-semcov-hrp", "CORR-HRP": "correlation-hrp",
           "V0": "hsp-baseline", "V1": "causal-hsp",
           "V2": "causal-hsp-feedback"}
METHOD_LABEL = {"dynotears": "DYNOTEARS", "varlingam": "VARLiNGAM", "granger": "ridge-Granger"}


def _windows(matrix: pd.DataFrame) -> list[int]:
    """Windows with at least one DYNOTEARS Phase-II cell, ascending."""
    m = matrix[matrix.method == "dynotears"]
    return sorted(int(w) for w in m.window.unique())


def _nav(tag: str) -> pd.Series | None:
    path = RESULTS / tag / "closed_loop.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)["backtest"].nav_net


# F1: Sharpe heat-map
def f1_heatmap(matrix: pd.DataFrame) -> None:
    m = matrix[matrix.method != "phase_i"]
    methods = [x for x in ("dynotears", "varlingam", "granger") if x in set(m.method)]
    ws = _windows(matrix)
    ncol = 2
    nrow = int(np.ceil(len(ws) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.0 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(ws):]:
        ax.set_visible(False)
    for ax, w in zip(axes, ws):
        sub = m[m.window == w].pivot(index="method", columns="allocator", values="sharpe")
        sub = sub.reindex(index=methods, columns=ALLOC_ORDER)
        vals = sub.to_numpy(dtype=float)
        im = ax.imshow(vals, cmap="RdYlGn", vmin=0.35, vmax=0.42, aspect="auto")
        ax.set_xticks(range(len(ALLOC_ORDER)), [DISPLAY[a] for a in ALLOC_ORDER],
                      fontsize=8, rotation=20, ha="right")
        ax.set_yticks(range(len(methods)), [METHOD_LABEL[x] for x in methods])
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center",
                            fontsize=8.5)
        # Box the D0 (symmetrised control) column.
        j0 = ALLOC_ORDER.index("D0")
        ax.add_patch(plt.Rectangle((j0 - 0.5, -0.5), 1, len(methods), fill=False,
                                   edgecolor="black", lw=2))
        corr = matrix[(matrix.method == "phase_i") & (matrix.allocator == "CORR-HRP")
                      & (matrix.window == w)].sharpe
        base = f"   (correlation-hrp baseline: {float(corr.iloc[0]):.3f})" if len(corr) else ""
        ax.set_title(f"{w}-day window{base}", fontsize=10)
    fig.colorbar(im, ax=list(axes[:len(ws)]), shrink=0.85, label="net Sharpe")
    fig.savefig(FIG / "phase_ii_heatmap.png", dpi=200)
    plt.close(fig)


# F2: fixed-graph contrast forest plot
def f2_forest(contrasts: pd.DataFrame) -> None:
    sub = contrasts[contrasts.contrast.str.match(r"D.*-D0$")].copy()
    sub["alloc"] = sub.contrast.str.replace("-D0", "", regex=False)
    methods = [x for x in ("dynotears", "varlingam") if x in set(sub.method)]
    ws = sorted(int(w) for w in sub.window.unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(10.5, 4.2),
                             sharey=True, constrained_layout=True)
    allocs = [a for a in ALLOC_ORDER if a != "D0"]
    offsets = np.linspace(-0.28, 0.28, len(ws)) if len(ws) > 1 else [0.0]
    for ax, m in zip(np.atleast_1d(axes), methods):
        for k, w in enumerate(ws):
            s = sub[(sub.method == m) & (sub.window == w)].set_index("alloc")
            y = np.arange(len(allocs)) + offsets[k]
            first = True
            for yi, a in zip(y, allocs):
                if a not in s.index:
                    continue
                row = s.loc[a]
                ax.errorbar(row.delta_sharpe, yi,
                            xerr=[[row.delta_sharpe - row.ci_lower],
                                  [row.ci_upper - row.delta_sharpe]],
                            fmt="o", ms=4.5, capsize=2.5, color=C[f"w{w}"],
                            label=f"{w} d" if first else None)
                first = False
                if row.p_value < 0.05:
                    ax.annotate("*", (row.ci_upper + 0.002, yi - 0.12),
                                color=C[f"w{w}"], fontsize=13)
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.set_yticks(range(len(allocs)), [DISPLAY[a] for a in allocs], fontsize=8)
        ax.set_title(METHOD_LABEL[m], fontsize=10)
        ax.set_xlabel("ΔSharpe vs skeleton control skeleton-hrp")
        # Legend goes upper left; lower right holds the D0s/w189 star.
        ax.margins(y=0.14)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.savefig(FIG / "phase_ii_forest.png", dpi=200)
    plt.close(fig)


# F3: NAV curves (w252)
def f3_nav() -> None:
    curves = {
        "correlation-hrp  correlation-distance HRP": ("phase_ii_corr_hrp_w252", "#000000", ":"),
        "hsp-baseline  HSP as published": ("phase_i_v0_w252", C["V0"], "-"),
        "skeleton-hrp  skeleton": ("phase_ii_dynotears_D0_w252", C["D0"], "-"),
        "semcov-hrp  SEM-implied covariance": ("phase_ii_dynotears_D1_w252", C["D1"], "-"),
        "topo-semcov-hrp  topological order + SEM cov.": ("phase_ii_dynotears_D2s_w252", C["D2s"], "-"),
        "causal-hsp  causal drivers + FFNN": ("phase_i_v1_w252", C["V1"], "--"),
    }
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for label, (tag, color, ls) in curves.items():
        nav = _nav(tag)
        if nav is not None:
            ax.plot(nav.index, nav.values, color=color, ls=ls, lw=1.4, label=label)
    ax.set_ylabel("cumulative net NAV (start = 1.0)")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.savefig(FIG / "phase_ii_nav.png", dpi=200)
    plt.close(fig)


# F4: seed audit
def f4_seed(matrix: pd.DataFrame) -> None:
    audit = pd.read_csv(RESULTS / "seed_audit.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    x_v1 = np.zeros(len(audit))
    ax.scatter(x_v1 + np.random.default_rng(0).uniform(-0.06, 0.06, len(audit)),
               audit.sharpe, s=28, color=C["V1"], zorder=3,
               label=f"causal-hsp FFNN seeds (n={len(audit)})")
    bp = ax.boxplot(audit.sharpe, positions=[0], widths=0.3, showfliers=False)
    for elem in ("boxes", "whiskers", "caps", "medians"):
        plt.setp(bp[elem], color="#666666")
    committed = float(audit.loc[audit.seed == 0, "sharpe"].iloc[0])
    ax.annotate("committed value\n(seed 0)", (0.09, committed),
                fontsize=8, color="#444444", va="center")
    # D-variants are deterministic: single points, no seed variance.
    m = matrix[(matrix.method == "dynotears") & (matrix.window == 252)]
    for k, (a, color) in enumerate((("D0", C["D0"]), ("D1", C["D1"]), ("D2s", C["D2s"]))):
        val = float(m.loc[m.allocator == a, "sharpe"].iloc[0])
        ax.scatter([k + 1], [val], marker="D", s=55, color=color, zorder=3)
        ax.annotate("deterministic", (k + 1, val - 0.004), fontsize=7,
                    ha="center", color=color)
    v0 = 0.371
    ax.axhline(v0, color=C["V0"], lw=1, ls=":")
    ax.annotate("hsp-baseline", (1.5, v0 + 0.0005), fontsize=8, color=C["V0"])
    ax.set_xticks(range(4), ["causal-hsp\n(FFNN)", "skeleton-hrp", "semcov-hrp", "topo-semcov-hrp"], rotation=12, ha="right",
               fontsize=8)
    ax.set_ylabel("net Sharpe (252-day window)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIG / "phase_ii_seed.png", dpi=200)
    plt.close(fig)


# F5: regime excess over hsp-baseline (w252)
def f5_regime() -> None:
    path = RESULTS / "regime_analysis" / "daily_metrics.csv"
    if not path.exists():
        print("F5 skipped: run scripts.regime_analysis first")
        return
    df = pd.read_csv(path)
    df = df[df.window == 252]
    regimes = ["nber_recession", "nber_expansion", "high_vol", "low_vol"]
    df = df[df.regime.isin(regimes)]
    base = df[df.variant == "V0"].set_index("regime")["sharpe"]
    show = [("V0prime", "skeleton-hrp", C["D0"]), ("DYNO-D1", "semcov-hrp", C["D1"]),
            ("DYNO-D2s", "topo-semcov-hrp", C["D2s"]),
            ("V1-DYNOTEARS", "causal-hsp", C["V1"])]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    width = 0.8 / len(show)
    xs = np.arange(len(regimes))
    for k, (v, label, color) in enumerate(show):
        s = df[df.variant == v].set_index("regime")["sharpe"]
        excess = [float(s.get(r, np.nan) - base[r]) for r in regimes]
        ax.bar(xs + (k - len(show) / 2 + 0.5) * width, excess, width,
               color=color, label=label)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(xs, ["NBER\nrecession", "NBER\nexpansion", "VIX\ntop quintile",
                       "VIX\nbottom quintile"], fontsize=9)
    ax.set_ylabel("Sharpe excess over hsp-baseline")
    ax.legend(fontsize=8.5)
    fig.savefig(FIG / "phase_ii_regime.png", dpi=200)
    plt.close(fig)


# F6: the decomposition
def f6_decomposition(matrix: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    """Net Sharpe of CORR, D0, D1/D2s with bootstrap deltas on each step."""
    def sharpe(method, alloc, w):
        m = matrix[(matrix.method == method) & (matrix.allocator == alloc)
                   & (matrix.window == w)]
        return float(m.sharpe.iloc[0])

    def delta(name, w, method="dynotears"):
        c = contrasts[(contrasts.contrast == name) & (contrasts.window == w)
                      & (contrasts.method == method)]
        if c.empty:
            return None
        return float(c.delta_sharpe.iloc[0]), float(c.p_value.iloc[0])

    bars = [("CORR-HRP", "#000000"), ("D0", C["D0"]), ("D1", C["D1"]),
            ("D2s", C["D2s"])]
    ws = _windows(matrix)
    all_vals = {w: [sharpe("phase_i", "CORR-HRP", w)] + [
        sharpe("dynotears", a, w) for a, _ in bars[1:]] for w in ws}
    lo = min(v for vs in all_vals.values() for v in vs) - 0.006
    hi = max(v for vs in all_vals.values() for v in vs) + 0.007

    ncol = 2 if len(ws) > 2 else len(ws)
    nrow = int(np.ceil(len(ws) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.7 * nrow),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(ws):]:
        ax.set_visible(False)
    # Each bar's step relative to its reference (CORR for D0; D0 for D1/D2s).
    steps = {1: ("D0-CORR", "+ skeleton"), 2: ("D1-D0", "+ orientation"),
             3: ("D2s-D0", "+ orientation")}
    for ax, w in zip(axes, ws):
        vals = all_vals[w]
        xs = np.arange(len(bars))
        ax.bar(xs, vals, 0.74, color=[c for _, c in bars])
        ax.axhline(vals[0], color="#000000", lw=0.9, ls=":")
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.3f}", (x, v + 0.0012), ha="center", fontsize=9)
            if x in steps:
                d = delta(steps[x][0], w)
                if d is not None:
                    dv, p = d
                    ax.annotate(f"{steps[x][1]}\n{dv:+.3f}\n(p={p:.2f})",
                                (x, lo + 0.55 * (min(vals) - lo) + 0.004),
                                ha="center", va="bottom", fontsize=7.4,
                                color="white", fontweight="bold")
        ax.set_xticks(xs, [DISPLAY[b] for b, _ in bars], fontsize=8, rotation=15, ha="right")
        ax.set_ylim(lo, hi)
        ax.set_title(f"{w}-day window", fontsize=10)
    for k in range(0, len(ws), ncol):
        axes[k].set_ylabel("net Sharpe")
    fig.savefig(FIG / "phase_ii_decomposition.png", dpi=200)
    plt.close(fig)


# F7: skeleton vs orientation additions across estimation horizons
def f7_window_gradient(contrasts: pd.DataFrame) -> None:
    """The decomposition as a function of estimation-window length."""
    def series(name: str, method: str):
        c = contrasts[(contrasts.contrast == name) & (contrasts.method == method)]
        c = c.sort_values("window")
        return (c.window.to_numpy(dtype=float), c.delta_sharpe.to_numpy(dtype=float),
                c.ci_lower.to_numpy(dtype=float), c.ci_upper.to_numpy(dtype=float),
                c.p_value.to_numpy(dtype=float))

    skel = series("D0-CORR", "dynotears")
    orient = series("D1-D0", "dynotears")
    orient_var = series("D1-D0", "varlingam")
    if len(skel[0]) == 0 or len(orient[0]) == 0:
        print("F7 skipped: D0-CORR / D1-D0 contrasts not found")
        return

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(10.5, 3.9),
                                  constrained_layout=True)
    for (w, d, lo, hi, _), label, color, marker in (
            (skel, "skeleton  (skeleton-hrp − correlation-hrp)", C["D0"], "o"),
            (orient, "orientation  (semcov-hrp − skeleton-hrp)", C["D1"], "s")):
        ax.errorbar(w, d, yerr=[d - lo, hi - d], fmt=f"-{marker}", ms=6,
                    capsize=3, lw=1.6, color=color, label=label)
    if len(orient_var[0]):
        ax.plot(orient_var[0], orient_var[1], "--^", ms=5, lw=1.1,
                color="#999999", label="orientation, VARLiNGAM")
    ax.axhline(0, color="grey", lw=1, ls=":")
    ax.set_xticks(skel[0], [f"{int(x)}" for x in skel[0]])
    ax.set_xlabel("estimation window (trading days)")
    ax.set_ylabel("ΔSharpe added")
    ax.legend(fontsize=8.5)

    # Right panel: stacked additions, the split of the total gain.
    wsx = np.arange(len(skel[0]))
    width = 0.55
    axr.bar(wsx, skel[1], width, color=C["D0"], label="skeleton")
    # Negative components extend below their own base.
    base = np.where(np.sign(skel[1]) == np.sign(orient[1]), skel[1], 0.0)
    axr.bar(wsx, orient[1], width, bottom=base, color=C["D1"],
            label="orientation")
    total = skel[1] + orient[1]
    for x, t in zip(wsx, total):
        axr.annotate(f"total {t:+.3f}", (x, max(t, 0) + 0.0015), ha="center",
                     fontsize=8)
    axr.margins(y=0.15)
    axr.axhline(0, color="grey", lw=1)
    axr.set_xticks(wsx, [f"{int(x)} d" for x in skel[0]])
    axr.set_ylabel("ΔSharpe over correlation-hrp")
    axr.legend(fontsize=8.5)
    fig.savefig(FIG / "phase_ii_window_gradient.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(RESULTS / "phase_ii_matrix.csv")
    contrasts = pd.read_csv(RESULTS / "phase_ii_contrasts.csv")
    f1_heatmap(matrix)
    f2_forest(contrasts)
    f3_nav()
    f4_seed(matrix)
    f5_regime()
    f6_decomposition(matrix, contrasts)
    f7_window_gradient(contrasts)
    print(f"figures → {FIG}/phase_ii_*.png")


if __name__ == "__main__":
    main()
