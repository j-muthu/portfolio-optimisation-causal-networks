"""Phase-II figure set (F1–F5) — direction-aware allocation results.

Regenerated entirely from committed artefacts (``results/phase_ii_matrix.csv``,
``results/phase_ii_contrasts.csv``, ``results/seed_audit.csv``, the regime
tables and the gitignored bundles), so the figures always match FINDINGS.md.
Saved to ``results/figures/``. House style: Okabe-Ito palette, matching
``plot_thesis_figures.py``.

F1 phase_ii_heatmap.png   allocator × method net-Sharpe heat-map, D0 column
                          boxed (the direction effect at a glance), both windows
F2 phase_ii_forest.png    fixed-graph contrasts D* − D0: ΔSharpe ± 95% CI
F3 phase_ii_nav.png       cumulative net NAV, w252: V0, D0 (≡V0′), D1, D2s, V1
F4 phase_ii_seed.png      E7 seed audit: V1 Sharpe across FFNN seeds vs the
                          deterministic D-variants
F5 phase_ii_regime.png    regime-conditional Sharpe excess over V0 (w252),
                          incl. the best direction-aware allocators

Run order: collate_phase_ii → regime_analysis → this script.
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
    "D0": "#009E73",        # = V0' green
    "D1": "#0072B2",        # blue — structural covariance
    "D2s": "#D55E00",       # vermillion — causal-ordered bisection + Σ_struct
    "V1": "#E69F00",        # orange — the HSP machinery route
    "w252": "#0072B2",
    "w504": "#D55E00",
}
ALLOC_ORDER = ["D0", "D0s", "D1", "D2", "D2s", "D3", "D4"]
METHOD_LABEL = {"dynotears": "DYNOTEARS", "varlingam": "VARLiNGAM", "granger": "GRANGER"}


def _nav(tag: str) -> pd.Series | None:
    path = RESULTS / tag / "closed_loop.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)["backtest"].nav_net


# ============================================================================
# F1 — Sharpe heat-map
# ============================================================================
def f1_heatmap(matrix: pd.DataFrame) -> None:
    m = matrix[matrix.method != "phase_i"]
    methods = [x for x in ("dynotears", "varlingam", "granger") if x in set(m.method)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    for ax, w in zip(axes, (252, 504)):
        sub = m[m.window == w].pivot(index="method", columns="allocator", values="sharpe")
        sub = sub.reindex(index=methods, columns=ALLOC_ORDER)
        vals = sub.to_numpy(dtype=float)
        im = ax.imshow(vals, cmap="RdYlGn", vmin=0.35, vmax=0.42, aspect="auto")
        ax.set_xticks(range(len(ALLOC_ORDER)), ALLOC_ORDER)
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
        v0 = matrix[(matrix.method == "phase_i") & (matrix.allocator == "V0")
                    & (matrix.window == w)].sharpe
        ax.set_title(f"{w}-day window   (V0 baseline: {float(v0.iloc[0]):.3f})",
                     fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.85, label="net Sharpe")
    fig.suptitle("Net Sharpe by discovery method × allocator (boxed = symmetrised control D0)",
                 fontsize=11)
    fig.savefig(FIG / "phase_ii_heatmap.png", dpi=200)
    plt.close(fig)


# ============================================================================
# F2 — fixed-graph contrast forest plot
# ============================================================================
def f2_forest(contrasts: pd.DataFrame) -> None:
    sub = contrasts[contrasts.contrast.str.match(r"D.*-D0$")].copy()
    sub["alloc"] = sub.contrast.str.replace("-D0", "", regex=False)
    methods = [x for x in ("dynotears", "varlingam") if x in set(sub.method)]
    fig, axes = plt.subplots(1, len(methods), figsize=(10.5, 3.6),
                             sharey=True, constrained_layout=True)
    allocs = [a for a in ALLOC_ORDER if a != "D0"]
    for ax, m in zip(np.atleast_1d(axes), methods):
        for k, w in enumerate((252, 504)):
            s = sub[(sub.method == m) & (sub.window == w)].set_index("alloc")
            y = np.arange(len(allocs)) + (0.16 if w == 504 else -0.16)
            for yi, a in zip(y, allocs):
                if a not in s.index:
                    continue
                row = s.loc[a]
                ax.errorbar(row.delta_sharpe, yi,
                            xerr=[[row.delta_sharpe - row.ci_lower],
                                  [row.ci_upper - row.delta_sharpe]],
                            fmt="o", ms=5, capsize=3, color=C[f"w{w}"],
                            label=f"w{w}" if yi == y[0] else None)
                if row.p_value < 0.05:
                    ax.annotate("*", (row.ci_upper + 0.002, yi - 0.12),
                                color=C[f"w{w}"], fontsize=13)
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.set_yticks(range(len(allocs)), allocs)
        ax.set_title(METHOD_LABEL[m], fontsize=10)
        ax.set_xlabel("ΔSharpe vs symmetrised control D0")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("The direction effect, graph held fixed (Politis–Romano 95% CI; * p<0.05)",
                 fontsize=11)
    fig.savefig(FIG / "phase_ii_forest.png", dpi=200)
    plt.close(fig)


# ============================================================================
# F3 — NAV curves (w252)
# ============================================================================
def f3_nav() -> None:
    curves = {
        "CORR  correlation-distance HRP": ("phase_ii_corr_hrp_w252", "#000000", ":"),
        "V0  correlation-HSP": ("phase_i_v0_w252", C["V0"], "-"),
        "D0 ≡ V0′  symmetrised causal": ("phase_ii_dynotears_D0_w252", C["D0"], "-"),
        "D1  structural covariance": ("phase_ii_dynotears_D1_w252", C["D1"], "-"),
        "D2s  causal-ordered bisection": ("phase_ii_dynotears_D2s_w252", C["D2s"], "-"),
        "V1  Causal-HSP (drivers + FFNN)": ("phase_i_v1_w252", C["V1"], "--"),
    }
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for label, (tag, color, ls) in curves.items():
        nav = _nav(tag)
        if nav is not None:
            ax.plot(nav.index, nav.values, color=color, ls=ls, lw=1.4, label=label)
    ax.set_ylabel("cumulative net NAV (start = 1.0)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Direction-aware allocation from the same DYNOTEARS graphs (252-day window, net of 5 bps)",
                 fontsize=10.5)
    fig.savefig(FIG / "phase_ii_nav.png", dpi=200)
    plt.close(fig)


# ============================================================================
# F4 — seed audit
# ============================================================================
def f4_seed(matrix: pd.DataFrame) -> None:
    audit = pd.read_csv(RESULTS / "seed_audit.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    x_v1 = np.zeros(len(audit))
    ax.scatter(x_v1 + np.random.default_rng(0).uniform(-0.06, 0.06, len(audit)),
               audit.sharpe, s=28, color=C["V1"], zorder=3,
               label=f"V1 FFNN seeds (n={len(audit)})")
    bp = ax.boxplot(audit.sharpe, positions=[0], widths=0.3, showfliers=False)
    for elem in ("boxes", "whiskers", "caps", "medians"):
        plt.setp(bp[elem], color="#666666")
    committed = float(audit.loc[audit.seed == 0, "sharpe"].iloc[0])
    ax.annotate("committed value\n(seed 0)", (0.09, committed),
                fontsize=8, color="#444444", va="center")
    # Deterministic D-variants: single points, zero seed variance by design.
    m = matrix[(matrix.method == "dynotears") & (matrix.window == 252)]
    for k, (a, color) in enumerate((("D0", C["D0"]), ("D1", C["D1"]), ("D2s", C["D2s"]))):
        val = float(m.loc[m.allocator == a, "sharpe"].iloc[0])
        ax.scatter([k + 1], [val], marker="D", s=55, color=color, zorder=3)
        ax.annotate("deterministic", (k + 1, val - 0.004), fontsize=7,
                    ha="center", color=color)
    v0 = 0.371
    ax.axhline(v0, color=C["V0"], lw=1, ls=":")
    ax.annotate("V0 baseline", (1.5, v0 + 0.0005), fontsize=8, color=C["V0"])
    ax.set_xticks(range(4), ["V1 (FFNN)", "D0", "D1", "D2s"])
    ax.set_ylabel("net Sharpe (w252)")
    ax.set_title("FFNN-seed variability of the Phase-I pipeline vs seed-free Phase-II allocators",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIG / "phase_ii_seed.png", dpi=200)
    plt.close(fig)


# ============================================================================
# F5 — regime excess over V0 (w252)
# ============================================================================
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
    show = [("V0prime", "D0 ≡ V0′", C["D0"]), ("DYNO-D1", "D1", C["D1"]),
            ("DYNO-D2s", "D2s", C["D2s"]), ("V1-DYNOTEARS", "V1", C["V1"])]
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
    ax.set_ylabel("Sharpe excess over V0")
    ax.set_title("Regime-conditional edge over correlation-HSP (252-day window)",
                 fontsize=10.5)
    ax.legend(fontsize=8.5)
    fig.savefig(FIG / "phase_ii_regime.png", dpi=200)
    plt.close(fig)


# ============================================================================
# F6 — the decomposition (the master question in one picture)
# ============================================================================
def f6_decomposition(matrix: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    """Net Sharpe of CORR → D0 (+skeleton) → D1/D2s (+orientation), DYNOTEARS,
    with the pairwise bootstrap Δ and p annotated on each step."""
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

    bars = [("CORR", "#000000"), ("D0", C["D0"]), ("D1", C["D1"]), ("D2s", C["D2s"])]
    all_vals = {w: [sharpe("phase_i", "CORR-HRP", w)] + [
        sharpe("dynotears", a, w) for a, _ in bars[1:]] for w in (252, 504)}
    lo = min(v for vs in all_vals.values() for v in vs) - 0.006
    hi = max(v for vs in all_vals.values() for v in vs) + 0.007

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True,
                             constrained_layout=True)
    # The step each bar adds relative to its reference (CORR for D0; D0 for D1/D2s).
    steps = {1: ("D0-CORR", "+ skeleton"), 2: ("D1-D0", "+ orientation"),
             3: ("D2s-D0", "+ orientation")}
    for ax, w in zip(axes, (252, 504)):
        vals = all_vals[w]
        xs = np.arange(len(bars))
        ax.bar(xs, vals, 0.62, color=[c for _, c in bars])
        ax.axhline(vals[0], color="#000000", lw=0.9, ls=":")
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.3f}", (x, v + 0.0012), ha="center", fontsize=9)
            if x in steps:
                d = delta(steps[x][0], w)
                if d is not None:
                    dv, p = d
                    ax.annotate(f"{steps[x][1]}\n{dv:+.3f}\n(p={p:.2f})",
                                (x, lo + 0.55 * (min(vals) - lo) + 0.004),
                                ha="center", va="bottom", fontsize=7.8,
                                color="white", fontweight="bold")
        ax.set_xticks(xs, [b for b, _ in bars])
        ax.set_ylim(lo, hi)
        ax.set_title(f"{w}-day window", fontsize=10)
    axes[0].set_ylabel("net Sharpe")
    fig.suptitle(
        "Decomposing the causal-graph gain over the correlation matrix: "
        "skeleton vs edge orientation (DYNOTEARS)", fontsize=10.5)
    fig.savefig(FIG / "phase_ii_decomposition.png", dpi=200)
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
    print(f"figures → {FIG}/phase_ii_*.png")


if __name__ == "__main__":
    main()
