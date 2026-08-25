"""Collate Phase II: headline matrix plus the pre-specified contrasts.

Reads the results/phase_ii_* bundles and Phase-I comparators, emits
results/phase_ii_matrix.csv and results/phase_ii_contrasts.csv
(bootstrap ΔSharpe CIs, 10000 resamples).

Usage:  python -m scripts.collate_phase_ii
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT
from pipeline.evaluation.bootstrap import sharpe_difference_ci
from pipeline.evaluation.metrics import annualised_sharpe

RESULTS = THESIS_ROOT / "results"
N_RESAMPLES = 10000
METHODS = ("dynotears", "varlingam", "granger")
ALLOCS = ("D0", "D0s", "D1", "D2", "D2s", "D3", "D4")
WINDOWS = (189, 252, 378, 504)
DIRECTION_AWARE = ("D1", "D2", "D2s", "D3", "D4")
# Mechanism controls, DYNOTEARS only. Kept out of ALLOCS so they never
# enter the family contrasts or the best(D*) selection.
CONTROL_ALLOCS = ("D0lw", "D0df", "D0pc")
# Graph-blind anchors: levels only, no family.
ANCHORS = {
    "EW": "phase_ii_ew_w{w}",
    "IVP": "phase_ii_ivp_w{w}",
}
# HERC cells (PREDICTIONS_HERC.md); outside both SPA families by pre-commitment.
HERC = {
    "HERCC": "phase_ii_herc_corr_w{w}",
    "HERC0": "phase_ii_dynotears_HERC0_w{w}",
    "HERC1": "phase_ii_dynotears_HERC1_w{w}",
}

PHASE_I = {
    "V0": "phase_i_v0_w{w}",
    "V0prime": "phase_i_v0prime_w{w}",
    "V1-DYNOTEARS": "phase_i_v1_w{w}",
    "V1-VARLiNGAM": "phase_i_v1_varlingam_w{w}",
    # Correlation control for the skeleton-vs-orientation decomposition.
    "CORR-HRP": "phase_ii_corr_hrp_w{w}",
}


def _load_returns(tag: str) -> pd.Series | None:
    path = RESULTS / tag / "closed_loop.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        bt = pickle.load(fh)["backtest"]
    r = bt.nav_net.pct_change().dropna()
    r.attrs["nav_final"] = float(bt.nav_net.iloc[-1])
    r.attrs["max_drawdown"] = float((bt.nav_net / bt.nav_net.cummax() - 1).min())
    r.attrs["mean_turnover"] = float(np.mean([x.turnover for x in bt.rebalances]))
    r.attrs["nav_index"] = bt.nav_net.index
    return r


def _metrics_row(r: pd.Series) -> dict:
    years = (r.index[-1] - r.index[0]).days / 365.25
    return {
        "sharpe": annualised_sharpe(r),
        "cagr": float(r.attrs["nav_final"] ** (1 / years) - 1),
        "max_drawdown": r.attrs["max_drawdown"],
        "mean_turnover": r.attrs["mean_turnover"],
        "nav_net": r.attrs["nav_final"],
    }


def _contrast(name: str, a: pd.Series, b: pd.Series, window: int, method: str) -> dict:
    ci = sharpe_difference_ci(a, b, n_resamples=N_RESAMPLES, seed=42)
    return {
        "contrast": name, "method": method, "window": window,
        "delta_sharpe": ci.point_estimate,
        "ci_lower": ci.ci_lower, "ci_upper": ci.ci_upper,
        "p_value": ci.p_value_two_sided,
    }


def main() -> None:
    # Load everything
    rets: dict[tuple[str, str, int], pd.Series] = {}
    for m in METHODS:
        for a in ALLOCS:
            for w in WINDOWS:
                r = _load_returns(f"phase_ii_{m}_{a}_w{w}")
                if r is not None:
                    rets[(m, a, w)] = r
    for a in CONTROL_ALLOCS:
        for w in WINDOWS:
            r = _load_returns(f"phase_ii_dynotears_{a}_w{w}")
            if r is not None:
                rets[("dynotears", a, w)] = r
    comparators: dict[tuple[str, int], pd.Series] = {}
    for name, stem in PHASE_I.items():
        for w in WINDOWS:
            r = _load_returns(stem.format(w=w))
            if r is not None:
                comparators[(name, w)] = r
    anchors: dict[tuple[str, int], pd.Series] = {}
    for name, stem in ANCHORS.items():
        for w in WINDOWS:
            r = _load_returns(stem.format(w=w))
            if r is not None:
                anchors[(name, w)] = r
    herc: dict[tuple[str, int], pd.Series] = {}
    for name, stem in HERC.items():
        for w in WINDOWS:
            r = _load_returns(stem.format(w=w))
            if r is not None:
                herc[(name, w)] = r

    # Matrix
    rows = []
    for (m, a, w), r in sorted(rets.items()):
        rows.append({"method": m, "allocator": a, "window": w, **_metrics_row(r)})
    for (name, w), r in sorted(comparators.items()):
        rows.append({"method": "phase_i", "allocator": name, "window": w, **_metrics_row(r)})
    for (name, w), r in sorted(anchors.items()):
        rows.append({"method": "anchor", "allocator": name, "window": w, **_metrics_row(r)})
    for (name, w), r in sorted(herc.items()):
        rows.append({"method": "herc", "allocator": name, "window": w, **_metrics_row(r)})
    matrix = pd.DataFrame(rows)
    matrix.to_csv(RESULTS / "phase_ii_matrix.csv", index=False)
    print("=== phase_ii_matrix.csv ===")
    pivot = matrix[matrix.method != "phase_i"].pivot_table(
        index=["method", "allocator"], columns="window", values="sharpe"
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nPhase-I comparators (net Sharpe):")
    print(matrix[matrix.method == "phase_i"]
          .pivot_table(index="allocator", columns="window", values="sharpe")
          .to_string(float_format=lambda x: f"{x:.3f}"))

    # Contrasts
    out = []
    for m in ("dynotears", "varlingam"):
        for w in WINDOWS:
            d0 = rets.get((m, "D0", w))
            if d0 is None:
                continue
            # 1. Direction effect, fixed graph.
            for a in ("D0s",) + DIRECTION_AWARE:
                r = rets.get((m, a, w))
                if r is not None:
                    out.append(_contrast(f"{a}-D0", r, d0, w, m))
            # 2. D0 - V0 (replication + its VAR analogue).
            v0 = comparators.get(("V0", w))
            if v0 is not None:
                out.append(_contrast("D0-V0", d0, v0, w, m))
            # 2b. Decomposition anchors vs the correlation control:
            #     total(D*) - CORR = skeleton(D0 - CORR) + orientation(D* - D0).
            corr = comparators.get(("CORR-HRP", w))
            if corr is not None:
                out.append(_contrast("D0-CORR", d0, corr, w, m))
                for a in ("D1", "D2s"):
                    r = rets.get((m, a, w))
                    if r is not None:
                        out.append(_contrast(f"{a}-CORR", r, corr, w, m))
            # 3. best(D*) - V1 under the same discovery method.
            v1_key = "V1-DYNOTEARS" if m == "dynotears" else "V1-VARLiNGAM"
            v1 = comparators.get((v1_key, w))
            cands = {a: rets[(m, a, w)] for a in ALLOCS if (m, a, w) in rets}
            if v1 is not None and cands:
                best_a = max(cands, key=lambda a: annualised_sharpe(cands[a]))
                out.append(_contrast(f"best({best_a})-V1", cands[best_a], v1, w, m))
    # 4. GRANGER vs DYNO at the best allocator (E2; skip until bundles exist).
    for w in WINDOWS:
        g_cands = {a: rets[("granger", a, w)] for a in ALLOCS if ("granger", a, w) in rets}
        if not g_cands:
            continue
        d_cands = {a: rets[("dynotears", a, w)] for a in ALLOCS if ("dynotears", a, w) in rets}
        best_a = max(d_cands, key=lambda a: annualised_sharpe(d_cands[a]))
        if best_a in g_cands:
            out.append(_contrast(
                f"GRANGER-DYNO@{best_a}", g_cands[best_a], d_cands[best_a], w, "granger",
            ))

    # 5. Mechanism controls: how much of the D1 - D0 gap does each
    #    direction-free covariance reproduce?
    for w in WINDOWS:
        d0 = rets.get(("dynotears", "D0", w))
        d1 = rets.get(("dynotears", "D1", w))
        if d0 is None:
            continue
        for a in CONTROL_ALLOCS:
            r = rets.get(("dynotears", a, w))
            if r is None:
                continue
            out.append(_contrast(f"{a}-D0", r, d0, w, "dynotears"))
            if d1 is not None:
                out.append(_contrast(f"D1-{a}", d1, r, w, "dynotears"))
        # Skeleton-channel control: does the graph-free partial-correlation
        # skeleton reproduce D0's gain over CORR?
        pc = rets.get(("dynotears", "D0pc", w))
        corr = comparators.get(("CORR-HRP", w))
        if pc is not None and corr is not None:
            out.append(_contrast("D0pc-CORR", pc, corr, w, "dynotears"))

    # 6. HERC: the two channels on the second tree-reading rule.
    for w in WINDOWS:
        hc, h0, h1 = (herc.get(("HERCC", w)), herc.get(("HERC0", w)),
                      herc.get(("HERC1", w)))
        if hc is not None and h0 is not None:
            out.append(_contrast("HERC0-HERCC", h0, hc, w, "herc"))
        if h0 is not None and h1 is not None:
            out.append(_contrast("HERC1-HERC0", h1, h0, w, "herc"))

    contrasts = pd.DataFrame(out)
    contrasts.to_csv(RESULTS / "phase_ii_contrasts.csv", index=False)
    print("\n=== phase_ii_contrasts.csv ===")
    print(contrasts.to_string(
        index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x:.0f}",
    ))


if __name__ == "__main__":
    main()
