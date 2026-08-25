"""Collate the J4 tables: K-sensitivity (J4a) and alpha/gamma sweep (J4b).

Computes net Sharpe per config and bootstrap ΔSharpe vs the matched baseline,
same methodology as J2/J3.

Run:  python -m scripts.collate_j4
Outputs: results/j4a_k_sensitivity.csv, results/j4b_alpha_gamma.csv
"""
from __future__ import annotations

import pathlib
import pickle

import pandas as pd

from pipeline.evaluation.bootstrap import sharpe_difference_ci
from pipeline.evaluation.metrics import annualised_sharpe, max_drawdown

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

KS = [10, 14, 17, 20, 25]
WINDOWS = [252, 504]
AG_GRID = [(0.4, 0.1), (0.4, 0.3), (0.4, 0.5),
           (0.6, 0.1), (0.6, 0.3), (0.6, 0.5),
           (0.8, 0.1), (0.8, 0.3), (0.8, 0.5)]


def _rets(tag: str) -> pd.Series | None:
    path = RESULTS / tag / "closed_loop.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        bt = pickle.load(fh)["backtest"]
    return bt.nav_net.pct_change().dropna()


# J4 configs run fresh with a suffix; committed phase_i_* bundles stay the
# headline. main() checks fresh K=17 reproduces the committed numbers.
def _v0_tag(w, k):  return f"phase_i_v0_w{w}_k{k}"
def _v1_tag(w, k):  return f"phase_i_v1_w{w}_k{k}"
def _v2_tag(a, g):  return f"phase_i_v2_w252_a{a}_g{g}"


def collate_j4a() -> pd.DataFrame:
    rows = []
    for w in WINDOWS:
        for k in KS:
            r0, r1 = _rets(_v0_tag(w, k)), _rets(_v1_tag(w, k))
            if r0 is None or r1 is None:
                print(f"  [skip] w{w} K={k}: missing "
                      f"{'V0 ' if r0 is None else ''}{'V1' if r1 is None else ''}")
                continue
            ci = sharpe_difference_ci(r1, r0, n_resamples=2000)
            rows.append({
                "window": w, "K": k,
                "sharpe_V0": round(annualised_sharpe(r0), 4),
                "sharpe_V1": round(annualised_sharpe(r1), 4),
                "dsharpe_V1_minus_V0": round(ci.point_estimate, 4),
                "p_value": round(ci.p_value_two_sided, 4),
                "maxdd_V1": round(max_drawdown(r1), 4),
            })
    return pd.DataFrame(rows)


def collate_j4b() -> pd.DataFrame:
    # Baseline is fresh V1 w252 K=17; fall back to committed if absent.
    r_v1 = _rets("phase_i_v1_w252_k17")
    if r_v1 is None:
        r_v1 = _rets("phase_i_v1_w252")
    rows = []
    for a, g in AG_GRID:
        r2 = _rets(_v2_tag(a, g))
        if r2 is None:
            print(f"  [skip] V2 a={a} g={g}: missing bundle")
            continue
        row = {"alpha": a, "gamma": g, "sharpe_V2": round(annualised_sharpe(r2), 4)}
        if r_v1 is not None:
            ci = sharpe_difference_ci(r2, r_v1, n_resamples=2000)
            row["dsharpe_V2_minus_V1"] = round(ci.point_estimate, 4)
            row["p_value"] = round(ci.p_value_two_sided, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _repro_check() -> None:
    """Confirm fresh K=17 (frozen-EEM) reproduces the committed headline."""
    from pipeline.evaluation.metrics import annualised_sharpe as sh
    print("Reproducibility — fresh K=17 (frozen EEM) vs committed headline:")
    for w in WINDOWS:
        for v in ("v0", "v1"):
            fresh = _rets(f"phase_i_{v}_w{w}_k17")
            comm = _rets(f"phase_i_{v}_w{w}")
            if fresh is not None and comm is not None:
                print(f"  {v.upper()} w{w}: fresh={sh(fresh):.4f}  committed={sh(comm):.4f}  "
                      f"Δ={sh(fresh)-sh(comm):+.4f}")


def main() -> None:
    _repro_check()
    print("\nJ4a — K-sensitivity (V1 vs V0):")
    j4a = collate_j4a()
    if not j4a.empty:
        j4a.to_csv(RESULTS / "j4a_k_sensitivity.csv", index=False)
        print(j4a.to_string(index=False))
        print("\n  pivot — net Sharpe by variant × K:")
        for w in WINDOWS:
            sub = j4a[j4a.window == w]
            if not sub.empty:
                print(f"  w{w}:")
                print(sub.set_index("K")[["sharpe_V0", "sharpe_V1",
                      "dsharpe_V1_minus_V0", "p_value"]].to_string())

    print("\nJ4b — alpha/gamma sweep (V2 vs V1 open-loop, w252):")
    j4b = collate_j4b()
    if not j4b.empty:
        j4b.to_csv(RESULTS / "j4b_alpha_gamma.csv", index=False)
        print(j4b.to_string(index=False))

    print(f"\nsaved → {RESULTS}/j4a_k_sensitivity.csv, {RESULTS}/j4b_alpha_gamma.csv")


if __name__ == "__main__":
    main()
