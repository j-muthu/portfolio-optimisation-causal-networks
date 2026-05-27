"""End-to-end smoke test on a ~20-asset subset.

Covers the verification plans from both Plan A and Plan B:

* DYNOTEARS on a small subset -> output is a valid DAG with sensible weights.
* VARLiNGAM on the same subset -> ``B0`` is triangular under the causal order;
  the non-Gaussianity / error-independence assumption is checked.
* Rolling windows for both methods -> graphs change but stay structurally close.
* Graph analysis & regime detection on the rolling sequences.
* Head-to-head comparison of DYNOTEARS ``W`` vs VARLiNGAM ``B0``.
* HRP portfolio construction from a causal matrix.

Run from the thesis root::

    .venv/bin/python -m pipeline.smoke_test
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from pipeline.data import build_dataset
from pipeline.discovery.diagnostics import (
    analyse_rolling,
    causal_order_drift,
    compare_rolling,
    detect_regime_changes,
)
from pipeline.portfolio import compare_hrp
from pipeline.discovery.dynotears import run_dynotears_window, run_rolling_dynotears
from pipeline.discovery.varlingam import run_rolling_varlingam, run_varlingam_window

# 20 large-cap, long-history names across several GICS sectors.
SMOKE_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",  # Tech / comms
    "JPM", "BAC", "GS",                                # Financials
    "XOM", "CVX",                                      # Energy
    "JNJ", "PFE",                                      # Health care
    "PG", "KO", "WMT", "HD",                           # Staples / discretionary
    "DIS", "VZ", "T",                                  # Comms / telecom
]
START, END = "2018-01-01", "2023-01-01"
WINDOW, STEP = 504, 189  # ~2-year window, ~9-month step -> a handful of windows


def _banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # ------------------------------------------------------------------
    # 1. Shared data pipeline
    # ------------------------------------------------------------------
    _banner("1. DATA PIPELINE")
    ds = build_dataset(start=START, end=END, tickers=SMOKE_TICKERS)
    print(f"  dataset: {ds!r}")
    print(f"  n (days) = {ds.n},  d (assets) = {ds.d}")
    print(f"  ADF p-values: max={ds.adf_pvalues.max():.2e} "
          f"(all < 0.01 => stationary: {bool((ds.adf_pvalues < 0.01).all())})")
    print(f"  dropped: {ds.dropped}")
    assert ds.n > WINDOW, f"need > {WINDOW} rows, got {ds.n}"
    assert ds.d >= 15, f"expected ~20 assets, got {ds.d}"

    # ------------------------------------------------------------------
    # 2. DYNOTEARS single window -> valid DAG?
    # ------------------------------------------------------------------
    _banner("2. DYNOTEARS SMOKE (single window)")
    window_df = ds.returns.iloc[:WINDOW]
    W, A, converged, removed = run_dynotears_window(
        window_df, p=1, lambda_w=0.05, lambda_a=0.05
    )
    intra = nx.DiGraph()
    intra.add_edges_from(zip(*(a.tolist() for a in np.nonzero(W))))
    is_dag = nx.is_directed_acyclic_graph(intra) if intra.number_of_nodes() else True
    print(f"  converged={converged}  intra-edges={int(np.count_nonzero(W))}  "
          f"inter-edges={int(np.count_nonzero(A[0]))}  "
          f"(edges removed to enforce acyclicity: {removed})")
    print(f"  W weight range: [{np.abs(W[W != 0]).min():.4f}, "
          f"{np.abs(W[W != 0]).max():.4f}]"
          if np.any(W != 0) else "  W is empty")
    print(f"  contemporaneous graph is a DAG: {is_dag}")
    assert is_dag, "DYNOTEARS contemporaneous graph must be acyclic"

    # ------------------------------------------------------------------
    # 3. VARLiNGAM single window -> triangular B0 + assumption check
    # ------------------------------------------------------------------
    _banner("3. VARLiNGAM SMOKE (single window)")
    vwin = run_varlingam_window(
        window_df, lags=1, criterion="bic", compute_error_independence=True
    )
    order = vwin.causal_order
    permuted = vwin.B0[np.ix_(order, order)]  # i->j: should be strictly upper-triangular
    lower = np.tril(permuted)  # includes diagonal
    print(f"  B0 shape={vwin.B0.shape}  contemp-edges={vwin.n_contemp_edges}  "
          f"selected lags={vwin.selected_lags}")
    print(f"  causal order (upstream first): {vwin.causal_order_tickers}")
    print(f"  max |B0| below diagonal after causal-order permutation: "
          f"{np.abs(lower).max():.2e} (should be ~0)")
    pvals = vwin.error_indep_pvalues
    frac_ok = float(np.mean(pvals[np.triu_indices_from(pvals, k=1)] > 0.05))
    print(f"  error-independence: {frac_ok:.0%} of pairs have p > 0.05")
    assert np.abs(lower).max() < 1e-6, "B0 not triangular under the causal order"

    # ------------------------------------------------------------------
    # 4. Rolling windows -- both methods
    # ------------------------------------------------------------------
    _banner("4. ROLLING WINDOWS")
    dyn = run_rolling_dynotears(ds, window=WINDOW, step=STEP, p=1,
                                lambda_w=0.05, lambda_a=0.05)
    var = run_rolling_varlingam(ds, window=WINDOW, step=STEP, lags=1, criterion="bic")
    print(f"  DYNOTEARS: {len(dyn)} windows")
    print(dyn.to_frame().to_string(index=False))
    print(f"\n  VARLiNGAM: {len(var)} windows")
    print(var.to_frame().to_string(index=False))
    assert len(dyn) == len(var) >= 3, "expected >= 3 comparable windows"

    # ------------------------------------------------------------------
    # 5. Graph analysis & regime detection
    # ------------------------------------------------------------------
    _banner("5. GRAPH ANALYSIS & REGIME DETECTION")
    dyn_metrics = analyse_rolling(dyn)
    print("  DYNOTEARS rolling metrics:")
    print(dyn_metrics.to_string())
    regimes = detect_regime_changes(dyn_metrics, n_sigma=1.0)
    print(f"  regime-change candidates (>1 sigma): "
          f"{[d.date().isoformat() for d in regimes.index]}")
    drift = causal_order_drift(var)
    print("\n  VARLiNGAM causal-order drift:")
    print(drift.to_string())

    # ------------------------------------------------------------------
    # 6. Head-to-head DYNOTEARS vs VARLiNGAM
    # ------------------------------------------------------------------
    _banner("6. HEAD-TO-HEAD (W vs B0)")
    cmp = compare_rolling(dyn, var)
    print(cmp[["frobenius_distance", "edge_jaccard", "sign_agreement",
               "weight_correlation"]].to_string())

    # ------------------------------------------------------------------
    # 7. Portfolio integration (HRP)
    # ------------------------------------------------------------------
    _banner("7. PORTFOLIO INTEGRATION (HRP)")
    last_window_returns = ds.returns.iloc[dyn.windows[-1].start_row:
                                          dyn.windows[-1].end_row]
    last_window_returns.columns = ds.tickers
    weights = compare_hrp(last_window_returns, dyn.windows[-1].W, distance="embedding")
    print(weights.to_string())
    print(f"  weight sums: correlation={weights['correlation_hrp'].sum():.6f}  "
          f"causal={weights['causal_hrp'].sum():.6f}")
    assert np.isclose(weights["causal_hrp"].sum(), 1.0), "HRP weights must sum to 1"

    _banner("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
