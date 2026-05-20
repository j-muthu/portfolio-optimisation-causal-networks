# Plan: Causal-HRP Portfolio Construction Framework

## Context

This is the **second stage** of the thesis pipeline. The first stage (already planned in `thesis/Causal Discovery Pipelines for S&P 500.md`) produces a time series of causal adjacency matrices from DYNOTEARS and VARLiNGAM over rolling 2-year windows with a 1-month step on S&P 500 daily log-returns.

This stage answers the central thesis question: **do causal discovery methods improve portfolio optimisation versus correlation-based approaches?** It does so by injecting the causal matrices into Hierarchical Risk Parity (HRP) and benchmarking the resulting portfolios against a range of standard alternatives.

The intended outcome is a single experiment harness that, given a discovery method (DYNOTEARS or VARLiNGAM), produces out-of-sample backtest results across three causal-HRP variants and four benchmarks, with metrics, statistical confidence intervals, and regime-conditional breakdowns.

---

## Module Structure

All under `thesis/pipeline/portfolio/`:

| File | Purpose |
|------|---------|
| `hrp.py` | Pure HRP implementation (from scratch) — pluggable input matrix |
| `causal_hrp.py` | Three causal-HRP variants (clustering-only, structural covariance, blended) |
| `benchmarks.py` | 1/N, min-variance, Markowitz MVO, cap-weighted index |
| `backtest.py` | Walk-forward simulation, rebalancing engine |
| `metrics.py` | Sharpe, Sortino, MDD, Calmar, turnover, concentration, block-bootstrap CIs |
| `experiment.py` | Top-level orchestration; takes a method config, runs all variants + benchmarks |

Implement HRP **from scratch** using `scipy.cluster.hierarchy` (~80 LOC) — academic transparency, full control over what's plugged in where. Avoid `riskfolio-lib` / `pyportfolioopt` (they hide the matrix-substitution points).

---

## HRP refresher (3 steps)

1. **Tree clustering**: distance matrix → `scipy.cluster.hierarchy.linkage` (single linkage) → hierarchical tree
2. **Quasi-diagonalisation**: reorder so similar assets sit adjacent
3. **Recursive bisection**: walk tree top-down, split allocation by **inverse-variance** between left/right subclusters

The input matrix enters in two places: **distance** (steps 1–2) and **variance/covariance** (step 3). The three causal variants differ in *which* of these gets the causal matrix.

---

## The three causal-HRP variants

Let `W` denote the contemporaneous causal adjacency (DYNOTEARS `W` or VARLiNGAM `B₀`), and `sym(W) = (|W| + |Wᵀ|) / 2`.

### V1 — Clustering-only
- Normalise `sym(W)` to `[0, 1]` by dividing by `max(sym(W))`
- Causal distance: `d_ij = sqrt(0.5 * (1 - sym_norm(W)_ij))` (López de Prado's distance form, treating normalised `sym(W)` as a similarity)
- Hierarchical tree + quasi-diag from this distance
- Allocation step: **standard sample variances**
- Interpretation: "do causal groupings produce better clusters than correlation groupings?"

### V2 — Structural covariance drop-in
- Compute `Σ_causal = (I - W)⁻¹ Σ_e (I - W)⁻ᵀ` where `Σ_e` is the diagonal residual covariance from the SVAR fit (available from both DYNOTEARS and VARLiNGAM outputs)
- Convert to correlation, then López de Prado distance
- Use `Σ_causal` for the inverse-variance allocation step
- **PSD safeguard**: if `Σ_causal` has negative eigenvalues, apply nearest-PSD projection or Ledoit-Wolf shrinkage toward sample covariance
- Interpretation: "does the implied causal covariance produce better risk parity than the sample covariance?"

### V3 — Blended
- Blended distance: `α * d_causal + (1-α) * d_correlation` for `α ∈ {0.25, 0.5, 0.75}`
- Allocation: blended covariance `α * Σ_causal + (1-α) * Σ_sample`
- Interpretation: "is the optimal mix of causal and statistical information non-trivial?"

---

## Matrix scope — primary vs sensitivity

**Primary results**: contemporaneous only (`W` / `B₀`).

**Sensitivity check**: combined matrix `W + Σᵢ Aᵢ` symmetrised. Report as a robustness section. Expectation (per Howard et al.): negligible difference for daily returns, but worth confirming.

This is parameterised via a `use_lagged: bool` flag in `experiment.py` so re-running for sensitivity is one config change.

---

## Backtest design (walk-forward)

Aligned with the discovery pipeline's rolling-window cadence:

- **Rebalance dates**: month-end (every ~21 trading days)
- **At each rebalance**: use the most-recent causal graph fitted on the 2-year window ending at that date
- **Hold**: 1 month, recording daily returns
- **Out-of-sample horizon**: 2014–2024 minus 2-year burn-in → ~8 years × 12 rebalances ≈ 96 rebalance events
- **Universe**: fixed S&P 500 constituents (Approach 1 from the discovery plan), with Approach 3 (intersection universe) as a robustness check
- **Long-only**, no leverage (HRP is naturally long-only)
- **Transaction costs**: model as fixed bps per unit turnover; report results both with and without

---

## Benchmarks (all rebalanced monthly on the same dates)

1. **Correlation-HRP** — plain HRP with sample correlation. The head-to-head test for the thesis question.
2. **Equal-weight (1/N)** — naive baseline, famously hard to beat.
3. **Min-variance** — `min wᵀΣw` s.t. `Σw = 1`, `w ≥ 0` using sample covariance with Ledoit-Wolf shrinkage.
4. **Markowitz MVO** — sample mean + shrunk covariance, target a moderate risk aversion.
5. **S&P 500 cap-weighted index** — market benchmark for absolute interpretation.

Use `cvxpy` for the optimisation-based benchmarks (3, 4).

---

## Metrics

Computed for every strategy:

- **Risk-adjusted**: annualised Sharpe, Sortino, Calmar
- **Drawdown**: max drawdown, time underwater
- **Total return**: cumulative, CAGR
- **Risk**: annualised volatility, downside deviation
- **Concentration**: Herfindahl-Hirschman index, effective N, max weight
- **Turnover**: one-way annualised turnover (proxy for trading cost burden)
- **Statistical**: stationary block bootstrap (Politis & Romano) → 95% CIs on Sharpe difference vs Correlation-HRP. Reject null `H₀: ΔSharpe = 0` at 5%.
- **Regime-conditional**: Sharpe / drawdown computed *within* regimes labelled by `graph_analysis.py` (from the discovery pipeline). This is where causal-HRP should shine if it does at all.

---

## Critical files (reused inputs from Stage 1)

| File | What we consume |
|------|-----------------|
| `thesis/pipeline/data.py` | Log-return DataFrame `(~2520, ~500)` |
| `thesis/pipeline/rolling_dynotears.py` | `{date: (W, A₁, Σ_e)}` mapping |
| `thesis/pipeline/rolling_varlingam.py` | `{date: (B₀, B₁, Σ_e, bootstrap_probs)}` mapping |
| `thesis/pipeline/graph_analysis.py` | Regime labels per date |

The portfolio module depends on these but does not modify them. If the residual covariance `Σ_e` is not currently exposed by the rolling modules, that's a small addition needed there.

---

## Verification plan

1. **Unit-test HRP**: reproduce López de Prado's textbook example (chapter 16 of *Advances in Financial Machine Learning*). Output weights should match to 4 decimal places.
2. **Smoke test**: 20 assets, 2 years, single rebalance. Verify weights sum to 1, all non-negative, no NaNs.
3. **Sanity 1 — V1 differs from correlation-HRP**: identical pipeline, different distance matrix, output weights should *not* be identical.
4. **Sanity 2 — V2 PSD check**: `Σ_causal` may have small negative eigenvalues from estimation noise; verify the PSD projection / shrinkage triggers cleanly and that downstream HRP runs without complex weights.
5. **Sanity 3 — Σ_causal sanity**: compare diagonal of `Σ_causal` to sample variances; large divergence flags W estimation problems.
6. **Walk-forward sanity**: cumulative return curves should be continuous (no look-ahead leaks). Run a deliberately broken version (using future graph) and confirm Sharpe is suspiciously high — sanity-check that the look-ahead would *be* detectable.
7. **Bootstrap CIs**: confirm that random shuffles of returns produce CIs that include zero (null behaves correctly).
8. **End-to-end**: full S&P 500 backtest, both methods (DYNOTEARS, VARLiNGAM), all three variants, all five benchmarks, with and without transaction costs, contemporaneous + lagged sensitivity. Save results to `thesis/pipeline/portfolio/results/` as Parquet for downstream plotting / thesis figures.

---

## Out of scope

- Long-short / market-neutral portfolios (HRP is long-only by construction; would need a separate framework)
- Higher-frequency rebalancing (intraday, weekly) — daily returns + monthly rebalance only
- Live execution / order routing
- Alternative portfolio constructors (HERC, NCO, risk parity variants) — could be follow-up work after HRP results are in
