# Plan: Closed-Loop Causal-HSP Portfolio Framework

## Context

This supersedes `Causal-HRP Portfolio Construction Framework.md` following the shift in research direction. Key changes:

| | Old plan | New plan |
|---|---|---|
| Base method | HRP variants only | HSP variants (HRP becomes a benchmark) |
| Locus of innovation | Distance matrix + allocation (V1, V2, V3) | Driver selection inside HSP's clustering step (allocation unchanged) |
| Loop | Open-loop (data → portfolio → end) | Closed-loop (backtest performance updates driver utility, which feeds back to selection) |
| Variants | Causal-HRP V1/V2/V3 | Causal-HSP V0/V1/V2 with V0' as asset-only ablation |
| Drivers | None (asset-asset causal graph) | Causally-selected exogenous drivers from Stage 1 |

Two important supervisor redirects baked in:
- **"Either work on clustering or allocation, not both"** → all causal innovation goes into clustering via driver selection; recursive bisection with sample-variance stays unchanged.
- **"Can't just symmetrise — need to take advantage of causal info"** → sensitivity-space distance (Euclidean by construction, no symmetrisation needed) replaces the symmetrised asset-asset adjacency.

This is **Stage 2** of the thesis pipeline. It consumes Stage 1 outputs (see `Causal_Factor_Discovery_Pipeline.md`).

---

## Locked implementation choices

Decisions made during the planning pass (mirrors the table in `Causal Factor Discovery Pipeline.md`, with Stage 2-specific implications):

| Decision | Choice | Stage 2 impact |
|---|---|---|
| Module layout | Nested `pipeline/{portfolio,feedback,evaluation}/` alongside the Stage 1 tree | `causal_hsp.py` reads Stage 1 Parquet directly; no shared globals |
| Asset universe data source | `fja05680/sp500` + **WRDS / CRSP** (primary) + `yfinance` fallback for the most recent ~2 trading days | Delisted prices honoured in the backtest's holding period — never silently dropped. CRSP also provides historical shares-outstanding so the S&P-100 market-cap selection is point-in-time. |
| FFNN framework | PyTorch multi-head model per window, weights cached by `(window_key, K)` | Hyperparameter sweeps over α / γ / linkage / K never retrigger Stage 1 training |
| Utility table keying | Keyed by **holding-period-end** date, not rebalance date | Selector at rebalance `t'` reads the latest row whose end ≤ `t'`; lookahead assertion `end ≤ t' - 21d` raises if violated |
| DYNOTEARS prior-knowledge | Native `causalnex` `tabu_edges` (no optimiser surgery) | Stage 1 cost-of-discovery risk that previously threatened the V2 timeline is gone |

---

## Central thesis question

> Does (a) replacing HSP's correlation-based driver selection with causal-discovery-based selection, and (b) closing the loop by letting realised out-of-sample portfolio performance update driver utility scores, produce more robust portfolios than HSP and HRP — particularly around regime changes (NBER recessions, vol spikes)?

The contributions are testable independently via the ablation matrix below.

---

## Pipeline overview (one rebalance cycle)

```
At rebalance date t:

  Stage 1:
    1. Causal discovery on [drivers | assets] over window [t-W, t]
       → W, A_p, bootstrap probabilities
    2. Greedy factor selection:
          score_d = α · causal_score_d + (1-α) · U_d[t-1]
       → selected_drivers[t]  (top K by score)
    3. FFNN + AAD per asset on selected drivers
       → sensitivities[t]
    
  Stage 2:
    4. Build sensitivity distance matrix D[t] from sensitivities[t]
    5. Project to nearest PSD if needed
    6. Hierarchical clustering on D[t] → linkage tree
    7. Quasi-diagonalisation → asset order
    8. Recursive bisection with sample-covariance variances → weights w[t]
    9. Hold w[t] from t to t+21 trading days; record realised returns
   10. Compute realised reward R[t] (Sharpe-excess vs equal-weight on holding period)
   11. Sensitivity-weighted credit attribution to selected drivers
   12. Update driver utility: U_d[t] = γ · credit_d + (1-γ) · U_d[t-1]
   13. Persist U[t]; next rebalance reads U[t-1]
```

Strict walk-forward: U[t] depends only on realised returns from holding periods ending at or before t.

---

## Strategy variants — the ablation matrix

The contributions of (causal selection) and (closed-loop feedback) are isolated by running:

| Variant | Driver selection | Feedback loop | Purpose |
|---|---|---|---|
| **V0 — Vanilla HSP** | Cumulative correlation (Rodriguez-Dominguez 2023) | Off | Primary baseline |
| **V1 — Causal-HSP open-loop** | Causal greedy (Stage 1, α=1) | Off | Isolates the causal-selection contribution |
| **V2 — Causal-HSP closed-loop** | Causal greedy + utility blend (α<1) | On | The proposed full method |
| **V0' — Asset-only Causal-HRP** | No drivers; asset-asset graph as distance | Off | Tests whether exogenous drivers add value over asset-only causal info |

V0' uses the symmetrised asset-asset adjacency (`(|W| + |Wᵀ|) / 2`) as a distance, with a López-de-Prado distance form on top. It's the closest surviving cousin of the old V1 from the superseded plan. Included to address the natural question: *do you need exogenous drivers at all, or is asset-asset causal structure enough?*

DYNOTEARS and VARLiNGAM are both run for V1 and V2 → 4 method combinations on top of V0 and V0'. Total: ~6 causal strategies.

NTS-NOTEARS, if Stage 1 succeeds with it, replaces both DYNOTEARS discovery and FFNN sensitivities in a unified stack — adds 1-2 more variants for the ablation if time permits.

---

## Feedback loop mechanics

### Reward signal

At rebalance `t`, observe realised holding-period reward:
```
R[t] = annualised_Sharpe(portfolio over [t, t+21d]) 
       - annualised_Sharpe(1/N over [t, t+21d])
```
Reward is excess Sharpe vs equal-weight, computed only on the realised holding period. Using excess over 1/N (rather than absolute Sharpe) controls for market-wide regime effects in the reward signal — drivers don't get punished for being selected during a market crash that hit everything.

Alternative reward: CER excess. Run as sensitivity check; report both.

### Sensitivity-weighted credit attribution

Drivers that more strongly shaped this period's portfolio get more credit (positive or negative):

```
For each driver d in selected_drivers[t]:
    influence_d = Σ_i |w_i[t] · s_{i,d}[t]|
    normaliser  = Σ_{d' ∈ S[t]} influence_{d'}
    credit_d[t] = R[t] · influence_d / normaliser
```
where `s_{i,d}[t]` is asset i's sensitivity to driver d at window t and `w_i[t]` is the portfolio weight.

This preserves total reward: `Σ_d credit_d[t] = R[t]`. A driver with high sensitivities-times-weights gets a large slice; a driver that was "barely used" by the portfolio gets a small slice.

Unselected drivers receive no credit (their utility carries unchanged).

### Utility EMA update

```
U_d[t] = γ · credit_d[t] + (1-γ) · U_d[t-1]    for d in selected_drivers[t]
U_d[t] =                       U_d[t-1]         for d not selected
```

`γ ∈ (0,1)` is the EMA decay. Primary value γ = 0.3 (moderate memory); sensitivity sweep over {0.1, 0.3, 0.5}.

### Coupling to Stage 1 selection

Stage 1's greedy selection (Stage B) augments its scoring with utility:
```
selection_score_d[t] = α · z(causal_score_d[t]) + (1-α) · z(U_d[t-1])
```
where `z(·)` is a z-score normalisation across the candidate pool (so causal and utility scales are commensurate).

`α ∈ [0,1]` mixes pure causal evidence (α=1, equivalent to V1 open-loop) with pure historical utility (α=0, "what worked before, do again"). Primary value α = 0.6 (lean toward causal evidence, let history correct).

Sensitivity sweep over {0.4, 0.6, 0.8, 1.0}. The α=1 case recovers V1 exactly.

### Burn-in handling

For the first 6 rebalances of the backtest (≈6 months), force α=1 — no feedback. After that, blend in utility per the formula above. This avoids basing selection on a handful of historical observations.

### Lookahead discipline — the critical bit

At rebalance t, the selection uses `U[t-1]`, which itself was computed using only realised returns from holding periods strictly before t. Implementation:

- **U is stored as a time-indexed Parquet table keyed by *holding-period-end date*** (not rebalance date). The credit for rebalance t cannot be computed until t+21d when the holding period ends; storing rows by holding-period-end makes "what was known at time t'" a single explicit lookup: latest row whose `end_date ≤ t'`.
- Stage 1's `factor_selection/selector.py` reads U via `feedback.storage.lookup_utility(t')`, which:
    1. Reads the latest row of `utility.parquet` with `end_date ≤ t'`.
    2. Asserts `end_date ≤ t' - pd.Timedelta(days=21)` (raises on violation).
  This is the single hardest-edged lookahead guard in the pipeline; it is an `assert`, not a warning.
- Deliberately-broken variant (`feedback/leak_canary.py`): a separate code path that calls `lookup_utility_unsafe(t')` returning the row with `end_date ≤ t' + 21d` (one rebalance ahead). Sharpe should be visibly inflated vs the safe variant. Run monthly during long backtests as a leak-detection canary; alarm if the gap closes.

**Canonical V2 entry-point:** `pipeline.closed_loop.run_closed_loop`. The conventional `run_stage1 → run_stage2` two-pass path is correct for the open-loop V0prime / V1 variants, but for V2 it pre-computes all Stage 1 selections before the backtest starts, so the utility update at rebalance t never feeds back into the selector at rebalance t+1. `run_closed_loop` fixes that by interleaving Stage 1, backtest, and feedback per rebalance: at each t it slices the lookback window, runs Stage 1 just-in-time with the *live* `UtilityStore.as_lookup()`, builds the V2 weight vector, then in the post-rebalance hook (fired by `run_backtest` before advancing) computes credit + EMA-updates U + appends to the store. The next rebalance's strategy call sees the freshly-written row.

The closed-loop discipline is formally verified by `tests/test_closed_loop.py`, which contains three integration tests:
1. **t1** — after burn-in, the selector's `utility_lookup_timestamp` is populated for at least one rebalance, confirming the per-rebalance interleave is genuinely closing the loop (not just bookkeeping after the fact).
2. **t2** — with `α=1.0` (selector ignores U), `run_closed_loop`'s per-rebalance weights match a `run_stage1 → run_stage2(variants=["V1"])` baseline to ≤ 1e-8.
3. **t3** — `feedback.leak_canary.make_leaky_lookup` returns U rows whose holding-period-end is strictly after the 21-day cutoff (the guard the strict lookup refuses), proving the canary exposes future information on at least one rebalance.

---

## Backtest design (walk-forward)

- **Universe**: approximate S&P 100 (top 100 by market cap from S&P 500 historical, per Stage 1 data layer)
- **Rebalance cadence**: monthly (every 21 trading days)
- **Backtest period**: 2007-01-01 → 2024-12-31, with 2-year burn-in (2005-2006) → ~216 rebalances. This covers the GFC (2008-09), European debt crisis (2011), 2015 China-led vol spike, 2018Q4 selloff, COVID 2020, and 2022 rate-hike drawdown — the major regime events of the last 18 years.
- **Holding**: 1 month, daily mark-to-market
- **Long-only**, no leverage
- **Transaction costs**: 5 bps per unit one-way turnover (primary); sensitivity sweep at {0, 5, 10, 20} bps. Report results both gross and net.
- **Liquidity filter**: drop any stock with < $10M average daily volume in the 21d prior to rebalance
- **Data availability caveats**: HYG ETF starts April 2007 (drop from driver pool until then; use Moody's BAA - 10Y Treasury spread from FRED as substitute pre-2007). VVIX starts 2007 (drop from burn-in pool, available from 2007 onward).

---

## Benchmarks

All rebalanced monthly on the same dates:

1. **1/N (equal-weight)** — naive baseline
2. **Min-variance** — `min wᵀΣw` s.t. `Σw = 1`, `w ≥ 0` with Ledoit-Wolf shrunk covariance
3. **Markowitz MVO** — sample mean + shrunk covariance, moderate risk aversion (γ_RA = 3)
4. **Vanilla HRP** — sample correlation distance, sample-variance bisection (López de Prado 2016)
5. **Vanilla HSP** — cumulative-correlation driver selection, FFNN+AAD sensitivities, sensitivity distance, sample-variance bisection (Rodriguez-Dominguez 2023). This is V0.
6. **S&P 100 cap-weighted index** — market benchmark for absolute interpretation

Optimisation benchmarks (#2, #3) implemented in `cvxpy`.

---

## Metrics

For every strategy:

- **Risk-adjusted**: annualised Sharpe, Sortino, Calmar
- **Drawdown**: max drawdown, time underwater
- **Total return**: cumulative, CAGR
- **Risk**: annualised volatility, downside deviation
- **Concentration**: Herfindahl-Hirschman index, effective N, max weight
- **Turnover**: one-way annualised turnover
- **CER (certainty-equivalent return)**: at risk aversion γ_RA ∈ {1, 3, 5}. Primary headline at γ_RA = 3 (per Howard et al.); γ_RA = 1 (light) and γ_RA = 5 (heavy) reported as robustness. Conclusions should be stated robust across this range; if a strategy ranks differently at different γ_RA, that's a real finding for the discussion chapter.
- **Statistical**: stationary block bootstrap (Politis-Romano) → 95% CI on Sharpe difference vs V0. Reject H₀: ΔSharpe = 0 at 5%.
- **Regime-conditional**: Sharpe / drawdown / turnover computed *within* regimes:
  - NBER recession dates (binary)
  - VIX-based vol regimes (top vs bottom quintile)
  - Causal-network density regimes (per Howard et al.'s market-timing indicator construction, applied to V2's discovery outputs)

The regime breakdown is where causal methods are expected to differentiate themselves if at all.

---

## Module structure

```
thesis/pipeline/
  portfolio/
    hrp.py                 # Pure HRP (López de Prado 2016), pluggable distance & covariance
    hsp.py                 # HSP wrapper around hrp with sensitivity distance
    causal_hsp.py          # V1 (open-loop), V2 (closed-loop) — wraps hsp with Stage 1 outputs
    benchmarks.py          # 1/N, min-var, MVO, cap-weighted
    backtest.py            # Walk-forward simulation, rebalancing engine
    
  feedback/
    utility.py             # U state, sensitivity-weighted credit, EMA updates
    storage.py             # Time-indexed U table with lookahead assertions
    leak_canary.py         # Deliberately-broken variant for ongoing leak detection
    
  evaluation/
    metrics.py             # Sharpe, Sortino, MDD, CER, turnover, HHI
    regime.py              # NBER dates, VIX regimes, network-density regimes
    bootstrap.py           # Politis-Romano block bootstrap for Sharpe-diff CI
    
  stage2_pipeline.py       # Orchestrates Stage 1 outputs → backtest → metrics → results
  experiment.py            # Top-level: runs all variants × all hyperparameter configs
```

---

## Verification plan

1. **HRP unit test**: reproduce López de Prado's textbook example (chapter 16 of *Advances in Financial Machine Learning*) — weights match to 4 decimal places.
2. **HSP unit test**: reproduce Rodriguez-Dominguez's published figures on a small subset (figure-by-figure where possible).
3. **Smoke test**: 20 assets, 10 drivers, 2 years, single rebalance — verify weights sum to 1, non-negative, no NaNs.
4. **Sanity 1 — variants differ**: V0, V0', V1, V2 with same data should produce non-identical weights. If V1 and V0 are identical, the causal selection isn't changing anything (bug or null result — investigate).
5. **Sanity 2 — feedback degenerates correctly**: V2 with α=1, γ=0 should be identical to V1 (no feedback effect).
6. **Sanity 3 — leak canary works**: deliberately-broken variant (uses future U) produces visibly inflated Sharpe. If not, leak detection is broken.
7. **Sanity 4 — feedback loop direction**: on synthetic regime-switching data with a known optimal driver set, drivers that consistently appear with high-reward portfolios should accumulate higher U over time. Plot U trajectories per driver.
8. **PSD safeguard**: sensitivity distance matrix may have small negative eigenvalues from estimation noise; verify nearest-PSD projection triggers cleanly and downstream HRP runs without complex weights.
9. **Bootstrap CIs**: random shuffles of returns produce CIs that include zero (null behaves correctly).
10. **VARLiNGAM misspecification spot-check** (HSIC residual independence): across the annually-sampled VARLiNGAM windows (`error_independence_every_n_windows=12`), the median rejection rate at α=0.05 should be ~5%. Materially higher periods (e.g. a window flagged "LIKELY MISSPECIFIED" with rejection rate > 0.20) are reported in the methodology chapter as VARLiNGAM-misspecification caveats — V1/V2 results using the VARLiNGAM backend in those regimes should be interpreted with caution.
11. **End-to-end**: full backtest, all variants, all benchmarks, with and without transaction costs, primary and sensitivity-check hyperparameters. Save results to `thesis/results/` as Parquet for downstream plotting.

---

## Indicative timeline (≈10 weeks, working daily)

Today: 22 May 2026. Target completion: early August 2026. Hard deadline: end September.


| Week | Focus | Deliverable |
|---|---|---|
| 1 | Data layer: asset universe with point-in-time membership, driver universe assembly, preprocessing pipeline | Clean Parquet store of assets + drivers, calendar-aligned |
| 2 | Discovery: adapt existing rolling DYNOTEARS + VARLiNGAM to joint variable set; implement prior-knowledge masking | Stage 1 discovery output for a single window, verified |
| 3 | Factor selection: Stage A prune + Stage B greedy; sensitivity stability checks | Stage 1 selection output for full backtest period |
| 4 | Sensitivities: FFNN + AAD per asset; HSP from scratch with pluggable inputs | Vanilla HSP (V0) working end-to-end |
| 5 | Causal-HSP V1 (open-loop); full benchmark suite | V0, V0', V1 all running on full backtest |
| 6 | Feedback loop: utility storage, sensitivity-weighted credit, EMA updates, lookahead assertions, leak canary | V2 closed-loop running on full backtest |
| 7 | Evaluation suite: metrics, regime breakdowns, bootstrap CIs; hyperparameter sweeps | Full results matrix written to Parquet |
| 8 | Analysis: plots, regime-conditional breakdowns, ablation table; write-up of methodology + results chapters | Methodology + results drafts |
| 9 | NTS-NOTEARS extension (if on track); robustness sensitivity checks; first full thesis draft | First-pass complete draft |
| 10 | Iteration: polish, additional sensitivity checks if time, supervisor feedback incorporation | Submission-ready draft |

**Critical-path items** (must work for the thesis to land): weeks 1-7. NTS-NOTEARS is firmly in the "if there's slack" bucket. VARLiNGAM as a robustness ablation is also a stretch — if it's not running cleanly by end of week 5, deprioritise to "compute on a subset of windows for a sanity-check appendix" rather than a full ablation.

**Risks**:
- ~~DYNOTEARS prior-knowledge masking ends up requiring deeper surgery on `causalnex` than expected~~ — resolved: `causalnex` exposes `tabu_edges` natively; the wrapper just enumerates `(lag, asset_col, driver_col)` tuples (~10 k entries per window).
- FFNN training is slow at 100 assets × 216 rebalances → **mitigated by design**: single multi-head PyTorch model per window (shared hidden layers, 100 output heads trained jointly with per-asset MSE), reducing 100 per-asset fits per window to one batched fit. Weights cached by `(window_end_date, K, arch)` so α/γ/linkage sweeps replay Stage 1 without retraining. MPS/CUDA auto-detect; CPU fallback.
- Feedback loop oscillates or fails to converge → fall back to smaller γ; document and analyse why
- Pre-2007 data availability for credit-spread drivers (HYG) requires substitute (BAA-10Y); document mapping clearly in methodology chapter
- yfinance silently returns nothing for many delisted tickers (~30-40 financials delisted in GFC) → **resolved by switching to WRDS / CRSP as the primary asset price source.** `data/assets.py` cascade is now `wrds → yfinance`; CRSP is survivorship-bias-free and also exposes historical shares-outstanding (replacing the prior "today's shares × historical price" market-cap proxy). Every requested ticker is logged with its resolved source; coverage check still fires if joint resolution falls below 95 %.

---

## Out of scope

- Long-short / market-neutral construction (HRP/HSP are long-only by design)
- Higher-frequency rebalancing (intraday, weekly)
- Live execution / order routing
- Alternative allocation methods (HERC, NCO, risk parity variants) — could be follow-up after V0/V1/V2 results
- Direct backpropagation through HRP/HSP for end-to-end gradient updates — the feedback loop here is a scalar-reward online-update scheme, not differentiable optimisation
- Reinforcement-learning framing with policy gradients — possible extension but adds scope
- Path-dependent HSP (Vasicek-modelled sensitivities, per Rodriguez-Dominguez chapter 11) — interesting but adds a layer of stochastic modelling on top of an already-complex pipeline