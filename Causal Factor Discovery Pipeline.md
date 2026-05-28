# Plan: Causal Factor Discovery Pipeline

## Context

This supersedes `Causal Discovery Pipelines for S&P 500.md` following a shift in research direction. The shift, prompted by supervisor input, is from asset-to-asset causal discovery → driver-to-asset causal discovery, framed within Rodriguez-Dominguez's HSP (Hierarchical Sensitivity Parity) paradigm rather than HRP.

Key changes vs the superseded plan:

| | Old plan | New plan |
|---|---|---|
| Variable set | S&P 500 asset returns only | S&P 100 asset returns + curated exogenous driver pool |
| Output | Time series of asset-asset adjacency matrices | Time series of (a) selected driver subsets, (b) per-asset sensitivities to those drivers |
| Universe | 500 assets | 100 assets + ~30-60 drivers |
| Methods | DYNOTEARS, VARLiNGAM | DYNOTEARS (primary), VARLiNGAM (robustness), NTS-NOTEARS (extension) |
| Prior knowledge | None | Forbid asset → driver edges (encodes directional hypothesis) |

This is **Stage 1** of the thesis pipeline. Stage 2 (`Closed-Loop_Causal-HSP_Portfolio_Framework.md`) consumes the outputs of this stage and runs the portfolio construction and backtest with the closed-loop feedback.

---

## Locked implementation choices

Decisions made during the planning pass for the implementation; binding for the code build-out:

| Decision | Choice | Rationale |
|---|---|---|
| Module layout | Nested `pipeline/{data,discovery,factor_selection,sensitivities,portfolio,feedback,evaluation}/` | Replaces the flat single-file layout of the legacy asset-only pipeline |
| Asset universe data source | `fja05680/sp500` (point-in-time S&P 500 membership) + **WRDS / CRSP** (primary; survivorship-bias-free prices + historical shares-outstanding) + `yfinance` (fallback for the most recent ~2 trading days CRSP hasn't yet ingested) | WRDS is the canonical academic source. Removes the GFC survivorship-bias risk entirely and lets `top_n_by_mcap_at` use point-in-time market cap instead of "today's shares × historical price". Stooq and Tiingo were considered and dropped — stooq moved CSV access behind a paid tier in late 2025; Tiingo's free tier lacks delisted coverage. |
| Driver pool scope | Full ~50-series pool assembled up front (FRED + Yahoo) in week 1 | K calibration needs to see the real pool from the start |
| FFNN framework | PyTorch — single multi-head model per window (shared hidden layers, 100 output heads), jacobian via `torch.func.jacrev` | Compatible with later NTS-NOTEARS swap; cache weights by `(window_key, K)` so γ/α hparam sweeps don't retrain |
| DYNOTEARS prior-knowledge enforcement | Native `tabu_edges` (see §Causal discovery below) | `causalnex` already supports this — no optimiser surgery required |
| Per-window normalisation | Inside the rolling loop, not globally | Existing `pipeline/data.py::standardise` standardises globally; must be moved |

---

## Interface contract — what Stage 2 consumes

For each rebalance date `t`, Stage 1 produces:

| Object | Shape | Source module |
|---|---|---|
| `selected_drivers[t]` | ordered list of ≤K driver names | `factor_selection/` |
| `sensitivities[t]` | dict `{asset_i: vector ∈ ℝ^K}` | `sensitivities/` |
| `discovery_metadata[t]` | dict with W, A_p, bootstrap probs, fit diagnostics | `discovery/` |

Stage 1 also reads driver-utility state `U[t-1]` written by Stage 2's feedback loop (see `Closed-Loop_Causal-HSP_Portfolio_Framework.md`). On the first rebalance, U is initialised uniformly to zero.

---

## Data layer

### Asset universe — approximate S&P 100, point-in-time

Free historical S&P 100 membership is paywalled (Bloomberg, Refinitiv, WRDS). S&P 100 cannot be deduced from S&P 500 directly because S&P 100 is selected "for sector balance" with undisclosed discretion. The pragmatic substitute:

- **Source**: `fja05680/sp500` GitHub repo (S&P 500 historical constituents, 1996–present, free, widely used).
- **Selection rule**: at each rebalance date `t`, take the top 100 by market capitalisation from that date's S&P 500 membership. This is the standard free approximation to S&P 100 used in academic work. Document the approximation explicitly in the methodology chapter — it lacks the sector-balance discretion of the official index.
- **Burn-in**: 2 years (504 trading days) before first rebalance.
- **Daily log-returns**: `r_t = log(p_t / p_{t-1})`.
- **Delisted constituents**: carry last available price; subsequent returns NaN. For backtest, NaN on holding day → zero return that day with a transaction-cost penalty (proxy for forced sale at last close). Fetch full price history (including delisted tickers) from Yahoo or stooq.com to avoid survivorship bias.
- **Normalisation**: zero mean, unit variance over the fit window, before passing to discovery algorithms.

### Driver universe — exogenous candidates only

Target: 30-60 series, all defensibly exogenous to the S&P 100 universe.

| Category | Examples | Source |
|---|---|---|
| Macro indicators | CPI, core CPI, unemployment rate, ISM PMI, industrial production, retail sales, housing starts, consumer sentiment | FRED |
| Commodities | WTI/Brent oil, natural gas, gold, silver, copper, GSCI agricultural composite | Yahoo / FRED |
| FX | DXY, EUR/USD, USD/JPY, GBP/USD, USD/CNY | FRED |
| Treasury rates | 3M, 2Y, 5Y, 10Y, 30Y yields; 10Y-2Y slope; 10Y-3M spread | FRED |
| Credit spreads | BAA-AAA corporate spread; HYG-LQD ETF spread as high-yield proxy | FRED / Yahoo |
| Vol indicators | VIX, VVIX | CBOE / Yahoo |
| International equity | EAFE (EFA), EM (EEM), country-specific ETFs (EWJ, EWG, EWU) | Yahoo |

**Explicitly excluded** (failed exogeneity test):
- US sector SPDRs (XLK, XLF, XLE, etc.) — mechanical aggregations of S&P 100 constituents
- Fama-French / AQR factor returns — constructed from long-short portfolios of US equities that may include S&P 100 members; not exogenous in HSP's sense
- S&P 500, Russell 1000, NASDAQ-100 index returns — direct/near-direct functions of the asset universe

VIX is a borderline case (derived from S&P 500 options, so semi-endogenous). Include in primary runs but report a robustness check with VIX excluded.

### Preprocessing

- Log-returns for price series (commodities, FX, equity ETFs)
- First-differences for yield series (levels are non-stationary)
- YoY % change for macro indicators (FRED data is typically reported in levels or YoY%; align to a single convention)
- Forward-fill low-frequency macro indicators to daily, then difference for stationarity
- Align all series to NYSE trading calendar
- Normalize per window (z-score on the fit window) — matches Howard et al. and DYNOTEARS convention

### Per-asset eligibility masking (G.5.b)

Implemented in `pipeline/data/alignment.py::build_joint_matrix(drop_na='drivers_only')`. The plan's original `drop_na='any'` mode dropped any row where *any* column had NaN — which meant a single late-inception asset (e.g. LIN, formed by the Linde + Praxair merger in 2018-10) could cost 200+ rows for the other 99 survivors. The fix:

1. Keep every row where all *drivers* are populated (driver NaNs still cause row-drops; discovery can't tolerate them).
2. For *assets*, zero-fill any NaN cells and record a per-(row, asset) boolean `asset_eligibility` mask on the returned `JointMatrix`.
3. Downstream consumers honour the mask:
   - **FFNN** (`pipeline/sensitivities/ffnn.py::fit_sensitivities_window`) accepts the mask and per-asset-masks its training loss so pre-inception zero-fills don't bias the Jacobian toward zero.
   - **Closed-loop strategy** (`pipeline/closed_loop.py::run_closed_loop`'s `universe_at(t)` callable) consults `joint.assets_eligible_in_window(t - lookback, t)` and excludes assets whose full lookback isn't observed.

This is what makes the GFC-period backtest viable — many current S&P 500 names didn't exist or had different tickers in 2007-2010 (Meta as Facebook FB→META rename; Berkshire BRK.B; Alphabet GOOG vs GOOGL; etc.). The mask handles each cleanly.

### Stationarity diagnostics

ADF + KPSS on each driver after preprocessing, per window. Drivers failing both tests are flagged (not dropped) — the flag is logged with discovery output for downstream filtering.

---

## Causal discovery — three options, prioritised

All operate on the joint variable matrix `X = [D | A]` where D is the driver block and A is the asset block. Total dimension d ≈ 130-160 with n = 504.

### DYNOTEARS — primary

- **Implementation**: `causalnex.structure.dynotears.from_pandas_dynamic`
- **Window**: 504 trading days (2y), step 21 trading days (1m)
- **Lag**: `p = 1` primary; sensitivity check with `p = 3` (drivers may have longer lead times than asset-to-asset effects)
- **Regularisation**: `λ_w = λ_a = 0.05` initial; grid-search over {0.01, 0.05, 0.1} on a validation window every 6 months
- **Prior knowledge constraint**: enforce W[asset_i, driver_j] = 0 and A_p[asset_i, driver_j] = 0 for all (i, j, p) — i.e. forbid asset → driver edges. This encodes the directional hypothesis that, within the analysis timescale, drivers cause assets.

  Implementation: `causalnex.structure.dynotears.from_pandas_dynamic` already accepts a `tabu_edges: List[Tuple[lag, from, to]]` argument; under the hood (`from_numpy_dynamic`, lines 213-242 of `causalnex/causalnex/structure/dynotears.py`) it sets the corresponding cells of W and A to `(0, 0)` L-BFGS-B bounds, giving an exact hard constraint with no optimiser modification needed. The wrapper enumerates `(lag, asset_col, driver_col)` for every lag ∈ {0..p}, asset, driver — roughly 100 × 50 × 2 ≈ 10 k tuples per window, negligible cost.

  Verification: a refit without the tabu mask is run periodically; the magnitude of the difference on the driver → asset block is logged with the discovery output (small delta ⇒ the data already respects the directional hypothesis; large delta ⇒ the constraint is doing real work).

- **Output per window**: `(W, A_1, ..., A_p, Σ_e, fit_loss)`

- **Empirical runtime scaling (G.7 measurement)**: at d=134 (99 assets + 35 drivers), one DYNOTEARS fit with `max_iter=100` on a 252-day window takes ~3-4 min wall (Apple Silicon, single-threaded L-BFGS-B inside `scipy.optimize.minimize`). Vs ~17s at d=65 in Phase H, the scaling is ≈26× — closer to **O(d³·⁵) than O(d²)**, suggesting L-BFGS-B's per-iteration cost dominates at thesis scale. Bears on full-backtest compute budgets (see §Runtime budget at end).

### VARLiNGAM — robustness comparison

- **Implementation**: `lingam.VARLiNGAM` with regularised Stage 1 (LASSO-VAR via `sklearn.linear_model.MultiTaskLasso` with α = 0.01 — OLS-VAR is underdetermined at d ≈ 150, n = 504)
- **Window / step**: same as DYNOTEARS
- **Lag**: BIC-selected, capped at 3
- **Bootstrap**: 100 samples per window for edge reliability (built into `lingam`)
- **Causal ordering**: tracked across windows as additional regime-change signal — shifts in `causal_order_` indicate structural change
- **Prior knowledge**: `lingam.VARLiNGAM` accepts a `prior_knowledge` matrix for B₀ (entries in {-1, 0, 1} pin "no edge" / "edge"). Verify in implementation week 2 whether this extends to the lagged B_p; if not, follow with post-fit projection (zero out the asset → driver block of each B_p, re-apply ICA on the residuals to recover the corrected mixing matrix). Document any divergence from DYNOTEARS under the same constraint.
- **Output per window**: `(B_0, B_1, ..., B_p, Σ_e, causal_order, bootstrap_probs, error_indep_pvalues)`
- **Misspecification check (HSIC residual independence)**: VARLiNGAM's identifiability hinges on the residuals being mutually independent (the LiNGAM assumption). The implementation exposes `compute_error_independence` per window — calls `lingam.VARLiNGAM.get_error_independence_p_values()` and returns a `(d, d)` matrix of pairwise HSIC p-values. Cost is O(d²) HSIC tests × O(n²) each — at d ≈ 135 that's prohibitive on every window. Run at a configurable **spot-check cadence** (`error_independence_every_n_windows` parameter; default 12 = annual). Windows where the off-diagonal rejection rate at α=0.05 exceeds ~20% emit a "LIKELY MISSPECIFIED" warning and the methodology chapter flags that period as a VARLiNGAM caveat.

### NTS-NOTEARS — extension / stretch goal

- **Implementation**: reference implementation from Sun et al. (https://github.com/xiangyu-sun-789/NTS-NOTEARS)
- **Window / step**: same
- **Lag (K)**: 3
- **Prior knowledge**: forbid asset → driver edges via L-BFGS-B bounds on first-layer kernel L2 norms (natively supported per Sun et al.)
- **Expected runtime**: 5-20× DYNOTEARS per window. If time-budget allows, run on a subset of windows (e.g. every 3rd month) for a sub-analysis comparing selected drivers and sensitivities to DYNOTEARS.
- **Critical caveat**: implementation is research-grade, not battle-tested at d ≈ 150. Budget a week for getting it running cleanly. If it fails to converge or runtime is prohibitive, drop and document.
- **Output**: `(W^k for k=1..K+1, kernel weights for sensitivity extraction)`

---

## Greedy factor selection (option b3 from prior discussion)

Per window:

### Stage A — Prune (cheap, runs always)

1. From discovery output, score each candidate driver by aggregate outgoing influence:
   ```
   score_d = Σ_{p ≥ 1} Σ_{asset i} |edge_d→i at lag p| × stability_d→i
   ```
   For DYNOTEARS, `stability_d→i = 1` if edge magnitude exceeds threshold τ (set τ such that ~10% of possible edges pass), else 0.

   For VARLiNGAM, `stability_d→i = bootstrap_prob(d → i)`.

2. Drop candidates with zero score.
3. Keep top `2K` survivors (where K is target selected count) as the pool for Stage B.

**Note**: only lagged edges (p ≥ 1) contribute to the score. Contemporaneous edges are dropped here because (a) they're where exogeneity worries bite hardest, and (b) drivers are supposed to be *predictive* of asset moves, which requires temporal precedence.

### Stage B — Conditional greedy refinement on the pool

1. Start with empty selected set S.
2. For each remaining candidate `d` in the pool, fit a small auxiliary model: regularised lagged regression of each asset's return on lagged values of `S ∪ {d}`. Score `d` by aggregate validation log-likelihood gain vs the model using `S` only (held-out 20% of the window).
3. Add the argmax candidate to S.
4. Repeat until `|S| = K` or the marginal gain falls below an absolute threshold ε.

**Hyperparameters**:
- K (selected drivers): determined adaptively by the Initial K Calibration step below, not fixed a priori
- pool size: 2K
- ε (stopping threshold): determined by null-permutation — fit Stage B on shuffled-driver data, take 95th percentile of gains as ε

### Initial K Calibration (one-off, run at start of backtest on burn-in window)

K is set data-adaptively rather than picked a priori. Procedure:

1. Run discovery on the full candidate driver pool over the burn-in window (no Stage B selection).
2. Compute aggregate causal score per candidate driver (per Stage A scoring rule).
3. Sort scores descending.
4. Apply two K-suggestion methods in parallel:
   - **Kneedle algorithm** on the sorted-score curve → `K_elbow`
   - **Permutation null with Benjamini-Hochberg FDR**: shuffle each candidate's time series independently (preserves marginal distribution, destroys temporal causal structure); refit discovery; collect the **full (B, d) per-driver score matrix** across B=50 shuffles. For each real driver, compute a one-sided p-value via the parametric z-score `z_d = (real_d − mean(null_{·,d})) / std(null_{·,d})`, converted via `1 − Φ(z)`. Apply Benjamini-Hochberg at α=0.05 to control the false-discovery rate across the d hypotheses. `K_perm` = count of drivers surviving BH-FDR.
5. Set primary K = `max(K_elbow, K_perm)` clipped to `[1, pool_size]` (more conservative — admits more drivers).
6. Set sensitivity sweep range: `{⌈K/2⌉, K, min(2K, |pool|/2)}` plus two interpolating values. Ceiling at `|pool|/2` enforces that selection remains meaningful (selecting more than half the pool isn't selection).

This produces a per-thesis K rather than a copied-from-HSP K, and lets the diagnostic plots in the verification step (distance concentration, half-window ARI, effective dimensionality) feed into validating the choice rather than just hyperparameter sweeping.

**Implementation notes (Phase H)**: lives in `pipeline/factor_selection/k_calibration.py`. Two runtime fixes were essential to make this practical at thesis scale:

- **`permuted_max_iter` cap on permuted DYNOTEARS fits** (default 20 vs the unmodified `max_iter=100` for the *real* fit). Shuffled drivers have no causal structure to converge on; capping reduces per-fit cost by ~5× without changing the resulting score distribution (verified empirically by KS-test, see `tests/test_k_calibration.py::test_h2_permuted_max_iter_cap_preserves_distribution`).
- **`n_jobs` joblib parallelisation** across the B permutation fits — embarrassingly parallel. Seeds drawn up-front from the RNG so result is deterministic regardless of n_jobs (`test_h3_n_jobs_determinism`).

Combined, these give a ~24× speedup at d=65 (162 min → 7 min). At d=134, K calibration took **187 min (~3h)** in Phase G.7 — the bulk of full-backtest setup cost lives here.

**`K_perm_legacy` diagnostic preserved**. The pre-Phase-H definition (real_d vs the 95th percentile of `max_d (null_scores)`) is structurally biased upward at large d (multiple-comparisons "max-of-d" tail statistic). It's kept alongside the BH-FDR `K_perm` as a side-channel diagnostic for the methodology chapter, because the comparison cleanly demonstrates the bias.

**Empirical reality (G.5 / Phase H / G.7)**: on the actual S&P-100 universe at both d=65 and d=134, BH-FDR reports **`K_perm = 0` at α=0.05** — no individual driver achieves FDR-controlled significance. The same windows have `K_elbow = 9` (d=65) and `K_elbow = 14` (d=134), so the operational K selector is **always Kneedle** in practice. The interpretation: Stage A causal scores rank drivers informatively (Kneedle finds a real elbow that scales with N) but no individual driver's score is high enough to clear a strict FDR threshold at the dimensionalities tested. The thesis claim is about the *combination* of causally-selected drivers in the FFNN + HSP pipeline, not the individual-driver significance — so K_perm = 0 is reported honestly in the methodology chapter as a finite-sample-causal-discovery caveat, not hidden.

### Output

`selected_drivers[t]` = ordered list of selected driver names (order = addition order in Stage B), plus per-step gain log for diagnostics.

---

## Sensitivity computation

### Primary — FFNN + AAD (HSP-style)

For each asset `i` and each window:

1. Inputs: lagged values of K selected drivers, lag horizon L (start with L=1; sensitivity check L=5).
2. Train a feed-forward NN: `r_{i,t} = f_i(D_{t-1:t-L})`.
3. Architecture search: depth ∈ {1, 2}, width ∈ {16, 32, 64}; select by RMSE on a held-out 20% slice of the training window.
4. Compute sensitivities via AAD (autodiff): `s_{i,d} = ∂f_i / ∂d` evaluated at training observations, averaged over the window.
5. Stack: `sensitivities[t][asset_i] = vector ∈ ℝ^K`.

This is identical to HSP's sensitivity module except the input set is causally-selected, not correlation-selected.

### Extension — NTS-NOTEARS gradients

If NTS-NOTEARS is run, sensitivities come directly from the per-asset 1D CNN:
```
s_{i,d} = ∂CNN_i / ∂d at lag k=1
```
averaged over the training window. No separate FFNN training step — discovery and sensitivity computation unified in one model.

---

## Module structure

```
thesis/pipeline/
  data/
    universe.py            # fja05680 S&P 500 membership replay + top-N-by-mcap-at-date
    assets.py              # WRDS/CRSP (primary, delisted+historical-shares) + yfinance (fallback)
    wrds_backend.py        # CRSP queries (SQLAlchemy + psycopg → wrds-pgdata, .pgpass auth)
    drivers.py             # ~50-series candidate pool (FRED + Yahoo), per-type preprocessing
    alignment.py           # NYSE calendar alignment, joint matrix builder, per-window z-score, ADF/KPSS
    
  discovery/
    dynotears.py           # Rolling DYNOTEARS with native tabu_edges asset→driver mask
    varlingam.py           # Rolling VARLiNGAM with LASSO-VAR Stage 1 + prior_knowledge / post-fit projection
    nts_notears.py         # Rolling NTS-NOTEARS (extension)
    diagnostics.py         # Stationarity, fit-quality summaries, network-density time series
    
  factor_selection/
    prune.py               # Stage A: pool-down by outgoing edge score
    k_calibration.py       # One-off Kneedle + permutation-null K selection on burn-in
    greedy.py              # Stage B: conditional greedy refinement
    selector.py            # Top-level: α · causal + (1-α) · U[t-1] blend; reads feedback.storage
    
  sensitivities/
    ffnn.py                # PyTorch multi-head per-window model, jacobian via torch.func.jacrev
    nts_grads.py           # NTS-NOTEARS gradient extraction (extension)
    sensitivity_matrix.py  # Per-window D[t]_{ij} = ||s_i - s_j||_2, nearest-PSD projection
    
  stage1_pipeline.py       # Orchestration: end-to-end Stage 1, writes outputs to Parquet for Stage 2
  closed_loop.py           # V2 canonical entry point: interleaves Stage 1 + backtest + feedback per
                           #   rebalance so U[t] genuinely affects selection at t+1 (Stage 2 docs)
```

---

## Verification plan

1. **Data sanity**: each driver series has no missing data within the rebalance period after preprocessing; stationarity diagnostics logged.
2. **Discovery smoke test**: run DYNOTEARS on 5 assets + 5 drivers over 2 years. Verify W is acyclic, asset→driver block is zero (constraint enforced), driver→asset block has non-trivial entries.
3. **Prior knowledge verification**: remove the asset→driver constraint and re-fit. Quantify how much the constraint changes the driver→asset block. If the constraint has zero effect on real data, the algorithm naturally satisfied it; if it has large effect, the constraint is enforcing something the data weakly contradicts (interesting either way).
4. **Selection sanity — economic priors**: known relationships should be recovered. Examples: oil futures should be selected when energy stocks are heavily weighted in S&P 100; 10Y-2Y slope should be selected in periods of yield-curve sensitivity; VIX should be selected in vol-regime transitions. Manual inspection of selected drivers per regime, especially around 2018Q4, 2020Q1, 2022.
5. **Selection stability**: Jaccard similarity of `selected_drivers[t]` between consecutive windows. Median ≥ 0.5 expected; near-1 suggests over-conservative selection, near-0 suggests over-fitting.
6. **Cross-method agreement**: top 5 drivers from DYNOTEARS and VARLiNGAM should overlap substantially (Jaccard ≥ 0.4). Large disagreement is itself a finding.
7. **Sensitivity stability**: per-asset sensitivities should be roughly invariant to small perturbations of the training window. Bootstrap the training window N=10 times; coefficient-of-variation of sensitivities should be < 0.3 for the majority of (asset, driver) pairs.
8. **K appropriateness diagnostics** (run during sensitivity sweep, plotted in results chapter):
   - **Distance concentration ratio**: track `σ(D_ij) / E[D_ij]` over pairwise sensitivity distances as K grows. Healthy embeddings have well-spread distances; concentration ratio falling toward zero indicates the curse of dimensionality (everything looks equidistant). The inflection point bounds K from above.
   - **Half-window ARI**: split each window in halves, run full pipeline on each half independently, compute Adjusted Rand Index between resulting cluster assignments. Plot ARI vs K — peak indicates the K at which the embedding captures real reproducible structure rather than noise.
   - **Effective dimensionality**: PCA on the per-window sensitivity matrix S ∈ ℝ^(N×K). Effective dim = smallest q such that top-q PCs explain 95% variance. If effective dim ≪ K, K is too high and most coordinates are redundant.
   - Concordance check: the three diagnostics should agree on a "good K" range. If they disagree, investigate (e.g., effective dim small but ARI high → real but low-rank structure, possibly fine; concentration ratio dropping but ARI rising → contradictory, investigate).
9. **Reproducibility**: fixed random seeds at every stochastic step (FFNN initialisation, bootstrap, train/val split, permutation nulls); full pipeline replays bit-for-bit from seed.

### Verification status (as of 2026-05-28)

Empirical findings from the build-out + four shakedown runs (G.3, G.5.b, F.2, Phase H, G.7):

| Check | Status | Notes |
|---|---|---|
| 1. Data sanity | ✅ | 100 tickers + 35 drivers resolve cleanly via WRDS+FRED+Yahoo. |
| 2. Discovery smoke test | ✅ | DYNOTEARS produces acyclic W; asset→driver block exactly zero (tabu_edges verified). |
| 3. Prior-knowledge verification | ⚠️ Pending | Refit-without-mask comparison not yet run; do during Phase I or after. |
| 4. Selection sanity (economic priors) | ✅ | Driver rotation matches macro regime: rates pre-COVID → dual-crude + credit at April 2020 crash → safe-haven mid-crisis → vol + rates during recovery (G.5.b + G.7). |
| 5. Selection stability (Jaccard) | ⚠️ Mixed | Median Jaccard between consecutive rebalances ≈ 0.6 in V1; V0 (cum-corr) is near 1.0 (tautology). |
| 6. Cross-method (DYNOTEARS ∩ VARLiNGAM) | ❌ Not yet | VARLiNGAM at thesis scale hasn't been run; Phase I deliberately defers it. |
| 7. Sensitivity stability | ⚠️ Pending | Bootstrap of training window not yet executed. |
| 8. K-appropriateness diagnostics | Partial | Kneedle elbow scales with N (9 at d=65 → 14 at d=134); concentration ratio + ARI not yet plotted. |
| 9. Reproducibility | ✅ | Fixed seeds everywhere; `n_jobs` parallelisation is deterministic (`test_h3_n_jobs_determinism`). |

### Runtime budget (empirical, from G.7 at d=134, N=99)

| Step | Wall time | Notes |
|---|---|---|
| Asset prices (100 tickers, 2005-2024) | ~5-10 min | WRDS-cached; first run is longer. |
| Driver pool (35 series, 2005-2024) | ~10 sec | FRED + Yahoo, mostly cached. |
| Joint matrix + eligibility mask | < 1 sec | |
| **K calibration** (B=50, n_jobs=-1, permuted_max_iter=20) | **~3 hours** | One-off on burn-in; dominant fixed cost. |
| **Per-rebalance Stage 1** (DYNOTEARS + Stage B + FFNN) | **~3.8 min** | × 216 rebalances = ~14 hours per variant. |
| **Per-variant total** | **~17 hours wall** | K-cal once + 216 rebalances. |
| **Three variants (V0 + V1 + V2)** | **~40 hours total wall** | V0 + V2 share V1's calibrated K; only V1 pays the K-cal cost. Two overnight runs. |

---

## Out of scope

- Real-time / streaming discovery (rolling batch is fine)
- Adaptive driver pool (candidates fixed at the start; selection from them varies)
- Discovery on tick / intraday data
- Constraint-based discovery methods (PC, FCI) — scalability and faithfulness concerns documented in `causal_ml_a_survey_and_open_problems.pdf` and Howard et al.
- Granger causality — known weaker baseline, would only re-confirm DYNOTEARS literature
- Latent-variable extensions (LPCMCI, etc.) — interesting but adds scope