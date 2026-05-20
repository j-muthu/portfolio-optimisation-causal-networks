# Causal Discovery Pipelines for S&P 500

Two separate plans for the two causal discovery methods. Each is self-contained.

- **[Plan A: DYNOTEARS](#plan-a-dynotears-on-sp-500)** — Score-based continuous optimisation
- **[Plan B: VARLiNGAM](#plan-b-varlingam-on-sp-500)** — ICA-based two-stage decomposition
- **[Shared Data Pipeline](#shared-data-pipeline)** — Common preprocessing used by both

---

# Shared Data Pipeline

**File**: `thesis/pipeline/data.py`

Both methods consume identical input. Build this once:

1. Download S&P 500 constituent list (current + historical for survivorship bias awareness)
2. Use `yfinance` to download daily adjusted close prices (~10 years, 2014–2024)
3. Compute **log-returns**: `log(P_t / P_{t-1})`
4. **Stationarity check**: Run ADF test per asset, flag/drop non-stationary series (p < 0.01)
5. **Standardise**: Zero mean, unit variance
6. Handle missing data: drop assets with >5% missing days, forward-fill small gaps, align all assets to common trading dates

**Output**: A DataFrame of shape `(~2520, ~500)` — 10 years of daily log-returns for ~500 assets. `n` is the number of time-series observations (rows), `d` is the number of variables (columns) = 500.

---

## Handling S&P 500 Constituent Changes

The S&P 500 is not a fixed list. Stocks are added and removed regularly (~20-30 changes per year). This creates **survivorship bias**: if you only use today's 500 constituents and backtest over 10 years, you're looking at "winners" — companies that survived long enough to still be in the index — and missing companies that were delisted, acquired, or dropped. This inflates backtested performance and distorts causal structure (you never see causal links *to* or *from* companies that disappeared).

### The Three Approaches

#### Approach 1: Fixed Universe (Simplest — Recommended Starting Point)

Use today's S&P 500 constituents, download their full history, and accept the bias.

**Pros**: Trivial to implement. Every rolling window has the same d=500 columns. Adjacency matrices are directly comparable across windows.

**Cons**: Survivorship bias. Missing companies like Lehman Brothers (bankrupt 2008), Worldcom (fraud 2002), or any stock that was removed. During the 2008 crisis window, your "S&P 500" includes companies that weren't actually in the index then.

**When this is acceptable**: For a first-pass / proof-of-concept. Acknowledge it as a limitation in the thesis. Most academic papers on DYNOTEARS for finance (including Howard et al.) use a fixed universe.

```python
# Simple: get current S&P 500 tickers
import pandas as pd
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = table[0]['Symbol'].tolist()
```

#### Approach 2: Point-in-Time Constituents (Gold Standard)

Use the actual S&P 500 membership list as it was at each point in time.

**Pros**: No survivorship bias. Causal graphs reflect the true investable universe at each date. Results are realistic for actual portfolio management.

**Cons**: Hard to get the data. S&P constituent change history is available from:
- **Wikipedia**: The second table on the S&P 500 page lists historical changes (additions/removals with dates) — free but requires parsing and may have gaps
- **WRDS/CRSP**: Definitive source, requires academic subscription
- **Siblis Research / similar**: Paid datasets of historical index constituents

**Implementation complexity**: Each rolling window may have a *different set of assets*. This means adjacency matrices change size across windows, making direct comparison harder. You need to handle:
- Assets entering mid-window (use NaN / partial data)
- Assets leaving mid-window (delisted, acquired — price series ends)
- Mapping between different-sized adjacency matrices across windows

```python
# Pseudocode for point-in-time approach
for window_start, window_end in rolling_windows:
    # Get constituents as of window_start (or window_end)
    constituents = get_sp500_members_at_date(window_start)
    window_data = all_prices[constituents].loc[window_start:window_end]
    # Drop assets with insufficient data in this window
    window_data = window_data.dropna(axis=1, thresh=min_obs)
    # Now d varies per window
```

#### Approach 3: Intersection Universe (Practical Compromise)

Take the **intersection** of all S&P 500 constituent lists across your study period — only include stocks that were in the index for the *entire* period.

**Pros**: Fixed d across all windows (matrices are comparable). Reduces survivorship bias vs. Approach 1 (though doesn't eliminate it — you still only see "survivors"). Easy to implement.

**Cons**: Shrinks the universe significantly. Over 10 years you might get ~350-400 stocks instead of 500. Still biased toward long-tenure large-caps.

```python
# Get historical changes from Wikipedia
changes_table = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
)[1]  # Second table = historical changes
# Parse additions/removals to build point-in-time membership
# Then intersect across all dates in study period
```

### Recommendation

**Start with Approach 1** (fixed universe) to get the pipeline working. Then upgrade to **Approach 2 or 3** for the final thesis results. The Wikipedia historical changes table gives you enough data to implement Approach 3 without needing a WRDS subscription.

In your thesis, run the main experiments with Approach 1, then include a **robustness check** section where you repeat key results with Approach 3 and show the conclusions hold. If they diverge materially, that's itself an interesting finding about survivorship bias in causal discovery.

### Additional Data Considerations for Constituent Changes

- **Delisted stocks**: yfinance may return partial data or errors for delisted tickers. Use `try/except` and log failures.
- **Ticker changes**: Some companies change tickers (e.g., FB → META). Wikipedia's changes table captures some of these. yfinance handles most ticker changes automatically.
- **Mergers/acquisitions**: When company A acquires company B, B's price series ends. The causal link A→B disappears — which is correct, since B no longer exists as a separate entity.
- **Spin-offs**: A company splitting into two creates a new ticker mid-series. Handle by treating the new entity as a new constituent from the spin-off date.

---
---

# Plan A: DYNOTEARS on S&P 500

## Context

You're building a causal discovery pipeline for your thesis: feed S&P 500 daily returns into DYNOTEARS and extract causal graphs that can later feed into portfolio optimisation (HRP with causal graph matrices, regime detection, etc.). No financial data pipeline exists yet — this is greenfield.

---

## Part 1: Conceptual Understanding

### What Goes In

**Input**: A pandas DataFrame (or list of DataFrames) where:
- Each **column** = one asset (e.g., `AAPL`, `MSFT`, ..., ~500 columns)
- Each **row** = one trading day
- Values = **log-returns** (not raw prices — the DYNOTEARS paper explicitly uses log-returns for S&P data and tests stationarity via ADF)
- Index = sequential integers (the transformer requires this)

```python
# Conceptual shape for 5 years of daily data, 500 assets:
# ~1260 rows × 500 columns
```

**Key parameter `p`** (lag order): How many past days can causally influence today. `p=1` means only yesterday matters; `p=2` means yesterday and the day before, etc.

### What Comes Out

**Output**: A `StructureModel` (NetworkX DiGraph) with two types of causal edges:

1. **Intra-slice edges** (contemporaneous): `AAPL_lag0 → MSFT_lag0` — "AAPL causally influences MSFT within the same trading day"
2. **Inter-slice edges** (lagged): `AAPL_lag1 → MSFT_lag0` — "AAPL's return yesterday causally influences MSFT's return today"

Each edge has a **weight** (causal effect magnitude) and can be extracted as a weighted adjacency matrix — exactly the format needed for HRP insertion.

**Important finding from Howard et al.**: For daily stock returns, inter-slice (lagged) weights are almost always zero. Most causal structure appears in the contemporaneous (intra-slice) graph. This means `p=1` is likely sufficient, and the real value is in the W matrix, not the A matrices.

### Key Parameters

| Parameter | What it does | Recommended starting point |
|-----------|-------------|---------------------------|
| `p` | Lag order (how many past days) | `1` (Howard et al. finding) |
| `lambda_w` | Sparsity of same-day edges | `0.05`–`0.1` (grid search) |
| `lambda_a` | Sparsity of lagged edges | `0.05`–`0.1` (grid search) |
| `w_threshold` | Drop edges below this weight | `0.01`–`0.05` |
| `max_iter` | Optimisation iterations | `100` (default) |

### Stationarity Assumption (Critical)

DYNOTEARS assumes the causal structure is **fixed across the entire input window**. This is the single biggest issue for financial data, where causal relationships change across market regimes. This is why **rolling windows are essential**.

---

## Part 2: Pipeline Plan

### Step 1: Rolling Window DYNOTEARS

**File**: `thesis/pipeline/rolling_dynotears.py`

Since DYNOTEARS assumes stationarity, you **must** use rolling windows to capture time-varying causal structure:

```
Window 1: [day 0 ... day 504]     → Graph G_1
Window 2: [day 21 ... day 525]    → Graph G_2  (step = 1 month)
Window 3: [day 42 ... day 546]    → Graph G_3
...
```

**Design choices**:
- **Window size**: ~2 years (504 trading days) — enough samples for 500 variables with regularisation. The paper uses n ≫ dp for reliable estimation; with d=500 and p=1, you want n ≫ 500
- **Step size**: 21 trading days (1 month) — produces a new causal graph monthly
- **Hyperparameter selection**: Cross-validate `lambda_w` and `lambda_a` on a held-out portion of each window using Frobenius norm (as the paper recommends for financial data)

For each window:
1. Slice the log-return DataFrame to the window
2. Reset index to sequential integers
3. Call `from_pandas_dynamic(window_df, p=1, lambda_w=..., lambda_a=...)`
4. Extract the weighted adjacency matrix from the StructureModel
5. Store: `{date: adjacency_matrix}` mapping

### Step 2: Graph Analysis & Regime Detection

**File**: `thesis/pipeline/graph_analysis.py`

From the sequence of causal graphs, compute:
- **Graph density** over time (number of edges / possible edges)
- **Average edge weight** over time
- **Sector-level causal flow** (which sectors cause which)
- **Graph distance** between consecutive windows (Frobenius norm of adjacency matrix differences) — large jumps = regime changes

This directly addresses your supervisor's priority: "regime change detection via causal structure."

### Step 3: Portfolio Integration (later phase)

**File**: `thesis/pipeline/portfolio.py`

- Extract the W matrix (contemporaneous causal adjacency) from each window
- Symmetrise it (e.g., `(|W| + |W^T|) / 2`) for use as a distance/similarity matrix
- Feed into HRP in place of the correlation matrix
- Compare against correlation-based HRP

---

## Other Things You Need to Consider

### 1. Survivorship Bias
See the [Handling S&P 500 Constituent Changes](#handling-sp-500-constituent-changes) section in the Shared Data Pipeline for the full treatment.

### 2. Computational Cost
- 500 assets × rolling windows = many DYNOTEARS runs
- Each run on d=500 can take minutes to tens of minutes
- Plan for parallelisation (`joblib` or `multiprocessing`)

### 3. Regularisation Tuning
With d=500 and n=504 (barely n > d), regularisation is **critical**. The paper says: "when n < dp, regularisation becomes more important." You'll need relatively strong `lambda_w` and `lambda_a` to get sparse, interpretable graphs.

### 4. Comparison Methods
Your tierlist identifies these as essential comparisons:
- **VARLiNGAM** (same pipeline, different algorithm — already in your `/lingam/` fork)
- **Classical Granger causality** (baseline)
- **Correlation matrix** (the null hypothesis)

### 5. Log-Returns vs Raw Prices
Always use log-returns. Raw prices are non-stationary and will violate DYNOTEARS assumptions. The paper's S&P 100 experiment explicitly uses log-returns.

---

## Verification Plan

1. **Smoke test**: Run DYNOTEARS on a small subset (~20 assets, 2 years) and verify output is a valid DAG with sensible edge weights
2. **Reproduce paper results**: Try to match the S&P 100 experiment from the DYNOTEARS paper (Appendix C.1) as a sanity check
3. **Scaling test**: Run on full 500 assets for one window, check runtime and memory
4. **Rolling window test**: Run 3 consecutive windows, verify graphs change but aren't wildly different (structural stability check)

---

## Critical Files

| File | Purpose |
|------|---------|
| `causalnex/causalnex/structure/dynotears.py` | Core algorithm — `from_pandas_dynamic()` and `from_numpy_dynamic()` |
| `causalnex/causalnex/structure/transformers.py` | `DynamicDataTransformer` — converts DataFrames to DYNOTEARS format |
| `causalnex/tests/structure/test_dynotears.py` | Tests showing expected input/output formats |
| `DYNOTEARS- Structure Learning from Time-Series Data.pdf` | Original paper with S&P 100 experiment details |
| `causal_discovery_methods_tierlist.md` | Your methods evaluation guide |

---
---

# Plan B: VARLiNGAM on S&P 500

## Context

VARLiNGAM is the head-to-head comparison partner for DYNOTEARS. It solves the same problem (causal discovery from time-series, outputting contemporaneous + lagged adjacency matrices) but via ICA-based decomposition rather than score-based optimisation. Crucially, VARLiNGAM **uniquely identifies the DAG** (not just an equivalence class) under non-Gaussianity — and financial returns are non-Gaussian (heavy tails, skewness), making this assumption a *strength*.

---

## Part 1: Conceptual Understanding

### How VARLiNGAM Works

A **two-stage** algorithm:

1. **Stage 1 — VAR estimation**: Fit a Vector Autoregression to the time series. The VAR coefficients become the lagged adjacency matrices (B₁, B₂, ..., Bₖ). Residuals are passed to Stage 2.

2. **Stage 2 — LiNGAM on residuals**: Apply DirectLiNGAM (ICA-based) to the VAR residuals to discover the contemporaneous causal structure (B₀). Exploits **non-Gaussianity** to uniquely identify the causal ordering.

### What Goes In

**Input**: A single numpy array or pandas DataFrame:
- Shape: `(n_samples, n_features)` — e.g., `(1260, 500)` for 5 years, 500 assets
- Rows = trading days (sequential order matters), columns = assets
- Values = **log-returns** (same as DYNOTEARS, from the shared data pipeline)

```python
model = lingam.VARLiNGAM(lags=1)
model.fit(log_returns_df)  # shape: (n_samples, n_features)
```

**Difference from DYNOTEARS**: VARLiNGAM takes a single matrix. DYNOTEARS's `from_pandas_dynamic()` accepts a list of DataFrames for multiple realisations.

### What Comes Out

**Output**: `model.adjacency_matrices_` — numpy array of shape `(lags+1, n_features, n_features)`

| Matrix | Shape | Meaning |
|--------|-------|---------|
| `adjacency_matrices_[0]` = **B₀** | (500, 500) | Contemporaneous causal effects — "AAPL causes MSFT today" |
| `adjacency_matrices_[1]` = **B₁** | (500, 500) | Lag-1 causal effects — "AAPL yesterday causes MSFT today" |

Entry `B₀[i, j]` = causal effect of variable *j* on variable *i* at the same time step.

**Additional outputs**:
- `model.causal_order_` — discovered causal ordering (which assets are "upstream" causes). Unique to VARLiNGAM.
- `model.residuals_` — VAR residuals, shape `(n_samples - lags, n_features)`

### Key Parameters

| Parameter | What it does | Recommended starting point |
|-----------|-------------|---------------------------|
| `lags` | Number of lags in VAR model | `1` (same reasoning as DYNOTEARS) |
| `criterion` | Lag order selection: `'bic'`, `'aic'`, `'hqic'`, `'fpe'`, or `None` | `'bic'` (default, most conservative) |
| `prune` | Apply adaptive LASSO to remove weak edges | `True` (default) |
| `lingam_model` | Which LiNGAM variant for Stage 2 | `None` (defaults to DirectLiNGAM) |
| `random_state` | Reproducibility seed | Set to a fixed int |

**Tuning advantage**: VARLiNGAM's sparsity comes from **adaptive LASSO pruning** (automatic) and **BIC-based lag selection** (automatic). Fewer hyperparameters than DYNOTEARS.

### Non-Gaussianity Assumption (Critical)

VARLiNGAM's identifiability requires **non-Gaussian error terms**. Financial returns have heavy tails, skewness, and volatility clustering — all non-Gaussian. Verify with `model.get_error_independence_p_values()` — p-values > 0.05 mean the independence assumption holds.

### Bootstrap for Statistical Reliability

Built-in `bootstrap()` method (DYNOTEARS lacks this):

```python
result = model.bootstrap(X, n_sampling=100)
probs = result.get_probabilities()           # Bootstrap probability per edge
effects = result.get_total_causal_effects()  # Median effects + probabilities
```

Edges in 90%+ of bootstrap samples are reliable; <50% are noise.

---

## Part 2: Pipeline Plan

### Step 1: Rolling Window VARLiNGAM

**File**: `thesis/pipeline/rolling_varlingam.py`

Same rolling window design as DYNOTEARS (VARLiNGAM also assumes stationarity):

```
Window 1: [day 0 ... day 504]     → Matrices [B₀, B₁]_1
Window 2: [day 21 ... day 525]    → Matrices [B₀, B₁]_2
...
```

For each window:
1. Slice the log-return DataFrame to the window
2. Fit VARLiNGAM:
   ```python
   model = lingam.VARLiNGAM(lags=1, criterion='bic', prune=True, random_state=42)
   model.fit(window_df.values)
   ```
3. Extract adjacency matrices: `B0 = model.adjacency_matrices_[0]`
4. (Optional) Run bootstrap: `result = model.bootstrap(window_df.values, n_sampling=100)`
5. Store: `{date: (B0, B1, causal_order, bootstrap_probs)}` mapping

**Design choices**:
- **Window size**: ~504 trading days (2 years) — same as DYNOTEARS
- **Step size**: 21 trading days (1 month)
- **Lag selection**: Let BIC decide automatically, cap at `lags=5`
- **Bootstrap**: 100 samples per window for edge reliability

### Step 2: Graph Analysis & Regime Detection

**File**: `thesis/pipeline/graph_analysis.py` — shared framework with DYNOTEARS

From the sequence of B₀ matrices:
- Graph density, average weight, sector-level flow, graph distance (same as DYNOTEARS)
- **VARLiNGAM-specific**: Track how `causal_order_` changes across windows — shifts in causal ordering signal regime changes

### Step 3: Portfolio Integration

**File**: `thesis/pipeline/portfolio.py` — shared framework

- Extract B₀ (contemporaneous) from each window
- Symmetrise: `(|B₀| + |B₀ᵀ|) / 2`
- Feed into HRP as causal distance matrix
- Compare: DYNOTEARS-HRP vs VARLiNGAM-HRP vs Correlation-HRP

---

## Scalability Warning for d=500

VARLiNGAM's Stage 1 (VAR estimation) fits a VAR(p) with d=500: 250,000+ coefficients from ~504 observations. **Severely underdetermined**.

**Mitigations**:
1. **Regularised VAR**: Use LASSO-VAR or Ridge-VAR instead of OLS-VAR for Stage 1
2. **Reduce d**: Start with S&P 100 (~100 assets) where n >> d, then scale up
3. **Sector representatives**: 5-10 assets per sector (~50-100 total)
4. **Pre-computed AR coefficients**: Estimate externally, pass via `ar_coefs` parameter

More acute than for DYNOTEARS, which has built-in L₁ regularisation.

---

## Verification Plan

1. **Smoke test**: Run on ~20 assets, 2 years. Verify B₀ is lower-triangular (after causal ordering permutation)
2. **Assumption check**: `model.get_error_independence_p_values()` — verify non-Gaussianity holds
3. **Bootstrap stability**: 100 samples, check high-probability edges (>80%) are consistent across adjacent windows
4. **Head-to-head**: Run both DYNOTEARS and VARLiNGAM on same 20-asset window. Compare B₀ vs W
5. **Scaling test**: Try d=100, d=200, d=500 — find where VAR step breaks down

---

## Critical Files

| File | Purpose |
|------|---------|
| `lingam/lingam/var_lingam.py` | Core VARLiNGAM class — `fit()`, `bootstrap()`, `estimate_total_effect()` |
| `lingam/examples/VARLiNGAM.ipynb` | Example notebook showing full usage |
| `lingam/examples/CommonEdgeAnalysisWithVAR-LiNGAM.ipynb` | Bootstrap & common edge analysis |
| `lingam/examples/data/sample_data_var_lingam.csv` | Sample synthetic data |
| `lingam/tests/test_var_lingam.py` | Tests showing expected input/output |

