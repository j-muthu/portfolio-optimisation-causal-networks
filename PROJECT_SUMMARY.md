# Project summary: Portfolio Optimisation with Causal Networks

MSc thesis, Imperial College London, September 2026. Author: Joshua Muthu. Supervisors: Dr Ce Guo and Prof. Wayne Luk. Second marker: Prof. Stephen Weston.

This document is a self-contained description of the whole project: why it was done, what prior work it builds on, how the experiment was designed, how the codebase implements it, what the results are, and what they mean. The report itself is `final_report/main.tex`; the code is under `pipeline/`, `scripts/`, `tests/` and `reproduction.py`.

## Contents

1. [One-paragraph overview](#1-one-paragraph-overview)
2. [Motivation](#2-motivation)
3. [Prior research reviewed](#3-prior-research-reviewed)
4. [Experimental design](#4-experimental-design)
5. [Layout of the codebase](#5-layout-of-the-codebase)
6. [The code in detail](#6-the-code-in-detail)
7. [How the literature maps onto the code](#7-how-the-literature-maps-onto-the-code)
8. [Results](#8-results)
9. [Implications](#9-implications)
10. [Limitations, threats to validity and future work](#10-limitations-threats-to-validity-and-future-work)
11. [Project history](#11-project-history)

---

## 1. One-paragraph overview

Hierarchical Risk Parity (HRP) clusters assets on a correlation matrix and allocates capital by recursive bisection. Correlation is undirected, so it cannot say which assets move which. Causal discovery algorithms such as DYNOTEARS and VARLiNGAM produce a directed acyclic graph (DAG) over the assets, which can be split into a skeleton (which pairs are linked) and edge orientations (which way each link points). The thesis asks: if the correlation matrix in HRP is replaced by a discovered causal graph, does performance improve, and how much of any change is due to the skeleton versus the orientations? The answer, over 99 large-cap US equities, 215 monthly rebalances from 2007 to 2024, four estimation windows and 146 evaluated configurations, is that the graph does not improve net Sharpe. The skeleton hurts at every window, the orientations recover a similar amount at every window, the total is indistinguishable from zero, and nothing is significant after correcting for the search. Mechanism controls show that neither component actually needs a graph: a density-matched partial-correlation network reproduces the skeleton's effect, and a graph-free de-factored covariance reproduces the orientation effect.

---

## 2. Motivation

### 2.1 The problem with correlation as an allocation input

Since Markowitz (1952), the covariance or correlation matrix has been the central object in portfolio optimisation. Mean-variance optimisation inverts that matrix; at realistic sizes (around 100 assets on a one-year window) it is ill-conditioned, so the resulting weights are concentrated and unstable. López de Prado's HRP (2016) avoids the inversion by clustering assets on correlation distance, ordering them by the dendrogram's leaf order, and bisecting recursively with inverse-variance weights.

But correlation is symmetric and undirected. It records co-movement, not influence, and is prone to spurious relationships (Simon 1954). The intuition behind the project is that a causal graph carries strictly more information than a correlation matrix: it says not just that A and B are related but that A drives B. If that information is real, an allocator that reads it should do better than one that discards it.

### 2.2 Why the question needs a decomposition

A DAG has two separable pieces of information:

- the **skeleton**: an undirected structure recording which asset pairs are causally linked, and
- the **edge orientations**: which direction each link runs.

The skeleton can be fed to HRP by any orientation-blind transform of the adjacency matrix (for example, symmetrising it). The orientations cannot, because HRP's clustering step requires a symmetric distance matrix, and any symmetrisation throws the direction away. So the research question has two parts, and the second part cannot be answered without new allocators:

> When the correlation matrix in hierarchical portfolio allocation is replaced by a causal graph, is there a performance improvement, and how much of any improvement is attributable to the graph's skeleton versus its edge orientations?

Two follow-up questions then ask about mechanism:

- Is the skeleton's effect specific to the discovered causal structure, or would any sparse dependence network of similar density do the same thing?
- Is the orientation effect due to edge direction, or to some other property of the DAG-implied covariance (shrinkage, removal of the market factor) that a graph-free estimator could reproduce?

### 2.3 The three technical challenges

1. **Attribution.** If the correlation matrix is simply swapped for a graph, skeleton, orientations and implied covariance all change at once. Isolating each requires an ablation design where the identical graph feeds allocators that differ in exactly one respect, plus graph-free controls to test whether the graph is necessary at all.
2. **Asymmetric inputs to hierarchical allocators.** There was no established way to give an asymmetric adjacency matrix to HRP's clustering or allocation step without symmetrising it.
3. **Inference at realistic effect sizes.** Sharpe differences between allocators are small relative to their standard errors over 18 years of heavy-tailed daily returns, and the study evaluates 146 configurations. Inference has to account for autocorrelation, fat tails, and the search over configurations, and has to report statistical power so that a null result is an upper bound rather than "no effect".

### 2.4 How the project got here (Phase I to Phase II)

The project did not start with this question. The first phase (May to June 2026) used Hierarchical Sensitivity Parity (HSP, Rodríguez Domínguez 2023) as its baseline. HSP keeps HRP's mechanism but clusters assets on their sensitivities to a set of macro-financial drivers, estimated by a per-asset feed-forward neural network, with the drivers chosen by correlation. Phase I asked whether selecting the drivers from a discovered causal graph (DYNOTEARS over assets and drivers jointly) instead of by correlation would improve HSP, and whether a monthly feedback loop that re-ranked drivers on realised excess Sharpe would help further.

Phase I found nothing detectable: causal driver selection gave +0.010 net Sharpe at one driver count and negative differences at three others, none significant, and the seed-to-seed spread of the neural network (0.021 Sharpe) was twice the effect. The feedback loop produced weights identical to the open-loop ones at every rebalance. Standard HRP beat HSP outright. This reframed the project around the sharper question above, with plain HRP as the baseline and the graph feeding the allocator directly rather than selecting drivers for it. Phase I's 41 configurations are retained in the multiplicity correction so that this change of direction is accounted for.

---

## 3. Prior research reviewed

The literature falls into three domains that had not previously been combined for allocation.

### 3.1 Hierarchical portfolio allocation and backtest rigour

| Work | What it contributes | How the thesis uses it |
| --- | --- | --- |
| Markowitz 1952 | Mean-variance optimisation | Motivation: the matrix inversion problem |
| López de Prado 2016, HRP | Cluster on correlation distance, quasi-diagonalise, recursive bisection | The baseline allocator (`correlation-hrp`) and the mechanism every treatment modifies |
| Raffinot 2017 (HCAA), López de Prado 2020 (NCO), Lohre et al. 2020 | HRP variants | Context |
| Raffinot 2018, HERC | Equalise risk contribution across dendrogram clusters instead of bisecting the leaf order | Second hierarchical allocator used as a robustness check on the decomposition |
| Rodríguez Domínguez 2023, HSP | Cluster on FFNN-estimated sensitivities to macro drivers | Phase I baseline; retained in the trial count |
| Ledoit and Wolf 2004 | Linear shrinkage of the covariance | Mechanism control for shrinkage (`lw-hrp`) and the source of the precision matrix for `pcorr-hrp` |
| Sharpe 1963, Laloux et al. 1999 | Single-factor model; the market factor as the top eigenvalue | Mechanism control for market-factor removal (`defactored-hrp`), using the equal-weight cross-sectional mean as the market proxy |
| Friedman et al. 2008, graphical lasso | Sparse precision estimation | Motivates the thresholded partial-correlation control (`pcorr-hrp`), which takes the simpler route of inverting a Ledoit-Wolf covariance and keeping the largest entries |
| Mantegna 1999 (MST), Tumminello et al. 2005 (PMFG), Pozzi et al. 2013, Peralta and Zareei 2016 | Allocate from a filtered correlation network; peripheral assets diversify better than central ones | Shows that allocating from an asset graph is not new; the thesis's graph comes from causal discovery instead of correlation filtering. The centrality-equals-market-exposure finding turns out to explain the orientation effect |
| Výrost et al. 2019 | Weights from directed Granger-causality networks | Directed-graph allocation is not unprecedented, but not hierarchical and not decomposed |
| Bailey and López de Prado 2012, 2014 | Probabilistic and Deflated Sharpe ratio | Non-normality correction and selection-bias correction over all 146 trials |
| White 2000, Hansen 2005, Hansen et al. 2011 | Reality Check, SPA test, Model Confidence Set | Family-wise tests of a set of candidates against a baseline |
| Politis and Romano 1994 | Stationary block bootstrap | Pairwise contrast inference, 10,000 resamples, mean block 21 days |

### 3.2 Causal graphs applied to investing

- **Howard, Lohre and Mudde 2025** is the closest prior work: DYNOTEARS on S&P 500 equities, re-estimated on a rolling basis. They find that causal networks carry regime information that correlation does not and outperform in some regards but not consistently. Two of their findings shape this thesis: the contemporaneous matrix carries almost all the structure (so the thesis fits one lag and uses only the contemporaneous block), and they never use the graph as a direct allocation input (they use it for similarity or compress it to a centrality score). That gap is what the thesis fills.
- **Tang 2024** applies time-series causal discovery to trading, again without hierarchical allocation.
- **López de Prado 2023** (Causal Factor Investing) argues for causal rather than associational reasoning in factor research and is the source of the "causal-washing" concern discussed under implications.

### 3.3 Causal discovery

- **Constraint-based methods** (PC, FCI, PCMCI, PCMCI+; Runge et al. 2019, 2020) were rejected: they scale poorly at 132 variables, rely on faithfulness (weak in markets where offsetting effects can produce near-zero correlation despite a real link), and PCMCI tests only lagged links while PCMCI+ returns a partially directed graph.
- **NOTEARS** (Zheng et al. 2018) recasts acyclicity as a smooth constraint so that structure learning becomes continuous optimisation. **DYNOTEARS** (Pamfil et al. 2020) extends it to a structural VAR, returning a contemporaneous matrix W and lagged matrices A. It is the primary method because it scales, accepts forbidden edges ex ante (needed for the directional prior), and returns a DAG.
- **Scale sensitivity of NOTEARS.** Peters and Bühlmann 2014 show the least-squares score assumes equal error variances, so NOTEARS orients edges from low-variance to high-variance variables. Reisach, Seiler and Weichwald 2021 show that on unstandardised data a trivial variance-sorting heuristic matches NOTEARS, and that standardising drops NOTEARS to PC-level accuracy. The thesis therefore standardises every window before fitting. Reisach et al. 2023 show that ordering by explained variance (R²) survives standardisation; this is left as an unaddressed threat.
- **LiNGAM family.** Shimizu et al. 2006 (LiNGAM), Shimizu et al. 2011 (DirectLiNGAM), Hyvärinen et al. 2010 (VARLiNGAM). Identification comes from non-Gaussian residuals, which daily returns exhibit (Cont 2001) but which weaken with aggregation. VARLiNGAM is the robustness comparator.
- **Granger 1969.** A ridge-regularised VAR(1) coefficient matrix gives a directed graph whose directions are informative but whose magnitudes are shrunken predictive coefficients rather than structural effects. Used as a control that separates direction from magnitude.
- **NTS-NOTEARS** (Sun et al. 2023), a non-linear MLP-based extension, was probed but too expensive to run at scale; it appears in future work.
- **SEM background.** Wright 1934 (path coefficients) and Bollen 1989 supply the linear structural equation model used to turn a DAG into a covariance.

### 3.4 The research gap

No prior work sets hierarchical portfolio weights from a discovered score-based causal graph, and none decomposes the allocation value of such a graph into skeleton and orientations. The gap table in the report (Table 2.1) contrasts HRP (clusters on correlation, no discovery), HSP (clusters on sensitivities, no discovery, symmetric), MST/PMFG networks (weights from a correlation network, symmetric), Granger networks (directed, but no clustering), and Howard et al. (discovery, but no allocation) with this work (discovery, weights from graph, asymmetric allocation, skeleton-versus-orientation decomposition).

---

## 4. Experimental design

### 4.1 Data

- **Universe.** 99 US equities, fixed for the whole sample. Built as the union of the largest S&P 500 constituents by market cap at three dates (start, midpoint, end of the 2018 to 2020 pilot period) as a proxy for the paywalled S&P 100 history. Frozen in `scripts/phase_i_universe.txt`. None delists over 2007 to 2024. This induces survivorship bias in absolute Sharpe levels but not in pairwise allocator comparisons, since every allocator sees the same assets.
- **Prices.** CRSP daily prices and shares outstanding via WRDS (direct PostgreSQL connection, not the `wrds` Python package). Yahoo Finance is the fallback and is used for the 2025 to 2026 out-of-sample slice because CRSP daily data ends on 2024-12-31.
- **Drivers.** 33 plausibly exogenous macro-financial series from FRED and Yahoo: year-on-year CPI, core CPI, industrial production, retail sales, housing starts; unemployment-rate change; Michigan sentiment change; 3-month, 2-, 5-, 10- and 30-year Treasury yield changes; 10y-2y and 10y-3m slope changes; Baa-Aaa and Baa-10y credit spread changes; dollar index change; EURUSD, JPYUSD, GBPUSD, CNYUSD log returns; WTI, Brent, gold, natural gas, silver, copper log returns; VIX level; EFA, EEM, EWJ, EWG, EWU log returns. Sector ETFs, US index returns and factor portfolios are excluded as non-exogenous.
- **Transformations.** Log returns for price-like series, first differences for yields and spreads, year-on-year changes for monthly macro levels, no transform for volatility indices. Stationarity checked with ADF and KPSS. Every window standardises all variables before discovery.

### 4.2 Discovery

At each rebalance date t, on the joint panel X = [D | A] of 33 drivers and 99 assets (132 variables) over the estimation window:

- **DYNOTEARS** (primary): lag p = 1, λ_W = λ_A = 0.05, minimum edge weight 0.01. A **directional prior** forbids every asset-to-driver edge at every lag via DYNOTEARS' tabu list. Refitting without the prior on six windows across 2008 to 2022 puts 36 to 48% of driver-asset edge mass on implausible asset-to-driver edges, so the prior binds.
- **VARLiNGAM** (robustness comparator): the lagged block is fitted as a ridge VAR where drivers regress only on lagged drivers; DirectLiNGAM identifies the contemporaneous block B_0 with the same restriction passed as a prior-knowledge matrix; lingam's own pruning is disabled; any residual asset-to-driver mass is zeroed.
- **Ridge-Granger** (magnitude control, 252-day window only, a few allocators): VAR(1) with ridge λ = 10⁻² scaled by window length; edge i to j is the absolute lagged coefficient of i in j's regression; the graph is thinned to DYNOTEARS' density. Not acyclic in general, so the total-effect series is truncated at 10 terms and cycles are broken before ordering.

The **asset graph** M is the asset-to-asset submatrix of the contemporaneous matrix, with M[i, j] the effect of i on j. Acyclicity is recorded. A sparsity threshold τ prunes edges below it (default 0; swept over {0, 0.01, 0.05, 0.10}). DYNOTEARS graphs have 614 to 673 directed edges on average (6.3 to 6.9% density) with a mean longest path of about 50 assets; VARLiNGAM's unpruned B_0 is 32% dense at 189 days.

Graph fits are cached in `cache/discovery/` keyed on the exact bytes of the window, so every allocator at a given (method, window, rebalance) reads the identical graph.

### 4.3 The structural equation model

The asset block defines a linear SEM: (I - Mᵀ) x = ε, so x = (I - Mᵀ)⁻¹ ε = T ε. Because M is nilpotent for a DAG, the total-effect matrix T = Σ_k (Mᵀ)^k is exact with at most N terms. T_ij is the total effect on asset i of a unit shock to asset j along all directed paths, and its asymmetry is how orientation information enters the allocators.

The implied covariance is Σ_SEM = D_σ (T Σ_ε Tᵀ) D_σ, with Σ_ε diagonal (independent structural shocks, variances estimated from the residuals E = X̃ (I - M) over the window) and D_σ rescaling from standardised to return units. A small ridge (10⁻⁶ tr(Σ)/N) is added to the diagonal and the result is projected to nearest-PSD. The off-diagonal of Σ_ε is deliberately not estimated, because that would feed the sample correlation back in.

### 4.4 The allocator family

Every allocator is long-only, fully invested, deterministic, and receives the same graph M plus the same return window. HRP has two places a graph can enter: the ordering (normally from the dendrogram) and the covariance used for cluster variances in bisection. Crossing those gives the primary family:

| Allocator | Ordering from | Covariance for bisection | Reads orientation? |
| --- | --- | --- | --- |
| `correlation-hrp` (baseline, code tag `CORR-HRP`) | correlation dendrogram, distance √(½(1-ρ)) | sample | no, graph-blind |
| `skeleton-hrp` (`D0`) | dendrogram on the edge-profile embedding e_i = [M_i·, M_·i], distance ‖e_i - e_j‖₂ | sample | no, transpose-invariant |
| `undirected-hrp` (`D0s`) | dendrogram on ½(|M| + |Mᵀ|), scaled to [0, 1], distance 1 - similarity | sample | no, per-edge invariant |
| `semcov-hrp` (`D1`) | same dendrogram as `skeleton-hrp` | Σ_SEM | allocation step |
| `topo-hrp` (`D2`) | topological sort of M, ties by total downstream influence Σ_r |T_ri| then name | sample | ordering step |
| `topo-semcov-hrp` (`D2s`) | topological sort | Σ_SEM | both steps |

Additional allocators evaluated (all in the trial count):

| Allocator | Role |
| --- | --- |
| `lw-hrp` | `skeleton-hrp` dendrogram, Ledoit-Wolf shrunk covariance. Tests whether shrinkage explains the orientation effect. |
| `defactored-hrp` (`D0df`) | `skeleton-hrp` dendrogram, covariance of residuals from regressing each asset on the equal-weight mean return. Tests whether market-factor removal explains the orientation effect. |
| `pcorr-hrp` | Dendrogram on a thresholded partial-correlation network with the DAG's edge count, from the inverse of a Ledoit-Wolf covariance. Tests whether any sparse network reproduces the skeleton effect. |
| `coancestry-hrp` (`D3`) | Dendrogram on shared upstream ancestry. In the pre-specified 7-allocator family; excluded from the narrowed 5. |
| `struct-rp` (`D4`) | Risk parity over structural shocks using Σ_SEM, no hierarchy. Also in the 7, excluded from the 5. |
| `correlation-herc`, `skeleton-herc`, `semcov-herc` | HERC versions of the three decomposition anchors. |
| Equal weight, inverse variance | Naive anchors. |

### 4.5 The ablation and its contrasts

Because each allocator differs from its neighbour in one respect, the decomposition is two subtractions:

- **Skeleton effect** = `skeleton-hrp` - `correlation-hrp`
- **Orientation effect** = `oriented` - `skeleton-hrp`, where `oriented` is `semcov-hrp` (allocation step), `topo-hrp` (ordering step) or `topo-semcov-hrp` (both)
- **Total effect** = `oriented` - `correlation-hrp` = skeleton effect + orientation effect

Mechanism contrasts: `semcov-hrp` - `lw-hrp` (shrinkage), `semcov-hrp` - `defactored-hrp` (market-factor removal), `pcorr-hrp` - `skeleton-hrp` (graph-specificity of the skeleton). Robustness: `undirected-hrp` as an alternative skeleton definition, VARLiNGAM and ridge-Granger graphs, HERC, the τ sweep, the transaction-cost sweep.

### 4.6 Backtest protocol

- Monthly rebalancing every 21 trading days: 215 rebalances from January 2007 to November 2024 at every window.
- Estimation windows of 189, 252, 378 and 504 trading days, used both for discovery and for the sample covariance. The 378- and 504-day windows use equal weights for their first 6 and 12 rebalances, because there is not yet enough history, leaving 209 and 203 graph-fed rebalances.
- Transaction cost 5 bps per unit one-way turnover, swept over {0, 5, 10, 20}.
- Performance is out-of-sample, net-of-cost, annualised Sharpe of daily portfolio returns; CAGR, maximum drawdown and turnover are also reported.

### 4.7 Inference

- **Pairwise contrasts.** Politis-Romano stationary block bootstrap, 10,000 resamples, mean block length 21 days, resampling the joint return panel so cross-strategy correlation is preserved. The two-sided p-value is the percentile one (one minus the largest confidence level that excludes zero).
- **Non-normality.** Daily returns have excess kurtosis of about 15.9. The Probabilistic Sharpe ratio corrects for skewness, kurtosis and length.
- **Selection bias.** The Deflated Sharpe ratio raises the hurdle to the best Sharpe expected from N = 146 zero-skill trials (41 Phase I, 105 Phase II).
- **Family-wise tests.** Hansen's SPA on two families: (i) the 7 graph-consuming allocators against `correlation-hrp` and (ii) the edge-orientation-aware allocators against DYNOTEARS `skeleton-hrp`, each at every window and pooled across windows, over both the full and the post hoc narrowed sets. The 90% Model Confidence Set is run over the 17 252-day strategies. Both use mean daily net return as the loss.
- **Power.** The one-sided minimum detectable effect at 5% size and 80% power is about 0.047 Sharpe for the total contrast and 0.037 for the orientation contrast.
- **Out-of-sample.** The DYNOTEARS allocators are rerun on 19 rebalances from January 2025 to August 2026 on Yahoo prices with separate caches, with bootstrap standard errors.
- **Regimes.** 252-day daily returns are split by NBER recession versus expansion and by top versus bottom VIX quintile.

---

## 5. Layout of the codebase

About 19,000 lines of Python, of which roughly 7,500 are project code (`pipeline/`, `scripts/`, `tests/`, `reproduction.py`) and the rest are pruned vendored copies of three third-party libraries.

```
thesis/
├── reproduction.py            # one-shot driver: runs every script in order, idempotent
├── README.md                  # setup, data, run instructions, the 2026-09-05 correction
├── CLAUDE.md                  # assistant instructions (gitignored)
├── pipeline/                  # the library
│   ├── __init__.py            # re-exports run_closed_loop, ClosedLoopResult
│   ├── _vendored.py           # sys.modules shims for causalnex / lingam; THESIS_ROOT
│   ├── _parallel.py           # joblib rolling-window runner with pickle checkpoints
│   ├── requirements.txt       # pinned Python 3.13 dependencies
│   ├── data/                  # universe, prices (WRDS -> Yahoo), drivers (FRED/Yahoo), alignment
│   ├── discovery/             # DYNOTEARS, VARLiNGAM, ridge-Granger, NTS-NOTEARS wrappers; cache; asset graph
│   ├── factor_selection/      # Phase I: Stage A prune, K calibration, Stage B greedy, alpha-blend, V0 correlation
│   ├── sensitivities/         # Phase I: PyTorch FFNN Jacobian sensitivities, HSP distance
│   ├── feedback/              # Phase I: credit attribution, EMA utility, lookahead-safe store
│   ├── portfolio/             # HRP, HERC, HSP, the D-family allocators, topological ordering, backtest
│   ├── evaluation/            # metrics (Sharpe, PSR, DSR, ...), stationary block bootstrap, regimes
│   ├── stage1_pipeline.py     # per-rebalance discovery -> selection -> sensitivities
│   ├── closed_loop.py         # the walk-forward driver that wires stage 1, allocator, backtest, feedback
│   └── shakedown.py           # end-to-end real-data entry point (universe -> prices -> drivers -> backtest)
├── scripts/                   # launchers and collation
│   ├── phase_i_universe.txt   # the frozen 99-ticker universe
│   ├── run_phase_i.py         # V0 / V0' / V1 / V2 cells
│   ├── run_phase_ii.py        # one (method, allocator, window, tau) cell
│   ├── run_oos_slice.py       # 2025-26 out-of-sample cell on Yahoo prices
│   ├── collate_phase_ii.py    # Sharpe matrix + bootstrap contrasts
│   ├── collate_oos.py         # OOS table + LaTeX macros
│   ├── collate_j4.py          # K sweep and alpha/gamma grid tables
│   ├── robust_stats.py        # PSR, DSR, Reality Check, SPA, MCS, MDE; writes LaTeX macros
│   ├── regime_analysis.py     # NBER / VIX regime-conditional metrics
│   ├── plot_phase_ii_figures.py
│   ├── extract_asset_graphs.py    # cache-hit gate + DAG diagnostics
│   ├── verify_directional_prior.py
│   ├── run_seed_audit.py
│   └── probe_nts_notears.py
├── tests/                     # 88 pytest cases across 9 files (+ an HSP reference CLI)
├── results/                   # 227 entries: one directory per backtest cell, collated CSVs, figures/
├── final_report/              # main.tex, references.bib, figures/, _generated/{robust_stats,oos_stats}.tex
├── cache/                     # gitignored: prices, shares, wrds, drivers, ffnn, discovery (2,051 fits, 532 MB)
├── HSP/                       # Rodríguez Domínguez reference spreadsheets (Assets_SPX, Drivers_no_SB_Sectors)
├── causalnex/                 # vendored causalnex 0.12.1: structure/{dynotears,structuremodel,transformers}.py
├── lingam/                    # vendored lingam 1.12.2: var_lingam, direct_lingam, base, bootstrap, hsic, utils
└── nts-notears/               # vendored NTS-NOTEARS: nts_notears, utils, lbfgsb_scipy, locally_connected, trace_expm
```

### 5.1 Data flow

```
universe.py (S&P 500 history, top-N by CRSP mcap)  ─┐
assets.py (CRSP via wrds_backend.py, Yahoo fallback) ─┼─► alignment.build_joint_matrix  X = [D | A]  (4781 × 132)
drivers.py (35-spec catalogue, 33 retained)          ─┘                │
                                                                        ▼
                                 per rebalance t (every 21 trading days, 215 of them)
                                                                        │
                       window of 189/252/378/504 rows ending before t, z-scored
                                                                        │
                  discovery.cache.load_or_compute_discovery (SHA-256 of the window bytes)
                     ├─ dynotears.run_dynotears_joint_window   (tabu: asset -> driver forbidden)
                     ├─ varlingam.run_varlingam_joint_window   (masked ridge VAR + DirectLiNGAM prior)
                     └─ granger.run_granger_joint_window       (ridge VAR(1), density-matched)
                                                                        │
                  asset_graph.asset_graph_from_discovery  ->  AssetGraphWindow(M, zscore_std, resid_var_z, is_dag)
                                                                        │
                  portfolio.directed.dispatch_allocator(name, graph, returns_window)  ->  weights
                                                                        │
                  portfolio.backtest.run_backtest  (21-day hold, 5 bps one-way, gross and net NAV)
                                                                        │
                  results/<tag>/closed_loop.pkl  ->  scripts/collate_*  ->  CSVs, LaTeX macros, figures
```

### 5.2 Naming conventions

- **Matrix convention.** Everywhere in the pipeline, `M[i, j]` is the effect of variable i on variable j. Vendored lingam uses the opposite convention, so the VARLiNGAM wrapper transposes at fit time and nothing downstream transposes again.
- **Result tags.** Phase I cells are `results/phase_i_{v0|v0prime|v1|v2}[_varlingam]_w{W}[_k{K}|_a{a}_g{g}]`. Phase II cells are `results/phase_ii_{dynotears|varlingam|granger}_{D*}_w{W}[_cost{c}|_tau...]`, with `phase_ii_corr_hrp_w{W}`, `phase_ii_{ew|ivp}_w{W}` and `phase_ii_herc_corr_w{W}` for the graph-blind cells. OOS cells are `results/oos_{...}`.
- **Allocator IDs versus report names.** The code keeps the short IDs that predate the report's renaming:

| Code ID | Report name |
| --- | --- |
| `CORR` | correlation-hrp |
| `D0` | skeleton-hrp |
| `D0s` | undirected-hrp |
| `D0lw` | lw-hrp |
| `D0df` | defactored-hrp |
| `D0pc` | pcorr-hrp |
| `D1` | semcov-hrp |
| `D2` | topo-hrp |
| `D2s` | topo-semcov-hrp |
| `D3` | struct-rp |
| `D4` | coancestry-hrp |
| `HERCC`, `HERC0`, `HERC1` | correlation-herc, skeleton-herc, semcov-herc |
| `EW`, `IVP` | equal weight, inverse variance |
| `V0`, `V0prime`, `V1`, `V2` | HSP as published, asset-only causal HRP (identical to D0), causal-selection HSP, closed-loop HSP |

---

## 6. The code in detail

### 6.1 Infrastructure

**`pipeline/_vendored.py`.** The three third-party trees are pruned copies (causalnex 0.12.1, lingam 1.12.2, NTS-NOTEARS) with their package `__init__` files removed. This module fabricates stub packages with `types.ModuleType`, sets `__path__` to the vendored directory, inserts them into `sys.modules`, and attaches `StructureModel` onto the `causalnex.structure` stub because `dynotears.py` imports it at module level. It also re-adds `pd.Index.is_integer` (removed in pandas 2.1) for the vendored `DynamicDataTransformer`. It re-exports `from_pandas_dynamic`, `from_numpy_dynamic`, `StructureModel`, `VARLiNGAM`, `VARBootstrapResult` and `THESIS_ROOT`, which is the root of every cache path. NTS-NOTEARS is imported differently, by prepending `nts-notears/notears` to `sys.path` lazily, because its modules use bare imports. No vendored file has been modified: `git log --follow` shows one commit each, and the only thesis-adjacent grep hit is upstream's own "VAR.fit() is not searching lags correctly" comment. Licences: causalnex Apache 2.0, lingam MIT, nts-notears Apache 2.0.

**`pipeline/_parallel.py`.** `execute_windows(jobs, call_fn, n_jobs, label, checkpoint_dir, log_every)` partitions rolling-window jobs into already-checkpointed pickles versus to-do, runs the rest serially or through `joblib.Parallel(return_as="generator_unordered")`, pickles each result as it lands, logs a linear ETA, and returns results sorted by window index.

**`pipeline/requirements.txt`.** Pinned: numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, scikit-learn 1.8.0, statsmodels 0.14.6, networkx 3.6.1, yfinance 1.3.0, joblib 1.5.3, matplotlib 3.10.9. Floors: arch ≥ 7.2 (SPA, MCS), torch ≥ 2.4 (FFNN), cvxpy ≥ 1.5, pandas-market-calendars ≥ 4.4, sqlalchemy ≥ 2.0 and psycopg ≥ 3.1 (direct WRDS Postgres, deliberately not the `wrds` package, which pins old pandas), openpyxl, igraph.

### 6.2 Data layer (`pipeline/data/`)

**`universe.py`.** Point-in-time S&P 500 membership from the `fja05680/sp500` GitHub repository, cached as parquet with a 24-hour filename check. `membership_at(date)` returns the constituent set active at a date. `top_n_by_mcap_at(date, n, prices, shares)` computes market cap as the last close within a 5-day lookback times shares outstanding, restricted to members. `rolling_top_n_universe` and `union_of_universes` produce the S&P 100 approximation: the union of the top N at three snapshots.

**`wrds_backend.py`.** Connects to `wrds-pgdata.wharton.upenn.edu:9737` with SQLAlchemy and psycopg, username from `WRDS_USERNAME` or `~/.pgpass`, retry backoff (2, 5, 10 s), results cached under `cache/wrds/` by SHA-256 of the query parts. PERMNOs are resolved from `crsp.stocknames` over the requested date span (tickers get reused; PERMNOs do not). Adjusted close is `ABS(prc) / NULLIF(cfacpr, 0)` from `crsp.dsf` (negative `prc` marks a bid-ask midpoint). Market cap at a snapshot is `ABS(prc) * shrout * 1000 / NULLIF(cfacshr, 0)`. Shares outstanding is the latest `shrout` on or before the date, in units.

**`assets.py`.** `fetch_prices(tickers, start, end, prefer=("wrds", "yfinance"))` runs a per-ticker cascade: cache hit if the cached span covers the request, else WRDS, else `yf.download(auto_adjust=True)`. Each series is cached as `cache/prices/<TICKER>.parquet` with its source recorded. `fetch_shares_outstanding` prefers CRSP point-in-time `shrout` with a current-yfinance constant fallback. `coverage_report` warns when per-rebalance coverage drops below 95%. A `.env` at the repo root is loaded at import.

**`drivers.py`.** A `DRIVER_CATALOGUE` of 35 `DriverSpec` entries (name, source in {fred, yahoo, derived}, identifier, preprocessing in {log_return, first_diff, yoy_pct, yoy_diff, level}, optional `available_from`). Groups: macro (7, FRED monthly levels transformed year-on-year, or first-differenced for UNRATE and UMCSENT), Treasury yields and slopes (7, first differences), credit (BAA-AAA derived, BAA10Y, HYG/LQD log return from 2007-04-11), FX (5, dollar index diff plus four log returns), commodities (6, WTI and Brent from FRED, gold, natural gas, silver, copper from Yahoo futures because FRED's LBMA fixings were discontinued), volatility (VIX level, VVIX level from 2007-03-15), international equity ETFs (5 log returns). The two late-start series (HYG/LQD, VVIX) are dropped for the 2007 start, giving 33. FRED is fetched through the fredgraph CSV endpoint with a browser user agent and a `curl` subprocess fallback; Yahoo caches record the requested span in a `.meta` file so that a later request inside that span never refetches, because live refetches of auto-adjusted Yahoo data (EEM in particular) changed the DYNOTEARS fit. Monthly series are forward-filled onto the trading calendar as "the value an observer would have known on day t".

**`alignment.py`.** `trading_calendar(start, end)` uses `pandas_market_calendars` NYSE with an SPY-derived fallback. `build_joint_matrix(drivers, assets, calendar, drop_na)` returns a `JointMatrix` with columns always drivers first then assets, integer index arrays `driver_idx` and `asset_idx`, and, under `drop_na="drivers_only"`, zero-filled asset NaNs plus an `asset_eligibility` boolean frame so the zeros are never mistaken for returns. `zscore_window(frame, ddof=0, eps=1e-12)` standardises one window and must only be called inside the rolling loop. `stationarity_flags` runs ADF (AIC lag selection) and KPSS (constant, automatic lags) per column as a diagnostic that never drops anything.

### 6.3 Discovery (`pipeline/discovery/`)

**`dynotears.py`.** Wraps the vendored `from_pandas_dynamic`. `run_dynotears_joint_window(joint_window, driver_columns, asset_columns, p=1, lambda_w=0.05, lambda_a=0.05, w_threshold=0.01, max_iter=100, enforce_acyclic=True, enforce_tabu=True)` z-scores the window (ddof 0, std floored at 1e-12), builds the tabu list with `make_tabu_edges_asset_to_driver` (every asset-to-driver edge at lags 0 to p, encoded by causalnex as (0, 0) L-BFGS-B bounds, so the prior is a hard constraint), fits, converts the `StructureModel` into a contemporaneous `W` and lagged `A[k]` via `structure_model_to_matrices` (edges named `var_lag{k}` into a `lag0` node), and records the unregularised Frobenius reconstruction error as `fit_loss`. Because the continuous acyclicity constraint holds only to `h_tol`, `enforce_dag` repeatedly finds a cycle with networkx and deletes its smallest-magnitude edge; the count is stored as `acyclic_edges_removed`. The result is a `JointDynotearsWindow` dataclass with block accessors `driver_to_asset_block(lag)`, `asset_to_driver_block(lag)` (should be all zero) and `asset_to_asset_block(lag)` (what the allocators consume), plus the z-score mean and std. `run_rolling_dynotears_joint` runs windows through `execute_windows`.

The vendored DYNOTEARS minimises `(1/2n)‖X(I − W) − X_lags A‖²_F + λ_w‖W‖₁ + λ_a‖A‖₁` subject to `h(W) = tr(exp(W∘W)) − d = 0`, by augmented Lagrangian with L-BFGS-B on a positive/negative split of the weights, `rho *= 10` while progress on h is under 25%, `alpha += rho·h`, stopping at `h ≤ h_tol` or `max_iter` dual-ascent steps. Only `W` is constrained; lagged blocks point forward in time and cannot form cycles. Weights below `w_threshold` are zeroed after optimisation.

**`varlingam.py`.** Two stages with the directional prior in both. `estimate_var_coefs_masked(X, lags, driver_idx, asset_idx, alpha=1.0)` fits the VAR equation by equation with `sklearn.Ridge(fit_intercept=False)`, where driver equations regress only on lagged drivers and asset equations are unconstrained (plain OLS VAR is underdetermined at d = 132). `make_prior_knowledge_asset_to_driver` builds lingam's prior matrix (0 = forbidden edge). `run_varlingam_joint_window(..., lags=1, prune=True, random_state=42, ridge_alpha=1.0, enforce_prior_knowledge=True, n_bootstrap=0)` z-scores, passes the precomputed `ar_coefs` so VARLiNGAM skips its own VAR, constructs `DirectLiNGAM(prior_knowledge=pk)` for the contemporaneous `B0`, re-zeroes any asset-to-driver mass that lingam's pruning refit may have reintroduced, and transposes everything into the pipeline's i-to-j convention. Optional bootstrap probabilities and an HSIC residual-independence check (warns "LIKELY MISSPECIFIED" above a 20% rejection rate) are available. Both Phase I and Phase II pass `prune=False`, because lingam's adaptive-lasso pruning fails at d = 132 when a late-order variable has more predecessors than window samples.

The vendored DirectLiNGAM uses the pairwise likelihood-ratio measure (`pwling`) to pick the most exogenous variable, regresses it out, and repeats to get a causal order; VARLiNGAM then forms `B_τ = (I − B0) M_τ` for the lagged blocks.

**`granger.py`.** `run_granger_joint_window(joint_window, driver_columns, asset_columns, lambda_ridge=1e-2, target_density=None, threshold=0.0)` z-scores the joint panel, takes the asset block, and solves the closed-form ridge VAR(1) `coef = (X_lagᵀX_lag + λ·n·I)⁻¹ X_lagᵀY` with the penalty scaled by sample size so its strength is window-length invariant. Edge weight is `|coef|` with the diagonal zeroed. With `target_density` set, the k largest off-diagonal magnitudes are kept where `k = round(density · N(N−1))`. The result records `is_dag`, which is generally false, and every downstream consumer branches on it. In `stage1_pipeline`, `density_match_dynotears=True` first loads the paired DYNOTEARS fit for the same window, measures its asset-block density, and injects it as the Granger target, so the two graphs are compared at matched sparsity and the density enters the cache key.

**`nts_notears.py`.** A reduced-scope probe. Builds `NTS_NOTEARS(dims=[d, 8, 1], number_of_lags=1, prior_knowledge=...)` with the asset-to-driver ban as zero-bound rules, trains on CPU in float32 with `lambda1=0.02, lambda2=0.01, w_threshold=0.1, max_iter=20, h_tol=1e-6`, and reshapes the output into a `JointDynotearsWindow` so it is drop-in for the DYNOTEARS interface. Each edge is the L2 norm of a 1-D convolution kernel (magnitudes only). The vendored model is a pair of positive and negative `Conv1d` layers over time followed by per-variable locally connected MLPs, with the acyclicity constraint on the instantaneous kernel slice only.

**`cache.py`.** `discovery_cache_key` is a 24-hex-character SHA-256 over the method string, the driver and asset column names, the full float64 buffer of the window, the index (normalised to microseconds, because a datetime unit change once silently re-keyed 1,684 identical fits), and the sorted discovery kwargs. `load_or_compute_discovery` is a passthrough when caching is off, recomputes on a corrupt pickle, and writes atomically through a PID-suffixed temp file and `os.replace`. The key deliberately excludes Phase I's K, α and γ so those sweeps reuse the same graph fits. Cache dir `cache/discovery/` holds 2,051 fits (532 MB).

**`asset_graph.py`.** The single chokepoint between discovery and allocation. `asset_graph_from_discovery(disc, joint_window, method, tau=0.0, universe=None)` slices the lag-0 asset-to-asset block (rows and columns together, in the caller's universe order), zeroes entries with `|M| < tau` and the diagonal, slices the z-score std, computes structural residuals in z-space as `E = X_z (I − M)` and their variance (ddof 0), and tests acyclicity with Kahn's algorithm. There is no driver marginalisation and no path composition here: drivers are dropped, not integrated out, which is the source of the "residuals share a driver component" caveat. The frozen `AssetGraphWindow(end_date, asset_names, M, zscore_std, resid_var_z, method, tau, is_dag, meta)` is what every allocator receives.

### 6.4 Allocators (`pipeline/portfolio/`)

**`hrp.py`.** `hrp_weights(distance, covariance, linkage_method="single")` forces exact symmetry and a zero diagonal, runs `scipy.cluster.hierarchy.linkage` on the condensed distance, takes the quasi-diagonal leaf order, and applies `recursive_bisection`: split the ordered list in half, compute each half's inverse-variance-weighted cluster variance, allocate `alpha = 1 − v_left / (v_left + v_right)` to the left, recurse. `herc_weights` shares the clustering but descends the actual dendrogram from the root, splitting each node's budget by inverse cluster variance (the full-recursion HERC of Raffinot 2018).

**`hsp.py`.** Covariance estimators used by the whole family: `sample_covariance`, `ledoit_wolf_covariance` (sklearn), and `defactored_covariance`, which demeans, takes the equal-weight cross-sectional mean as the single factor, regresses each asset on it, and returns the residual covariance with the same ddof as `DataFrame.cov`. `hsp_weights_from_S` is HRP with the correlation distance replaced by the Euclidean distance between sensitivity rows.

**`_old_v123.py`.** Retained helpers: `symmetrise(M) = ½(|M| + |Mᵀ|)`, `nearest_psd`, `correlation_distance(corr) = sqrt(½(1 − corr))`, `symmetrise_distance` (symmetrise, rescale by the max off-diagonal to [0, 1], distance = 1 − similarity), and `causal_embedding_distance`, which embeds asset i as `e_i = [M[i, :], M[:, i]]` (out-edges then in-edges) and takes Euclidean distances. The embedding is invariant to reversing every edge at once but not to reversing one edge, which is why the report calls `skeleton-hrp` "transpose-invariant" rather than orientation-free.

**`directed.py`.** The Phase II family, all with signature `(graph, returns_window, linkage_method="single") -> pd.Series`:

- `total_effect_matrix(M, is_dag=True, k_trunc=10, spectral_target=0.95)`: `T = (I − Mᵀ)⁻¹` solved exactly for a DAG; for a cyclic Granger graph a truncated Neumann series of 10 terms, with `M` rescaled to spectral radius 0.95 if it is at or above 1.
- `structural_covariance_v2(graph, ridge=1e-6)`: `Σ_SEM = D_σ (T Σ_ε Tᵀ) D_σ` with `Σ_ε = diag(max(resid_var_z, 1e-12))` and `D_σ = diag(zscore_std)`, then nearest-PSD and a ridge of `1e-6 · tr(Σ)/N`. Condition numbers above 1e10 are logged.
- `erc_weights(cov)`: long-only equal risk contribution by cyclical coordinate descent on the log-barrier objective (Spinu 2013), deterministic, raising if risk contributions do not equalise to 1e-8.
- The allocator table (`ALLOCATORS`): `CORR` (correlation distance, sample covariance, graph supplies only the asset set), `D0` (embedding distance, sample), `D0s` (symmetrised adjacency distance, sample), `D0lw` (embedding, Ledoit-Wolf), `D0df` (embedding, de-factored), `D0pc` (embedding distance of a partial-correlation matrix built from the inverse of a Ledoit-Wolf covariance, thresholded to the graph's non-zero count, sample covariance), `EW`, `IVP`, `HERCC`/`HERC0`/`HERC1` (HERC on correlation, embedding+sample, embedding+Σ_SEM), `D1` (embedding, Σ_SEM), `D3` (ERC on Σ_SEM, no hierarchy), `D4` (co-ancestry distance `sqrt(2(1 − S))` where `S` is the cosine similarity of row-normalised total-effect rows, sample covariance).
- `dispatch_allocator(name, ...)` routes `D2` and `D2s` to `topological.d2_weights` lazily.
- Every HRP and HERC path accepts `psd_project_distance=False`. Setting it to True reinstates the pre-2026-09-05 nearest-PSD projection of the distance matrix for replaying the original Phase I bundle.

**`topological.py`.** `topological_order(M)` is Kahn's algorithm, upstream first, with ties broken by total downstream influence `Σ_r |T[r, i]|` descending and then by name. `remove_feedback_arcs` isolates the cyclic core and repeatedly zeroes its weakest edge (Granger graphs only). `d2_weights(graph, returns_window, covariance="sample"|"structural")` runs `recursive_bisection` directly on the topological order with no clustering step. `dag_diagnostics` reports edge count, density, longest path (`dag_depth`), roots and leaves; `order_stability` gives Kendall's tau between consecutive orders.

**`causal_hsp.py`.** Phase I: `v0_vanilla_hsp`, `v0prime_asset_only_causal_hrp` (embedding distance on the asset block plus sample covariance, byte-identical to `D0`), and `v1`/`v2` wrappers that are mathematically identical to V0 because the variants differ only in how the sensitivity matrix was selected upstream.

**`backtest.py`.** `run_backtest(rebalance_dates, universe_at, strategy, prices, holding_days=21, transaction_cost_bps=5.0, on_rebalance_complete=None)`: per rebalance, get the eligible universe, call the strategy, reindex and renormalise weights to sum to one (long-only by construction, no caps), hold for the next 21 trading days strictly after t, compute one-way turnover `½ Σ|w_new − w_prev|` over the union of names, charge `turnover · bps / 10,000` on the first holding day of the net track only, record the holding reward (portfolio Sharpe minus equal-weight Sharpe of the same names, the Phase I feedback signal), and compound gross and net NAV independently. The callback fires before advancing so a closed-loop driver can update state the next strategy call reads.

### 6.5 The walk-forward drivers

**`stage1_pipeline.py`.** `fit_stage1_rebalance(...)` z-scores the window and dispatches on `selection_method`: `"asset_only"` (V0' and all of Phase II) runs discovery through the cache and returns an empty selection; `"correlation"` (V0) runs the cumulative-correlation selector with no discovery; `"causal_greedy"` (V1, V2) runs discovery then `select_drivers`. All three then fit sensitivities (a zero-width placeholder for empty selections). `run_stage1` is the batch version; the closed loop calls the per-rebalance function directly.

**`closed_loop.py`.** `run_closed_loop(joint_frame, asset_returns, rebalance_dates, universe_at, ..., K=10, window_size=504, lookback_days=504, holding_days=21, transaction_cost_bps=5.0, gamma_ema=0.3, selection_method, discovery_method, allocator=None, graph_tau=0.0, discovery_cache=False, ...)`. The inner `strategy(t, asset_names)` slices the window ending strictly before t, calls stage 1 with the live utility lookup, and then either (Phase II) builds the `AssetGraphWindow` restricted to the eligible assets and dispatches the named allocator, or (V0') calls the asset-only HRP, or (V0/V1/V2) calls the HSP allocator on the sensitivity matrix. The inner `on_rebalance_complete` computes sensitivity-weighted credit, applies the EMA update, and appends to the `UtilityStore`. Prices for the backtest are synthesised as cumulative products of the log-return panel. The run ends by pickling `closed_loop.pkl` (config, backtest, credit history, utility frame) into `results/<tag>/`.

**`shakedown.py`.** The end-to-end entry point that `scripts/run_phase_i.py` and `run_phase_ii.py` call. Builds the universe (override or top-N union at three snapshots), fetches prices and converts to log returns, builds the driver pool on the NYSE calendar, builds the joint matrix with `drop_na="drivers_only"`, derives rebalance dates as `cal[first_backtest_pos : len − holding_days − 1 : 21]`, optionally calibrates K on the burn-in window (permuting each driver column and refitting DYNOTEARS with `max_iter` capped at 20 for the null), defines `universe_at(t)` as the assets eligible on every day of the lookback, runs the closed loop, and writes `shakedown_bundle.pkl`.

### 6.6 Phase I modules (retained, and still in the trial count)

**`factor_selection/prune.py`** (Stage A): the causal influence score of a driver is the sum over lagged driver-to-asset edges of `|edge| × stability`, where for DYNOTEARS the stability mask keeps the top 10% of non-zero lagged magnitudes (data-driven quantile threshold so the stage is scale-invariant) and for VARLiNGAM it is bootstrap probability or presence. Only lagged edges count because contemporaneous edges have no temporal precedence. The pool is the top 2K positive scores.

**`factor_selection/greedy.py`** (Stage B): forward stepwise selection. Repeatedly add the candidate whose lagged ridge regression (`alpha=1.0`, chronological 80/20 split) of all asset returns on the selected drivers gives the largest held-out gain in negative half-MSE, until K or the gain falls below epsilon.

**`factor_selection/k_calibration.py`**: `K = max(K_elbow, K_perm)`, where `K_elbow` is Kneedle on the sorted Stage A curve and `K_perm` counts drivers whose real score is significant against a permutation null under Benjamini-Hochberg FDR at 5% with z-score p-values (the Monte Carlo p-value floor of 1/(B+1) is above BH's rank-1 threshold at typical B, so it is fail-closed; the old max-of-d statistic is kept as `K_perm_legacy`). Seeds are drawn up front so `n_jobs` does not change the result. The report's K is 17.

**`factor_selection/correlation_selector.py`** (V0): sum of `|corr(driver_{t−lag}, asset_t)|` over lags {0, 1} and all assets, top K. The published HSP rule.

**`factor_selection/selector.py`**: the α-blend `score = α·z(causal) + (1 − α)·z(utility)`, α forced to 1 during a 6-rebalance burn-in, pool restricted to drivers with strictly positive causal evidence so utility alone never selects, then Stage B.

**`sensitivities/ffnn.py`**: a shared-body MLP (GELU, depth in {1, 2}, width in {16, 32, 64}, chosen by validation RMSE) with one output head per asset, trained full-batch with AdamW (lr 1e-2, weight decay 1e-4, 200 epochs, patience 20) on lagged drivers to contemporaneous returns, with eligibility masks excluding pre-inception cells. Sensitivities are the training-window mean of the `torch.func.jacrev` Jacobian, lag blocks summed, giving `S ∈ R^{N×K}`. Cached under `cache/ffnn/` by a hash of the inputs. Not bit-reproducible on Apple MPS, hence the seed audit.

**`sensitivities/sensitivity_matrix.py`**: `distance_from_S` (Euclidean in sensitivity space, the HSP distance), `correlation_from_S`, `lopez_de_prado_distance`, and two K diagnostics (distance concentration, effective dimensionality at 95% variance).

**`feedback/utility.py`** and **`feedback/storage.py`**: credit for driver d is the holding reward split in proportion to `Σ_i |w_i s_{i,d}|`; the EMA update `γ·credit + (1 − γ)·prior` touches selected drivers only; the `UtilityStore` keys rows by holding-period end date and asserts a gap of at least 21 days before any lookup, which is the lookahead guard.

### 6.7 Scripts and reproduction

**`reproduction.py`** runs twelve steps as subprocesses of `python -m scripts.<name>` from the repo root, logs to `results/reproduction_logs/<name>.log`, stops at the first failure, and skips any backtest cell whose `results/<tag>/closed_loop.pkl` exists unless `--force`. Steps: `preflight` (Python ≥ 3.13, vendored trees, universe file, HSP spreadsheets, WRDS credential warning), `tests` (pytest), `phase1` (V1 first at 252 and 504 to calibrate K, then V0/V2/V0' at that K, then VARLiNGAM V1/V2: 12 cells), `phase1_sweeps` (20 K-sweep cells, 9 α/γ cells, collate), `phase2` (2 methods × 7 graph allocators × 4 windows = 56, plus 9 controls × 4 windows = 36, plus 4 Granger cells at 252), `phase2_sweeps` (9 τ cells, 18 cost cells), `gates` (cache-hit gate, DAG diagnostics, and a replication gate asserting `max|ΔNAV| ≤ 1e-3` between DYNOTEARS D0 at 252 and Phase I V0'), `checks` (directional prior, 20-seed audit, NTS probe), `collate` (Phase II matrix and contrasts, regimes, robust stats in both trial universes), `figures`, `oos` (7 allocators × 4 windows, collate), `compile` (latexmk).

**Cell launchers.** `run_phase_i.py --variant {V0,V0prime,V1,V2} --window --discovery-method --k --alpha --gamma --transaction-cost-bps --tag-suffix --discovery-cache`. `run_phase_ii.py --method {dynotears,varlingam,granger} --allocator --window --tau --transaction-cost-bps --granger-lambda --tag-suffix`, always with the discovery cache on and `selection_method="asset_only"`. `run_oos_slice.py --allocator --window` redirects the price and driver caches to `cache/prices_oos` and `cache/drivers_oos`, stubs out WRDS, and backtests 2025-01-02 to 2026-08-14 with data from 2022-07-01 so the 504-day lookback is burned in.

**Collation.** `collate_phase_ii.py` writes `results/phase_ii_matrix.csv` (Sharpe, CAGR, max drawdown, mean turnover, final NAV per cell) and `results/phase_ii_contrasts.csv` (the pre-specified contrast families with 10,000-resample bootstrap CIs and p-values, seed 42): `{D0s, D1, D2, D2s, D3, D4} − D0`, `D0 − V0`, the decomposition `D0 − CORR`, `D1 − CORR`, `D2s − CORR`, `best(D*) − V1`, Granger versus DYNOTEARS, the mechanism controls, and the HERC pair. Controls and naive anchors are kept out of the `best(D*)` search. `collate_oos.py` writes the OOS Sharpe table, the five OOS contrasts with bootstrap standard errors and no p-values (by the pre-committed protocol), and `final_report/_generated/oos_stats.tex`. `collate_j4.py` writes the K-sweep and α/γ tables.

**`robust_stats.py`.** The trial universe is 41 Phase I configurations by default and 146 with `--phase-ii`, which is the mode that owns the report macros. It computes annualised Sharpe, PSR against zero and against the published-HSP baseline, and DSR deflated against all trial Sharpes; deduplicates byte-identical return series (V2 equals V1, D0 equals V0') before SPA and MCS because duplicates give the studentised MCS zero differential variance; runs White's Reality Check and Hansen's SPA via `arch.bootstrap.SPA` (block 21, 10,000 reps, seed 42) and the 90% MCS; runs every family test twice, once on the pre-registered family and once on the narrowed one; adds a pooled SPA over the union of all four windows; and emits the full-sample Sharpe SE (Lo 2002), the contrast SEs and minimum detectable effects, and the closed-loop reward signal-to-noise. Output is `final_report/_generated/robust_stats.tex`.

**Other scripts.** `regime_analysis.py` (NBER and VIX-quintile conditional Sharpe for 21 variants, with a self-check that the "all" regime reproduces the full-sample Sharpe to 1e-9, plus named stress windows). `extract_asset_graphs.py` (fails with `CACHE-HIT GATE FAILED` on any cache miss; writes `phase_ii_dag_diagnostics.csv`). `verify_directional_prior.py` (six windows with and without the tabu mask). `run_seed_audit.py` (FFNN seeds with the FFNN cache off, resumable). `probe_nts_notears.py` (four windows, 25 assets, 504 days). `plot_phase_ii_figures.py` (regenerates the five figures from the committed CSVs only, Okabe-Ito palette).

### 6.8 Tests (`tests/`, 88 cases)

| File | What it pins down |
| --- | --- |
| `test_asset_graph.py` (9) | Slice round-trip, no driver leakage, τ pruning, universe order, residual variance formula `E = X_z(I − M)`, no double transpose for VARLiNGAM, cycle detection. |
| `test_closed_loop.py` (4) | The loop actually closes after burn-in and every lookup respects the 21-day gap; V0 skips discovery; VARLiNGAM path is live; V0 and V1 select differently. |
| `test_directed_allocators.py` (17) | Total-effect matrix equals the Neumann sum and the inverse; spectral guard on a 2-cycle; Σ_SEM limits (M = 0, unit chain grows variance 1, 2, 3, quadratic in std); ERC parity and determinism; `D0` equals V0' exactly; every allocator is long-only, sums to one and is deterministic; CORR is graph-blind; HERC variants are orientation-blind / graph-blind / orientation-sensitive as claimed. |
| `test_discovery_cache.py` (4) | Cache hit is byte-exact and writes one pickle; `use_cache=False` touches no disk; keys isolate on kwargs, method and data; key is invariant to the datetime64 unit. |
| `test_fixed_graph_ablation.py` (16) | Controls `{D0, D0s, D0lw, D0df}` are transpose-invariant; treatments `{D1, D2s, D3, D4}` respond to edge reversal; D2 responds to topology and to reversal on an odd-sized universe (on a power-of-two universe reversal mirrors the order and bisection gives the same tree); `D0pc` sees the graph only through its edge count. |
| `test_granger.py` (4) | Ridge VAR(1) recovers planted support with precision > 0.8; density matching within 1.5 edges; driver blocks exactly zero; the chokepoint consumes a Granger window. |
| `test_k_calibration.py` (3) | BH-FDR recovers 5 planted signals with ≤ 3 false positives while Monte Carlo BH is fail-closed; capping `max_iter` on null fits changes the null distribution by KS < 0.25; `n_jobs` does not change the result. |
| `test_robust_stats.py` (8) | PSR pinned to 0.675943 on a fixture and monotone in T and Sharpe; expected max Sharpe grows with trials; DSR collapses to PSR at one trial; Reality Check, SPA and MCS return valid output. |
| `test_topological.py` (7) | Order respects every edge over 100 random DAGs; deterministic under relabelling; 3-cycle raises; feedback-arc removal drops the weakest cycle edge; D2 valid under both covariances and survives a non-DAG; diagnostics on a chain. |

`test_hsp_reference.py` is a CLI (`python -m tests.test_hsp_reference`) that replays the Rodríguez Domínguez reference spreadsheets in `HSP/` (14 S&P names 2011 to 2022, and 211 driver series 2015 to 2022) rather than a pytest module.

---

## 7. How the literature maps onto the code

| Concept in the report | Source | Where it lives in the code |
| --- | --- | --- |
| HRP: correlation distance, quasi-diagonalisation, recursive bisection | López de Prado 2016 | `portfolio/hrp.py` (`hrp_weights`, `quasi_diagonal_order`, `recursive_bisection`); distance in `_old_v123.correlation_distance` and `sensitivity_matrix.lopez_de_prado_distance`; baseline allocator `directed.corr_hrp_weights` |
| HERC full-recursion variant | Raffinot 2018 | `portfolio/hrp.py::herc_weights`; allocators `HERCC`, `HERC0`, `HERC1` in `directed.py` |
| HSP: cluster on FFNN sensitivities to drivers, drivers chosen by cumulative correlation | Rodríguez Domínguez 2023 | `sensitivities/ffnn.py`, `sensitivity_matrix.distance_from_S`, `portfolio/hsp.py`, `factor_selection/correlation_selector.py`; reference data in `HSP/`; V0 in `causal_hsp.py` |
| DYNOTEARS: SVAR structure learning with the `tr(exp(W∘W)) − d` acyclicity constraint and L1 penalties | Pamfil et al. 2020, Zheng et al. 2018 | vendored `causalnex/causalnex/structure/dynotears.py`; wrapper `discovery/dynotears.py` (λ = 0.05, p = 1, threshold 0.01) |
| Directional prior: drivers cause assets, never the reverse | Thesis design choice | `dynotears.make_tabu_edges_asset_to_driver`, `varlingam.make_prior_knowledge_asset_to_driver` and `estimate_var_coefs_masked`, `nts_notears._asset_to_driver_prior`; verified by `scripts/verify_directional_prior.py` |
| Standardisation before NOTEARS-type fitting to remove varsortability | Reisach et al. 2021, Peters and Bühlmann 2014 | `alignment.zscore_window`, called inside every discovery wrapper; z-score stats stored on the window so `Σ_SEM` can be converted back to return units |
| VARLiNGAM: VAR then DirectLiNGAM on residuals, non-Gaussian identification | Hyvärinen et al. 2010, Shimizu et al. 2011 | vendored `lingam/lingam/{var_lingam,direct_lingam}.py`; wrapper `discovery/varlingam.py` with `prune=False` |
| Ridge-Granger VAR(1) directed graph as a magnitude-degraded orientation control | Granger 1969 | `discovery/granger.py`, density-matched in `stage1_pipeline.py`; allocator support for cyclic input via `total_effect_matrix(is_dag=False)` and `topological.remove_feedback_arcs` |
| NTS-NOTEARS non-linear extension | Sun et al. 2023 | vendored `nts-notears/notears/`; wrapper `discovery/nts_notears.py`; probe `scripts/probe_nts_notears.py` |
| Linear SEM, total-effect matrix `T = (I − Mᵀ)⁻¹`, implied covariance `T Σ_ε Tᵀ` | Wright 1934, Bollen 1989 | `directed.total_effect_matrix`, `directed.structural_covariance_v2`; residual variances from `asset_graph.asset_graph_from_discovery` |
| Skeleton-only allocators | Thesis design | `D0` (`causal_embedding_distance`), `D0s` (`symmetrise_distance`) |
| Edge-orientation-aware allocators: Σ_SEM at the allocation step, topological order at the ordering step | Thesis contribution | `D1` (`directed.d1_weights`), `D2` and `D2s` (`topological.d2_weights`), plus `D3` (`erc_weights` on Σ_SEM) and `D4` (co-ancestry) |
| Ledoit-Wolf shrinkage control | Ledoit and Wolf 2004 | `hsp.ledoit_wolf_covariance`; allocator `D0lw` |
| Single-factor de-factored covariance control | Sharpe 1963, Laloux et al. 1999 | `hsp.defactored_covariance` (equal-weight mean as the factor); allocator `D0df` |
| Density-matched partial-correlation network control | Friedman et al. 2008 (motivation) | `directed.d0pc_weights`: precision from the inverse Ledoit-Wolf covariance, thresholded to `count_nonzero(graph.M)` |
| Equal risk contribution by cyclical coordinate descent | Spinu 2013 | `directed.erc_weights` |
| Stationary block bootstrap for Sharpe contrasts | Politis and Romano 1994 | `evaluation/bootstrap.py` (geometric block lengths, mean 21, joint panel resampled, percentile p-value); used with 10,000 resamples in the collation scripts |
| Probabilistic and Deflated Sharpe ratio | Bailey and López de Prado 2012, 2014 | `evaluation/metrics.py` (`probabilistic_sharpe_ratio`, `expected_max_sharpe`, `deflated_sharpe_ratio`); tables in `scripts/robust_stats.py` |
| Reality Check, SPA, Model Confidence Set | White 2000, Hansen 2005, Hansen et al. 2011 | `scripts/robust_stats.py` via `arch.bootstrap.SPA` and `MCS`, with a hand-rolled Reality Check fallback |
| Sharpe standard error and minimum detectable effect | Lo 2002 | `robust_stats.full_sample_sharpe_se`, `contrast_se_mde` |
| Regime information in causal networks | Howard et al. 2025 | `evaluation/regime.py` (NBER USREC from FRED, VIX quintiles, network-density regimes); `scripts/regime_analysis.py` |
| Greedy factor selection, knee detection, BH-FDR permutation null (Phase I) | Standard methods | `factor_selection/{greedy,k_calibration}.py` |
| Point-in-time membership and market-cap screens (survivorship control at the data level) | CRSP conventions | `data/universe.py`, `data/wrds_backend.py` |

---

## 8. Results

### 8.1 Headline: no gain from the graph

`correlation-hrp` net Sharpe at 189, 252, 378 and 504 days: 0.411, 0.427, 0.407, 0.383.

Net annualised Sharpe of the graph-fed allocators, 2007 to 2024:

| Graph | Window | skeleton-hrp | undirected-hrp | semcov-hrp | topo-hrp | topo-semcov-hrp | struct-rp | coancestry-hrp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DYNOTEARS | 189 | 0.392 | 0.389 | 0.407 | 0.392 | 0.408 | 0.400 | 0.410 |
| DYNOTEARS | 252 | 0.399 | 0.401 | 0.410 | 0.399 | 0.406 | 0.397 | 0.396 |
| DYNOTEARS | 378 | 0.404 | 0.412 | 0.413 | 0.405 | 0.419 | 0.400 | 0.403 |
| DYNOTEARS | 504 | 0.375 | 0.394 | 0.396 | 0.382 | 0.399 | 0.387 | 0.390 |
| VARLiNGAM | 189 | 0.376 | 0.387 | 0.340 | 0.387 | 0.394 | 0.400 | 0.398 |
| VARLiNGAM | 252 | 0.394 | 0.392 | 0.385 | 0.388 | 0.386 | 0.392 | 0.418 |
| VARLiNGAM | 378 | 0.399 | 0.400 | 0.410 | 0.398 | 0.399 | 0.394 | 0.419 |
| VARLiNGAM | 504 | 0.364 | 0.377 | 0.371 | 0.372 | 0.375 | 0.375 | 0.394 |
| ridge-Granger | 252 | 0.386 | – | 0.352 | 0.393 | – | 0.350 | – |

Of the 20 DYNOTEARS comparisons against the baseline (5 allocators, 4 windows), 6 are positive and all 6 are at the two longer windows. Only 1 of 20 VARLiNGAM comparisons is positive. At the two shorter windows the baseline beats every graph-fed allocator under every discovery method. Hansen's SPA never rejects the null that no family member beats `correlation-hrp`.

### 8.2 The decomposition (DYNOTEARS, ΔSharpe with bootstrap p-values)

| Contrast | 189 | 252 | 378 | 504 |
| --- | --- | --- | --- | --- |
| skeleton: skeleton-hrp - correlation-hrp | -0.019 (0.21) | -0.028 (0.056) | -0.003 (0.80) | -0.008 (0.56) |
| orientation, allocation step: semcov-hrp - skeleton-hrp | +0.015 (0.22) | +0.011 (0.35) | +0.009 (0.45) | +0.021 (0.051) |
| orientation, both steps: topo-semcov-hrp - skeleton-hrp | +0.016 (0.31) | +0.007 (0.61) | +0.015 (0.33) | +0.024 (0.086) |
| total: semcov-hrp - correlation-hrp | -0.004 (0.82) | -0.017 (0.36) | +0.006 (0.73) | +0.013 (0.44) |
| total: topo-semcov-hrp - correlation-hrp | -0.003 (0.87) | -0.021 (0.25) | +0.012 (0.50) | +0.016 (0.33) |

The skeleton is negative at every window, the orientations positive at every window, and all 12 DYNOTEARS orientation contrasts are non-negative to three decimals. Nothing reaches p < 0.05; the smallest orientation p-value (0.051) is roughly what multiplicity predicts under the null.

Against the stricter `undirected-hrp` skeleton, the skeleton effect becomes -0.022, -0.026, +0.005, +0.011 and the orientation effect for `semcov-hrp` becomes +0.017, +0.009, +0.001, +0.001. So the orientation effect at the long windows only holds against `skeleton-hrp`; `undirected-hrp` beats `skeleton-hrp` by +0.008 (378) and +0.019 (504, p = 0.072).

### 8.3 Mechanism controls (DYNOTEARS)

| Allocator | Role | 189 | 252 | 378 | 504 |
| --- | --- | --- | --- | --- | --- |
| lw-hrp | shrinkage, no orientation | 0.389 | 0.394 | 0.382 | 0.356 |
| defactored-hrp | de-factored, no orientation | 0.411 | 0.409 | 0.425 | 0.402 |
| pcorr-hrp | graph-free, density-matched skeleton | 0.372 | 0.410 | 0.399 | 0.382 |

- **Shrinkage does not explain the orientation gain.** `lw-hrp` is below `skeleton-hrp` at every window and below `semcov-hrp` by 0.016 to 0.040.
- **Market-factor removal does.** `defactored-hrp` matches `semcov-hrp` at every window; the residual `semcov-hrp` - `defactored-hrp` is between -0.011 and +0.001.
- **The skeleton's cost is not graph-specific.** `pcorr-hrp` matches `skeleton-hrp` at every window (differences -0.020, +0.012, -0.005, +0.007; p ≥ 0.07) and both underperform the baseline (`pcorr-hrp` - `correlation-hrp` = -0.039 at p = 0.008, then -0.016, -0.008, -0.001).
- **Edge magnitudes still matter.** Under ridge-Granger graphs (directions kept, magnitudes degraded) `topo-hrp` is unchanged (0.393 vs 0.399) but `semcov-hrp` falls to 0.352 (-0.058, p = 0.13). Under VARLiNGAM the orientation contrasts show no systematic sign (6 of 12 positive, mean 0.000).

### 8.4 Robustness

- **Sparsity τ** (DYNOTEARS, 252 days): `skeleton-hrp` 0.399, 0.399, 0.386, 0.395 and `topo-hrp` 0.399, 0.399, 0.403, 0.402 at τ = 0, 0.01, 0.05, 0.10. `semcov-hrp` was not swept.
- **Transaction cost** (0, 5, 10, 20 bps): `semcov-hrp` - `skeleton-hrp` at 504 days grows from +0.020 to +0.023 as cost rises, because `semcov-hrp` has lower turnover (0.089 versus 0.107 per rebalance). `topo-semcov-hrp`'s gain at 252 days shrinks from +0.008 to +0.005 because its turnover is 0.140.
- **HERC** reproduces both signs at every window with larger magnitudes: `semcov-herc` - `skeleton-herc` = +0.088, +0.052, +0.010, +0.049 (p = 0.10, 0.38, 0.90, 0.46); `skeleton-herc` - `correlation-herc` = -0.258, -0.182, -0.028, -0.076 (p ≥ 0.07). HERC levels: correlation 0.454, 0.410, 0.345, 0.364; skeleton 0.197, 0.228, 0.316, 0.288; semcov 0.285, 0.280, 0.326, 0.337.
- **Return terms at 252 days** (DYNOTEARS): equal weight 0.364 Sharpe, 5.32% CAGR, -61.9% max drawdown; inverse variance 0.388, 5.21%, -51.0%; `correlation-hrp` 0.427, 5.77%, -49.7%, turnover 0.128; `skeleton-hrp` 0.399, 5.37%, -50.2%, 0.118; `semcov-hrp` 0.410, 5.60%, -50.0%, 0.110; `topo-semcov-hrp` 0.406, 5.52%, -50.6%, 0.140.
- **Regimes at 252 days**: in every regime except the bottom VIX quintile the graph-fed allocators are below the baseline (NBER recession: -0.567 baseline vs -0.591 / -0.590 / -0.606; high-vol: -1.316 vs -1.326 / -1.351 / -1.361; low-vol: 5.746 vs 5.789 / 5.887 / 5.876). This contradicts the hypothesis that causal structure helps most under stress.

### 8.5 Multiplicity-aware inference

- **PSR** at 252 days: 0.964 (`correlation-hrp`), 0.953 (`skeleton-hrp`), 0.958 (`semcov-hrp`). Non-normality does not threaten the sign of any level.
- **DSR** over 146 trials: no configuration reaches 0.95. `correlation-hrp` at 252 days is among the highest at 0.925; `semcov-hrp` 0.915; `skeleton-hrp` 0.907. The spread across configurations is under 0.02, so the DSR does not rank them.
- **SPA p-values**:

| Family | Set | 189 | 252 | 378 | 504 | Pooled |
| --- | --- | --- | --- | --- | --- | --- |
| (i) vs correlation-hrp | full (7) | 0.440 | 0.785 | 0.346 | 0.235 | 0.823 |
| (i) vs correlation-hrp | narrowed (5) | 0.673 | 0.886 | 0.273 | 0.213 | 0.827 |
| (ii) vs skeleton-hrp | full | 0.217 | 0.299 | 0.328 | 0.173 | 0.451 |
| (ii) vs skeleton-hrp | narrowed (3) | 0.361 | 0.333 | 0.175 | 0.076 | 0.399 |

- **MCS** at 90% retains 16 of the 17 252-day strategies and excludes only HSP as published.
- **Power.** Bootstrap standard errors at 252 days are 0.019 (total) and 0.015 (orientation, larger of the two); minimum detectable effects 0.047 and 0.037, roughly twice the observed effects.

### 8.6 Out-of-sample, January 2025 to August 2026 (19 rebalances, Yahoo prices)

| Contrast | 189 | 252 | 378 | 504 |
| --- | --- | --- | --- | --- |
| skeleton: skeleton-hrp - correlation-hrp | +0.107 (0.076) | +0.071 (0.072) | -0.009 (0.065) | +0.100 (0.056) |
| orientation: semcov-hrp - skeleton-hrp | +0.035 (0.078) | +0.050 (0.086) | +0.105 (0.061) | +0.036 (0.065) |
| orientation: topo-semcov-hrp - skeleton-hrp | +0.002 (0.105) | +0.013 (0.107) | +0.079 (0.107) | +0.110 (0.087) |
| total: semcov-hrp - correlation-hrp | +0.142 (0.122) | +0.122 (0.122) | +0.096 (0.077) | +0.136 (0.076) |
| residual: semcov-hrp - defactored-hrp | -0.090 (0.138) | -0.001 (0.116) | +0.005 (0.110) | -0.001 (0.107) |

Standard errors in parentheses. The orientation effect is positive at every window, as in-sample. The total is positive because the skeleton's sign flipped out of sample at three of four windows, so the skeleton effect is not stable over time. The de-factored residual stays within 0.006 at the three longer windows but is -0.090 at 189 days. All of these are sign checks; the standard errors are far too large for significance.

OOS Sharpe levels for reference (`results/oos_slice.csv`):

| Allocator | 189 | 252 | 378 | 504 |
| --- | --- | --- | --- | --- |
| CORR | 0.826 | 0.859 | 0.900 | 0.850 |
| D0 | 0.934 | 0.930 | 0.891 | 0.950 |
| D0s | 0.927 | 0.919 | 0.877 | 0.879 |
| D1 | 0.969 | 0.980 | 0.996 | 0.986 |
| D2 | 0.942 | 0.889 | 0.910 | 0.947 |
| D2s | 0.936 | 0.943 | 0.970 | 1.060 |
| D0df | 1.059 | 0.981 | 0.991 | 0.987 |

### 8.7 Phase I (HSP) results

Phase I net Sharpe (`results/phase_ii_matrix.csv`, V rows):

| Variant | 252 | 504 |
| --- | --- | --- |
| V0 (HSP as published, K = 17) | 0.371 | 0.370 |
| V0' (asset-only causal HRP, identical to D0) | 0.399 | 0.375 |
| V1 DYNOTEARS (causal driver selection) | 0.381 | 0.371 |
| V1 VARLiNGAM | 0.399 | 0.357 |

- **K sweep** (V1 minus V0, 2,000 resamples): at 252 days the difference is -0.015 (K = 10, p = 0.13), +0.003 (14), +0.010 (17, p = 0.27), -0.016 (20, p = 0.10), -0.010 (25); at 504 days it is between -0.012 and +0.007, never significant, and V1's Sharpe is 0.3715 for every K from 14 to 25, so the selection saturates. The calibrated K is 17 at both windows.
- **α/γ grid**: all nine V2 cells at 252 days have Sharpe 0.3811, exactly V1's value, with ΔSharpe = 0.0 and p = 1.0. The closed loop is inert because the assignment function can only re-rank drivers inside the causally selected pool and the monthly reward has mean -0.048, standard deviation 1.055, signal-to-noise about 0.05.
- **Seed audit** (20 FFNN seeds, V1 DYNOTEARS 252 K = 17): net Sharpe in [0.375, 0.396], median 0.389, range 0.021. The committed seed 0 result (0.381) sits at the 5th percentile of its own distribution.
- **Asset-only over HSP**: D0 − V0 is +0.028 at 252 days (p = 0.003, bootstrap; Reality Check and SPA p = 0.004) and +0.005 at 504 days (p = 0.61). This is the one significant in-sample result in the whole battery, and it says that dropping HSP's driver route in favour of the asset skeleton helps, not that the graph beats HRP.
- HRP beats HSP as published at 252 days: 0.427 versus 0.371, and the 90% MCS excludes only HSP as published.

### 8.8 Graph and pipeline checks

**DAG diagnostics** (`results/phase_ii_dag_diagnostics.csv`; the cache-hit gate passed 215/215, 215/215, 209/209 and 203/203 windows at the four horizons):

| Method | Window | Windows | Mean density | Mean depth | Mean edges | All DAGs |
| --- | --- | --- | --- | --- | --- | --- |
| DYNOTEARS | 189 | 215 | 0.0633 | 50.0 | 613.7 | yes |
| DYNOTEARS | 252 | 215 | 0.0644 | 49.7 | 624.8 | yes |
| DYNOTEARS | 378 | 209 | 0.0674 | 50.1 | 653.5 | yes |
| DYNOTEARS | 504 | 203 | 0.0694 | 50.8 | 673.2 | yes |
| VARLiNGAM | 189 | 215 | 0.3195 | 63.3 | 3099.8 | yes |
| VARLiNGAM | 252 | 215 | 0.1462 | 40.0 | 1418.8 | yes |
| VARLiNGAM | 378 | 209 | 0.0693 | 29.9 | 672.6 | yes |
| VARLiNGAM | 504 | 203 | 0.0661 | 31.1 | 641.1 | yes |

DYNOTEARS density is stable across windows; VARLiNGAM's collapses five-fold from 189 to 504 days, which is the most plausible reason its `semcov-hrp` at 189 days (0.340) is the worst cell in the matrix.

**Directional prior** (`results/directional_prior_verification.csv`, DYNOTEARS, 252 days, six windows ending 2008-10, 2011-09, 2014-06, 2018-12, 2020-03, 2022-06): removing the tabu mask puts 187 to 250 edges and 36 to 48% of total edge mass on asset-to-driver links, and shifts the protected driver-to-asset block by 52 to 72% in L1. Top-17 driver Jaccard between masked and unmasked fits ranges 0.62 to 0.89. The prior binds.

**NTS-NOTEARS probe** (`results/j5_nts_probe.csv`, 25 assets, 504 days, four windows): the asset-to-driver block is exactly zero in every window; top-10 driver Jaccard against DYNOTEARS is 0.25, 0.25, 0.25, 0.67 (mean 0.35); Spearman on Stage A scores 0.52, -0.07, 0.49, -0.06; wall clock about 214 s per fit versus about 22 s for DYNOTEARS. Extrapolated to 215 rebalances at the full 132 variables this is days per configuration, so it stays in future work.

**Determinism.** DYNOTEARS and VARLiNGAM fits are bit-identical across processes and machines, and the replication gate confirms DYNOTEARS D0 at 252 days equals Phase I V0' to 16 digits. The Phase I FFNN on Apple MPS is not reproducible at a fixed seed, hence the seed audit.
- **Distance-projection correction (2026-09-05).** Every HRP/HERC allocator, including the baseline, had inherited a nearest-PSD projection of its clustering distance matrix from the Phase I code. A Euclidean distance matrix has one positive eigenvalue, so the projection collapsed it to rank one and single linkage degenerated into a chain ordered by one scalar per asset. Removing it raised `correlation-hrp` by +0.020 to +0.046 and moved `skeleton-hrp` and `semcov-hrp` by at most 0.005, which turned an apparent gain from the graph into the null result now reported. The old behaviour is kept behind `psd_project_distance=True` for replay.

---

## 9. Implications

### 9.1 Answers to the research question

1. **Does the causal graph improve on the correlation matrix?** No. The total effect of a DYNOTEARS graph is -0.017 to +0.013 Sharpe across windows, never significant (p ≥ 0.36), SPA never rejects, the MCS cannot separate anything.
2. **Skeleton versus orientations.** The skeleton costs -0.003 to -0.028 at every window; the orientations recover +0.009 to +0.021 at every window; they roughly cancel. Both are below the minimum detectable effect, which bounds the orientation gain at about 0.04 Sharpe. The same signs appear under HERC.
3. **Is the skeleton effect graph-specific?** No. A density-matched partial-correlation network built without discovery reproduces it. The discovered graph adds nothing at the clustering step beyond its sparsity pattern.
4. **Is the orientation effect due to edge direction?** No. A de-factored covariance with no graph reproduces it at every window; Ledoit-Wolf shrinkage does not. The gain comes from Σ_SEM's assumption of independent shocks, which removes the common market factor. Reliable magnitudes are still required (ridge-Granger and VARLiNGAM graphs do not produce the gain), but the direction itself is not what helps.

### 9.2 What this says about causal networks in allocation

- The findings agree with Howard, Lohre and Mudde: causal networks do not consistently beat correlation. They also agree with their observation that the contemporaneous matrix carries the structure, since the orientation effect appears only for DYNOTEARS' contemporaneous block.
- The finding connects to the network-portfolio literature (Pozzi et al., Peralta and Zareei): under-weighting central nodes helps because centrality is market exposure. The de-factored control shows the orientation-aware allocators are exploiting exactly that mechanism, via the SEM's independence assumption.
- A DAG is not a free upgrade over correlation for hierarchical allocation. The parts of it that help are reproducible with cheaper, graph-free estimators.

### 9.3 Methodological implications

- Decomposition plus mechanism controls is the right way to test a "causal" allocator. Without the controls, `semcov-hrp`'s positive orientation contrast would have looked like evidence for edge direction. With them, it is market-factor removal.
- Backtest rigour changes conclusions. The pre-correction pipeline showed a graph gain that vanished once the baseline's dendrogram was fixed, and effects that look consistent across windows are still well inside the noise once the search over 146 configurations is accounted for.
- "Causal-washing": labelling a method causal invites trust it may not earn. Here the causal graph's skeleton is a worse clustering input than correlation, and its orientations only help through a mechanism that needs no graph.

### 9.4 For practitioners

For a long-only large-cap equity HRP portfolio, replacing the correlation matrix with a DYNOTEARS or VARLiNGAM graph is not worth the cost. If the goal is the market-factor-removal effect, regress returns on the equal-weight market proxy and use the residual covariance in bisection; it matches the SEM allocator at every window and requires no discovery step.

---

## 10. Limitations, threats to validity and future work

### Limitations stated in the report

1. Effects of at most 0.03 Sharpe against minimum detectable effects of about 0.037 to 0.047; nothing survives correction for the search.
2. The orientation effect is DYNOTEARS-specific and, per the de-factored control, not attributable to direction anyway.
3. Long-only US large caps only.
4. Survivorship bias from a universe selected in 2018 to 2020 and held fixed from 2007 (absolute levels only).
5. The SEM is built from the asset block, so driver effects sit in the residuals and the independent-shock assumption is violated.
6. Three post hoc design changes: the research question came from Phase I, the windows went from 2 to 4 after seeing a temporal pattern, and the SPA family was narrowed from 7 to 5 after seeing results. All are disclosed and included in the 146-trial correction; the full 7-allocator family remains the primary SPA test, and the two dropped allocators underperform the baseline in 12 of 16 configurations, so they were not dropped for their results.
7. `undirected-hrp` and `skeleton-hrp` disagree at the two longer windows, adding estimator uncertainty there.
8. Edge persistence between rebalances was not measured.
9. R²-sortability (Reisach et al. 2023) was not tested: DYNOTEARS may orient edges from low- to high-R² assets, which for equities is market beta.
10. λ_W = λ_A = 0.05 was never swept.

### Future work

- Repeat the ablation with a different discovery family (for example NTS-NOTEARS) to test method dependence.
- Equalise risk contributions across structural shocks (a long-short generalisation of `struct-rp`).
- Broaden the asset universe and classes.
- Use T for orientation-aware risk attribution and validate edge directions against firm-specific news shocks.
- Quantify survivorship bias with a point-in-time S&P 500 panel including delisting returns.
- Deferred internal items: λ sweep, point-in-time survivorship check, HERC cluster-cut choice, R²-sortability measurement, sparsity sweep on `semcov-hrp`.

---

## 11. Project history

125 commits from 2026-05-20 to 2026-09-07, all on a single laptop.

| Period | What happened |
| --- | --- |
| 2026-05-20 to 05-31 | Phase I pipeline: joint data layer (universe, prices, drivers, alignment), factor selection (prune, greedy, K calibration), PyTorch FFNN sensitivities, HRP/HSP/V0-V2 allocators, closed-loop feedback, backtest, metrics, shakedown harness. Full 2007 to 2024 Phase I run: causal selection robust, closed loop inert. |
| 2026-06-12 to 06-17 | Ablation matrix completed (VARLiNGAM discovery, V0' asset-only), content-keyed discovery cache, J4a K sweep and J4b α/γ grid, J5 NTS-NOTEARS probe, EEM determinism fix, interim report, first final-report draft. |
| 2026-06-25 | Distribution- and multiplicity-aware Sharpe battery (PSR, DSR, RC, SPA, MCS); report reoriented around the precise question. |
| 2026-07-08 to 07-19 | Phase II: direction-aware allocation from asset-asset graphs (D0 to D4), CORR-HRP control, decomposition contrasts, seed audit, figures; 189 and 378-day windows added; ridge-Granger reframed as a magnitude control. |
| 2026-08-01 to 08-16 | Allocator family restricted to the 2×2 crossing and renamed; direction-free covariance controls, partial-correlation control, HERC, pre-committed 2025 to 2026 out-of-sample slice, 10k-resample battery with pooled SPA. |
| 2026-08-22 to 09-03 | Report copy-editing, supervisor review items, no-HSP variant, reproduction README, Pangram tooling. |
| 2026-09-05 to 09-07 | Distance-projection bug found and fixed; every HRP/HERC cell, battery and figure rerun; findings rewritten to the null result; reproduction harness; repository pruned to the modules the pipeline imports; final report. |
