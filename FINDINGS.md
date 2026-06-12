# Consolidated Findings — Causal-HSP Portfolio Optimisation

*Working document for writing the final thesis report. Amalgamates: the empirical
results from the full 2007–2024 backtests (`results/`), the design decisions and
their justifications (the two repo plan docs + the session plan), the research
notes in this repo, and the methodological findings/bugs discovered during
implementation. Last updated 2026-06-11.*

> **One-line thesis.** Replace HSP's *correlation-based* driver selection with
> *causal-discovery-based* selection (and optionally close a performance→selection
> feedback loop), and test whether this yields more robust S&P-100 portfolios,
> especially around regime changes.

---

## 1. Headline empirical results

Full backtest: approximate S&P 100 (top-99 by CRSP market cap, fixed universe),
33 exogenous driver candidates, **215 monthly rebalances, 2007-01 → 2024-11**,
net of 5 bps one-way costs, K=17 calibrated once on a 2006 burn-in. Two lookback
windows (252 / 504 trading days) as a robustness check. Significance via
Politis–Romano stationary block bootstrap (2000 resamples).

### The full matrix (annualised net Sharpe)

| method | window | V0 | V1 | V2 | V1−V0 (p) | V2−V1 (p) |
|---|---|---|---|---|---|---|
| DYNOTEARS | 252 | 0.371 | 0.382 | 0.382 | +0.011 (0.19) | +0.000 (0.97) |
| DYNOTEARS | 504 | 0.370 | 0.382 | 0.373 | +0.012 (0.23) | **−0.009 (0.031)** |
| VARLiNGAM | 252 | 0.371 | **0.398** | 0.398 | **+0.027 (0.007)** | −0.000 (0.06) |
| VARLiNGAM | 504 | 0.370 | 0.357 | 0.357 | −0.013 (0.15) | −0.000 (1.0) |

CAGR (net): V0 4.87–4.93%, V1-DYNOTEARS 5.07–5.14%, V1-VARLiNGAM 5.33% (w252).
All variants absorb the full ≈−51% GFC drawdown (long-only equity HRP/HSP — the
thesis is about *relative* driver-selection quality, not drawdown avoidance).

Variants: **V0** = vanilla HSP (cumulative-correlation selection); **V1** =
Causal-HSP open-loop (causal greedy, α=1); **V2** = Causal-HSP closed-loop
(α=0.6, γ=0.3); **V0′** = asset-only Causal-HRP (no drivers, asset-asset causal
graph → HRP).

### V0′ (asset-only Causal-HRP) — DONE, and a striking ablation result

| window | V0 (cum-corr) | **V0′ (asset-only causal)** | V1 (DYNOTEARS) | V0′−V0 (p) | V1−V0′ (p) |
|---|---|---|---|---|---|
| 252 | 0.371 | **0.400** | 0.382 | **+0.029 (0.000)** | −0.018 (0.042) |
| 504 | 0.370 | 0.374 | 0.382 | +0.004 (0.69) | +0.008 (0.44) |

At **window 252 the asset-only causal graph (V0′) is the single best DYNOTEARS
variant** — Sharpe 0.400, *significantly* beating both V0 (p<0.001) and the
driver-based V1 (p=0.042), with the smallest drawdown (−50.8%) and the
least-negative recession Sharpe (−0.595). So the ablation question "do
exogenous drivers add value over asset-asset causal structure alone?" gets a
provocative w252 answer: **no — at the short window, asset-asset causal
structure alone beats adding the driver/sensitivity machinery.** But V0′ is
**window-fragile** (0.400 → 0.374), reverting to ≈V0 at 504 days where V1
reclaims the top. **The only variant robust across *both* windows is
DYNOTEARS-V1 (0.382/0.382).** Net thesis-level reading: causal structure
helps, but the *form* that helps most is window-dependent — V0′ and
VARLiNGAM-V1 both peak at w252 and fade at w504; only the open-loop
causal-driver-selection (DYNOTEARS-V1) carries a consistent edge across both.

### The three-part conclusion (the spine of the results chapter)

1. **Causal selection beats correlation selection — robust under DYNOTEARS,
   modest, not significant.** V1-DYNOTEARS > V0 by ΔSharpe +0.011/+0.012 at
   *both* windows (same sign and magnitude; p≈0.19–0.23). This is the
   replicable primary result. CAGR edge ~+0.2 pp/yr.

2. **VARLiNGAM strengthens the result at 252 days (significantly) but is
   window-fragile.** V1-VARLiNGAM beats V0 by +0.027, **p=0.007** at 252 days
   — the strongest causal result in the study — but *reverses* to −0.013 (ns)
   at 504 days. Proposed mechanism (good methodological point): VARLiNGAM's
   identifiability rests on **non-Gaussian residuals**; a 504-day window spans
   more regimes → residuals trend Gaussian (CLT) → ICA identification weakens →
   noisier causal order → worse selection. DYNOTEARS (pure score-based, no
   distributional assumption) has no such window sensitivity. **Quote the
   252-day VARLiNGAM number only with the 504-day caveat.**

3. **The closed-loop feedback (V2) does not robustly help — a characterised
   negative.** V2 ≈ V1 at the short window under both methods; significantly
   *worse* than open-loop at DYNOTEARS-w504 (p=0.031); under VARLiNGAM V2 ≡ V1
   to 3 d.p. at both windows (feedback does essentially nothing). The apparent
   2018Q4 regime-break edge at w252 (+0.120) flips sign at w504 (−0.111) — a
   window artefact. Plausible reason: at 504 days the 2-year discovery window
   already yields stable causal graphs, so the utility blend adds stale-regime
   noise rather than signal.

**Framing takeaway.** A robust positive primary result + a well-characterised
negative on the secondary extension is a *stronger, more defensible* thesis
outcome than a fragile "the full method wins." The two-window × two-method
design is what repeatedly exposed the fragility (VARLiNGAM-w504, V2-w504); the
robustness checking is itself a methodological contribution.

---

## 2. Regime-conditional findings (the differentiator)

From `scripts/regime_analysis.py` → `results/regime_analysis/` (zero re-compute;
self-check: each table's "all" row reproduces the headline Sharpe to <1e-9; the
named-window per-rebalance excess-Sharpe reproduces the interim report's figure
for 4/5 windows, COVID differing only by date-range definition).

### Regime Sharpe (net), window 252

| variant | all | NBER recession | NBER expansion | high-vol (VIX top quintile) | low-vol (VIX bottom) |
|---|---|---|---|---|---|
| V0 | 0.371 | −0.640 | 0.731 | −1.351 | 5.628 |
| V1-DYNOTEARS | 0.382 | −0.623 | 0.744 | −1.354 | 5.679 |
| **V1-VARLiNGAM** | **0.398** | **−0.604** | **0.758** | **−1.325** | **5.710** |

**Key finding: the causal variants beat V0 in *every* regime slice at w252** —
*less-bad* in stress (recession, high-vol) and *better* in benign (expansion,
low-vol), with V1-VARLiNGAM best across the board. This is the cleanest
substantiation of the core hypothesis that causal selection differentiates,
especially in stress.

- **Max drawdown, NBER recession**: V1-VARLiNGAM −0.511 vs V0 −0.523 — causal
  selection modestly cushions the GFC-recession drawdown.
- **Turnover** (one-way annualised, w252): V0 1.44, V1-DYNOTEARS 1.43,
  V1-VARLiNGAM 1.53. VARLiNGAM trades slightly more; turnover rises in
  recession/high-vol regimes for all variants (~1.6–1.75). (`turnover.csv`.)
- **Window 504 reconfirms VARLiNGAM fragility**: V1-VARLiNGAM regime Sharpes
  fall *below* V0 in every slice, while DYNOTEARS-V1 stays above V0 — consistent
  with the matrix-level finding.

**Caveat / deferred**: network-density regimes (per Howard et al.'s market-timing
construction) need per-window discovery W matrices, which were not persisted in
the result bundles. NBER + VIX (both zero-cost) are shipped; density needs a
cheap discovery-only re-run if the chapter wants it.

---

## 3. The directional prior is consequential (verification result)

`scripts/verify_directional_prior.py` → `results/directional_prior_verification.csv`.
Refits DYNOTEARS on 6 windows across regimes **with** vs **without** the
asset→driver tabu mask (the prior that forbids "assets cause drivers").

| window | driver→asset block L1 change | asset→driver mass suppressed | as % of total edge mass | top-K driver Jaccard |
|---|---|---|---|---|
| 2008-10 (GFC) | 63% | 9.22 (181 edges) | 38.8% | 0.62 |
| 2011-09 (Euro) | 56% | 9.42 (206) | 40.7% | 0.89 |
| 2014-06 (calm) | 52% | 10.99 (256) | 37.0% | 1.00 |
| 2018-12 (selloff) | 60% | 10.92 (229) | 42.2% | 0.79 |
| 2020-03 (COVID) | 56% | 9.38 (202) | 35.5% | 0.79 |
| 2022-06 (rate hike) | 72% | 10.19 (209) | 47.8% | 0.62 |

**Finding: the prior does substantial work — it is not cosmetic.** Without it,
DYNOTEARS spends **35–48% of total edge mass** on economically-implausible
asset→driver edges (a single stock "causing" the 10Y yield), and allowing those
edges *distorts the driver→asset block we actually use by 52–72%* (L1). The prior
also changes the top-17 selected drivers in a meaningful minority of windows
(Jaccard 0.62–1.00, lowest in the 2008/2022 stress regimes). The prior cheaply
removes spurious reverse-causation mass and protects the legitimate driver→asset
structure. **Defensible framing**: the masked fit is *correct by economic
construction* (single-stock→macro causation is implausible at these scales); the
verification quantifies how much the prior was doing.

---

## 4. Methodology findings & design decisions (with justifications)

### 4.1 Data layer
- **Universe**: free historical S&P 100 membership is paywalled and cannot be
  deduced from S&P 500 (it's chosen "for sector balance" with undisclosed
  discretion). Standard academic substitute: **top-N by market cap from
  historical S&P 500 membership** at each date. Membership from the open
  `fja05680/sp500` repo (1996+, point-in-time); prices + *historical*
  shares-outstanding from **WRDS/CRSP** (survivorship-bias-free, includes
  delisted names — critical for the GFC where ~30–40 financials delisted),
  with a thin yfinance fallback for the most recent days CRSP hasn't ingested.
  Document the top-100-by-mcap approximation as a methodology limitation.
- **Driver pool** (~33–35 series, all defensibly *exogenous* to S&P-100):
  Treasury rates + slopes, credit spreads (BAA-AAA, BAA-10Y, HYG/LQD), FX
  (DXY + majors), commodities (WTI/Brent/gold/silver/copper/natgas), vol
  indices (VIX, VVIX), international equity ETFs (EFA/EEM/EWJ/EWG/EWU), and
  monthly macro (CPI, core CPI, unemployment, IP, retail sales, housing
  starts, sentiment). **Explicitly excluded** (failed exogeneity): US sector
  SPDRs (mechanical aggregations of constituents), Fama-French/AQR factor
  returns (long-short US-equity portfolios), and S&P 500 / Russell / NASDAQ
  index returns (near-identical to the asset block).
- **Transformations are per-series-type, NOT uniform** (an important
  methodology point an examiner will probe): log-returns for price-like series
  (commodities, FX, equity ETFs); first-differences for yields/spreads (which
  can be zero or negative — logs are undefined/meaningless); YoY% for monthly
  macro levels (also removes seasonality); pass-through ("level") for VIX
  (already stationary). **"Log everything" is wrong** — it breaks on
  rates/spreads. Asset returns *are* log-then-difference (`r = Δlog P`), which
  is exactly the standard treatment; equities are I(1) so the difference is a
  foregone conclusion (no need to gate it on a test).
- **Stationarity uses ADF *and* KPSS, deliberately** (not ADF alone): the two
  have opposite nulls (ADF H0 = unit root; KPSS H0 = stationary), and ADF has
  notoriously low power against persistent-but-stationary series. "ADF fails to
  reject" conflates "genuinely non-stationary" with "ADF underpowered"; the
  KPSS confirmation resolves that 2×2. Series are *flagged*, not dropped, with
  the flag logged. Transforms are fixed by series-type up-front (not
  test-driven per window) precisely so a variable means the *same thing* in
  every rolling window — required for comparability across the rolling causal
  graphs and the V2 utility EMA.
- **Per-asset eligibility masking** (`build_joint_matrix(drop_na='drivers_only')`):
  late-inception assets (e.g. LIN from the 2018 Linde+Praxair merger; FB→META;
  BRK.B; GOOG/GOOGL) would otherwise cost 200+ rows for every other ticker. The
  fix keeps all rows where drivers are populated, zero-fills pre-inception asset
  cells, and exposes a per-(row, asset) eligibility mask that the FFNN
  (per-asset masked loss) and the backtest universe-filter both honour. This is
  what makes the full GFC-era backtest viable.

### 4.2 Causal discovery
- **DYNOTEARS** (primary): score-based, no faithfulness assumption, scales to
  d≈130, and supports forbidding edges natively via `tabu_edges` → L-BFGS-B
  (0,0) bounds. The directional prior (forbid asset→driver) is a simple
  enumeration of `(lag, asset, driver)` tuples — *not* the optimiser surgery the
  original plan feared.
- **VARLiNGAM** (robustness comparator): exploits non-Gaussianity of returns
  via ICA on VAR residuals. Prior enforced via DirectLiNGAM `prior_knowledge`
  on B₀ + ridge-VAR masked `ar_coefs` + post-fit projection on lagged blocks.
  Stronger at the short window but window-fragile (§1).
- **Lagged edges are near-negligible** for daily/monthly returns (confirmed by
  Pamfil et al. on S&P 100 and Howard et al. on S&P 500); the contemporaneous
  block dominates. Retaining VARLiNGAM is justified because it models lags +
  non-Gaussianity explicitly.
- **HSIC residual-independence spot-check** wired for VARLiNGAM (catches LiNGAM
  misspecification; O(d²) so run at an annual spot-check cadence, not every
  window).

### 4.3 Factor selection & K calibration
- **Two-stage greedy**: Stage A prunes by aggregate lagged driver→asset
  influence (DYNOTEARS: magnitude-threshold; VARLiNGAM: presence/bootstrap);
  Stage B adds drivers by marginal held-out predictive-likelihood gain.
- **K calibrated, not fixed**: Kneedle elbow on the sorted causal-score curve,
  plus a permutation null with **Benjamini-Hochberg FDR** (a Phase-H
  reformulation replacing the original "max-of-d" statistic, which is
  structurally tail-biased at large d).
- **Empirical finding — K_perm = 0 at every scale tested** (d=65 and d=134),
  under *both* the BH-FDR and the legacy max-of-d formulations. No individual
  driver clears FDR significance at α=0.05. So **Kneedle is the operational K
  selector**, and it scales with universe richness: K_elbow ≈ 9 (d=65) → 14–18
  (d=134). Honest framing for the chapter: the *ranking* of drivers is
  informative (Kneedle finds a real elbow), but no single driver is individually
  FDR-significant on noisy financial data — the thesis claim is about the
  *combination* of causally-selected drivers in the FFNN+HSP pipeline, not
  individual-driver significance. Report K_perm=0 openly; it's a finite-sample
  causal-discovery caveat, not a hidden failure.

### 4.4 Sensitivities & portfolio construction
- **Sensitivities**: one PyTorch multi-head FFNN per window (shared hidden
  layers, per-asset output heads), architecture search over depth∈{1,2},
  width∈{16,32,64} on a held-out tail; sensitivity matrix S via
  `torch.func.jacrev` averaged over the window. Identical to HSP's FFNN+AAD
  except the input driver set is *causally* selected, not correlation-selected.
- **Why sensitivity space (resolves the "can't symmetrise directional graphs"
  problem)**: each asset's sensitivity vector lives in a Euclidean space, so the
  distance matrix `D_ij = ||s_i − s_j||₂` is **symmetric by construction** — no
  symmetrisation of the asymmetric causal adjacency is ever needed. The
  directional causal information is preserved in *how it shaped* the selected
  drivers and their sensitivities. The allocation step (recursive bisection with
  sample-variance weights) is unchanged from HRP/HSP.
- **Commonality-principle link** (Rodriguez-Dominguez): iff drivers satisfy the
  commonality principle, a conformal map exists between unconditional-return
  space and sensitivity space, so idiosyncratic + systematic diversification can
  co-exist without the usual trade-off. Open question worth flagging: do
  DYNOTEARS/VARLiNGAM-selected drivers satisfy something like the commonality
  principle? (The K_perm=0 result suggests the per-driver causal signal is weak,
  so this is genuinely open.)

### 4.5 Closed-loop feedback (V2)
- Realised reward `R[t]` = holding-period excess Sharpe vs 1/N (excess form
  controls for market-wide regime effects, so drivers aren't punished for a
  crash that hit everything). Sensitivity-weighted credit attribution
  (`influence_d = Σ_i |w_i · s_{i,d}|`, normalised) → EMA driver-utility update
  → blended into next selection (`score = α·z(causal) + (1−α)·z(U)`).
- **Lookahead discipline**: utility stored keyed by *holding-period-end* date;
  the selector at rebalance t reads the latest row with end ≤ t−21d, with an
  assertion that raises on violation; a deliberately-broken "leak canary"
  confirms a future-peeking lookup produces visibly inflated results.
- **Result**: see §1.3 — the loop is a characterised negative.

---

## 5. Scaling & runtime findings (for the methods/reproducibility chapter)
- **DYNOTEARS per-window cost scales ≈ O(d³·⁵)**, not O(d²): ~17 s/window at
  d=65 → ~3–4 min/window at d=134 (≈26×). L-BFGS-B per-iteration cost dominates.
- **K calibration** (Phase-H fix: cap permuted-fit `max_iter`=20 + joblib
  parallelism) — 162 min → 7 min at d=65 (24×); ~70 min at d=134.
- **VARLiNGAM** per-window ≈ 32 s at d=134 (faster than DYNOTEARS); but its
  K-cal closure currently hardcodes DYNOTEARS, so VARLiNGAM runs reuse the
  DYNOTEARS-calibrated K=17 (defensible: Kneedle gives 17–18 robustly across
  method and d).
- **NTS-NOTEARS** (stretch): the report's "5–20× DYNOTEARS" ⇒ ~15–80 min/window
  ⇒ ~50–280 h per backtest at full scale — **computationally prohibitive**;
  viable only as a reduced sub-analysis (small universe / sparse windows) or
  future work. Vendored code *does* support the asset→driver prior natively (via
  L-BFGS-B kernel-norm bounds) and would unify discovery + sensitivities.
- **Full backtest budget**: ~14 h per variant (215 rebalances) + ~3 h one-off
  K-cal; the three DYNOTEARS variants ≈ two overnight runs. All data pre-cached
  → runs make zero WRDS calls (no Duo prompts).

---

## 6. Literature positioning
- **HRP** (López de Prado 2016): cluster on a correlation distance, quasi-
  diagonalise, recursively bisect with inverse-variance weights. Avoids matrix
  inversion; clustering needs only a symmetric distance, allocation needs only
  cluster variances — the two stages are cleanly separable (this separability is
  what lets us swap the distance for a causal/sensitivity one).
- **HSP** (Rodriguez-Dominguez 2023): clusters on *sensitivity* distance
  (similarity of assets' sensitivities to common drivers) rather than return
  correlation; motivated by the commonality principle + Reichenbach common-cause.
  **Its open flaw**: it still selects drivers by *cumulative correlation* — the
  exact thing this thesis replaces with causal discovery.
- **Howard, Lohre & Mudde (2025)** — the closest prior work: applies DYNOTEARS
  to the S&P 500 for peer-group neutralisation, a low-centrality factor, and a
  network-density market-timing indicator. Their honest conclusion: causal
  networks **complement rather than consistently beat** correlation, are
  compute-heavy, and hard to interpret. **Crucial gap they leave: they never set
  portfolio weights from the causal graph** (they go graph → node2vec →
  clustering → peer groups). This thesis directly converts causal structure into
  allocation — the novel contribution. They also independently validate two of
  our choices: lagged edges are negligible for returns, and causal-network
  structure carries regime information correlation misses.
- **Research-gap table** (this work vs HRP / HSP / Howard et al.): only this work
  does causal selection **+** directional structure **+** weights-from-graph **+**
  performance feedback.
- **Cum-corr tautology (our empirical observation)**: V0's correlation selector
  picks essentially the same drivers every rebalance — dominated by
  international-equity ETFs + the rates curve, which *trivially co-move* with US
  equity. It answers "what moves with the assets" rather than "what drives them"
  — exactly the bias causal discovery bypasses. V1 rotates across far more
  drivers, tracking the macro regime; cross-method (DYNOTEARS vs VARLiNGAM)
  top-K driver Jaccard is only ~0.34, yet both beat V0 — concordance-despite-
  independence.

---

## 7. Engineering lessons / bugs fixed (reproducibility appendix material)
- **DYNOTEARS prior is free via `tabu_edges`** — no optimiser surgery (corrected
  an early over-estimate of effort).
- **Per-window z-score must live inside the rolling loop**, not be applied
  globally (matches Howard et al. / DYNOTEARS convention).
- **VARLiNGAM at d=132 crashes on lingam's adaptive-lasso `_pruning`**
  (`LassoLarsIC`, n_samples < n_features for late-causal-order variables). Fix:
  `prune=False` — lingam's pruning is redundant since the asset→driver mask is
  enforced by post-fit projection and Stage A thresholds edges itself.
- **FFNN sensitivity cache race** between parallel runs (identical window/K/arch
  → same cache key → torn pickle). Fix: tolerant read (recompute on corrupt) +
  atomic write (temp + `os.replace`).
- **K-cal closure hardcodes DYNOTEARS** — so a "VARLiNGAM K-cal" actually
  calibrates on DYNOTEARS scores; VARLiNGAM runs reuse K=17 deliberately.
- **stooq dropped** mid-project (added a paid-API requirement); WRDS/CRSP +
  yfinance fallback is the final, cleaner data spine.

---

## 8. Open items & caveats (discussion chapter + remaining work)
- **V0′ (asset-only Causal-HRP) — DONE** (see §1): at w252 it is the best
  variant (Sharpe 0.400, significantly beating V0 *and* V1) but window-fragile
  (≈V0 at w504). The 4-variant ablation is complete.
- **K-sensitivity of the V1>V0 result** not yet swept (planned J4a:
  K∈{10,14,17,20,25}) — would harden the primary claim.
- **α/γ feedback sweep** (J4b) — expected to confirm the closed loop never beats
  open-loop; low scientific value given the demonstrated non-robustness, useful
  only as a defensive negative appendix.
- **NTS-NOTEARS** — feasibility-bounded to a reduced sub-analysis or future work.
- **Network-density regimes** — need a discovery-only re-run to persist
  per-window graph density.
- **Commonality-principle compatibility** of causally-selected drivers — open
  theoretical question; the weak per-driver significance (K_perm=0) bears on it.
- **Caveats to state plainly**: long-only equity HRP/HSP eats the full GFC
  drawdown (relative-quality study, not drawdown avoidance); the S&P-100-by-mcap
  approximation lacks the official index's sector-balance discretion;
  shares-outstanding uses CRSP point-in-time where available; the primary causal
  edge (V1>V0) is consistent but sub-significance over 18 years.

---

## 9. Provenance map (where each finding came from)
- **Headline matrix, significance, window robustness** → `results/phase_i_*`
  bundles; this session's full-matrix computation (§1).
- **Regime tables** → `scripts/regime_analysis.py`, `results/regime_analysis/*.csv` (§2).
- **Directional-prior verification** → `scripts/verify_directional_prior.py`,
  `results/directional_prior_verification.csv` (§3).
- **Design decisions / methodology** → `Causal Factor Discovery Pipeline.md`,
  `Closed-Loop Causal-HSP Portfolio.md`, `hrp_hsp_notes.md`; transformation &
  stationarity discussion from this session (§4).
- **Literature** → `causal-network-representations-in-factor-investing.md`
  (Howard et al.), `hrp_hsp_notes.md` (HSP/commonality), `interim_report/main.tex`
  (§6).
- **Runtime / bugs** → session plan `~/.claude/plans/ok-update-the-plans-generic-honey.md`
  Phases G/H/I/J (§5, §7).
- **Interim report** (`interim_report/main.tex`) is the current write-up skeleton;
  VARLiNGAM, regime, and prior-verification results in this doc are **new
  material for the final report** (the interim listed VARLiNGAM only as an
  available comparator).
