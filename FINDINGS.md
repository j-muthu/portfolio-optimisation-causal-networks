# Consolidated Findings — Causal-HSP Portfolio Optimisation

*Working document for writing the final thesis report. Amalgamates: the empirical
results from the full 2007–2024 backtests (`results/`), the design decisions and
their justifications (the two repo plan docs + the session plan), the research
notes in this repo, and the methodological findings/bugs discovered during
implementation. Last updated 2026-06-15.*

> **One-line thesis.** Replace HSP's *correlation-based* driver selection with
> *causal-discovery-based* selection (and optionally close a performance→selection
> feedback loop), and test whether this yields more robust S&P-100 portfolios,
> especially around regime changes.

### Figures (final report) — `scripts/plot_thesis_figures.py` → `results/figures/`

All regenerated from the frozen-EEM bundles + result CSVs (reproducible; match
the tables here). The submitted `interim_report/figures/` are left as-is.

| figure | supports |
|---|---|
| `nav_curves.png` | §1 — cumulative net NAV, V0/V0′/V1-DYNO/V1-VAR, both windows |
| `sharpe_matrix.png` | §1 — net Sharpe by variant × window (w252 edge, w504 convergence) |
| `k_sensitivity.png` | §1b J4a — V1>V0 edge is K-fragile |
| `feedback_grid.png` | §1b J4b — closed loop inert (V2≡V1 across the α/γ grid) |
| `regime_excess.png` | §2 — causal beats V0 in every regime (V0′ most) |
| `directional_prior.png` | §3 J1 — the asset→driver prior is consequential |
| `nts_probe.png` | §1c J5 — NTS-NOTEARS vs DYNOTEARS agreement + cost |

Report-ready captions for every figure: `results/figures/CAPTIONS.md`. Corrected
(frozen-EEM) versions of the two *interim*-report figures, separate from the
submitted ones: `interim_report/figures_corrected/` (V2 now coincides with V1;
the old 2018Q4 "V2 edge" is gone) — via `scripts/plot_interim_corrected.py`.

---

## 1. Headline empirical results

Full backtest: approximate S&P 100 (top-99 by CRSP market cap, fixed universe),
33 exogenous driver candidates, **215 monthly rebalances, 2007-01 → 2024-11**,
net of 5 bps one-way costs, K=17 calibrated once on a 2006 burn-in. Two lookback
windows (252 / 504 trading days) as a robustness check. Significance via
Politis–Romano stationary block bootstrap (2000 resamples).

### The full matrix (annualised net Sharpe)

**All numbers are the deterministic (frozen-EEM) re-run** (2026-06-15) — the
canonical bundles after the EEM-determinism fix (§1b, §7).

| method | window | V0 | V1 | V2 | V1−V0 (p) | V2−V1 (p) |
|---|---|---|---|---|---|---|
| DYNOTEARS | 252 | 0.371 | 0.381 | 0.381 | +0.010 (0.27) | +0.000 (1.00) |
| DYNOTEARS | 504 | 0.370 | 0.372 | 0.372 | **+0.001 (0.92)** | +0.000 (1.00) |
| VARLiNGAM | 252 | 0.371 | **0.399** | 0.399 | **+0.028 (0.004)** | +0.000 (1.00) |
| VARLiNGAM | 504 | 0.370 | 0.357 | 0.357 | −0.014 (0.14) | +0.000 (1.00) |

**Corrections vs the original (jittery-EEM) commit** — all consequences of the
determinism fix (§1b): (a) DYNOTEARS **w504 V1−V0 collapsed +0.012 → +0.001**, so
"robust across *both* windows" is really **w252-only**; (b) the headline "V2
*significantly worse* at DYNOTEARS-w504 (p=0.031)" **disappears** — **V2 ≡ V1
exactly at every cell** (the closed loop is inert, not harmful — J4b); (c)
**VARLiNGAM is robust to the fix** — its w252 result is essentially unchanged
(+0.028, p=0.004, still the strongest causal result) and still reverses at w504
(−0.014, ns). So VARLiNGAM's ICA-on-residuals discovery is far less sensitive to
the EEM micro-jitter than DYNOTEARS's non-convex L-BFGS — itself a robustness
observation worth a line in the methods chapter.

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
| 252 | 0.371 | **0.403** | 0.381 | **+0.032 (0.00)** | **−0.022 (0.01)** |
| 504 | 0.370 | 0.372 | 0.372 | +0.002 (0.84) | −0.001 (0.94) |

(Frozen-EEM re-run, 2026-06-15.) At **window 252 the asset-only causal graph
(V0′) is the single best variant** — Sharpe 0.403, *significantly* beating both
V0 (+0.032, p<0.001) and the driver-based V1 (V0′ over V1 by 0.022, **p=0.01**),
with the smallest drawdown (−50.5%). So the ablation question "do exogenous
drivers add value over asset-asset causal structure alone?" gets a provocative
w252 answer: **no — at the short window, asset-asset causal structure alone
beats adding the driver/sensitivity machinery.** But the edge is
**window-specific**: at 504 days V0′, V1 and V0 all converge to ≈0.372 (every
DYNOTEARS edge vanishes). Net thesis-level reading: **causal structure helps at
the 1-year window — most via V0′ — and essentially not at the 2-year window.**
The earlier "DYNOTEARS-V1 robust across both windows" reading does **not**
survive the determinism fix (w504 V1−V0 is +0.001); the honest claim is a
**w252 effect**, strongest for V0′.

### The three-part conclusion (the spine of the results chapter)

1. **Causal selection beats correlation selection — at the 1-year window only,
   and not robust to K** *(revised under the frozen-EEM re-run)*. V1-DYNOTEARS >
   V0 by +0.010 at w252 (p=0.27) but only **+0.001 at w504** (p=0.92) — the
   originally-claimed w504 edge (+0.012) was an EEM artefact. The J4a K-sweep
   further shows the w252 edge is **K-fragile** (positive at the Kneedle K=17,
   sign-flipping at other K; §1b). Honest primary claim: a modest, window- and
   K-specific w252 effect — and the strongest w252 effect is actually **V0′**
   (asset-only), not V1.

2. **VARLiNGAM strengthens the result at 252 days (significantly) but is
   window-fragile** *(confirmed robust to the frozen-EEM re-run: +0.028, p=0.004
   at w252; −0.014, ns at w504 — essentially unchanged, unlike DYNOTEARS-w504)*.
   V1-VARLiNGAM beats V0 by +0.028, **p=0.004** at 252 days
   — the strongest causal result in the study — but *reverses* to −0.013 (ns)
   at 504 days. Proposed mechanism (good methodological point): VARLiNGAM's
   identifiability rests on **non-Gaussian residuals**; a 504-day window spans
   more regimes → residuals trend Gaussian (CLT) → ICA identification weakens →
   noisier causal order → worse selection. DYNOTEARS (pure score-based, no
   distributional assumption) has no such window sensitivity. **Quote the
   252-day VARLiNGAM number only with the 504-day caveat.**

3. **The closed-loop feedback (V2) is inert — a clean characterised negative**
   *(revised)*. Under frozen EEM, **V2 ≡ V1 to 4 d.p. at both windows**
   (DYNOTEARS); the originally-headline "significantly *worse* at w504 (p=0.031)"
   was itself an EEM artefact. The J4b α/γ sweep confirms inertness across the
   *entire* feedback grid (§1b): the utility blend only re-ranks *within* the
   causal-selected set, so it never changes the selected drivers. **The loop
   does nothing, rather than hurting** — a cleaner negative than the original.
   (The earlier 2018Q4 regime-break sign-flip story was a jittery-EEM,
   w252-vs-w504 artefact and is superseded; regime tables in §2 will be
   refreshed once the VARLiNGAM re-run lands.)

**Framing takeaway** *(revised after the frozen-EEM re-run + J4)*. The honest
outcome: causal structure delivers a modest, **w252-localised** improvement over
correlation selection — strongest via the asset-only **V0′** — with **no robust
2-year-window edge**, **K-sensitivity** at the short window, and an **inert**
closed loop. The robustness/reproducibility machinery (two windows × two methods,
the J4a K-sweep, and the EEM-determinism fix) is precisely what exposed each of
these — and is itself the methodological contribution. A well-characterised,
*reproducible* set of modest/null results is more defensible than the original
"robust across both windows" reading, which the determinism fix showed rested
partly on a single non-deterministic driver (EEM).

> **⚠️ Qualified by the J4 sweeps + the EEM-determinism fix (§1b).** The w252
> V1>V0 edge reproduces at the operating K=17 but is **not robust to K**; and a
> discovered data-nondeterminism bug means the **committed w504 V1 edge (+0.012)
> was partly an artefact** — under the fix it falls to ≈+0.001. Read §1 point 1
> together with §1b.

---

## 1b. K-sensitivity (J4a), feedback grid (J4b) & the EEM-determinism fix

> **Bottom line:** the w252 headline (V1>V0 ≈ +0.010 at the operating K=17)
> reproduces, but the K-sweep shows the edge is **not robust to K** (it
> sign-flips at other K), and a discovered data-nondeterminism bug (EEM) means
> the **committed w504 V1 edge was substantially an artefact** — under the fix
> it falls from +0.012 to ≈+0.001. The closed loop is **inert across the whole
> α/γ grid**. All J4 numbers are fresh, fully deterministic (frozen-EEM) and
> self-consistent; the committed headline bundles (§1) are untouched.

Enabled by a content-keyed **discovery cache** (`pipeline/discovery/cache.py`):
the DYNOTEARS/VARLiNGAM graph is K/α/γ-independent, so each window's fit is
computed once (~15h populate per window) and every sweep config reuses it
(~80 s each). 418 cached graphs (215 w252 + 203 w504), ≈700× fan-out speedup.

### J4a — K-sensitivity of V1 vs V0 (net Sharpe, both windows)

| window | K | V0 | V1 | V1−V0 (p) |
|---|---|---|---|---|
| 252 | 10 | 0.391 | 0.376 | −0.015 (0.13) |
| 252 | 14 | 0.387 | 0.389 | +0.003 (0.77) |
| 252 | **17** | 0.371 | 0.381 | **+0.010 (0.27)** |
| 252 | 20 | 0.400 | 0.384 | −0.016 (0.10) |
| 252 | 25 | 0.393 | 0.383 | −0.010 (0.24) |
| 504 | 10 | 0.370 | 0.370 | −0.001 (0.92) |
| 504 | 14 | 0.371 | 0.372 | +0.000 (1.00) |
| 504 | **17** | 0.370 | 0.372 | +0.001 (0.92) |
| 504 | 20 | 0.365 | 0.372 | +0.007 (0.42) |
| 504 | 25 | 0.383 | 0.372 | −0.012 (0.18) |

**Findings.** (i) At the Kneedle operating point **K=17 the w252 edge is
positive (+0.010)** and matches the committed headline. (ii) But the edge is
**K-fragile** — it goes negative at K=10/20/25 (w252), driven mostly by V0's
cum-corr Sharpe bouncing 0.37–0.40 with K while V1 stays ≈0.38; **none** of the
ΔSharpes is significant (p 0.10–1.00). (iii) V1 w504 Sharpe is **flat at 0.372
for K≥14** because Stage-B greedy pool-exhausts at ≈13 drivers under the 2-year
window, so K≥14 all select the same set. **Honest reading: the causal-vs-
correlation edge holds at the data-driven K but is not robust to arbitrary K —
report the full curve, not just K=17.**

### J4b — α/γ feedback grid (V2 vs V1 open-loop, w252, DYNOTEARS)

All nine (α∈{0.4,0.6,0.8} × γ∈{0.1,0.3,0.5}) combos give V2 Sharpe = **0.381,
identical to V1 to 4 d.p.** (ΔSharpe 0.000, p=1.0 throughout). Verified the
feedback is genuinely active (utility table populated, 215 non-zero rows) yet
**every rebalance's weights are identical to V1** (max Δ = 0.0). Mechanism: the
utility blend can only re-rank *within* the causally-selected set — it cannot
promote a never-selected driver — so when the causal top-K is stable the
selected set never changes and V2 collapses to V1 exactly. This is the
**strongest form of the closed-loop negative**: across the entire
feedback-strength grid the loop is inert, not merely unhelpful (mirrors the
committed "V2≡V1 to 3 d.p. under VARLiNGAM").

### The EEM data-nondeterminism bug (found during J4; reproducibility-critical)

The discovery cache initially never reused — each run re-keyed. Root cause:
`fetch_yahoo_series` re-fetched **EEM** live on *every* call because EEM's
inception (2003-04-14) falls inside the 2-year pre-`start` padding window, so
the cache's coverage check `index.min() <= pad_start` never passed.
`auto_adjust=True` returns values that jitter ≈3e-7 run-to-run, which DYNOTEARS
(non-convex L-BFGS) amplifies to ‖ΔW‖≈0.14 — i.e. **the pipeline was not
bit-reproducible**, and each committed run used a one-off EEM realisation. Fixed
(`pipeline/data/drivers.py`): reuse a Yahoo cache when the previously-requested
*span* covers the request (sidecar `.meta`) + atomic writes; two independent
processes now produce byte-identical windows.

**Reproducibility impact (fresh frozen-EEM K=17 vs committed):** V0 w252/w504 and
V1 w252 reproduce to ≤0.0006, but **V1 w504 shifts 0.382 → 0.372** (V0 w504
unchanged at 0.370). So the committed **w504 V1−V0 edge of +0.012 becomes
≈+0.001** under the deterministic pipeline — the w504 leg of the "robust across
both windows" claim was substantially an EEM-realisation artefact. **The w252
edge (+0.010) is solid; the w504 edge is not.**

**Done (2026-06-15):** the canonical DYNOTEARS bundles (V0/V1/V2/V0′ × both
windows) were regenerated under frozen EEM (~8 min, warm cache) and §1 now
reflects them. This also corrected a *second* artefact: the committed "V2
significantly worse at DYNOTEARS-w504 (p=0.031)" disappears — V2 ≡ V1 exactly at
both windows under frozen EEM. The VARLiNGAM rows were regenerated likewise and
are **robust to the fix** (w252 +0.028 p=0.004 unchanged; w504 still reverses) —
VARLiNGAM's ICA discovery is far less sensitive to the EEM micro-jitter than
DYNOTEARS's L-BFGS. §2 regime tables refreshed from the all-frozen bundles.

Source: `scripts/collate_j4.py` → `results/j4a_k_sensitivity.csv`,
`results/j4b_alpha_gamma.csv`.

---

## 1c. J5 — non-linear-discovery probe (NTS-NOTEARS)

A full NTS-NOTEARS backtest is compute-prohibitive, so J5 is a **reduced-scope
probe**: fit NTS-NOTEARS (per-variable 1D-CNN structure learning) and DYNOTEARS
on a handful of regime windows at a reduced universe (25 assets + 33 drivers,
d=58, 504-day windows) and compare the discovered driver→asset structure.
Wrapper `pipeline/discovery/nts_notears.py` (edge strength = L2-norm of each
CNN kernel → a `(W, A)` pair slotting into the DYNOTEARS interface); probe
`scripts/probe_nts_notears.py` → `results/j5_nts_probe.csv`.

| window | top-10 Jaccard (NTS vs DYNO) | Spearman (Stage-A scores) | NTS asset→driver max | NTS fit | DYNO fit |
|---|---|---|---|---|---|
| 2008-10 GFC | 0.25 | +0.52 | 0.0 | 217 s | 17 s |
| 2014-06 calm | 0.25 | −0.07 | 0.0 | 206 s | 29 s |
| 2020-03 COVID | 0.25 | +0.49 | 0.0 | 216 s | 20 s |
| 2022-06 hike | 0.67 | −0.06 | 0.0 | 217 s | 22 s |

**Findings.**
1. **The directional prior transfers to NTS-NOTEARS** — its native
   `prior_knowledge` bound-dicts drive the asset→driver block to exactly 0 in
   every window (the non-linear analogue of DYNOTEARS `tabu_edges`; validates
   the interim report's claim that NTS supports the prior).
2. **Non-linear discovery agrees only modestly with the linear graph** — mean
   top-10 driver Jaccard 0.35 (coincidentally close to the DYNOTEARS-vs-VARLiNGAM
   0.34), mean Stage-A Spearman 0.22, *higher in stress windows* (GFC/COVID
   ρ≈0.5) than calm/hike (≈0). So NTS-NOTEARS is a genuinely different lens, not
   a re-derivation of DYNOTEARS — a non-linear backtest could plausibly differ,
   which is precisely why it's worth flagging as future work.
3. **Compute confirms infeasibility at scale** — NTS is ~10× DYNOTEARS even at
   d=58 (~214 s vs ~22 s/window); extrapolating, a full 215-rebalance backtest
   is ~13 h/variant *at this reduced d*, and several× that at the thesis d≈130
   (~50-90 h+/variant). **Full NTS-NOTEARS backtest integration is future work**;
   the probe establishes feasibility + the cross-method agreement signal.

(Requires `igraph`, a vendored-NTS dependency, added to `pipeline/requirements.txt`.)

---

## 1d. Phase II — direction-aware allocation (the fixed-graph ablation)

**The question (pre-registered in `PHASE_II_PLAN.md` §0.2):** *holding the
discovered asset–asset graph fixed, does a direction-aware allocator outperform
its own symmetrised counterpart?* V0′ symmetrises the asset–asset block
(`causal_embedding_distance`) before clustering — it discards exactly the
directional information causal discovery paid for. Phase II keeps it, through
five allocators fed the byte-identical per-window `M`: **D1** (embedding
distance + SEM-implied structural covariance `Σ = D_σ B Σ_ε Bᵀ D_σ`,
`B = (I−Mᵀ)⁻¹`), **D2/D2s** (topological order replaces the dendrogram —
recursive bisection over the DAG's causal order, sample/structural cov),
**D3** (no hierarchy: long-only ERC on Σ_struct — structural-shock risk
parity), **D4** (co-ancestry distance from `B̃B̃ᵀ`), plus a second
symmetrisation control **D0s**. All deterministic, seed-free, no drivers, no
FFNN. Same universe/costs/calendar as Phase I; every graph a cache hit
(836/836; zero refits). New code: `pipeline/discovery/asset_graph.py`
(chokepoint), `pipeline/portfolio/{directed,topological}.py`,
`pipeline/discovery/granger.py`, `scripts/run_phase_ii.py` + 50 unit tests.

**Replication gate passed byte-perfectly**: D0 (the V0′ math through the new
harness) reproduces the committed `phase_i_v0prime_w252` bundle with
ΔSharpe = 0.0 and max|Δweight| = 0.0 over all 215 rebalances.

**E1 matrix (net Sharpe; V0 baseline 0.371/0.370):**

| allocator | DYNO w252 | DYNO w504 | VAR w252 | VAR w504 |
|---|---|---|---|---|
| D0 (sym control ≡ V0′) | 0.403 | 0.372 | 0.397 | 0.377 |
| D0s (2nd sym control) | 0.398 | 0.383 | 0.396 | 0.390 |
| **D1 (Σ_struct)** | **0.411** | **0.395** | 0.385 | 0.383 |
| D2 (topo order) | 0.399 | 0.382 | 0.388 | 0.372 |
| **D2s (topo + Σ_struct)** | **0.406** | **0.399** | 0.386 | 0.375 |
| D3 (shock-space ERC) | 0.397 | 0.387 | 0.392 | 0.375 |
| D4 (co-ancestry) | 0.393 | 0.387 | 0.402 | 0.370 |

**The §0.2 answer, stated precisely:** *direction-aware allocation does not
significantly outperform its symmetrised counterpart once the multiplicity of
allocators tried is priced in.* The strongest effect — D1−D0 = **+0.023 at
w504 (pairwise Politis–Romano p = 0.049)**, with D2s−D0 +0.027 (p = 0.067)
and *all five* DYNOTEARS direction-aware contrasts positive at w504 — does
**not survive family-wise correction** (SPA over the 10-variant
direction-aware grid vs D0: p ≈ 0.15 at w504, p ≈ 0.33 at w252). At w252
direction adds nothing over D0 (D1−D0 +0.009, ns), and under VARLiNGAM graphs
the effect is absent everywhere (all contrasts ≈ 0 or negative) — the
direction signal is DYNOTEARS-specific. A characterised, honest near-null:
suggestive at the long window, not significant family-wise.

**What direction *does* buy — cross-window consistency.** The Phase-I story's
sharpest negative was V0′'s window-fragility (0.403 → 0.372). D1 and D2s are
the only variants strong at *both* windows (0.411/0.395 and 0.406/0.399):
keeping direction substantially repairs the w504 collapse of the undirected
skeleton. The repair is cost-invariant (D1−D0 at w504: +0.022→+0.025 across
0→20 bps; D1's turnover 0.102 is *below* D0's 0.112) and τ-robust (E3 sweep:
D2/D3 flat-to-better as the edge threshold rises to 0.1; D3 0.397→0.407).
Additionally, **best(D1) − V1 = +0.030 (p = 0.04) at w252**: the
direction-aware graph route beats the entire Phase-I driver/FFNN machinery
(same multiplicity caveat applies).

**E2 — the GRANGER comparator (Option 2 verdict).** A ridge-VAR(1) directed
graph, density-matched per window to DYNOTEARS: fine for *ordering* (D2 0.393
vs DYNO-D2 0.399) but poor for *structural covariance* (D1 0.352, D3 0.350) —
cheap directed structure suffices for the topology-only allocator but its
coefficient magnitudes cannot support Σ_struct. GRANGER−DYNO at the best
allocator: −0.060 (p = 0.12).

**E4 — the 82-trial battery (Bailey–LdP/White/Hansen, extended).**
`robust_stats --phase-ii` (separate output `robust_stats_phase_ii.csv`; the
committed 41-trial Phase-I macros stay frozen). **DYNO-D1 w252 tops the whole
82-config universe** (Sharpe 0.4114, DSR 0.9442) with Phase-II cells filling
the top of the table. But family-wise: D-variants vs V0 at w252 give SPA
p = 0.12 (vs p = 0.0015 for the Phase-I headline set) — the bigger family and
noisier differentials eat the significance; and the joint 90% MCS over all 21
distinct w252 strategies keeps 20, excluding **only V0** — with this many
highly-correlated causal variants the MCS has no discriminating power beyond
re-confirming the correlation baseline's inferiority.

**E5 — the stress hypothesis is rejected.** The directional-prior verification
(J1) had direction assignment most consequential in 2008/2022, motivating
"direction should matter most in stress". The regime tables say otherwise: the
D1/D2s edge over V0 concentrates in the *VIX bottom quintile* (+0.22/+0.25
Sharpe) and is ≈ 0 (D1) to negative (D2s) in the top quintile; in NBER
recession D0 ≥ D1. Direction-aware allocation earns its keep in calm markets,
not crises.

**E7 — the seed audit (Ce Guo's instability critique, quantified).**
Re-running the committed V1-DYNOTEARS w252 K=17 pipeline over 20 FFNN seeds
(discovery cached; FFNN cache off; everything else frozen). The first 10 and
the full 20 give the same distribution (median 0.390 → 0.389), so the
estimate is stable:

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| V1 net Sharpe across 20 seeds | 0.375 | 0.385 | 0.389 | 0.391 | 0.396 |

The committed value (0.381, seed 0 — bit-reproduced on re-run) sits at the
**5th percentile** of its own seed distribution, and the seed range (0.021)
is **more than twice the committed V1−V0 edge (+0.010)** — at the worst seed
(0.375) that edge all but vanishes (+0.004 over V0's 0.371): the
causal-driver-selection result lives inside its own seed noise. Every
Phase-II D-variant is deterministic by construction (two consecutive D0 runs
byte-identical), and D0/D1/D2s all sit *above the seed-cloud maximum*
(0.396 < 0.403) — the direction-aware graph route dominates the FFNN route
under any seed (`results/seed_audit.csv`,
`results/figures/phase_ii_seed.png`).

**Addendum (2026-07-13) — the CORR-HRP control and the decomposition
reading.** For the final report's master question ("how much of the causal
graph's gain over the correlation matrix is skeleton vs edge orientation?")
the like-for-like control is plain correlation-distance HRP (`CORR`,
`corr_hrp_weights`: distance √(½(1−ρ)), same allocator, graph-blind;
pre-registered as an optional comparator in §2-E1). Results
(`phase_ii_corr_hrp_w{252,504}`): **net Sharpe 0.381 / 0.363** — notably,
plain HRP *beats* the cum-corr HSP baseline V0 (0.371) at w252, i.e. the HSP
driver/FFNN machinery subtracts value relative to textbook HRP. Against this
stronger control the decomposition reads: **w252 — skeleton +0.022 (p=0.14),
orientation +0.009 on top (p=0.47); w504 — skeleton +0.009 (p=0.55),
orientation +0.023–0.027 on top (p=0.05–0.07)**. All 12 D-vs-CORR contrasts
are positive (consistent direction) but none is pairwise-significant, and the
family-wise R1 test (SPA, causal-structure variants vs CORR) gives **p=0.123
(w252) / 0.094 (w504)** — the causal graph's improvement over the pure
correlation matrix is directionally consistent at both windows but does not
survive data-snooping correction. The earlier D0−V0 significance (p<0.001)
was therefore substantially a statement about V0's machinery, not about the
correlation matrix per se. Unified battery: n_trials = 84 (+CORR ×2); the
joint 90% MCS over 22 w252 strategies keeps 21, excluding only V0; report
macros are now owned by `robust_stats --phase-ii` (the 41-trial default mode
no longer writes them).

**Artefacts.** `results/phase_ii_matrix.csv`, `phase_ii_contrasts.csv`,
`robust_stats_phase_ii.csv`, `seed_audit.csv`, `phase_ii_dag_diagnostics.csv`,
regime tables (extended), figures `results/figures/phase_ii_{heatmap,forest,
nav,seed,regime,decomposition}.png`. Compute: the whole phase ran in ~1 afternoon on the
warm discovery cache (each 215-rebalance D-variant backtest ≈ 40 s; the naïve
estimate without the cache was ~15 h/run). DAG diagnostics: DYNOTEARS
asset-graphs average ~625 edges (density 0.064) with DAG depth ~50 of 99 —
deep directed chains, so the ablation had real direction to work with.

---

## 2. Regime-conditional findings (the differentiator)

From `scripts/regime_analysis.py` → `results/regime_analysis/` (zero re-compute;
self-check: each table's "all" row reproduces the headline Sharpe to <1e-9; the
named-window per-rebalance excess-Sharpe reproduces the interim report's figure
for 4/5 windows, COVID differing only by date-range definition).

### Regime Sharpe (net), window 252

(Frozen-EEM re-run, 2026-06-15; V2 ≡ V1 in every cell — omitted.)

| variant | all | NBER recession | NBER expansion | high-vol (VIX top quintile) | low-vol (VIX bottom) |
|---|---|---|---|---|---|
| V0 | 0.371 | −0.638 | 0.731 | −1.351 | 5.630 |
| **V0′ (asset-only)** | **0.403** | **−0.575** | **0.759** | **−1.317** | **5.744** |
| V1-DYNOTEARS | 0.381 | −0.616 | 0.740 | −1.342 | 5.642 |
| V1-VARLiNGAM | 0.399 | −0.604 | 0.759 | −1.325 | 5.710 |

**Key finding: every causal variant beats V0 in *every* regime slice at w252** —
*less-bad* in stress (recession, high-vol) and *better* in benign (expansion,
low-vol). **V0′ (asset-only) is the standout**: best `all`-Sharpe and the
**least-bad recession Sharpe (−0.575 vs V0 −0.638)** — the cleanest
substantiation that causal structure differentiates, especially in stress.

- **Max drawdown, NBER recession**: V0′ −0.506, V1-VARLiNGAM −0.511 vs V0 −0.523
  — causal structure modestly cushions the GFC-recession drawdown (V0′ most).
- **Turnover** (one-way annualised, w252): broadly comparable across variants;
  rises in recession/high-vol regimes for all. (`turnover.csv`.)
- **Window 504 reconfirms the fragility**: V1-VARLiNGAM regime Sharpes fall
  *below* V0 in every slice, and at w504 DYNOTEARS-V1 ≈ V0′ ≈ V0 (all ≈0.372) —
  the regime edge is a **w252 phenomenon**, consistent with the matrix.

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
- **EEM driver was non-deterministic (reproducibility-critical, found in J4).**
  `fetch_yahoo_series` re-fetched EEM live on every call (its 2003-04 inception
  falls inside the 2-yr pre-`start` pad, so the cache coverage check never
  passed); `auto_adjust=True` jitters ≈3e-7 run-to-run, amplified by DYNOTEARS to
  ‖ΔW‖≈0.14. Fix: reuse a Yahoo cache when the previously-requested *span* covers
  the request (sidecar `.meta`) + atomic writes. Impact: shifted the committed
  **w504 V1** Sharpe 0.382→0.372 (see §1b) — the pipeline is now bit-reproducible.
- **K-cal closure hardcodes DYNOTEARS** — so a "VARLiNGAM K-cal" actually
  calibrates on DYNOTEARS scores; VARLiNGAM runs reuse K=17 deliberately.
- **stooq dropped** mid-project (added a paid-API requirement); WRDS/CRSP +
  yfinance fallback is the final, cleaner data spine.

---

## 8. Open items & caveats (discussion chapter + remaining work)
- **V0′ (asset-only Causal-HRP) — DONE** (see §1): at w252 it is the best
  variant (Sharpe 0.400, significantly beating V0 *and* V1) but window-fragile
  (≈V0 at w504). The 4-variant ablation is complete.
- **K-sensitivity (J4a) — DONE** (see §1b): V1>V0 holds at the operating K=17
  (w252) but is **not robust to K** (sign-flips at other K, never significant).
  Qualifies the primary claim — report the full K curve.
- **α/γ feedback sweep (J4b) — DONE** (see §1b): V2 ≡ V1 *exactly* across all 9
  (α,γ) combos — the closed loop is inert (utility re-ranks only within the
  causal-selected set). Strongest form of the closed-loop negative.
- **Headline re-run under frozen EEM — DONE** (2026-06-15) for the *full* matrix
  (DYNOTEARS + VARLiNGAM, V0/V1/V2/V0′ × both windows) + §2 regime tables; §1/§2
  updated. Corrected two DYNOTEARS EEM artefacts (w504 V1−V0 +0.012→+0.001;
  "V2 sig worse at w504" → V2≡V1). VARLiNGAM proved **robust to the fix**
  (w252 +0.028 p=0.004 unchanged). The whole §1/§2 is now internally consistent
  and reproducible.
- **NTS-NOTEARS (J5) — DONE as a reduced-scope probe** (see §1c): integrates +
  enforces the prior, agrees modestly with DYNOTEARS (Jaccard 0.35), ~10× the
  cost → full backtest remains future work.
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
