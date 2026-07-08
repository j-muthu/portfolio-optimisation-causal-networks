# PHASE II PLAN — Direction-Aware Allocation from Asset–Asset Causal Graphs

*Execution plan for Claude Code. Written 2026-07-08. Companion to `FINDINGS.md`
(Phase-I results) and the two repo plan docs. This document is self-contained:
it carries the rationale, the design decisions and their justifications, the
experiment matrix, the file-by-file implementation plan, tests, acceptance
criteria, and the runtime budget.*

---

## RESULT (2026-07-08) — executed in full; see FINDINGS.md §1d

All acceptance criteria met the same day (the warm discovery cache made each
D-variant backtest ≈ 40 s, vs the ~15 h naïve estimate):

1. **Replication gate passed byte-perfectly** — D0(DYNO, w252) ≡ the committed
   `phase_i_v0prime_w252` bundle: ΔSharpe = 0.0, max|Δweight| = 0.0 across all
   215 rebalances. Cache-hit gate 836/836 (both methods × both windows), zero
   refits, zero WRDS calls.
2. **Every E1 cell ran at both windows** (28 runs) + E2 GRANGER (4), E3 τ-sweep
   (9), E6 cost sweep (18), E4 82-trial battery, E5 regime tables, E7 seed
   audit (10 seeds) — all with bootstrap p-values.
3. **The §0.2 answer:** direction-aware allocation does *not* significantly
   beat its symmetrised counterpart family-wise. Strongest effect: D1−D0 =
   +0.023 at w504 (pairwise p = 0.049; all five DYNO direction-aware contrasts
   positive there) but SPA over the direction-aware grid vs D0 gives p ≈ 0.15
   (w504) / 0.33 (w252); nothing at w252; absent under VARLiNGAM. What
   direction *does* buy is cross-window consistency: D1 (0.411/0.395) and D2s
   (0.406/0.399) are the only variants strong at both windows — repairing
   V0′'s w504 collapse — cost-invariant (0→20 bps) and τ-robust. DYNO-D1 w252
   tops the whole 82-config universe (DSR 0.944). best(D1)−V1 = +0.030
   (p = 0.04). Deviation from plan hypothesis: the direction edge concentrates
   in *calm* (VIX-bottom-quintile) regimes, not stress. GRANGER: suffices for
   ordering (D2) but not for Σ_struct (Option 2 verdict).
4. **Seed audit (implemented deviation: 10 seeds × V1 only, user-approved
   minimum):** V1 Sharpe spans 0.381–0.396 across FFNN seeds — the seed range
   (0.015) exceeds the committed V1−V0 edge (+0.010), and the committed value
   is the most pessimistic seed. All D-variants deterministic and above the
   seed-cloud maximum.
5. FINDINGS.md §1d written; figures F1–F5 in `results/figures/phase_ii_*.png`;
   the new test files green alongside the full suite; pipeline
   bit-reproducible.

*Implementation deviation (documented):* instead of a parallel harness, the
D-variants thread through the proven `asset_only` closed-loop path (a new
`allocator=` dispatch in `pipeline/closed_loop.py`, mirroring how J3 added
V0′), so the replication gate is structural — everything upstream of the
allocator is the identical Phase-I code path. §3's `asset_graph.py`,
`directed.py`, `topological.py`, `granger.py` and all five test files exist as
specified; `run_phase_ii.py` is a thin launcher over `run_shakedown`.

---

## 0. Context: why this phase exists

### 0.1 The scientific question (the spine of the final report)

> **Where, if anywhere, in a hierarchical allocation pipeline does discovered
> causal structure add value over correlation?**

The final report answers this by testing three *entry points* for causal
information:

1. **Through driver selection** (Phase I: V0 vs V1/V2, the HSP route).
   Verdict, already established: modest, window- and K-fragile; the FFNN/driver
   machinery is where the instability lives (EEM bug, K-fragility, inert
   feedback loop). A clean negative about an entry point.
2. **Through the undirected asset–asset skeleton** (Phase I: V0′).
   Verdict: **yes** — Sharpe 0.403 at w252, significantly beats V0 (+0.032,
   p<0.001) *and* V1 (p=0.01), best recession Sharpe, in the MCS90 —
   achieved while *deleting* the entire HSP driver/FFNN machinery.
3. **Through edge direction itself** — *this phase*. V0′ symmetrises the
   asset–asset block (`causal_embedding_distance`) before clustering, i.e. it
   discards exactly the directional information that motivated causal
   discovery. Nobody (including Howard, Lohre & Mudde 2025) has tested whether
   direction per se carries allocation value.

### 0.2 The precise Phase-II sub-question

> **Holding the discovered graph fixed, does a direction-aware allocator
> outperform its own symmetrised counterpart?**

The *fixed-graph ablation* is the whole design: the same per-window `W` feeds
both the symmetric and the direction-aware allocators, so direction is the sole
treatment variable. This yields a knockdown headline either way:

- Direction-aware wins → a positive methodological contribution (allocation
  from asymmetric matrices).
- It doesn't → "asset–asset causal *structure* significantly improves HRP, but
  edge *direction* adds nothing beyond the undirected skeleton" — a clean,
  well-identified negative.

### 0.3 Supervisor constraints this phase must satisfy

- Ce Guo (7 Jul): need a clear, novel methodological contribution; HSP is
  seed-dependent/unstable; suggested **asymmetric matrices in clustering/
  allocation** (Option 1) and low-variance direct asset–asset learning
  (Option 2). This phase is Option 1 with Option 2's comparators folded in.
- Second marker (25 Jun): one precise scientific question; not just Sharpe —
  DSR, SPA/White's Reality Check, Model Confidence Set (already partially in
  `scripts/robust_stats.py`); precise definitions of "returns".
- No FFNN anywhere in the new path → deterministic, seed-free by construction
  (the frozen-EEM fix already makes data bit-reproducible). The **seed audit**
  of the Phase-I FFNN path (§5, E7) turns Ce Guo's critique into a table.

---

## 1. Mathematical design (rationale for the allocators)

### 1.1 The structural model and why the DAG makes it exact

Both DYNOTEARS (contemporaneous `W`) and VARLiNGAM (`B0`) are stored in the
repo's `i → j` convention (`M[i, j]` = effect of `i` on `j`) and imply the SEM

```
(I − Mᵀ) x = ε        ⇒        x = (I − Mᵀ)⁻¹ ε  =:  B ε
```

Key fact: **`M` restricted to the asset–asset block is a DAG** (DYNOTEARS
enforces acyclicity; LiNGAM's `B0` is strictly lower-triangular in causal
order). A DAG adjacency is nilpotent, so `(I − Mᵀ)⁻¹ = Σₖ (Mᵀ)ᵏ` terminates in
at most `N` terms and **exists exactly** — no spectral-radius assumption. `B`
propagates influence through *all* directed paths; its asymmetry is where edge
direction enters allocation natively.

### 1.2 The structural covariance (already implemented)

`pipeline/portfolio/_old_v123.py::structural_covariance(matrix, residual_cov)`
computes `Σ_struct = B Σ_ε Bᵀ` (+ nearest-PSD projection). Two required
upgrades (see §4, file `directed.py`):

1. **Estimate `Σ_ε` from data instead of defaulting to identity.** Compute the
   structural residuals on the fit window, `E = X_z (I − M)` (row-vector
   convention; equivalently `ε = (I − Mᵀ) x` per observation), and take
   `Σ_ε = diag(var(E))`. Diagonality is the SEM's own assumption (LiNGAM:
   independent non-Gaussian errors) — document this, don't estimate a dense
   `Σ_ε`, which would smuggle sample correlation back in.
2. **De-standardise.** Discovery is fit on per-window z-scored data (the
   z-stats are stored on each window dataclass: `zscore_mean`, `zscore_std`).
   `Σ_struct` therefore lives in z-units. Allocation needs return units:
   `Σ = D_σ Σ_struct^z D_σ` with `D_σ = diag(zscore_std[asset_idx])`.
   **This is mandatory** — recursive bisection compares cluster variances, and
   skipping the rescale silently equalises asset vols.

### 1.3 The allocator family (D-variants)

All long-only, fully invested, monthly, same universe/costs as Phase I — so
every number is comparable to the Phase-I matrix.

| tag | clustering / ordering | allocation covariance | direction enters via | status |
|---|---|---|---|---|
| **D0** | embedding distance (sym) | sample cov | nowhere (baseline = V0′ as-is) | exists |
| **D0s** | `(|M|+|Mᵀ|)/2` distance (sym) | sample cov | nowhere (2nd symmetrisation, robustness) | exists (`symmetrise_distance`) |
| **D1** | embedding distance (sym) | **Σ_struct** | allocation step (through `B`) | resurrect old "v2" + §1.2 fixes |
| **D2** | **topological order** replaces the dendrogram | sample cov | ordering step | new |
| **D2s** | topological order | **Σ_struct** | ordering + allocation | new |
| **D3** | none (no hierarchy at all) | **Σ_struct** | pure shock-space ERC | new |
| **D4** | co-ancestry distance from `BBᵀ` (sym, but built from directed paths) | sample cov | distance construction | new, cheap |

**D2 — Causal-Ordered Bisection.** HRP's dendrogram exists only to produce a
quasi-diagonalising leaf order. A DAG carries a topological order for free:
sort assets upstream→downstream (ties broken by total downstream influence
`Σⱼ B[i, j]`, then alphabetically for determinism), then run the *existing*
`recursive_bisection` on that order. Replaces clustering entirely — Ce Guo's
"we couldn't use clustering" point — at near-zero implementation cost.

**D3 — Structural-shock Equal Risk Contribution (SRP).** Skip hierarchy
altogether: run ERC (Spinu 2013 cyclical coordinate descent, long-only,
closed-loop-free, deterministic) on `Σ_struct`. Portfolio risk decomposes as
`wᵀΣw = ‖Σ_ε^{1/2} Bᵀ w‖²` — parity of contributions on `Σ_struct` is parity
over *structural shock origins* rather than over correlated returns
(Leontief-inverse logic; cite input–output economics in the write-up). Note
for the report: exact per-shock parity (`|(Bᵀw)_i| σ_{ε,i}` equal ∀i) generally
requires shorting; the long-only ERC on `Σ_struct` is the practical projection.
State this rather than clip signed weights.

**D4 — Co-ancestry clustering.** Similarity `S = B̃ B̃ᵀ` (row-normalised `B̃`),
distance `√(2(1 − S))`: two assets are close iff they inherit shocks from the
same *upstream sources*, even with no direct edge. Direction-aware input to an
unchanged symmetric pipeline — the halfway house between D0 and D2/D3 that
lets the results chapter decompose *where* any gain comes from.

### 1.4 Discovery variants (rows of the matrix)

- **DYNO** — asset–asset block of the cached joint DYNOTEARS fits
  (`JointDynotearsWindow.asset_to_asset_block(0)`); **zero recompute**, 418
  windows already cached.
- **VAR** — `B0` asset block of the cached joint VARLiNGAM fits (same
  accessor pattern; check `JointVarLingamWindow` exposes it, else add).
- **GRANGER** — new low-variance comparator (Option 2): ridge-regularised VAR(1)
  on asset returns, `M[i, j] = |ridge coefficient of asset i's lag in asset j's
  equation|`, thresholded to the same edge density as DYNO's window (density
  matching makes the comparison fair). Deterministic, closed-form, no ICA/L-BFGS.
  *Note:* Granger `M` is lagged, not contemporaneous, and is **not** guaranteed
  acyclic → for GRANGER use `B ≈ Σ_{k≤K_trunc} (Mᵀ)ᵏ` with `K_trunc = 10` and a
  spectral-radius guard (scale `M` by `0.95/ρ(M)` if `ρ ≥ 1`); document as an
  approximation. Lag-1 only (Phase I + Howard et al.: lagged edges are
  near-negligible; the comparator exists to test "cheap directed graph vs
  fancy directed graph", not to model lags richly).

### 1.5 What is deliberately out of scope

- Anything with drivers, the FFNN, sensitivities, or the feedback loop (Phase I
  is frozen; its bundles are the comparators).
- NTS-NOTEARS (J5 probe stands as-is; future work).
- Short positions, leverage, signal timing / lead-lag trading overlays (drifts
  into a different literature and invites data-mining critiques).
- Network-density regime definitions (nice-to-have; NBER + VIX ship).

---

## 2. Experiment matrix

### E1 — Headline fixed-graph ablation (the core result)

Grid: `{DYNO, VAR} × {D0, D0s, D1, D2, D2s, D3, D4} × {w252, w504}` = 28 runs,
plus existing comparators loaded from Phase-I bundles: correlation-HRP proxy
(V0), V1-DYNOTEARS, V1-VARLiNGAM, 1/N, and (new, cheap) plain HRP on sample
correlation + min-variance on sample cov if not already bundled.
215 rebalances, 2007-01→2024-11, net 5 bps, monthly, top-99-by-mcap universe —
**identical protocol to Phase I** so all tables merge.

Primary contrasts (report these, in this order):
1. `D{1,2,2s,3,4} − D0` per method/window — *the direction effect, fixed graph.*
2. `D0 − V0` — replication of the Phase-I V0′ result under the new harness.
3. `best(D*) − V1-*` — direction-aware vs the full HSP-machinery route.
4. `DYNO vs GRANGER` at the best allocator — does cheap directed structure
   suffice? (Option 2 verdict.)

### E2 — GRANGER discovery arm

`GRANGER × {D0, D1, D2, D3} × {w252}` (w504 only if w252 shows signal).
Requires a discovery pass (~seconds/window, ridge is closed-form) → cache under
the same content-keyed discovery cache with `method="granger_ridge"`.

### E3 — Sparsity-threshold sensitivity (the Phase-II analogue of J4a)

The asset–asset block's edge set depends on the magnitude threshold τ applied
to `W`. Sweep `τ ∈ {0.0 (none), 0.01, 0.05, 0.10}` × `{D0, D2, D3}` ×
DYNO × w252. Pre-empts "is it robust to the regularisation knob". Also record
per-window edge density and DAG depth (longest path) as diagnostics.

### E4 — Robustness battery (second-marker requirements)

Extend `scripts/robust_stats.py`'s trial universe with every Phase-II config
(`n_trials` grows from 41 → ~75; DSR penalises us for our own sweep — good,
report it). Add/refresh: PSR vs zero & vs V0, DSR, White's RC + Hansen's SPA
(benchmark = V0; candidates = all D-variants), MCS at 90%. Politis–Romano
block bootstrap CIs on all E1 contrasts (2000 resamples, same block length as
Phase I).

### E5 — Regime analysis

Extend `scripts/regime_analysis.py` variant table with the D-variants
(NBER recession/expansion, VIX top/bottom quintile; Sharpe, MaxDD, turnover
per slice). Hypothesis to check: direction should matter *most* in stress
(the directional-prior verification found direction assignment most
consequential in 2008/2022 windows).

### E6 — Cost/turnover sweep

`{0, 5, 10, 20}` bps for the headline D-variants; D2/D3 may have different
turnover profiles than clustering-based allocators (ordering can be less
stable window-to-window than a dendrogram — measure, don't assume).

### E7 — Seed audit of the Phase-I FFNN path (Ce Guo's critique, quantified)

Re-run `V0` and `V1-DYNOTEARS` (w252, K=17) with `n_seeds = 20` FFNN seeds
(everything else frozen). Report the Sharpe distribution (min/median/max, IQR),
the position of the committed number within it, and alongside it a one-line
statement that every D-variant is deterministic (zero seed variance by
construction). Budget note: this is the expensive item (~FFNN refit per seed
per rebalance) — see §6; if the full 20×2 grid is infeasible, 10 seeds × V1
only is the minimum publishable version.

---

## 3. Repository file plan

Repo layout assumed as in Phase I: `pipeline/{data,discovery,selection,
sensitivity,portfolio,evaluation}/`, `scripts/`, `tests/`, `results/`.
**NEW** = create; **MOD** = modify; everything else untouched. Phase-I result
bundles under `results/phase_i_*` are read-only inputs.

```
pipeline/
  discovery/
    asset_graph.py                 NEW   single chokepoint: per-window asset–asset M for all methods
    granger.py                     NEW   ridge-VAR(1) directed comparator
    cache.py                       MOD   (only if needed) accept method="granger_ridge"
    varlingam.py                   MOD   add asset_to_asset_block(0) accessor if JointVarLingamWindow lacks it
  portfolio/
    directed.py                    NEW   B-matrix machinery + Σ_struct v2 + D1/D3/D4 allocators
    topological.py                 NEW   DAG utilities + D2/D2s causal-ordered bisection
    causal_hsp.py                  MOD   nothing removed; optionally re-export D-variant wrappers
    _old_v123.py                   ---   read-only donor (structural_covariance, distances, nearest_psd)
  evaluation/
    (reuse bootstrap.py, metrics.py, regime.py as-is)
scripts/
  run_phase_ii.py                  NEW   launcher for the D-variant grid (mirrors run_phase_i.py CLI)
  run_granger_discovery.py         NEW   populate GRANGER graphs into the discovery cache
  run_seed_audit.py                NEW   E7
  collate_phase_ii.py              NEW   merge bundles → results/phase_ii_matrix.csv + contrast tables
  robust_stats.py                  MOD   extend trial universe with Phase-II tags (E4)
  regime_analysis.py               MOD   add D-variants to VARIANTS table (E5)
  plot_phase_ii_figures.py         NEW   figures F1–F5 (§7)
tests/
  test_asset_graph.py              NEW
  test_directed_allocators.py      NEW
  test_topological.py              NEW
  test_fixed_graph_ablation.py     NEW   integration: same W → D0 vs D2 differ only by design
  test_granger.py                  NEW
results/
  phase_ii_<method>_<D>_w<win>[ _tau<τ> ]/    bundles, same schema as phase_i_*
  phase_ii_matrix.csv              headline table
  phase_ii_contrasts.csv           bootstrap ΔSharpe + p per contrast
  seed_audit.csv                   E7
  granger_cache/ …                 via existing discovery cache dir
PHASE_II_PLAN.md                   this file (repo root, next to FINDINGS.md)
```

### 3.1 `pipeline/discovery/asset_graph.py` (NEW) — the single chokepoint

Everything downstream consumes one type. Do **not** let allocators touch
window dataclasses directly — the fixed-graph ablation depends on every
allocator seeing byte-identical `M`.

```python
@dataclass(frozen=True)
class AssetGraphWindow:
    end_date: pd.Timestamp
    asset_names: list[str]          # ordering is canonical for M
    M: np.ndarray                   # (N,N) contemporaneous, i -> j, DAG for DYNO/VAR
    zscore_std: np.ndarray          # per-asset window std (de-standardisation, §1.2)
    method: str                     # "dynotears" | "varlingam" | "granger_ridge"
    tau: float                      # magnitude threshold applied
    is_dag: bool                    # verified at construction
    meta: dict

def extract_asset_graphs(method: str, window: int, tau: float = 0.0,
                         universe: ... ) -> list[AssetGraphWindow]:
    """DYNO/VAR: load cached JointXWindow per rebalance via
    load_or_compute_discovery (cache hit path — zero refit), take
    asset_to_asset_block(0), apply |M| < tau -> 0, restrict rows/cols to the
    rebalance-date eligible universe (reuse the per-asset eligibility mask —
    late-inception names must be dropped from M *and* asset_names together).
    GRANGER: read from granger cache. Verify DAG (nx.is_directed_acyclic_graph
    or a topo-sort attempt); for granger set is_dag honestly and let callers
    branch (§1.4 truncated-Neumann path)."""
```

Gotchas encoded here, once: (a) asset ordering between `M` and the returns
panel must be asserted equal, not assumed; (b) `tau` thresholding happens
here so E3 is one flag; (c) VARLiNGAM's stored `B0` is already transposed to
`i → j` (`varlingam.py` line ~216) — do not re-transpose.

### 3.2 `pipeline/portfolio/directed.py` (NEW)

```python
def total_effect_matrix(M, is_dag=True, k_trunc=10) -> np.ndarray
    # B = (I − Mᵀ)⁻¹ exact when DAG (solve, don't invert); truncated Neumann
    # + spectral guard otherwise (§1.4). Unit-tested against series sum.

def structural_covariance_v2(M, residuals_z: np.ndarray | None,
                             zscore_std) -> pd.DataFrame
    # §1.2: Σ_ε = diag(var(E)) from window residuals E = X_z(I − M) when
    # residuals available, else identity (log a warning — identity is only for
    # tests); z-space Σ via _old_v123.structural_covariance-equivalent math;
    # THEN de-standardise by D_σ; nearest_psd; return labelled DataFrame.

def d1_weights(graph: AssetGraphWindow, returns_window) -> pd.Series
    # embedding distance (reuse causal_embedding_distance) + hrp_weights with
    # covariance = structural_covariance_v2.

def d3_srp_weights(graph, returns_window) -> pd.Series
    # ERC on Σ_struct: Spinu cyclical coordinate descent, long-only,
    # deterministic init w=1/N, tol 1e-10, max_iter 10_000; assert convergence.

def d4_coancestry_weights(graph, returns_window) -> pd.Series
    # S = row-normalised(B) @ row-normalised(B).T; D = sqrt(2(1−S)) clipped;
    # hrp_weights with sample covariance.

def d0s_weights(graph, returns_window) -> pd.Series
    # symmetrise_distance(M) + sample cov (reuse _old_v123 donors).
```

Design rule: every allocator takes `(AssetGraphWindow, returns_window)` and
returns a name-indexed weight Series summing to 1 — same contract as
`v0prime_asset_only_causal_hrp`, so `run_phase_ii.py` is a thin dispatch.

### 3.3 `pipeline/portfolio/topological.py` (NEW)

```python
def topological_order(M, asset_names, tie_break="downstream_influence")
    # Kahn's algorithm; deterministic tie-break: total downstream influence
    # Σ_j B[i,j] desc, then name asc. Raises on cycle (callers pre-check
    # is_dag; GRANGER graphs use a feedback-arc-set fallback: drop the
    # smallest-|M| edges until acyclic, log count dropped).

def d2_weights(graph, returns_window, covariance="sample")   # D2 / D2s
    # order = topological_order(...); recursive_bisection (existing, hrp.py)
    # on sample cov (D2) or structural_covariance_v2 (D2s). No linkage call.

def dag_diagnostics(graph) -> dict
    # edge density, longest path (DAG depth), n roots/leaves, order stability
    # inputs for E3/E6 (Kendall's τ of consecutive-window orders).
```

### 3.4 `pipeline/discovery/granger.py` (NEW)

Ridge-VAR(1) per window on the z-scored asset panel (`sklearn.linalg` /
closed-form `(XᵀX + λI)⁻¹XᵀY`, λ by GCV or fixed 1e-2 — fixed is fine, sweep
not required), `M[i, j] = |coef(x_i,t−1 → x_j,t)|`, threshold to match the
paired DYNO window's edge density. Emits `AssetGraphWindow(method=
"granger_ridge", is_dag=<verified>)`. Cache via `load_or_compute_discovery`
with a distinct method key.

### 3.5 `scripts/run_phase_ii.py` (NEW)

Mirror `run_phase_i.py`'s CLI and bundle schema exactly (same `BacktestResult`
persistence via `run_backtest`), so `robust_stats.py`/`regime_analysis.py`
extensions are pure tag additions:

```
--method {dynotears,varlingam,granger} --allocator {D0,D0s,D1,D2,D2s,D3,D4}
--window {252,504} --tau FLOAT=0.0 --transaction-cost-bps FLOAT=5.0
--output-tag STR (default phase_ii_{method}_{allocator}_w{window})
```

Strategy closure per rebalance: pull the `AssetGraphWindow` for date t
(pre-extracted list, dict-keyed by date), slice to `universe_at(t)`, dispatch
to the allocator. **No discovery, no FFNN, no selection inside the loop** —
per-rebalance cost is linear algebra (~ms), so a full 215-rebalance run is
minutes, not hours.

### 3.6 `scripts/run_seed_audit.py` (NEW — E7)

Loop `seed ∈ range(20)`: set torch/numpy seeds, re-run the V0 and V1 w252
K=17 pipelines with the FFNN sensitivity step live (reuse `run_phase_i.py`
internals; discovery comes from cache so only the FFNN refits). Persist
per-seed Sharpe/CAGR/MaxDD → `results/seed_audit.csv`. Output table: variant ×
{min, p25, median, p75, max, committed-value percentile}.

### 3.7 Modifications

- `scripts/robust_stats.py`: append Phase-II tags to `_all_trial_tags()`;
  add an SPA run with benchmark `V0_w252` vs candidates `{all D-variants w252}`;
  keep `drop_duplicate_configs` (D0 may duplicate V0′ byte-for-byte — that is
  the desired replication check, dedupe for MCS only).
- `scripts/regime_analysis.py`: add D-variant rows to the variants table.
- `pipeline/discovery/varlingam.py`: add `asset_to_asset_block(0)` to the
  joint-window dataclass if absent (mirror `dynotears.py` line ~527).

---

## 4. Tests (write these first)

`tests/test_directed_allocators.py`
- `total_effect_matrix`: on a hand-built 4-node DAG, equals the explicit
  Neumann sum; `(I−Mᵀ)B = I` to 1e-12.
- `structural_covariance_v2`: with `M=0`, Σ_struct reduces to
  `diag(zscore_std²)` de-standardised → sanity anchor. With a chain A→B→C and
  unit shocks, downstream variance strictly increases along the chain.
- De-standardisation: doubling one asset's `zscore_std` quadruples its Σ entry.
- D3 ERC: risk contributions equal to 1e-8 on a random PSD Σ; weights ≥ 0,
  sum 1; identical across two calls (determinism).

`tests/test_topological.py`
- Order respects all edges on random DAGs (property test, 100 draws).
- Deterministic under permuted input ordering (tie-break works).
- Cycle → raises; feedback-arc fallback drops the minimum-|M| edge on a
  crafted 3-cycle.

`tests/test_asset_graph.py`
- Round-trip: cached DYNO joint window → `extract_asset_graphs` → M equals
  `asset_to_asset_block(0)` after τ-threshold; asset ordering matches panel.
- Universe slicing drops rows *and* columns consistently.

`tests/test_fixed_graph_ablation.py` (integration, small synthetic universe)
- Same `AssetGraphWindow` into D0 and D2: weights differ (direction is live).
- Symmetrise M (`(|M|+|Mᵀ|)/2` fed as-if-directed): D2's topological order
  degenerates gracefully / D1 ≡ D0-with-Σ_struct-symmetric — i.e. the ablation
  isolates direction and nothing else.
- Leak canary pattern (reuse `leak_canary.py` approach): shift graphs one
  window forward → results visibly change (guards against off-by-one
  graph/date joins).

`tests/test_granger.py`
- Simulated VAR(1) with known sparse coefficients: ridge recovers the support
  at the matched density (precision > 0.8 on an easy SNR).

---

## 5. Execution order (dependency-sorted; each step ends green)

1. **Scaffold + tests** — `asset_graph.py`, `directed.py`, `topological.py`,
   `granger.py` + the five test files. All unit tests pass on synthetic data.
2. **Graph extraction dry-run** — extract DYNO w252 graphs for all 215
   rebalances from cache (zero refit expected — assert cache-hit rate = 100%;
   if the cache keys miss, stop and reconcile keying before anything else).
   Emit `dag_diagnostics` summary (density, depth) → sanity-read.
3. **Replication gate** — run `D0` (DYNO, w252) through `run_phase_ii.py`;
   Sharpe must reproduce the Phase-I V0′ bundle to ≤ 1e-3 (ideally byte-equal
   weights). **Do not proceed past a failed gate** — it means the new harness
   diverges from Phase I and every downstream comparison is void.
4. **E1 core** — D1/D2/D2s/D3/D4/D0s × DYNO × w252; then VAR × w252; then
   both × w504. Collate + bootstrap contrasts after each block.
5. **E2** — granger discovery pass + its allocator runs.
6. **E3** — τ sweep (reuses cached graphs; threshold applied at extraction).
7. **E4/E5/E6** — robust-stats extension, regime tables, cost sweep.
8. **E7** — seed audit (long-running; launch early in background once step 3
   passes, it's independent of steps 4–7).
9. **Figures + collation** — §7; refresh `FINDINGS.md` with a `## Phase II`
   section mirroring the §1/§1b house style (headline matrix, contrasts,
   honest caveats).

## 6. Runtime budget

| item | est. | notes |
|---|---|---|
| graph extraction (cache hits) | minutes | 418 cached fits reused |
| each D-variant backtest | 2–10 min | linear algebra per rebalance; no NN |
| full E1 grid (28 runs) | < 1 day wall | embarrassingly parallel, `_parallel.py` |
| GRANGER discovery (215+203 win) | < 1 h | closed-form ridge |
| τ sweep (E3) | ~1 h | extraction-level flag |
| robust stats / regimes / costs | ~1 h | zero recompute, bundle-fed |
| seed audit (E7) | ~2–4 h/seed/variant worst case | FFNN refits; discovery cached. 20×2 ≈ multiple overnights → start early; fall back to 10×V1 |

## 7. Deliverables

**Tables** — `phase_ii_matrix.csv` (method × allocator × window Sharpe/CAGR/
MaxDD/turnover, net); `phase_ii_contrasts.csv` (the four §2-E1 contrasts with
bootstrap CIs + p); extended `robust_stats.csv` (PSR/DSR/SPA/RC/MCS over ~75
trials); regime tables; `seed_audit.csv`.

**Figures** (`plot_phase_ii_figures.py`, house palette from
`plot_thesis_figures.py`): F1 allocator × method Sharpe heat-map with the D0
column highlighted (the direction effect at a glance); F2 fixed-graph contrast
forest plot (ΔSharpe ± CI per allocator, both windows); F3 NAV curves
(V0, D0, best-D, V1) w252; F4 seed-audit violin (V0/V1 seed distributions vs
a point for the deterministic D-variants); F5 regime excess bars incl. best-D.

**Acceptance criteria for the phase**
1. Replication gate passed (D0 ≡ V0′).
2. Every E1 cell run at both windows with bootstrap p-values; no cell blocked
   on compute.
3. A one-sentence answer to §0.2 exists and is supported in *both* directions
   of outcome (positive: which allocator, how big, robust to τ/window/method?
   negative: flat across all five direction-aware allocators, both methods,
   both windows — a characterised null like J4b, not an absence of evidence).
4. Seed audit table exists with ≥ 10 seeds on at least V1.
5. `FINDINGS.md` Phase-II section written; figures regenerated; all tests green;
   pipeline remains bit-reproducible (two consecutive runs byte-identical).

## 8. Risks & mitigations

- **Σ_struct ill-conditioned in dense-graph windows** (2008/2020 density
  spikes): nearest-PSD already guards PSD-ness; additionally ridge-load
  `Σ + 1e-6·tr(Σ)/N·I` before bisection/ERC. Log condition numbers per window.
- **Topological-order instability window-to-window → turnover blow-up**:
  measured in E6 (Kendall's τ diagnostic); if turnover is the story, report it
  — cost-adjusted results are the headline anyway.
- **D-variants ≈ D0 everywhere**: that *is* the negative result, and it is
  publishable by design (§0.2). Resist adding allocators mid-flight to chase
  a positive; the pre-registered grid is the credibility.
- **VARLiNGAM joint-window accessor missing / B0 convention confusion**: fixed
  at the chokepoint (§3.1) with a convention test in `test_asset_graph.py`.
- **Cache-key misses at step 2**: reconcile keys, never refit silently — a
  silent refit would decouple Phase-II graphs from the Phase-I bundles and
  invalidate the replication gate.
