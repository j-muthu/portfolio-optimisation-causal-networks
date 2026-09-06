# main.tex verification audit (2026-09-05)

Nine parallel read-only audits of `final_report/main.tex` (one per chapter slice plus one whole-document
cross-chapter sweep), checked against the post-fix results (`results/phase_ii_matrix.csv`,
`phase_ii_contrasts.csv`, `oos_contrasts.csv`, `robust_stats*.csv`, `regime_analysis/`, the sweep bundles,
`_generated/*.tex`), the code in `pipeline/`, `FINDINGS.md` §1e, `PREDICTIONS_*.md`, `README.md` and
`archive/results_psd_legacy/`. Line numbers refer to the working-tree `main.tex` at the time of the audit.

**Coverage verdict.** Every table was checked cell by cell and every `\rs*` macro against the CSVs. The
numbers in the abstract, Ch1 Quantified results, Ch5 tables (decomp, tau, cost, regime, return-terms, psd,
matrix-full, HERC, SPA, DSR/MCS, OOS) and Ch6 summary are correct to the printed precision, with the
exceptions listed under A below. No residual "hump"/"U-shape", 0.403, D0s−D0 +0.025, skeleton-beats-pcorr,
narrowed-SPA-rejection or "every HRP cell beats EW" narrative survives, except the items flagged in A4, A11.
No HSP-era framing is presented as the main method outside Ch2 literature, the Ch5 caveat, MCS bookkeeping
and the HSP appendix. All 66 `\ref` targets resolve.

---

## A. Hard errors: contradictions and stale numbers (fix before submission)

| # | Line(s) | Claim | Problem | Evidence |
|---|---|---|---|---|
| A1 | 2161-2163 | Hyperparameters: "every clustering distance projected to the nearest PSD matrix before linkage" | States the removed defect as the current setting. Contradicts Ch4 L790-810 and app:psd L2014-2035. | `directed.py` L157, L252 and `causal_hsp.py` L43: `psd_project_distance=False` default. Flagged independently by 5 agents. |
| A2 | 899-903 | "Recursive bisection is mirror-invariant: reversing the leaf order yields the same partition tree. This means \topohrp responds to orientation through the partition structure … not through the upstream/downstream direction itself." | False for non-power-of-2 cluster sizes. `recursive_bisection` (`hrp.py` L47-66) splits at `len//2`, so odd clusters split asymmetrically. For N=99 reversed order gives max abs weight difference 2.6e-3 (≈26% of a 1/99 weight); exact only at N=8, 16, … The inference that \topohrp cannot see direction is therefore unsupported, and `tests/test_fixed_graph_ablation.py` L83-84 exempts D2 from the edge-reversal test on the same false premise. | Verified by orchestrator. |
| A3 | 530 | Universe: "largest market-cap assets from the S&P 500 at each rebalance, and take the union across those rebalances … over 2007–2024" | Not how the list was built. `scripts/phase_i_universe.txt` is the G.7 shakedown universe (commit 0587fbc): union of top-N by CRSP mcap at three snapshots {start, midpoint, end} of the shakedown period (`shakedown.py` L211-214; defaults 2018-01-02 to 2020-12-31). The list contains no 2007-2024 delisting (no LEH, BSC, WB, MER, AIG, WM) and includes post-2007 listings (META, ABBV, PSX, GM, TFC, CHTR, TMUS). This is survivorship in universe selection and is undisclosed; the delisting paragraph L532-536 is moot because none of the 99 delists in-sample. Matches the deferred "PIT survivorship" item. Referee-level. | Verified by orchestrator. |
| A4 | 1066-1069 | "ridge-Granger and Ledoit–Wolf cells, several of which fall below equal weight" | D0lw is above EW at every window: 0.3895/0.3939/0.3822/0.3563 vs EW 0.3636/0.3638/0.3608/0.3554. Pre-fix leftover (legacy D0lw w504 0.353 < EW). Granger D1 0.352, D3 0.350 and HERC0/HERC1 do fall below EW; only the LW clause is wrong. Also self-contradicts the preceding sentence (all DYNOTEARS HRP cells beat EW). | `phase_ii_matrix.csv` |
| A5 | 1253 | "\toposemcovhrp has the highest turnover in the family (0.140 at 252 days)" | \undirhrp 0.1451 (and D4 0.1444) exceed 0.1400; tab:return-terms L1926 itself lists 0.145 for \undirhrp. Pre-fix (legacy D0s turnover 0.1255). | `phase_ii_matrix.csv` |
| A6 | 1083-1085 | total "−0.004 to +0.013 for \semcovhrp and −0.003 to +0.016 for \toposemcovhrp" | These are the w189 and w504 endpoints, not ranges. w252 totals are −0.017 and −0.021 (tab:decomp L1149). Understates the w252 shortfall and weakens "close to zero at every window". | `phase_ii_contrasts.csv` D1−CORR, D2s−CORR |
| A7 | 1422 | "no pairwise contrast reaches conventional significance" | False unqualified. VARLiNGAM D0−CORR w189/w252 p=0.032/0.032, D1−CORR w189/w252 p=0.036/0.046, D2s−CORR w252 p=0.041, VARL D4−D0 w504 p=0.012; DYNOTEARS D0pc−CORR w189 p=0.008 (quoted at L1226), D1−D0pc w189 p=0.017. True only for the DYNOTEARS decomposition contrasts (min p=0.051). | `phase_ii_contrasts.csv` |
| A8 | 1457 | "the null hypothesis rejections from testing the restricted family can only be suggestive" | Presupposes rejections that no longer exist (narrowed min p=0.076, tab:spa). Pre-fix wording. Rephrase as "any rejection would be". | `robust_stats.tex` |
| A9 | 1455-1456 (also 1438) | "The initial plan was to use a family of 7 allocators, and this is indeed the main family I discuss in the report" | Contradicts L926-927 and tab:variants (5 graph-consuming allocators). The 7 appear only in SPA full sets and the allocator map. | main.tex |
| A10 | 2041-2043 | Correction "moved the graph-fed HRP cells by at most 0.005 in either direction, so the whole of the change in the decomposition is a change in the baseline" | True only for D0/D1 (max 0.0053). Legacy→current: D0s w189 −0.022, D0s w504 +0.012, D0pc w252 +0.032, D0pc w504 +0.013, D0df w378 +0.008; VARL D0 w189 −0.017, D1 w189 −0.024, D4 w252/378/504 +0.016/+0.022/+0.024. The vanished D0s−D0 +0.025 caveat was itself a graph-fed-cell change. Qualify to "the two graph-fed cells of the decomposition". | `archive/results_psd_legacy/phase_ii_matrix.csv` vs current |
| A11 | 1821-1822 | "Table matrix-full provides every cell of the fixed-graph ablation" | Table omits \structrp (D3) and \coanchrp (D4) for both methods. DYNO D3 0.400/0.398/0.400/0.387, D4 0.410/0.396/0.403/0.390; VARL D3 0.400/0.392/0.394/0.375, D4 0.398/**0.418**/**0.419**/0.394. VARL-D4 w378 ties the best DYNOTEARS cell and w252 beats every VARLiNGAM cell shown; these Sharpe levels appear nowhere in the report though they are in the 146 trials. Either add rows or drop "every cell". | `phase_ii_matrix.csv`, `robust_stats_phase_ii.csv` |
| A12 | 1322; 1160-1163 | \dfhrp called a "graph-fed cell" at L1322 and L1318/L1851; heatmap note L1160 says \toposemcovhrp w378 (0.419) is "the best graph-fed cell" | \dfhrp w378 = 0.4247 > 0.419. Abstract and L1375 call \dfhrp "graph-free"/"built without any graph" (its covariance is graph-free; its clustering uses the D0 distance). Pick one term and use it everywhere; L1160 needs "in the primary family". Also L1322's DSR list omits VARL-D4 w378 (0.9202) and w252 (0.9197), both above \semcovhrp w378 (0.9168). | `robust_stats_phase_ii.csv` |
| A13 | 1308-1309 | "Howard et al. had anticipated this outcome, with their finding that their centrality alpha resides in stable regimes (Section howard-theme)" | §2.2 (L445-452) never mentions centrality, alpha or stable regimes; it says denser networks accompany recessions. Add the finding to §2.2 or drop the cross-ref. | main.tex |
| A14 | 1288-1310, fig:regime, L1978 | "all three graph-fed allocators" / "the three DYNOTEARS allocators" | `plot_phase_ii_figures.py` f5_regime plots a fourth series, V1-DYNOTEARS labelled "causal-hsp" (bottom quintile −0.104), visible in `phase_ii_regime.png`. Never mentioned in Ch5; conflicts with HSP being appendix-only; the floatnote ("beat the baseline only in the VIX bottom quintile") is false for that bar. Drop the series or mention it. | script + PNG |

## B. Framing not earned by the evidence (referee standard)

| # | Line(s) | Issue |
|---|---|---|
| B1 | 1368-1371, 1401 (OOS); 1611, 1636-1660 (Ch6) | OOS presented as confirming the in-sample findings ("verify that my in-sample findings generalise", "came back positive at every window", "every sign I predicted came out as predicted"). Post-fix in-sample total is −0.004/−0.017/+0.006/+0.013 (≈0 or negative); OOS total is +0.142/+0.122/+0.096/+0.136, driven by a skeleton that flipped sign (+0.107/+0.071/−0.009/+0.100). Rule 1 of PREDICTIONS_OOS.md was written when the total was believed positive. Ch5 L1373-1379 does disclose the skeleton reversal and the w189 residual −0.090 (rule 4); Ch6 Summary omits both entirely. State plainly that the OOS total sign does not match in-sample and that the skeleton reversed. |
| B2 | 731-732, 874-879 | "any difference in performance can be attributed to orientation, and to nothing else"; "\semcovhrp−\skelhrp isolates the effect of orientation on the allocation step alone"; "orientation enters through the asymmetric propagation of shocks in B". Σ_SEM also imposes independent shocks (de-factoring) and shrinkage, the objection PREDICTIONS_COVARIANCE_CONTROLS.md pre-registers, and Ch5 L1476 concludes the recovery "cannot be attributed to the edge direction itself" (D0df reproduces D1 at every window). Ch4 should qualify these when it introduces the contrast, not leave them for Ch5 to undo. |
| B3 | 1176-1177, 1277, 1533-1534, 1611-1612; abstract 89-90; 343 | w504 orientation gain explained as "where the sample moments are stalest" and "most at 2 years", but L1119-1123 and L1662-1668 concede the w504 gain is \skelhrp-control-specific (against \undirhrp: +0.001). The mechanism is asserted four times for a control-specific number; the abstract and Ch1 carry no caveat. |
| B4 | abstract 86-89, 306, 1528-1530, 1609-1610 | "skeleton costs performance at every window" stated unconditionally. Against \undirhrp the skeleton contribution is +0.005 (w378) and +0.011 (w504) (D0s−CORR), and it reversed OOS at 3 of 4 windows. Ch5 caveat (vii) and Summary item 3 disclose the control dependence; abstract/Ch1 do not. |
| B5 | 90-91, 343, 1096-1097, 1122 vs 1115-1116, 1130, 1475, 1499-1502, 1529 | "Roughly cancel" / "total flat at zero" vs "orientations recover roughly half of what the skeleton loses" vs "at the two shorter windows it costs performance". Point estimates are read as real costs at w189/w252 (−0.004 p=0.82, −0.017 p=0.36) but as noise at w378/w504 (+0.006, +0.013). Apply one standard. "Roughly half" holds only at w252 (0.011/0.028); at w189 it is ≈78%. |
| B6 | 1564-1565 | "the stress hypothesis is rejected" while Ch5 L1298-1300 says the regime analysis is "purely descriptive. I ran no formal regime-level test". Use "contradicted by the point estimates". |
| B7 | 1560-1561; 356 | "Under VARLiNGAM graphs the orientations add nothing at any window" (D1−D0 w378 +0.011, w504 +0.007; Ch5 L1252 says 6/12 positive, mean 0.000) and Ch1 "no effect … at any of the 4 windows" (w189 is −0.036, larger in magnitude than any DYNOTEARS gain). Say "no systematic gain". |
| B8 | 1537-1540 | "demonstrating that the recovery is accomplished by the removal of the market factor" vs Ch5 L1204 "indicates". D0df−D1 = +0.004/−0.001/+0.011/+0.006, untested. Keep Ch5's verb. |
| B9 | 1703-1704, 2178-2179, 1758-1761 | Reproducibility over-claimed: "each number in this report is exactly replicable" and "two independent end-to-end runs … identical results" vs README (Phase I FFNN path not bit-reproducible on Apple Silicon; seed_audit spread 0.375-0.396; appendix L2114 says HSP is non-deterministic). Data declaration "results can still be reproduced from the repository" vs README ("a Yahoo-only rebuild will not reproduce the reported numbers") vs appendix L2182 ("may differ slightly"). Scope to the Ch4 allocator family and add "given WRDS/CRSP access". |
| B10 | 1722-1723 vs 929, 344, 1548, 1606, 1781, 1814 | Family provenance named four ways: "original plan", "experiment plan", "pre-specified", "pre-registered". Only PREDICTIONS_OOS.md is genuinely pre-committed (L2212); "pre-registered" implies an external registry. Also L305 describes the narrowed family without the post-hoc disclosure Ch5 L1331 makes. |
| B11 | 1616-1668 | Limitations omit Ch5's fourth threat (R²-sortability, L1462-1471, "I did not assess this threat"), the unswept λ_W=λ_A=0.05 (L2156) while τ/cost sweeps are presented as robustness, the deferred PIT-universe quantification (Ch5 L1460), and delisting returns/look-ahead (L1455-1457). Future work omits Raffinot's HERC cluster-cut, which appendix L1885 calls "the next logical member to test". |
| B12 | 1052-1056 | "the largest shortfall, \skelhrp against \corrhrp at 252 days, is −0.028 at p=0.056" after covering both methods. VARLiNGAM D1−CORR w189 is −0.071 (p=0.036), D2s−CORR w252 −0.041 (p=0.041), \pcorrhrp−CORR w189 −0.039 (p=0.008). Needs "within the DYNOTEARS primary family". |
| B13 | 345, 1218-1219, 1226-1228, 1539 | \pcorrhrp called "comparable density" while L1218-1219 concedes it links half as many asset pairs. |
| B14 | 1042 | "at the conventional windows it is worse"; L1433 defines conventional windows as 252 and 504, and at 504 the total is +0.013/+0.016. Use "two shorter windows". |
| B15 | 1208-1209 vs 1267 | Granger graphs "see no performance improvement from the edge-aware allocators" vs \topohrp 0.393 > \skelhrp 0.386 under Granger. |

## C. Method descriptions that do not match the code

| # | Line(s) | Text | Code |
|---|---|---|---|
| C1 | 421 | \pcorrhrp control via graphical lasso | `directed.py` `d0pc_weights` L302-326: invert Ledoit-Wolf covariance → partial correlations → hard-threshold by magnitude to the DAG's edge count. Ch5 L1216 has it right. |
| C2 | 411-414 | De-factoring: market factor "identified by the largest eigenvalue"; "I therefore use these estimators" | `hsp.py` `defactored_covariance` L36-49 regresses on the equal-weight cross-sectional mean. |
| C3 | 571 | VARLiNGAM "pruning step refits the lagged block without referencing that list, so I set asset-to-driver edges to zero after the fact" | Phase II passes `prune=False` (`run_phase_ii.py` L71; appendix L2158). Actual: masked ridge VAR for the lagged block (driver equations restricted to lagged drivers), B0 via DirectLiNGAM `prior_knowledge` matrix; post-fit zeroing is belt-and-braces (`varlingam.py` L359-400, L497-535). |
| C4 | 650 | "minimal feedback-arc removal" | `topological.py` L31-32: greedy removal of smallest-|M| edges until acyclic. Say "greedy". |
| C5 | 590 | "verifies it is acyclic" | `asset_graph.py` L133-138 computes `is_dag` and logs at debug level; Granger graphs pass through non-acyclic (L648 says so). "records". |
| C6 | 574, 582 | drivers ranked by "total weight of their edges to assets" / "edge mass" | Stage A score uses **lagged** (A_1) driver→asset weight only (`prune.py` L74-108). The 36-48% / 52-72% figures stack lag 0 and lag 1 (`verify_directional_prior.py` L99-118). Numbers right; say "lagged edge mass". |
| C7 | 866-867 | \undirhrp distance "taken from ½(|M|+|M^T|)" | `symmetrise_distance` (`_old_v123.py` L40-51) peak-normalises off-diagonals to [0,1] and uses 1 − similarity. |
| C8 | 793-795 | projection "applied to every allocator in this chapter alike" | \topohrp/\toposemcovhrp never pass a distance through linkage and were unchanged (L2030-2033; FINDINGS §1e). |
| C9 | 796-802 vs 2016-2018 | Synthetic check: "100 one-factor assets … up to 0.08 on a range of 0.40 to 0.70" | No script in the repo produces these numbers (orchestrator's agent reproduced the order of magnitude: shift 0.076, projected rank 1). Ch4 describes the defect as moving distances "by up to 0.08"; app:psd says it collapsed the matrix to rank one and the dendrogram to a chain. Same defect, very different severity; align them. |
| C10 | 956-958 | gate "exactly reproduces the committed first-phase bundle" | The V0′ bundle was re-run post-fix; gate holds against the re-run (V0′ = D0 exactly at w252 0.398754 and w504 0.375307). The originally committed bundle (legacy 0.4028) no longer matches. Say "re-run". |
| C11 | 959-960; 950 | "Appendix app:tables records the figures for these gates"; "60+ tests" | Appendix records only the replication max-weight-diff (L2035, in app:psd not app:tables). Cache figures are in Ch3 L674-675; 836/836 appears nowhere. 64 `def test_` in total, 43 in the five Phase II files. |
| C12 | 1358-1359 | MCS universe "all of the 252-day causal allocators from all three algorithms" | `robust_stats.py` `mcs_universe` excludes D3, D4, mechanism controls and Granger D3; 17 = 4 headline + DYNO {D0s,D1,D2,D2s} + VARL {D0,D0s,D1,D2,D2s} + GRAN {D0,D1,D2} + CORR. |
| C13 | 1368, 1381 | OOS "January 2025 to July 2026" | Bundle NAV runs 2025-01-02 to 2026-08-07 (19 rebalances, last 2026-07-09 held into August). The 19-rebalance count and Yahoo source switch (L2209) are not in the section itself. |
| C14 | 2168-2170 | "215 / 209 / 203 rebalances" | Every bundle has 215 rebalances at every window; 209/203 = 215 minus the 6/12 burn-in equal-weight fallbacks. Say so. |
| C15 | 1823-1828 | Granger scope "one representative allocator per way of using the graph" | A fourth Granger cell (D3, 0.350) exists and is counted in \rsNtrials but is unreported. |
| C16 | 2183, 2190-2191, 2205-2207 | "99 assets … committed to the repository" (only the ticker list is); "causalnex and lingam are vendored" (nts-notears too); "a cache miss halts the run" (the halt is the separate gate in `extract_asset_graphs.py` L109 run by `reproduction.py`; backtests themselves would refit). |

## D. Minor numeric / rounding

- L1852 `\rsDlwSharpeWone` prints 0.390; matrix 0.38948 → 0.389. L1854 `\rsDdfSharpeWone` 0.410; matrix 0.41053 → 0.411. `scripts/robust_stats.py` double-rounds from the 4-dp CSV.
- L1949 tab:return-terms \corrhrp turnover 0.129; matrix 0.12845 → 0.128.
- L1303-1304 regime top quintile "−0.04/−0.05"; daily_metrics −0.0345/−0.0448 → −0.03/−0.04 (FINDINGS §1e carries the same rounding).
- L1058-1060 "within 0.2 pp of net CAGR" holds at w252 only; D1−CORR at w504 is +0.32 pp, D2s−CORR at w378 +0.30 pp.
- L354 "gains roughly half the MDE": 0.009-0.021 vs 0.037 is 0.24-0.57; "at most about half".
- L303 SPA bound quotes `\rsSpaSkelWfivePre` (0.235, full family) while Ch5 L1352 uses the narrowed 0.213; say "full family" or switch macro.
- L177/190 "three orientation-aware allocators" vs L306 "5 orientation-aware allocators from the experiment plan" with \structrp/\coanchrp named only in the appendix. L1328 says the narrowed corr-family has "4 allocators"; L1055, L1513 and the map footnote say 5.
- L1247 \structrp "0.397 to 0.407 (see Table tau)" but tab:tau has no \structrp row (values 0.3974/0.3974/0.4010/0.4069 exist in the tau bundles).
- L1251 cost sweep "+0.020 to +0.022": unrounded 0.3864−0.3638 = +0.0226.
- L1402 OOS "widest SEs for total and residual": D2s−D0 SE at w378/w504 (0.107/0.087) exceeds total's (0.077/0.076). L1375 "within 0.005": w378 residual is +0.0052.
- L1316 "despite … long sample length": sample length raises PSR.
- L1467 "≈50% GFC drawdown": HRP family −49.3% to −57.1%; HERCC w189 −43%.
- L1616 "components are 0.01-0.03": skeleton w378 is −0.003.
- L1168 fig:forest caption "each orientation-aware allocator" also plots \undirhrp (orientation-blind).
- \rsNtrials=146 counts byte-identical duplicates (V2=V1, D0=V0′) as separate trials; conservative but unstated.

## E. Typos, broken sentences, cross-refs

- L401 "is therefore argues" → "argued". L466 "coefficent". L530 "to get set of". L532 "When an allocator delists" → "asset". L1186 "could themselves by the drivers". L1475 unclosed parenthesis.
- L851-853 broken sentence: "…is that it coincides those same differences would change the dendrogram as well."
- L568 `\ref{app:tables}` lands on the allocator map; the Hyperparameters section is unlabelled. L449 lag order is set in sec:discovery-run (L564) not sec:asset-graph. L592 Σ_SEM residuals referenced to sec:d-family; derived in sec:sem (L740). L1273 attributes the symmetric-matrix statement to Ch2; it is in Ch4 sec:symm (L726-729).
- fig:nav (L1238) never referenced in text. `\label{sec:d2}` (L888) sits mid-paragraph so L391 resolves to that number. `\label{chap:sensitivity}` is an HSP-era name (cosmetic).
- Roman numerals reused for SPA families (i)/(ii) and limitations (i)-(viii); L1558 "(limitation (ii))" is ambiguous.
- L406 HERC "mixed results" vs Ch5/Ch6 "signs generalise, sizes do not". L420 skeleton's "performance improvements" (pre-fix framing). L437 "battery applied to every claim in Ch5" overstates (regime and sweeps untested; MCS 252 only). L545 "removes any co-movement" → "linearly explained". L830-831 \skelhrp "does not account for edge directions" is immediately contradicted by L835-838; say "transpose-invariant" as tab:variants does.

---

## Status (2026-09-05, later the same day)

All items above were actioned in `final_report/main.tex` (119 patches applied by seven section agents plus one orchestrator fix), with these side changes:
- `tests/test_fixed_graph_ablation.py`: false mirror-invariance docstring rewritten; new test `test_d2_responds_to_edge_reversal_on_odd_universe` (N=9). 17 tests pass.
- `scripts/plot_phase_ii_figures.py`: regime figure no longer plots the HSP V1 series; `results/figures/phase_ii_regime.png` regenerated (three series).
- `scripts/robust_stats.py`: Sharpe rounded once at write time (was double-rounded). NOT rerun (10,000-rep SPA); `_generated/robust_stats.tex` hand-edited for `\rsDlwSharpeWone` 0.389 and `\rsDdfSharpeWone` 0.411. Rerun `python -m scripts.robust_stats --phase-ii` when convenient to regenerate from source.
- `\label{app:hyper}` and `\label{sec:caveats}` added; Ch3 now references the Hyperparameters appendix directly.
- Universe paragraph (Ch3 Data) rewritten to state the three-date 2018--2020 construction and its survivorship consequence; echoed in Ch5 scope caveat and Ch6 limitation (iv) and future work.
- Line numbers in the tables above predate the edits and the user's concurrent rewrite of the Premise 1 section; they are historical.

Not actioned / left for Josh:
- `\label{sec:d2}` still sits mid-paragraph (no heading to attach to); the Ch1 reference to it resolves to the section number of the D-family section. Add a subsection heading or point the Ch1 reference at `sec:d-family`.
- Chapter 3 L680 "a key miss halts the run" still describes the halt as part of the backtest; the halt is the separate cache-hit gate run by `reproduction.py`.
- Howard et al. centrality-in-stable-regimes sentence (Ch2 and Ch5) was verified against the literature note, not the paper.
- `tab:cost` rounded cells still imply +0.022 while the text now says +0.023 (unrounded).
