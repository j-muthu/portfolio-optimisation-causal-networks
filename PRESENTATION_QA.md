# Q&A preparation: Portfolio Optimisation with Causal Networks

Prepared for the 10-minute Q&A after the presentation on 15 September 2026.
Panel: Dr Ce Guo and Prof. Wayne Luk (supervisors, Custom Computing group) and Prof. Stephen Weston (second marker, quant-finance practitioner).

Every number below is taken from `final_report/main.tex`, `final_report/_generated/*.tex`, `results/*.csv` or the cached graph fits. Numbers marked **[computed today, not in the report]** were pulled from the cache for this document and are not in the thesis; quote them only if you are comfortable doing so.

---

## 0. The ten answers to have word-perfect

1. **"So what did we learn from a null result?"** Three things a practitioner can act on: a DYNOTEARS graph is not a free upgrade over correlation in HRP; the part of the graph that helps (the SEM covariance) is reproduced by regressing out the equal-weight market and using the residual covariance, which needs no discovery step; and any "causal" allocator should be tested by ablation plus graph-free controls, because without the controls the orientation contrast would have looked like evidence for edge direction.
2. **"Why is the skeleton a worse clustering input than correlation?"** Because it is sparse. A 6 to 7% dense graph gives each asset an edge profile over roughly 12 neighbours, so the dendrogram is built on much less information than the full correlation matrix. A density-matched partial-correlation network, built with no discovery, produces the same cost at every window (differences of -0.020 to +0.012, p >= 0.07), so the loss is a property of sparsity, not of the discovered structure.
3. **"Why does the orientation effect appear at all if direction does not matter?"** Σ_SEM assumes independent structural shocks. With a diagonal Σ_ε, every off-diagonal entry of the implied covariance comes only from asset-to-asset paths, so the common market factor is removed by construction. The de-factored control does exactly that with a regression and no graph, and matches semcov-hrp to within 0.011 Sharpe at every window. Ledoit-Wolf shrinkage does not match, so it is de-factoring, not shrinkage.
4. **"Nothing is significant. Is the study just underpowered?"** Yes, and I say so. The minimum detectable effect at 80% power is about 0.047 Sharpe for the total and 0.037 for the orientation; the observed effects are 0.003 to 0.028. The null is therefore an upper bound: whatever the graph adds is under about 0.04 Sharpe. The informative part is not the p-values but the mechanism controls, which are point-estimate matches at all four windows.
5. **"Tell me about the bug."** Every HRP and HERC allocator, including the baseline, inherited a nearest-PSD projection of its clustering distance matrix from the Phase I code. A Euclidean distance matrix has one positive eigenvalue, so the projection collapsed it to rank one and single linkage degenerated into a chain ordered by one scalar per asset. Removing it raised correlation-hrp by +0.020 to +0.046 and moved the graph-fed allocators by at most 0.005. I found it on 5 September, reran every cell, and rewrote the findings from an apparent gain to the null. The old behaviour is kept behind a flag for replay. No new configurations were added, so the trial count for the multiplicity correction is unchanged.
6. **"How do you know the graphs are credible without ground truth?"** I cannot validate them against truth. I check that the directional prior binds (without it, 36 to 48% of driver-asset edge mass goes on asset-to-driver edges), that the graphs are non-degenerate DAGs with stable density across windows, that fits are bit-reproducible, and that the strongest edges are economically sensible (same-sector pairs such as HD and LOW, GS and MS, MA and V, DUK and SO). Validating orientations against firm-specific news shocks is future work.
7. **"Why DYNOTEARS?"** It scales to 132 variables at every monthly rebalance, it accepts the directional prior as a hard constraint before fitting, and it returns a DAG, which I need for an exact total-effect matrix and a topological order. Constraint-based methods scale poorly, rely on faithfulness, and PCMCI+ returns a partially directed graph.
8. **"Is the result specific to DYNOTEARS?"** Yes, and I state that as a limitation. Under VARLiNGAM the orientation contrasts have no consistent sign (6 of 12 positive, mean 0.000). Under ridge-Granger, which keeps directions but degrades magnitudes, semcov-hrp falls by 0.058. So the effect needs reliable edge magnitudes, and even then it is de-factoring.
9. **"Out of sample the total is positive at every window. Doesn't that contradict the null?"** Only the sign of the orientation effect agrees with the in-sample result. The total is positive because the skeleton flipped sign at three of four windows, which shows the skeleton effect is not stable over time. With 19 rebalances the standard errors are 0.06 to 0.14, so these are sign checks, not tests.
10. **"What would you do next?"** Repeat the ablation on a second discovery family (NTS-NOTEARS was 10x too slow at full scale), test R²-sortability as an explicit control on the topological order, sweep λ, quantify survivorship with a point-in-time panel, and validate edge directions against firm-specific shocks using the total-effect matrix for risk attribution.

---

## 1. Framing, novelty and contribution

**Q: What exactly is novel here? Howard et al. already ran DYNOTEARS on the S&P 500.**
- Howard, Lohre and Mudde never set portfolio weights from the graph. They use it for similarity or compress it to a centrality score.
- My contributions are (i) two ways of getting an asymmetric matrix into HRP without symmetrising (Σ_SEM at the allocation step, topological order at the ordering step), (ii) the skeleton-versus-orientation decomposition with graph-free mechanism controls, and (iii) the empirical finding under multiplicity-aware inference.
- Nobody had previously decomposed the allocation value of a discovered causal graph into its two information sources.

**Q: Why HRP and not a mean-variance or minimum-variance optimiser fed with Σ_SEM?**
- The research question is about hierarchical allocation, where the correlation matrix's role is clustering plus cluster variances, so it is the natural place to ask "what does a graph add over correlation".
- I did run a non-hierarchical consumer of Σ_SEM: struct-rp, equal risk contribution over the SEM covariance. It scores 0.387 to 0.400, no better than HRP. It is in the full 7-allocator SPA family.
- Mean-variance would reintroduce the inversion instability that HRP exists to avoid, and would confound the graph's effect with the optimiser's.

**Q: Why should edge direction matter for a risk-based allocator at all? What is the economic hypothesis?**
- Shocks propagate along directed paths. The total-effect matrix T tells you which assets are sources of systemic risk and which are sinks, and that is invisible to correlation.
- Two testable implications: bisection on Σ_SEM should treat a downstream asset as carrying its upstream exposures; and ordering upstream-to-downstream should separate sources from sinks before bisection.
- The network-portfolio literature (Pozzi et al., Peralta and Zareei) found that under-weighting central nodes helps because centrality is market exposure. My de-factored control shows the orientation-aware allocators are exploiting exactly that mechanism.

**Q: Is contemporaneous "causality" between daily stock returns even meaningful?**
- At daily aggregation, intraday lead-lag collapses into contemporaneous co-movement, so the contemporaneous block is where the structure is. Howard et al. report the same.
- I do not claim the edges are true causes. I test whether the structure a causal discovery algorithm produces has allocation value, which is precisely the "causal-washing" question the thesis answers.
- Example from the July 2026 graph: HD→LOW at 0.80, GS→MS at 0.77, MA→V at 0.75. The links are sensible; the directions between such pairs are hard to defend economically, which is consistent with orientation carrying no value beyond de-factoring. **[edges computed today from the cache; the pairs are not listed in the report]**

**Q: Why is this thesis worth a distinction if the answer is "no"?**
- A negative result with a mechanism identified is more useful than a positive result without one. I show not only that the graph does not help but why each component behaves as it does, and that both behaviours are reproducible without a graph.
- The methodological pieces stand on their own: the two entry points, the ablation, the controls, the inference battery, and a bit-reproducible pipeline with 88 tests and a one-command reproduction script.
- The pre-correction result looked positive. Finding and disclosing the bug, and letting it flip the headline, is what backtest rigour is supposed to do.

---

## 2. Data and universe (expect these from Prof. Weston)

**Q: Survivorship bias. You picked 99 survivors in 2018 to 2020 and backtested from 2007. Doesn't that contaminate the comparisons too, not just the levels?**
- The same panel feeds every allocator, so the bias is common to all of them. That is why I say it affects levels.
- The honest caveat: it could interact with the allocator if one allocator tilts more towards the eventual winners. I did not quantify that, and the report lists a point-in-time S&P 500 panel with delisting returns as future work.
- No asset delists over the sample, and each asset only becomes eligible once it has a full estimation window of prices, so there is no forward-fill of missing history.

**Q: Why 99 assets and not the S&P 100 or 500?**
- S&P 100 historical membership is paywalled. I took the union of the largest S&P 500 constituents by market cap at three dates during the 2018 to 2020 pilot; the union had 99 names.
- Compute: each DYNOTEARS fit at 132 variables takes around four minutes, and there are 215 rebalances at four windows for each discovery method. The full S&P 500 would be roughly 15x the variables and far more than 15x the time because the acyclicity constraint is cubic in d.

**Q: Are these Sharpe ratios not low for a survivorship-biased universe?**
- They are annualised from daily net returns over a sample containing 2008, 2020 and 2022, long-only, unlevered, after 5 bps costs. Equal weight over the same 99 names is 0.364 at the 252-day window, so all allocators are being compared against a like-for-like anchor.
- The absolute level is not the object of study; the pairwise differences are.

**Q: Why these 33 drivers, and why include them at all if you only use the asset block?**
- The asset-to-asset block is estimated conditional on the drivers, so co-movement linearly explained by rates, credit, FX, commodities, VIX and international equity is removed before the asset graph is read. Without them, macro co-movement would be forced onto asset-to-asset edges.
- They are chosen to be plausibly exogenous. Sector ETFs, US index returns and factor portfolios are excluded because assets clearly cause those.
- The directional prior (assets cannot cause drivers) is enforced as a hard constraint. Without it, 36 to 48% of driver-asset edge mass lands on asset-to-driver edges, so it binds.

**Q: Any look-ahead?**
- Every window ends strictly before the rebalance date. Standardisation, the graph fit, the sample covariance and D_σ all use only that window. Monthly macro series are forward-filled as the value an observer would have known on day t.
- Graph fits are cached by a hash of the exact window bytes, so every allocator at a given (method, window, date) reads the identical graph and the cache cannot leak across dates.

**Q: Why Yahoo for the out-of-sample slice?**
- CRSP daily data ends on 31 December 2024. Yahoo adjusts differently, so OOS levels are not comparable with in-sample levels, but every OOS allocator sees the same Yahoo panel so the OOS contrasts are like-for-like.

---

## 3. Discovery and graph credibility

**Q: Why only one lag, and what happens to the lagged matrix A1?**
- I fit p = 1 because Howard et al. find the contemporaneous matrix carries almost all of the structure, and lags beyond one add little at daily frequency.
- The lag-1 block is fitted but not used by the allocators. Its L1 mass is roughly 40% of the contemporaneous block and its largest edges are about a third the size of the largest contemporaneous edges. **[computed today, not in the report]** Lagged edges are predictive coefficients rather than structural effects, and for a risk allocator with a monthly hold the contemporaneous covariance is the relevant object. Folding A1 into a VAR-implied unconditional covariance is a natural extension.

**Q: Are the graphs stable from month to month?**
- The report says I did not measure edge persistence, and lists it as a limitation.
- Aggregate diagnostics are stable: density 6.3 to 6.9% across windows, mean longest path about 50 assets, all fits are DAGs, and fits are bit-identical across processes.
- If pressed: consecutive 252-day windows share 231 of 252 days, and the edge-set Jaccard between consecutive fits is about 0.4 on average. **[computed today, not in the report]** That means individual edges are noisy even though the aggregate structure is stable. It strengthens the conclusion rather than weakening it: an allocator that reads individual edge directions is reading noise, and the part that survives (de-factoring) does not depend on which edges are present.

**Q: How do you handle the fact that DYNOTEARS' acyclicity constraint only holds to a tolerance?**
- After optimisation, any remaining cycle is broken by deleting its smallest-magnitude edge, and the count is recorded. Every one of the 842 DYNOTEARS fits used by the allocators is a DAG.

**Q: Why is VARLiNGAM's graph 32% dense at 189 days?**
- Slide 6's "noisy adaptive lasso penalties" is the right mechanism. The two methods have opposite defaults. DYNOTEARS starts from zero and only adds an edge if it beats an explicit L1 penalty (λ = 0.05) and a 0.01 threshold. DirectLiNGAM starts dense: once it has a causal order, it regresses every variable on all of its predecessors, so before pruning the asset block is a full lower triangle (50% of ordered pairs). Sparsity has to come from its adaptive lasso.
- That adaptive lasso is weak at 189 samples. Its weights are 1/|OLS coefficient| from a first-stage regression with up to 131 predictors on 189 observations, and its BIC uses a noise variance from the same near-saturated OLS (about 57 residual degrees of freedom). Both are noisy, so the lasso keeps most coefficients: at 189 days it retains 64% of the allowed lower-triangle entries, at 252 days 29%, at 378 and 504 days 13 to 14%. **[shares computed today from the cache; the report gives only the 32% and 6 to 7% densities]**
- What the report's "lingam pruning disabled" refers to is VARLiNGAM's second, joint pruning pass over contemporaneous plus lagged coefficients (up to 263 predictors), which cannot run at all on 189 samples. Disabling it does not remove the per-target adaptive lasso inside DirectLiNGAM.
- The density collapse from 32% to 7% as the window grows is the likely reason VARLiNGAM's semcov-hrp at 189 days (0.340) is the worst cell in the matrix, and why the orientation effect has no consistent sign under VARLiNGAM.

**Q: Does VARLiNGAM even apply? Its identification needs non-Gaussian residuals.**
- Daily returns are heavily non-Gaussian (pooled excess kurtosis about 15.9), so identification holds at this frequency. It weakens with aggregation, which is one reason it is the comparator and not the primary method.

**Q: Reisach's variance-sortability critique. Are your orientations just variance ordering?**
- I standardise every window before fitting, which removes the marginal-variance sorting that Reisach et al. (2021) describe.
- Their 2023 follow-up shows ordering by R² survives standardisation. For equities R² is mostly market beta, so part of the topological order may be a beta ordering. I did not test this; it is the ninth limitation. The clean test is an R²-ordered bisection as a control for topo-hrp, the same way pcorr-hrp controls for skeleton-hrp.
- It does not affect the decomposition, because every allocator receives the same graph.

**Q: Why λ = 0.05 and no sweep?**
- It gives 6 to 7% density and deep DAGs without saturation. I did not sweep it, which is a stated limitation. The τ sweep on the extracted graph (0 to 0.10) changed Sharpe by under 0.013, which is weak evidence that the results are not knife-edge in sparsity, but it is not a λ sweep.

**Q: What does the ridge-Granger control actually control for? It is a lagged graph, not a contemporaneous one.**
- Fair point, and it is why it is only an ancillary control at one window on four allocators. Its role is to separate direction from magnitude: its directions are informative (Granger predictability) but its magnitudes are shrunken lagged coefficients, not structural effects. It is density-matched to DYNOTEARS so the comparison is at equal sparsity.
- Result: topo-hrp is unchanged (0.393 vs 0.399) but semcov-hrp falls to 0.352. So the SEM covariance needs reliable magnitudes, not just directions.

---

## 4. The SEM and the allocators

**Q: Walk me through Σ_SEM. Why is Σ_ε diagonal?**
- (I − Mᵀ)x = ε, so x = Tε with T = (I − Mᵀ)⁻¹, exact for a DAG because M is nilpotent. Σ_SEM = D_σ (T Σ_ε Tᵀ) D_σ.
- Σ_ε is diagonal because the SEM assumes independent structural shocks. If I estimated the off-diagonals from the residuals, I would feed the sample correlation back in and the allocator would no longer be testing the graph.
- The consequence, which I state as a limitation, is that the drivers are marginalised out, so ε absorbs driver-explained variance and the independence assumption is violated in practice. That is also exactly why Σ_SEM de-factors: whatever common variation is in ε is dropped from the off-diagonals.

**Q: Isn't the de-factoring result then obvious from the construction?**
- In hindsight, partly yes: a diagonal Σ_ε removes the common factor by construction. What was not obvious is whether the directed propagation through T adds anything on top of that. The answer is no: the regression-based de-factored covariance, which has no T at all, matches semcov-hrp at every window.
- That is the point of the control. Without it, the positive orientation contrast would have been reported as evidence for edge direction.

**Q: Matching Sharpe does not prove the same mechanism. Two different mechanisms could give the same number.**
- Agreed; Sharpe equivalence is necessary, not sufficient. Three things make me confident: the match holds at all four windows to within 0.011; it holds out of sample at three of four windows; and the mechanism is present analytically, since a diagonal Σ_ε has no common-factor term.
- A weight-level comparison between semcov-hrp and defactored-hrp (weight correlation, overlap) would strengthen this and I did not do it.

**Q: Is skeleton-hrp really orientation-blind? The embedding [M_i·, M_·i] distinguishes in-edges from out-edges.**
- It is transpose-invariant (reversing all edges at once leaves distances unchanged) but not per-edge invariant, and the report says so.
- I keep it as the primary skeleton allocator because the treatments and controls must differ from it in exactly one respect: semcov-hrp, lw-hrp and defactored-hrp all keep its dendrogram and change only the covariance.
- undirected-hrp is the strictly orientation-blind check. Against it the skeleton effect is -0.022, -0.026, +0.005, +0.011 and the orientation effect for semcov-hrp is +0.017, +0.009, +0.001, +0.001. So the orientation gain at the two long windows only holds against skeleton-hrp; the two skeleton definitions disagree by up to 0.019 there, which is the seventh limitation.

**Q: Why single linkage?**
- It is López de Prado's original HRP choice, and the baseline must be standard HRP. Every clustering allocator uses the same linkage so it cancels in the contrasts.

**Q: How does topo-hrp order assets with ties, and how many roots and leaves are there?**
- Kahn's algorithm, upstream first; ties broken by total downstream influence Σ_r |T_ri|, then by name. Graphs average about 3 roots and 7 leaves with a longest path of about 50, so most assets sit on a long chain and ties are rare.

**Q: Why long-only?**
- To match HRP and the practitioner setting. A true risk parity over structural shocks needs shorting, which is why struct-rp is only a long-only approximation and shock-parity is future work.

**Q: Why did Ledoit-Wolf do worse than the sample covariance?**
- I did not investigate why; it was a control, and it fails to reproduce the gain. One plausible reason is that shrinking towards a scaled identity compresses the cluster-variance ratios that recursive bisection uses, which pushes weights towards equal split. I would say that as a conjecture only.

---

## 5. The decomposition and the mechanism controls

**Q: The skeleton is -0.028 at p = 0.056 and the orientation is +0.021 at p = 0.051. Those are nearly significant. Why not call them effects?**
- They are uncorrected p-values from 12 orientation contrasts and 146 configurations. With 12 correlated tests, one p near 0.05 is what the null predicts.
- The family-wise tests are the right ones. Hansen's SPA never rejects for either family at any window, and the pooled test across windows gives p = 0.83 and 0.40.
- What I do claim is consistency of sign: all 12 DYNOTEARS orientation contrasts are non-negative and the skeleton is negative at every window, and the same signs hold under HERC.

**Q: The slide says the skeleton cost "rises at larger windows". The numbers are -0.019, -0.028, -0.003, -0.008. That is not monotone.**
- Correct. The cost is largest at 252 days and small at the two longer windows. The safer statement is that the skeleton's cost shrinks at the two longer windows, where the graph is fitted on more data.

**Q: Does pcorr-hrp really match the DAG's density? A symmetric matrix uses two entries per pair.**
- I match the count of non-zero entries, so pcorr-hrp links half as many pairs as the DAG's skeleton. I flag this in the report: sparsity itself could drive part of the difference. Even so it reproduces the skeleton's cost at every window and underperforms correlation-hrp at every window.

**Q: The defactored control uses the skeleton dendrogram. Did you try the de-factored covariance inside standard HRP with the correlation dendrogram?**
- No. That cell was not run. The control was designed to differ from semcov-hrp in one respect only, so it had to keep the skeleton dendrogram.
- Slide 11's "defactoring alone yields the same results as HRP" rests on defactored-hrp versus correlation-hrp being 0.000, -0.018, +0.018, +0.019 across the four windows, so it is within noise of HRP but not a clean test of de-factoring on its own.

**Q: Why does the orientation effect disappear under VARLiNGAM but not under HERC?**
- HERC changes the allocator, not the graph. It keeps the same DYNOTEARS dendrograms and reproduces both signs with larger magnitudes (skeleton -0.258 to -0.028, orientation +0.010 to +0.088), all insignificant.
- VARLiNGAM changes the graph. Its edges are denser and less reliable at short windows, and the SEM covariance depends on edge magnitudes. So the decomposition is allocator-robust but discovery-method-dependent, which is the main caveat.

**Q: coancestry-hrp beats skeleton-hrp by +0.030 at p = 0.012 under VARLiNGAM at 504 days. Why is that not a finding?**
- It is one of the uncorrected contrasts I list explicitly in the caveats. It is a single cell out of 146, from an allocator outside the narrowed family, under the comparator method. The SPA over the full family that contains it does not reject. Reporting it as a finding would be the selection bias the DSR and SPA exist to prevent.

**Q: HERC's skeleton cost is enormous (-0.258 at 189 days). Doesn't that say something?**
- HERC concentrates weights by following the dendrogram topology, so a poor dendrogram is punished much harder than under HRP's bisection of the leaf order. It confirms the sign and says the size of the decomposition depends on the allocator. The HERC cluster-cut choice was not tuned, which is a deferred item.

---

## 6. Inference, multiplicity and power

**Q: How is the bootstrap p-value defined?**
- Stationary block bootstrap (Politis-Romano), 10,000 resamples, mean block 21 days, resampling the joint two-strategy daily return panel so cross-strategy correlation is preserved. The two-sided p is twice the fraction of bootstrap differences on the other side of zero from the point estimate.

**Q: Why 146 trials, and are they independent? DSR assumes independent trials.**
- 146 is every configuration evaluated: 41 from Phase I (HSP) and 105 from Phase II, including sweeps, controls and HERC. Pre-correction reruns of the same configurations are not extra trials.
- Many trials are highly correlated, so the effective number is lower than 146. That makes 146 a conservative deflation, and no configuration reaches the 0.95 bar even so.

**Q: SPA uses mean return, not Sharpe. Why?**
- Hansen's SPA and the MCS are defined on a loss series; I use negative daily net return. Their hypotheses are about mean return. The Sharpe-based inference is the pairwise bootstrap and the DSR.
- SPA is one-sided for superiority, so the significant negative pairwise contrasts under VARLiNGAM (for example -0.071 at p = 0.036) do not cause rejection. That is by design: the question is whether any candidate beats the baseline.

**Q: The MCS keeps 16 of 17 strategies. Is that not just saying the test has no power?**
- Yes. With 17 highly correlated variants whose Sharpe ratios span 0.35 to 0.43, the MCS cannot separate treatments from controls. I report it because it is part of the standard battery, and the only exclusion (HSP as published) is consistent with everything else.

**Q: How did you compute the MDE?**
- Block-bootstrap standard error of the annualised Sharpe difference for the contrast at 252 days (0.019 total, 0.015 orientation, taking the larger of the two orientation contrasts), times (z_0.95 + z_0.80) for a one-sided test at 5% size and 80% power. That gives about 0.047 and 0.037.

**Q: How long a sample would you need to detect a 0.02 effect?**
- Standard error scales with 1/√T, so halving the MDE needs four times the sample, roughly 70 years. That is why I frame the null as an upper bound rather than pursuing significance.

**Q: Why 5 bps and 21-day holds?**
- 5 bps one-way is a conventional large-cap cost; I sweep 0, 5, 10 and 20 bps. Turnover is 0.09 to 0.15 per rebalance so costs move Sharpe by about 0.01 to 0.02 across the sweep and do not change any sign. Monthly rebalancing matches the HRP literature and gives 215 rebalances.
- semcov-hrp has the lowest turnover in the family (0.089 at 504 days versus 0.107 for skeleton-hrp), so its gain over the skeleton grows slightly with cost.

---

## 7. Post hoc changes and the bug (expect the panel to push here)

**Q: When did you find the projection bug, how, and how do I know the fix is right?**
- Found on 5 September 2026 while auditing the allocator code. The projection was applied to the clustering distance, not the covariance, and a Euclidean distance matrix is not PSD (it has one positive eigenvalue), so nearest-PSD projection zeroed everything but that eigenvector and collapsed the matrix to rank one.
- Verification: the corrected baseline is standard HRP as published (correlation distance straight into linkage); the old path is kept behind `psd_project_distance=True` and reproduces the pre-fix bundles; the tests assert the allocators' invariances; every HRP and HERC cell, statistic and figure was rerun.
- Effect: correlation-hrp +0.020 to +0.046, graph-fed allocators at most 0.005. The graph-fed allocators barely moved because their distance matrices were already low-rank-ish (sparse embeddings), so the projection hurt the baseline far more than the treatments. That asymmetry is what manufactured the apparent gain.

**Q: You changed the research question, the windows and the family after seeing results. Why should I trust the inference?**
- All three are disclosed in the caveats, and all evaluated configurations are counted in the 146-trial deflation, so the multiplicity correction spans both phases and all windows.
- The full 7-allocator family remains the primary SPA test; the narrowed 5-allocator family is labelled indicative. The two excluded allocators underperform the baseline in 12 of 16 configurations, so they were not dropped for their results.
- The added windows (189 and 378) were run identically to the original two, and the answer is the same at all four.

**Q: Why did you abandon Phase I (HSP)?**
- Causal driver selection gave +0.010 at one driver count and negatives at three others, none significant; the neural network's seed-to-seed spread (0.021) was twice the effect; the feedback loop produced weights identical to open-loop at every rebalance; and plain HRP beat HSP outright (0.427 versus 0.371). The sharper question was whether the graph itself helps HRP, so that became Phase II. Phase I's 41 configurations stay in the trial count.

**Q: Were the mechanism controls pre-registered?**
- pcorr-hrp was pre-registered on 15 August 2026. The two covariance controls were added after I recognised the shrinkage and de-factoring confound. They count towards the 146 trials, and they are controls rather than candidates, so they cannot inflate the headline.

---

## 8. Robustness, regimes and out of sample

**Q: Howard et al. say causal structure helps in stress. Did you see that?**
- No. Conditioning the 252-day portfolios on NBER recessions and VIX quintiles, the graph-fed allocators are below the baseline in every regime except the bottom VIX quintile. In recessions: -0.567 baseline against -0.591, -0.590 and -0.606.

**Q: Out of sample the skeleton flips sign and the 189-day residual against the de-factored control is -0.090. Explain.**
- The skeleton's effect is time-unstable; that is the OOS finding. The orientation sign is stable (positive at all four windows, both allocators).
- The 189-day residual is one cell with a standard error of 0.138. At the three longer windows the residual is within 0.006, matching in-sample. I do not read anything into a single 19-rebalance cell with that error.
- The OOS protocol was committed before the slice was run: no p-values, standard errors only.

**Q: Does the τ sweep cover the headline allocator?**
- No. It covers skeleton-hrp, topo-hrp and struct-rp. semcov-hrp's sensitivity to τ is untested, and the report says so. It is on my deferred list.

---

## 9. Implementation and compute (expect these from Prof. Luk and Dr Guo)

**Q: What is the computational cost, and where does it go?**
- One DYNOTEARS fit at 132 variables and one lag takes about four minutes on a laptop. The bottleneck is the acyclicity constraint h(W) = tr(exp(W∘W)) − d, whose matrix exponential and gradient are O(d³) per L-BFGS-B evaluation inside an augmented-Lagrangian loop.
- 2,051 cached fits across methods and windows (532 MB); 227 result cells; the reproduction script runs the whole thing end to end.
- Fits across rebalance dates are embarrassingly parallel and are run through joblib. Allocation and backtesting are negligible next to discovery.

**Q: Could this be accelerated, and would it change the conclusions?**
- The matrix exponential and the L-BFGS-B inner loop are dense linear algebra, so GPU or custom hardware would give a large constant-factor speed-up, and NTS-NOTEARS (10x slower per fit in my probe) would become feasible at full scale. That is the most direct route to testing method dependence, which is my main caveat.
- It would not change the statistical conclusion, which is limited by 18 years of data, not compute.

**Q: How do you guarantee reproducibility?**
- Deterministic fits (bit-identical across processes and machines), content-hashed discovery cache, a replication gate asserting DYNOTEARS skeleton-hrp at 252 days equals the Phase I asset-only allocator to 16 digits, 88 pytest cases including invariance tests (controls are transpose-invariant, treatments respond to edge reversal, pcorr-hrp sees the graph only through its edge count), pinned dependencies, vendored causalnex, lingam and nts-notears, and `reproduction.py` running preflight, tests, every cell, collation, figures and the report.
- The only non-reproducible piece is Phase I's neural network on Apple MPS, which is why there is a 20-seed audit.

**Q: Why vendor the libraries?**
- causalnex pins old pandas and numpy; lingam and nts-notears conflict with each other. Vendoring with package init files removed and a small stub loader gives one Python 3.13 environment. No vendored file is modified.

**Q: Why not use the WRDS Python package?**
- It pins an old pandas that silently changes DataFrame behaviour in the pipeline. I connect to the WRDS Postgres endpoint directly.

---

## 10. Big-picture and hostile questions

**Q: Isn't the whole premise wrong? HRP uses correlation because it wants co-movement, and a DAG is the wrong object for a risk allocator.**
- That is close to what the evidence says, and I would agree with the conclusion while defending the test. The premise in the literature (López de Prado 2023, Howard et al.) is that causal structure carries information correlation lacks. The right response is to test it with an ablation rather than assume either way. The answer is that the extra information has no allocation value here.

**Q: Could you have reached the de-factoring conclusion analytically, without 146 backtests?**
- The de-factoring property of a diagonal Σ_ε, yes. Whether directed propagation adds anything on top of it, no; that needed the control. And the skeleton result (sparsity is what hurts) is purely empirical.

**Q: What is the practical recommendation?**
- For long-only large-cap HRP, do not replace correlation with a discovered graph. If you want the market-factor-removal effect, regress returns on the equal-weight market and bisect on the residual covariance; it matches the SEM allocator at every window and costs nothing.
- Treat "causal" labels on allocators with scepticism until they have been ablated against graph-free controls.

**Q: What would change your conclusion?**
- A discovery method whose orientation gain survives the de-factored control, or a market where the skeleton is not sparse relative to the correlation structure (for example cross-asset-class universes with genuinely block-sparse dependence). Both are in future work.

**Q: What are you least confident about?**
- Method dependence: the orientation gain only exists under DYNOTEARS, and I have one alternative family. And the skeleton's sign reversal out of sample, which says the skeleton effect is not stable in time.

---

## 11. Slide-specific traps (things on the slides a careful reader may pick on)

| Slide | Issue | What to say if asked |
| --- | --- | --- |
| 6 | "32% for VARLiNGAM at 189 days (due to noisy adaptive lasso penalties)" | Correct as stated. DirectLiNGAM regresses each variable on all predecessors and relies on a per-target adaptive lasso to prune; at 189 samples with up to 131 predictors the lasso's weights and BIC noise estimate are too noisy to prune, so 64% of allowed entries survive. Density falls to 7% by 378 days. The report's "pruning disabled" refers to a separate second pass in VARLiNGAM (see Section 3). |
| 6 | "Survivorship bias affects levels, not pairwise comparisons" | Common to all allocators, so it cancels to first order; an allocator-specific interaction is possible and unquantified. |
| 10 | "Skeleton's performance effect rises at larger windows" | Not monotone (-0.019, -0.028, -0.003, -0.008). Say the cost shrinks at the two longer windows. |
| 11 | "Defactoring alone yields same results as HRP" | defactored-hrp minus correlation-hrp is 0.000, -0.018, +0.018, +0.019. Within noise of HRP. The control keeps the skeleton dendrogram; de-factoring inside standard HRP was not run. |
| 12 | Title says OOS "agrees on signs only" | Orientation sign agrees; skeleton sign flips at 3 of 4 windows; total is positive OOS. |
| 7 | Σ_ε "diagonal by design" | Also violated in practice because drivers are marginalised into ε; that violation is what de-factors. |
| 9 | "Last-minute bugfix" | Have the 5 September timeline, the rank-one explanation and the replay flag ready (Section 7 above). |
| 8 | "Deflated Sharpe over all 146 configurations" | 41 Phase I + 105 Phase II; pre-fix reruns not counted; correlated trials make 146 conservative. |
| 13 | "Three disclosed post hoc changes" | Research question from Phase I, windows 2 to 4, SPA family 7 to 5. The bug fix is a correction, not a design change. |

---

## 12. Number cheat-sheet

| Item | Value |
| --- | --- |
| Universe, sample | 99 US large caps, 2007-01 to 2024-11, 215 monthly rebalances (209 at 378 d, 203 at 504 d fed by DAGs) |
| Drivers | 33 (rates, slopes, credit, FX, commodities, VIX, international ETFs) |
| Windows | 189, 252, 378, 504 trading days |
| correlation-hrp net Sharpe | 0.411, 0.427, 0.407, 0.383 |
| Equal weight / inverse variance (252 d) | 0.364 / 0.388 |
| Skeleton effect | -0.019 (0.21), -0.028 (0.056), -0.003 (0.80), -0.008 (0.56) |
| Orientation, allocation step | +0.015 (0.22), +0.011 (0.35), +0.009 (0.45), +0.021 (0.051) |
| Orientation, both steps | +0.016 (0.31), +0.007 (0.61), +0.015 (0.33), +0.024 (0.086) |
| Total (semcov-hrp − correlation-hrp) | -0.004 (0.82), -0.017 (0.36), +0.006 (0.73), +0.013 (0.44) |
| lw-hrp, defactored-hrp, pcorr-hrp levels | 0.389/0.394/0.382/0.356; 0.411/0.409/0.425/0.402; 0.372/0.410/0.399/0.382 |
| semcov-hrp − defactored-hrp | -0.004, +0.001, -0.011, -0.006 |
| pcorr-hrp − skeleton-hrp | -0.020 (0.073), +0.012 (0.23), -0.005 (0.62), +0.007 (0.48) |
| Ridge-Granger (252 d) | skeleton 0.386, semcov 0.352, topo 0.393, struct-rp 0.350 |
| VARLiNGAM orientation contrasts | 6 of 12 positive, mean 0.000; semcov − corr = -0.071 (p 0.036) at 189 d |
| HERC orientation / skeleton | +0.088, +0.052, +0.010, +0.049 / -0.258, -0.182, -0.028, -0.076 |
| DYNOTEARS graphs | 614 to 673 edges, 6.3 to 6.9% density, longest path about 50, about 3 roots and 7 leaves, all DAGs |
| VARLiNGAM density | 32%, 15%, 7%, 7% |
| Directional prior | without it, 36 to 48% of driver-asset edge mass on asset-to-driver edges |
| Bootstrap | Politis-Romano, 10,000 resamples, mean block 21 d; excess kurtosis about 15.9 |
| SE and MDE (252 d) | total 0.019 / 0.047; orientation 0.015 / 0.037 |
| Trials | 146 (41 Phase I, 105 Phase II); no DSR reaches 0.95; corr-hrp DSR 0.925 |
| SPA pooled p | 0.82 to 0.83 (vs corr-hrp), 0.40 to 0.45 (vs skeleton-hrp); closest single cell 0.076 (narrowed, 504 d) |
| MCS 90% | keeps 16 of 17, drops HSP as published |
| Turnover per rebalance (252 d) | corr 0.128, skeleton 0.118, semcov 0.110, topo-semcov 0.140, undirected 0.145 |
| Max drawdown (252 d) | about -50% for every HRP variant; -61.9% equal weight |
| OOS (19 rebalances, 2025-01 to 2026-08) | orientation +0.035, +0.050, +0.105, +0.036; skeleton +0.107, +0.071, -0.009, +0.100; total +0.142, +0.122, +0.096, +0.136; SEs 0.06 to 0.14 |
| Bug fix | 5 Sept 2026; baseline +0.020 to +0.046; graph-fed at most 0.005 |
| Phase I | causal selection +0.010 at K = 17 (p 0.27); seed range 0.021 over 20 seeds; HRP 0.427 vs HSP 0.371; closed loop inert |
| Compute | about 4 min per fit at d = 132; 2,051 cached fits; 227 cells; 88 tests |
| Code | github.com/j-muthu/portfolio-optimisation-causal-networks; `reproduction.py` |
