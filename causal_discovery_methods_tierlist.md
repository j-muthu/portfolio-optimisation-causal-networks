# Causal Discovery Methods Tierlist for Causal Portfolio Optimisation

**Evaluated against**: Thesis on whether causal discovery methods improve portfolio optimisation versus correlation-based approaches, with emphasis on regime changes, time-series data, and practical integration with portfolio construction (including HRP with causal graph matrices).

**Supervisor priorities**: (1) Are causal discovery algos useful for portfolio optimisation? (2) Can we improve on causal discovery algorithms? (3) Use causal structure to adjust portfolio weights. (4) Regime change detection via causal structure. (5) Comparison/evaluation platform. (6) HRP with causal graph matrix insertion.

---

## TIER S — Core Methods (implement and compare these)

### DYNOTEARS (Pamfil et al., 2020)
**What it is**: Time-series extension of NOTEARS. Models the data as a structural vector autoregressive (SVAR) system X = XW + Y₁A₁ + ... + YₚAₚ + Z, where W captures contemporaneous causal effects and each Aᵢ captures lagged causal effects at lag i. The acyclicity constraint h(W) = tr(e^(W∘W)) - d = 0 is applied only to W, since temporal ordering already prevents cycles in the lagged relationships. Solved via augmented Lagrangian with L₁ sparsity penalty.

**Why Tier S**: This is your primary causal discovery method. It's the only well-established algorithm purpose-built for time-series causal discovery that outputs explicit weighted adjacency matrices — exactly the format you need for inserting into HRP or any other portfolio optimisation pipeline. Howard et al. have already demonstrated its application to S&P 500 factor investing. Your supervisor specifically names it. The weighted adjacency matrix W can be directly substituted for a correlation matrix in hierarchical clustering (your supervisor's HRP idea), and the edge weights provide causal effect magnitudes, not just directions.

**Pros**: Natively handles time series. Outputs weighted directed adjacency matrix. Doesn't require faithfulness assumption. Handles contemporaneous and lagged effects simultaneously. Proven scalable to ~500 assets (Howard et al.). Established codebase available.

**Cons**: Assumes linear structural equations (misses nonlinear regime-dependent dynamics). Non-convex optimisation with no global optimality guarantee. The acyclicity constraint h(W) can create numerical instability for large d. Howard et al. found interslice (lagged) weights are almost always zero for daily stock returns, which raises questions about whether the temporal modelling is actually adding value over static NOTEARS in the financial setting. Your supervisor specifically flags "hidden assumptions behind DYNOTEARS" — the linearity assumption and the choice of L₁ regularisation strength λ both embed substantive modelling choices that affect the recovered graph.

**Your angle**: DYNOTEARS is the method you test, compare, and potentially improve upon. The comparison targets are RCCP (correlation-based), other causal methods from this tierlist, and traditional methods like GICS industry classification.

---

### VARLiNGAM (Hyvärinen et al., 2010)
**What it is**: Time-series extension of LiNGAM (Linear Non-Gaussian Acyclic Model). LiNGAM exploits a key identifiability result: if the structural equations are linear and the exogenous noise terms are non-Gaussian, then the causal DAG is uniquely identifiable from observational data — you can recover the exact DAG, not just a Markov equivalence class. VARLiNGAM extends this to a VAR (vector autoregression) framework, modelling both contemporaneous and lagged effects just like DYNOTEARS, but using independent component analysis (ICA) rather than continuous optimisation to recover the causal structure.

**Why Tier S**: Your supervisor explicitly names it alongside DYNOTEARS ("dynotears, valingam?"). It solves the same problem as DYNOTEARS but via a completely different algorithmic approach — ICA-based decomposition rather than score-based continuous optimisation. This makes it the ideal head-to-head comparison partner. Importantly, VARLiNGAM can uniquely identify the DAG (not just an equivalence class) under its assumptions, while DYNOTEARS under purely Gaussian assumptions can only identify up to an equivalence class. Since financial returns are well-known to be non-Gaussian (heavy tails, skewness), VARLiNGAM's non-Gaussianity assumption is actually *better suited* to financial data than DYNOTEARS's implicit Gaussianity in the least-squares loss.

**Pros**: Uniquely identifies the DAG (not just equivalence class) under non-Gaussianity, which holds for financial returns. Handles time series via VAR framework. No acyclicity constraint needed — the ICA decomposition naturally recovers a causal ordering. Well-established theoretical foundations. The non-Gaussianity assumption is a *strength* in finance, not a limitation.

**Cons**: Assumes linear structural equations (same limitation as DYNOTEARS). ICA can be sensitive to the number of components and initialisation. Less scalable than DYNOTEARS for very high-dimensional settings (hundreds of variables). Assumes no hidden confounders, which is unrealistic for financial data where unobserved macro factors drive multiple assets.

**Note**: VARLiNGAM is not discussed in the Kaddour et al. survey (it's in the broader causal discovery literature), but your supervisor names it, and it's arguably the most natural comparison method for DYNOTEARS in a financial setting.

---

## TIER A — Strong Candidates (implement one or two as key comparisons or extensions)

### NTS-NOTEARS (Sun et al., 2023) ~~and GraphNOTEARS (Fan et al., 2023)~~
**What they are**: These are recent NOTEARS extensions specifically referenced in Howard et al.'s paper on causal network representations. NTS-NOTEARS extends NOTEARS to handle nonparametric dynamic Bayesian networks — it learns non-linear time-varying causal structures without specifying a parametric functional form, using prior knowledge constraints to guide the discovery. ~~GraphNOTEARS extends the framework to handle graph-structured data and higher-order interactions.~~

**Why Tier A**: These represent the current frontier of the NOTEARS family and are explicitly positioned as alternatives to DYNOTEARS in the financial causal discovery literature. If your supervisor wants you to explore "different causal discovery approaches for time series/temporal data," these are the most directly comparable alternatives that stay within the score-based continuous optimisation paradigm. Including them in your evaluation platform would strengthen the comparison significantly.

**Pros**: Directly comparable to DYNOTEARS (same optimisation framework). NTS-NOTEARS handles nonparametric relationships and time-varying structure. Already cited in the financial causal discovery literature. Represent the state-of-the-art within the NOTEARS family.

**Cons**: Newer methods with less established codebases and less community testing. NTS-NOTEARS's nonparametric flexibility comes with increased computational cost and potential overfitting risk. May require more careful hyperparameter tuning than DYNOTEARS.