## Howard, Lohre & Mudde (2025): Causal Network Representations in Factor Investing

*Robeco/Lancaster paper, ISAF 2025. The closest existing work to your thesis: applies DYNOTEARS to S&P 500, builds three investment use-cases.*

### Motivation

Factor investing has leaned on correlation-based methods for peer groups, factor construction, network analysis. Two well-known criticisms:

- Correlation **doesn't condition on anything** → spurious associations (Simon 1954), overfitting, false factor discoveries (López de Prado 2023).
- Correlation is **symmetric and static** → can't tell direction of influence, can't capture lead–lag, can't track regime changes.

Causal discovery offers: directional, conditional-on-everything, lead–lag-aware structure. Howard et al.'s claim is *not* that causal beats correlation everywhere — it's that causal **complements** correlation, especially where direction or dynamics matter.

*Your one-line summary of the abstract:* causal networks **don't consistently beat traditional approaches, are computationally complex, and aren't very interpretable**. That's the honest read — the paper is making a softer "complement, not substitute" claim. Worth remembering when framing your own contribution: surpassing correlation isn't the bar, but adding interpretable structure that correlation misses is.

*Your framing of the project:* financial markets *as* networks → modelling that network → correlation is the wrong tool for that job. Then: causal identification (e.g. DYNOTEARS) is the right tool. This is the cleanest version of the motivation — useful for the thesis intro.

*On factor investing:* factors = specific drivers of asset returns → rules-based investing approach. Examples = the Fama–French stable (MKT, SMB, HML, UMD, RMW, CMA, etc.).

### Theoretical framework (their Sec. 2.3–2.4)

- DAG over $p$ random variables $X$ is a **Bayesian network** if $P(X) = \prod_{i=1}^p P(X_i \mid \text{Pa}(X_i))$ — the **local Markov condition**. Each $X_i$ depends only on its parents and is independent of everything else conditional on them.
- **Faithfulness:** distribution $P$ is faithful w.r.t. DAG $D$ iff *all* conditional independencies are encoded by $D$. Equivalent formulation via d-separation. This is the *additional assumption* on top of CMC.
- **Strong faithfulness** (Gaussian case, Zhang & Spirtes 2002): for $\lambda \in (0,1)$,
$$
\min \{ |\text{corr}(X_i, X_j \mid X_S)| : j \text{ not d-separated from } i \mid S \} > \lambda
$$
i.e. d-connected pairs must show correlation above $\lambda$.
- PC-style constraint methods require strong faithfulness. *Your verdict:* **any algorithm relying on this is weak** — exactly because in financial markets you have lots of opposite-moving paths that can produce d-connected pairs with arbitrarily small marginal correlation. PC is also exponential in nodes and assumes causal sufficiency, which is bad given factor structure.
- They pick **score-based methods**, specifically the NOTEARS / DYNOTEARS line, which (a) doesn't need faithfulness, (b) is tractable via standard solvers thanks to Zheng et al.'s (2018) smooth acyclicity reformulation. Score-based methods (BIC, regularised likelihood) are slow when the acyclicity constraint is imposed combinatorially — Zheng et al. **recast the constraint as smooth + continuous + exact**, which is why it's tractable now.

### DBN → SVAR setup

Standard BNs ignore time. **Dynamic Bayesian networks (DBNs)** include lagged variables; edges split into:

- **Intra-slice edges** $W$: contemporaneous relationships between assets at time $t$.
- **Inter-slice edges** $A_1, \dots, A_p$: lagged relationships from $t - k$ to $t$.

*Your compact framing of the model choice:* DBN → formulate as **SVAR**. You could also use a **time-varying-parameter SVAR (TVP-SVAR)** — and only the intra-slice weights $W$ need to be acyclic (the lagged $A_i$ point forward in time, so they can't create cycles by construction). The paper notes TVP-SVAR as an alternative formulation in a footnote but doesn't pursue it; this is one of the cleanest "improve on the algorithm" directions for the thesis.

### DYNOTEARS (Pamfil et al. 2020), as used by Howard et al.

Models the panel as a SVAR:
$$
X = XW + Y_1 A_1 + \dots + Y_p A_p + Z
$$
- $X$ is $n \times d$ (samples × assets), $Y_i$ are lagged versions, $Z$ is noise.
- Acyclicity constraint applies **only to $W$** (the contemporaneous block) because the $A_i$ point forward in time and can't create cycles:
$$
h(W) = \text{tr}(e^{W \circ W}) - d = 0.
$$
- Optimisation problem:
$$
\min_W \tfrac{1}{2n} \|X - XW\|_F + \lambda_W \|W\|_1 \quad \text{s.t. } W \text{ is acyclic}
$$
- Solved with augmented Lagrangian + L-BFGS-B; hyperparameters $\lambda_W, \lambda_A$ (regularisation) and $\tau_W, \tau_A$ (threshold small weights to zero). *Your question:* **can you tune these?** Howard et al. don't (computational cost), but if compute allows, this is one of the most impactful design choices.

### Howard et al.'s implementation choices

- **Universe:** S&P 500 constituents, Dec 1989 – Dec 2022 (results from Jan 1993, first 3 years for calibration). Avg ~357 stocks per network (only stocks with full estimation-window history).
- **Lag $p = 0$.** They observe that off-diagonal inter-slice weights are almost always ~zero at all estimation points → contemporaneous-only model. This matches the Pamfil et al. observation for S&P 100. **Important caveat** for our work: this argues against DYNOTEARS *as a temporal model* for daily/monthly returns — but the temporal modelling may matter more at different frequencies or with macro vars included.
- **Pre-processing:** log-returns, then z-score each asset (mean 0, var 1). Without this, the L1 regulariser would systematically prefer low-vol stocks.
- **Rolling window:** 4 years, 1-month increment. Permits time-varying network. *Methodological question:* **could you use TVP-SVAR instead?** This would avoid the abrupt re-estimation that rolling windows produce — though Howard et al. note feature counts may not justify the parameter explosion of TVP. So the answer depends on whether you've reduced the universe to a small enough set of representative stocks/sectors first.
- **Reg parameter $\lambda_W = 0.1$.** Held fixed (computationally infeasible to tune per window). They borrow the value from Pamfil et al.'s grid search.
- **Cluster step:** apply **node2vec** to the DYNOTEARS graph to embed nodes, then standard clustering on embeddings. *Their pipeline summary:* cluster a causal-network graph. Node2vec uses hyperparameters embedding dimension, walk length, return parameter $p$, and in-out parameter $q$ ($p$ = probability of revisiting just-visited node; $q$ = exploration vs. staying close incentive). Embeddings then fed to K-means.

### Three experiments

#### Experiment 1: Peer group neutralisation

*Your unpacking of what peer-group neutralisation does:*
- Split firm characteristics into across-industry + within-industry components.
- Ignore the across-industry component because firms within a given industry have very similar characteristics, so an overall portfolio would over-weight specific whole industries rather than picking the best firms within them.
- Just focus on within-industry components.

Compare three peer-group selection methods for cross-sectional long–short factor strategies:

1. **GICS** (baseline industry classification)
2. **DYNOTEARS** + node2vec + clustering
3. **Statistical clustering (SC):** hierarchical clustering (Ward's) on PCA-transformed returns, $K$ set to GICS sector count

*Why causal beats GICS conceptually:* both are doing the same job — **classifying stocks that behave similarly into peer groups** — but causal networks can do it data-adaptively without committing to a static industry taxonomy. So this experiment is **identifying peer groups causally, not sectorally.**

12 firm characteristics: 12-1M momentum, 1M reversal, beta, B/P, cash-to-assets, E/P, EBITDA/EV, 1y-fwd E/P, gross profitability, residual 12-1M momentum, ROE. Quintile long–short within each peer group; both EW & VW versions; 1M and 12M holding periods.

**Findings:**
- Causal-based peer groups often deliver **higher Sharpe ratios** than GICS or SC, primarily via better *stock selection* (not lower vol). Sharpe ratio = $(R_p - R_f) / \sigma_p$, i.e. excess return per unit of risk — so a higher SR through stock selection rather than vol reduction is the "harder" win.
- **GICS still wins on volatility reduction** in many cases — different schemes optimise different things.
- The advantage is **context-dependent**: causal helps for characteristics with clear *inter-firm transmission* (momentum, value) but less for firm-specific attributes (quality).
- No-neutralisation strategies generally have higher vol → suggests these characteristics carry unpriced industry exposure.

#### Experiment 2: Low-centrality factor

- For each month, compute **eigenvector centrality** of each stock in the DYNOTEARS causal graph. (Choice motivated by Dablander & Hinne 2019, who argue eigenvector centrality is best-suited for causal networks vs. degree/closeness/betweenness which "are a poor substitute for causal inference".)
- Construct value-weighted quintile long–short: **long the most peripheral stocks**, short the most central. *I.e. long on low-centrality stocks.*
- Rebalanced monthly.

*Why low-centrality stocks should be desirable:* **high-centrality stocks are undesirable because they carry more risk during market crashes — less diversified, more systemic exposure.** This is the intuition for shorting them.

**Findings:**
- **Centrality factor has high $\alpha$** **[m]**. Significant alphas vs CAPM (2.21%), FF4 (2.78%), FF6 (5.56%), q5 (5.02%) — annualised. The factor provides **additional explanatory power beyond standard factor models** **[m]**.
- But Sharpe ratio is *negative*. Reconciliation: factor loads negatively on MKT and HML/IA — both carry positive risk premia, so unconditional return is dragged down even though alpha is positive.
- **Cyclical performance:** best before NBER recessions, worst after them. Looks like a growth-style factor (negative HML loading of −0.34 in FF3). *On the value/growth definition:* a growth stock = an inflated valuation, opposite of a value stock (low B/P).
- **Inverse implication for central stocks:** since the factor is long-peripheral / short-central, reversing the loadings tells us **central stocks are large, value-heavy companies (high book-to-price)** **[m]**. This aligns with Buraschi & Porchia (2012).
- **Strong stable-regime alpha, weak unstable-regime alpha** → suggests it acts as a *cheap hedge* in calm periods but not a free hedge (significantly underperformed 2020–2023).

#### Experiment 3: Market timing via network density

- Indicator $d_t$ = average eigenvector centrality of S&P 500 stocks at $t$. Use *change* in density rather than level (structural breaks make level uninformative). *Your one-liner:* **network density reflects risk + return (and timing of crashes).** That's the whole hypothesis.
- Bivariate predictive regression:
$$
r_{t+1} = \alpha + \beta x_t + \varepsilon_{t+1}
$$
where $r$ is S&P 500 excess return.

**Findings:**
- Bivariate: $R^2 = 0.75\%$ ($R^2_{OS} = 0.55\%$, sig. at 10%), CER gain 1.83%. *In short: weak correlation* — the $t$-stat on $\hat\beta$ is −1.62 in the bivariate spec, only just significant at 10%. The result strengthens substantially in the multivariate spec.
- Multivariate (controlling for 21 Welch–Goyal + Hammerschmid–Lohre predictors collapsed to 3 PCA factors): $R^2 = 5.98\%$, $R^2_{OS} = 3.40\%$ (sig. at 5%), CER gain 8.16%. Construction: **regress returns on density alongside the controls** — the density indicator adds predictive content beyond what the standard predictors capture.
- Sign is *negative*: increase in network density → lower expected returns. Consistent with Kaya (2015), Lenzu & Tedeschi (2012): denser networks ⇒ more systemic risk.
- Comparable to or better than established technical/macro indicators (Neely et al. 2014 reported $R^2_{OS}$ 0.44–0.88% for technicals).
- Likely concentrated in recessions: density is a good recession indicator, and Henkel et al. (2011) show return predictability is concentrated in recessions.

*On the CER metric:* this is the **certainty-equivalent return** — what an investor would be indifferent between (a) getting risk-free vs. (b) using the predictive model. The **CER gain** = the annual management fee the investor would pay to access the density forecast instead of the historical-average forecast. It's the natural economic benchmark for a market-timing indicator.

### Conclusions (theirs)

- Causal discovery (specifically DYNOTEARS) is a useful **complement** to correlation methods, not a clear superior.
- Peer-group selection: causal often higher Sharpe via stock selection; GICS still better for vol reduction. *Combine methods.*
- Centrality has economic content: low-centrality is a hedge-like factor with significant alpha vs FF/q5.
- Network density times the market reasonably well.

### Limitations they acknowledge

- DYNOTEARS is **computationally heavy** at scale — hyperparameter tuning per window is infeasible.
- Only one causal discovery algorithm tested. No horse race against alternative causal methods.
- Interpretability of recovered graphs is hard for practitioners.

---

## Hooks back to the thesis

The Howard et al. paper is the closest existing application of DYNOTEARS to a large equity universe and ratifies several of our design choices, but it leaves the HRP integration completely open. Specific implications:

- **Lag $p = 0$ for returns** is now empirically supported on two universes (S&P 100 in Pamfil, S&P 500 in Howard). For the v1/v2/v3 HRP pipelines, this means the causal adjacency matrix $W$ is effectively contemporaneous → simplifies the symmetrisation step. *But:* this is for log-returns at monthly/daily frequency on returns alone. Including macro vars or moving to higher-freq data could resurrect the lagged structure. Worth keeping VARLiNGAM in the pipeline as a robustness check since it explicitly models lags and uses non-Gaussianity (a known property of returns).
- **Faithfulness is the right thing to worry about** — and is exactly why constraint-based methods are off the table. Score-based (DYNOTEARS) and SCM-based (VARLiNGAM) families are the defensible choices.
- **Centrality as a node feature** (their Experiment 2) is an alternative or complement to the row-vector embedding $\boldsymbol{e}_i = [\boldsymbol{e}_i^{\text{out}}, \boldsymbol{e}_i^{\text{in}}]$ used in v1. Could swap in eigenvector centrality as the feature vector for the distance metric and see if HRP-with-causal-centrality differs meaningfully from HRP-with-row-embedding.
- **Regime framing is independently validated** — they find centrality alpha-add concentrates in stable regimes and density predicts recessions. Both are direct evidence that causal network structure carries regime information that correlation misses. Aligns with our planned NBER-conditional evaluation.
- **Gap they leave:** they don't use the causal graph to build portfolio weights directly. Node2vec → clustering → peer groups → factor neutralisation is several steps removed from "use $W$ to set $w$". The HRP integration (v1/v2/v3) is therefore genuinely novel relative to their work and a clean contribution.

### Open methodological questions raised

- **Commonality principle compatibility.** Howard et al. don't engage with Rodriguez-Dominguez's framework. Do DYNOTEARS-derived networks satisfy something like the commonality principle? The fact that low-centrality stocks have positive alpha but negative Sharpe suggests *no* — the network features aren't capturing all the systematic risk, otherwise loadings on MKT/HML wouldn't be doing the work.
- **node2vec vs. direct row embedding.** They embed via node2vec before clustering — adds a learned representation step on top of the causal graph. The v1/v2/v3 plan uses direct row vectors. Comparing the two could be an ablation.
- **$\lambda_W$ fixed at 0.1.** Their justification is computational, not statistical. If we can afford a sweep, this is the single most impactful hyperparameter on graph sparsity and therefore everything downstream.
- **Hidden assumptions behind DYNOTEARS** (supervisor's phrase): linearity, Gaussian noise (implicit in least-squares), single fixed $\lambda$, contemporaneous-only model, sliding window with abrupt re-estimation rather than smooth evolution. Each is a plausible target if we're trying to "improve on the causal algorithm".

---

## From the marginalia: standalone framings to keep

These are the punchy reformulations from your pen annotations that are worth preserving as standalone framings — useful when writing the thesis intro, methodology section, or defending the contribution. Loosely ordered by where they fit in a thesis structure.

**On the project's premise:**

> Financial markets *as* networks → modelling that network → correlation is the wrong tool for that job. Causal identification (e.g. DYNOTEARS) is the right tool.

**On the modesty of existing claims:**

> Causal networks don't consistently beat traditional approaches, are computationally complex, and aren't very interpretable.

(Useful for honest framing: this is the bar Howard et al. set, and your contribution is improving on each of these axes by integrating with HRP rather than competing as a standalone classifier.)

**On peer-group construction:**

> Classify stocks that behave similarly into peer groups. Causal networks can do this — identifying peer groups *causally*, not sectorally.

**On the centrality finding:**

> High-centrality stocks are undesirable: they carry more risk during market crashes, less diversified. Central stocks are large, value-heavy companies (high B/P). Low-centrality factor = long peripheral / short central = high alpha but negative Sharpe due to MKT/HML loadings.

**On the timing finding:**

> Network density reflects risk + return (and timing of crashes). Weak bivariate correlation, stronger multivariate effect when controlling for standard predictors.

**On methodological extensions you flagged:**

- **Could you use TVP-SVAR** instead of rolling-window DYNOTEARS? (Avoids abrupt re-estimation; would let weights evolve smoothly rather than jump every month. Trade-off: parameter explosion unless universe is reduced.)
- **Can you tune $\lambda_W$, $\lambda_A$, $\tau_W$, $\tau_A$?** (Howard et al. don't; they fix $\lambda_W = 0.1$ and borrow $\lambda_A$ likewise. Most impactful single hyperparameter on downstream behaviour.)
- **Compare selection methods** for the same downstream task (peer groups, distance matrices, etc.) — Howard does this for sectors but not for portfolio weighting.