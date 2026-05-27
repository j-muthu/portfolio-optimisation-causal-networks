## Causal discovery: concepts

### Interventions: $P(Y \mid \text{do}(X))$

- $\text{do}(X = x)$ means *force* $X$ to value $x$, breaking all incoming arrows to $X$. Distinct from *observing* $X = x$.
- $P(Y \mid \text{do}(X)) \neq P(Y \mid X)$ in general — observing $X = x$ also gives info about $X$'s parents, while intervening overrides them.
- Strategy in causal inference: apply do-calculus rules to rewrite $P(Y \mid \text{do}(X))$ as an expression in conditional probabilities only — the effect is then **identifiable** from observational data.

### d-separation

Whether information can flow between two nodes in a DAG **given what you condition on**.

- If yes: nodes are **d-connected** (assumed to be statistically dependent under faithfulness).
- If no: nodes are **d-separated** (independent under CMC).

Three local structures around a middle node $B$ on a path $A - B - C$:

| Structure | $A \to B \to C$ (chain) | $A \leftarrow B \to C$ (fork) | $A \to B \leftarrow C$ (collider) |
|-----------|------------------------|-------------------------------|-----------------------------------|
| Default   | d-conn                  | d-conn                         | d-sep                              |
| Condition on $B$ | d-sep            | d-sep                          | d-conn ("explaining away")         |

Collider example: $A$ = break-in, $B$ = car alarm, $C$ = earthquake. Marginally $A \perp C$. Conditional on $B$, learning $A$ tells you about $C$ (one explains the other away).

→ Crucial for finance: lots of "opposite-moving paths" exist (e.g. supply chain effects, hedging flows). Reading independence in data as "no causal connection" can wreck a recovered graph. PC-style algorithms suffer most here.

### Do-calculus (Pearl)

Three rewrite rules that strip $\text{do}(\cdot)$ from a target query until you're left with a purely observational expression (or stuck — non-identifiable without experiments).

- **Rule 1 (insert/delete observation):** if $W$ is d-separated from $Y$ given the rest in the manipulated graph, $P(Y \mid \text{do}(X), Z, W) = P(Y \mid \text{do}(X), Z)$. Drop irrelevant observations.
- **Rule 2 (action/observation exchange):** under certain d-separation conditions, $P(Y \mid \text{do}(X), \text{do}(Z)) = P(Y \mid \text{do}(X), Z)$. When back-door paths are blocked, intervening = observing.
- **Rule 3 (insert/delete action):** under certain conditions, $P(Y \mid \text{do}(X), \text{do}(Z)) = P(Y \mid \text{do}(X))$. If $Z$ has no causal path to $Y$, intervening on $Z$ is inert.

Punchline: iterate until either every $\text{do}(\cdot)$ vanishes (identifiable) or you're stuck (need experiment). Rodriguez-Dominguez's $P(a_i \mid \text{do}(\mathbf{D}))$ in the commonality theorems is exactly this kind of interventional quantity — "force the drivers to particular values".

### Causal Markov Condition (CMC)

Every variable independent of its non-descendants given its parents:
$$
P(a_1, \dots, a_N \mid \mathbf{D}) = \prod_{i=1}^N P(a_i \mid \mathbf{D})
$$
if the $a_i$ all have parent set $\mathbf{D}$. Parents screen off ancestors. Two variables d-separated in the DAG ⇒ independent in the data.

In causal discovery: minimise **residual pairwise dependence** after conditioning on putative parents.

### Faithfulness (converse of CMC)

If $A, B$ are independent in the data, they are d-separated in the DAG. I.e. **the only independencies in the data are the ones the graph structurally implies** — no accidental cancellations of opposing paths.

Easy failure mode: $A \xrightarrow{+} B \xrightarrow{+} C$ and $A \xrightarrow{-} C$ — effects can cancel and faithfulness rules that out by assumption. Strong faithfulness (Zhang & Spirtes 2002) tightens this: d-connected pairs must have correlation above some threshold $\lambda$.

→ Uhler et al. (2013) show this is *restrictive* for most DAGs. Algorithms that need it (PC and most constraint-based) are less credible in finance precisely because counter-acting paths are commonplace. Score-based methods (e.g. Van de Geer & Bühlmann 2013, NOTEARS) don't need it.

### Markov equivalence class

Set of DAGs encoding the same conditional independencies — observationally indistinguishable from each other. Constraint-based algorithms recover the *equivalence class* (CPDAG), not a specific DAG.

- **Unshielded colliders** ($A \to B \leftarrow C$ with no direct $A$–$C$ edge) **are identifiable** from observational data alone: statistical signature is $A, C$ marginally dependent (correlate), independent conditional on anything *except* $B$, dependent conditional on $B$.
- Once unshielded colliders are oriented, more edges can be oriented by extension rules.
- Prone to statistical mistakes since only submodels are compared.

DYNOTEARS sidesteps this by using **temporal ordering** (cause before effect) to orient lagged edges — identifies a specific DAG, not an equivalence class.

### Algorithm families

- **Constraint-based** (PC, FCI, GFCI): conditional independence tests + orientation rules. Test marginal indep for all pairs, then condition on 1 var and test, then 2 vars, ... until no test passes. Orient edges from unshielded colliders, propagate. Best for cond. indep. tests + inferring d-sep. Needs faithfulness; runtime exponential in nodes. FCI handles latent confounders; PC assumes causal sufficiency.
- **Score-based** (GES, NOTEARS, DYNOTEARS, etc.): $\max P(G \mid \text{Data})$ — maximise posterior over graph (BIC, BGe, regularised likelihood). GES starts from empty graph and adds/removes edges greedily. NOTEARS recasts the acyclicity constraint $h(W) = \text{tr}(e^{W \circ W}) - d = 0$ as smooth/continuous so standard solvers work. Does *not* require faithfulness.
- **Functional/SCM-based** (LiNGAM, VARLiNGAM): exploit non-Gaussianity to identify a unique DAG. Two-stage: fit reduced-form VAR, apply ICA-based LiNGAM to residuals. Non-Gaussianity is a *strength* in finance (heavy tails).

Common assumption to be wary of: **causal sufficiency** — all common drivers observed. In financial markets where macro factors, sentiment, hidden flows etc. are exogenous and often unobserved, this is suspect. FCI and PAG-returning algorithms try to relax it.