Split asset return into systematic risk and idiosyncratic risk.
- To minimise systematic risk, you use the factor model, which specifies explicit external variables or factors. Each asset has a vector of betas, which are sensitivity to each factor.
- For idiosyncratic risk, you reduce risk by diversification. Give it a covariance matrix of asset returns. It gives weights that minimise portfolio variance for a given expected return.
Factor model and diversification trade off against each other.

Rodriguez Dominguez: iff factors satisfy the commonality principle, then both types of diversification can exist with no trade-off, because a conformal map (preserves angles and hence correlation and clustering structure) exists between the unconditional return space and the sensitivity space.
Do the structures from dynotears and VARLiNGAM actually satisfy something like the commonality principle? Otherwise, we still get this trade-off.

Commonality principle: the optimal drivers for a portfolio are the specific drivers most frequently selected across assets, both in terms of persistence (remains driver over meaningful time horizon) & probability of causality

Unconditional return space vectors are assets' expected returns as a vector of returns at different time points.
- Dimensions are those time points
- Angles between two vectors yield the correlation coefficient between two assets.
Sensitivity/beta space dimensions are common drivers. 
Each asset is a vector of expected returns with respect to each driver:
$$\hat{\beta}_i = (\frac{\partial E[r_i | \tilde{CD}]}{\partial CD_1}, \frac{\partial E[r_i | \tilde{CD}]}{\partial CD_2}, \dots)$$
Expected change in return, conditional on all other drivers, when you change the kth common driver cd_k.

Conditional return space: expected returns conditional on common drivers at different points in time (dimensions are time points)_.

Rodriguez-Dominguez: a sensitivity space with idiosyncratic and systematic diversification both happening. Sensitivities of coordinates: portfolio optimisation based on distance matrix in this sensitivity coordinate system (whether assets respond similarly to factors). 
HSP selects top k common drivers based on highest cumulative correlation with all constituents of portfolio (assets) over a window. 

P[a and b|d] = P[a|d] * P[b|d], as d screens off correlation, so d is a probabilistic common cause. 

Dynotears & Varlingam don't need driver selection. We get the causal relationship between assets themselves. We don't need to curate drivers, but the causal structure is endogenous to the asset set, with no exogenous causes.

For each asset, train a feed-forward neural network mapping driver vectors to asset return, then get sensitivity vectors, then average over the training window, gives a single point for each asset. 
- FFNN predicts sensitivity vector per asset

sensitivity distance matrix 
- symmetric by consturction
- not necessarily PSD (M PSD iff x^T M x >= 0 for all x \in R^n) -> have to correct to nearest PSD

hierarchical clustering - single linkage clustering to sensitivity matrix (or PSD vers) -> dendrogram as in HRP

recursive bisection of dendrogram (starting from top, bifurcate into 2 clusters)
- inverse variance weights of each cluster w_k = diag(V_k)^(-1) / tr(diag(V_k)^(-1))

V_k is covariance matrix of assets' returns in cluster k
variance of cluster k = w_k^T V-k w_k

allocate between both clusters in inverse proportion to variance
alpha_1 = 1 - var_1 / (var_1 + var_2), alpha_2 = 1 - alpha_1
then scale w_k <- w_k * alpha_k
-> weight calc is identical between HRP + HSP

HSP: sensitivity. space tells who you clusters w/ whom
return covar tells you how risky each cluster is

dynotears/varlingam output: lagged causal adjacency matrix
-> to put it into HRP recursive bisection + var weighting, it needs to be symmetric

each asset could be a row vector of causal links
A_{i,:} (outgoing edges) or A^T_{:,i} (incoming edges) or concatenation, and then compute D_{ij} for all i, j
Treat each row of M as a feature vector for that asset:
\boldsymbol{e}_i^{\text{out}} = M_{i,:} = (M_{i,1}, M_{i,2}, \ldots, M_{i,N})
which captures asset i's outgoing causal effects. 
Or each column for incoming:
\boldsymbol{e}_i^{\text{in}} = M_{:,i}^\top = (M_{1,i}, M_{2,i}, \ldots, M_{N,i})
or the concatenation:
\boldsymbol{e}_i = [\boldsymbol{e}_i^{\text{out}}, \boldsymbol{e}_i^{\text{in}}] \in \mathbb{R}^{2N},
giving each asset a 2N-dimensional embedding that contains its full causal signature. 

Then define the distance matrix as
D_{ij} = \|\boldsymbol{e}_i - \boldsymbol{e}_j\|_2

cleaner than alternatives (graph-edit dist, 1/2*(A+A^T), shortest path dist)

HSP: commonality principle iff optimal diversification (conformal mapping)
- depends on drivers being generallly causal
- but Rodriguez Dominguez uses comulative correlation to select drivers!
-> DY & VA are more causal but have restrictive assumptions

regime-change framing:
- correlation-based distances collapse in crises
- but sensitivity-based distances retain structure

use multiple hyperparam configurations per backtest
use causal rather than correlational driver selection step
single-linkage clustering fragile to outliers
- use multiple linkage methods
regime conditional

do we want work to lie inside HSP's sensitivity space paradigm
- if yes, DY/VA are driver-discovery tools
- if no, work on outputted causal graph directly

1. clustering stage (build linkage tree, quasi-diagonalise)
- needs distance matrix (symmetric, non-negative, zero diagonal)
- which assets are similar to which
- hierarchical algos can't run on an asymmetric input

2. allocation stage (recursive bisection w/ inverse var matrix)
- needs covar matrix
- at each split, HRP looks up variance of each sub-cluster, and uses those numbers to set relative weights between the 2 children
- how risky is each cluster?

options for injecting the causal distance matrix:
**v1**:
causal adjacency matrix symmetrised & normalised, treat as similarity matrix and then use it as a distance metric
distance metric used for single linkage clustering
allocation step unchanged
so overall claim - causal structure better for clustering than correlational, but data directly trusted for how risky the cluster is

**v2**:
DY & VA produce SVAR X=XW+(lagged terms)+ε
find the estimated covariance of asset returns from the SVAR (i.e. the returns' covariance conditional on the model being true):
Σ_causal​ = (I−W)^-1 Σ_ε ​(I−W)^−⊤
(note that directionality is maintained)
then compute correlation, convert to distance, build tree, and use it in the bisection step (rather than the sample covar.)
need to convert to nearest PSD.

**v3**:
at clustering stage, take convex combination of causal & correlation distances $d = \alpha \cdot d_{\text{causal}} + (1 - \alpha) \cdot d_{\text{correlation}}$
- so blends v1 distance matrix (causal, symmetrised) w/ standard HRP correlational distance matrix

and at allocation stage do the same with covariances $\Sigma = \alpha \cdot \Sigma_{\text{causal}} + (1 - \alpha) \cdot \Sigma_{\text{sample}}$
- blend v2 structural covariance w/ standard hrp sample covar

sweep over a few different mixing coefficients (0, 0.25, 0.5, 0.75, 1) - note that these coeffs can/should be different over clustering & alloc stages
tests whether statistical & causal info are complementary.

symmetrisation in v1 & v3 means you lose causal directionality
but clustering isn't directional - assets are either in the same cluster or not