# Pre-committed interpretations: skeleton-channel control (D0pc) and naive anchors (EW, IVP)

Committed BEFORE any of these configurations was run, mirroring
`PREDICTIONS_COVARIANCE_CONTROLS.md`. The covariance controls (D0lw, D0df)
asked whether the *orientation* channel's gain needs a graph. This file asks
the symmetric question for the *skeleton* channel, and adds the naive
anchors the report currently lacks.

## D0pc — graph-free skeleton control

Construction: D0's allocator with the discovered skeleton replaced by a
thresholded partial-correlation matrix. Ledoit-Wolf covariance on the
lookback window, inverted to a precision matrix, converted to partial
correlations; the largest-|rho| cells are kept, density-matched by
nonzero-cell count to the paired DYNOTEARS graph (the graph enters only
through that scalar). Same embedding distance, same sample covariance,
same HRP recursion as D0. No causal discovery anywhere.

Question: does the skeleton channel's gain over CORR-HRP (D0 - CORR,
+0.022 at one year) require the *discovered* skeleton, or does any sparse
conditional-dependence structure of the same density suffice?

Pre-committed interpretations of the D0 - D0pc residual per window
(bootstrap p at the usual 10,000/block-21 protocol):

1. **D0pc ~ D0 (residual within ±0.01, p > 0.1 at every window):** the
   skeleton channel does not require causal discovery; a
   partial-correlation sparsification of the same density reproduces it.
   The skeleton half of the decomposition is then read as "sparse
   conditional-dependence clustering", not as a causal-discovery
   dividend, and the report's mechanism story becomes symmetric: neither
   channel's gain needs the causal machinery. This is the deflationary
   outcome and will be reported as such.
2. **D0 > D0pc (positive residual, at least one window p < 0.05, no
   negative-signed window):** the discovered skeleton carries allocative
   structure beyond generic partial-correlation sparsification; the
   skeleton channel earns its causal-discovery label.
3. **D0pc > D0 (negative residual anywhere with p < 0.05):** the
   discovered skeleton is *worse* than a cheap statistical skeleton;
   the skeleton channel's gain is then attributed to sparsification
   alone and DYNOTEARS' particular edges subtract value.
4. **Mixed signs across windows without significance:** inconclusive;
   reported as bounding neither reading, with the point estimates shown.

D0pc joins the deflation universe (it is an evaluated trial) and neither
SPA family, exactly as D0lw/D0df.

## EW and IVP — naive anchors

Construction: equal weight (1/N) and inverse-variance weights over the
identical universe, calendar, holding period and cost model. Graph unused.

Purpose: situate the absolute Sharpe levels (0.36-0.42) against the
cheapest possible allocators. These are anchors, not treatments; whatever
their level, no claim in the thesis changes, but the reader can see
whether the whole hierarchical family clears 1/N on this universe. Both
join the deflation universe and neither SPA family.

Predictions (weakly held, for the record): on a large-cap long-only
universe 1/N is expected to be competitive on raw Sharpe with the
hierarchical family (the 1/N literature) while carrying higher turnover
at rebalance parity; IVP is expected between EW and CORR-HRP. If EW
matches or beats the graph allocators on net Sharpe, the honest reading
is that the entire family's absolute edge on this universe is small, and
the thesis's claims (which are about *contrasts within* the family)
inherit that context explicitly.
