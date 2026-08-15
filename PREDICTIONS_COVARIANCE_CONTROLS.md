# Pre-committed predictions: shrinkage and de-factoring controls

Date: 2026-08-15. Written and committed BEFORE either control was run, at
Dr Guo's direction ("Before you run either, write down what each result
would mean for the orientation claim. That is what makes it evidence
rather than a choice."). Drafted with Claude; the interpretations below
are the ones the report will apply whatever the numbers come back as.

## The objection being tested

The covariance-channel orientation gain is measured as D1 − D0: identical
skeleton clustering, sample covariance swapped for the SEM-implied
covariance Σ_SEM. But Σ_SEM differs from the sample covariance in more
ways than direction: it is heavily structured (far fewer effective
parameters, shrinkage-like), and because structural shocks are treated as
independent it also strips the common market factor. A sceptic can
therefore attribute D1 − D0 to generic shrinkage or to de-factoring,
neither of which needs a causal graph.

## The two controls

Both are byte-identical to D0 (embedding-distance clustering, single
linkage, recursive bisection, 5 bps, monthly, DYNOTEARS graphs) except
for the allocation covariance. Neither has any directional content: the
covariance is built from the return window alone, and the distance is
D0's own (invariant to reversing every edge).

* **D0lw** — Ledoit-Wolf shrunk sample covariance
  (`pipeline.portfolio.hsp.ledoit_wolf_covariance`). Isolates "generic
  shrinkage with no graph".
* **D0df** — single-factor residual covariance: each asset's returns are
  regressed on the equal-weight cross-sectional mean of the window
  (the market proxy), and the covariance of the residuals is used.
  Isolates "de-factoring with no graph".

Runs: DYNOTEARS × {D0lw, D0df} × {189, 252, 378, 504}, the standard
Phase-II protocol, cache-fed. The two controls join the DSR deflation
universe (they are evaluated trials) but NOT the family-wise SPA/MCS
candidate sets of either the pre-registered or the reported family: they
are diagnostic controls introduced after both families were fixed, and
they answer a mechanism question, not a selection question.

## Quantity of interest

Per window w, the gap-closure ratio

    rho_x(w) = [Sharpe(D0x, w) − Sharpe(D0, w)] / [Sharpe(D1, w) − Sharpe(D1's anchor D0, w)]

i.e. how much of the D1 − D0 gap the direction-free control reproduces,
alongside stationary-block-bootstrap contrasts D0x − D0 and D1 − D0x
with the same block length as the rest of the study. The windows that
matter most are the extremes (189, 504), where the direction claim and
its U-shape live; current gaps there are +0.021 and +0.023 net Sharpe.

## Pre-committed interpretations

1. **Neither control closes the gap** (rho ≲ 0.5 at both extreme
   windows, and D1 − D0x remains positive there): the generic-shrinkage
   and de-factoring explanations are rebutted directly. The report keeps
   the orientation-as-regulariser mechanism and cites these controls as
   the direct test, replacing the current indirect rebuttal.
2. **D0lw closes the gap** (rho ≳ 0.5 at the extreme windows): the
   covariance-channel gain is not distinguishable from generic shrinkage.
   The report will say so, will stop attributing the D1 − D0 gap to
   directional content, and the orientation claim narrows to (a) the
   ordering channel (D2/D2s vs their anchors) and (b) the cross-method
   contrast (the same construction on VARLiNGAM graphs gains nothing),
   which is weaker and will be stated as weaker.
3. **D0df closes the gap**: same as 2 with "de-factoring" in place of
   "shrinkage"; additionally the report will note that Σ_SEM's
   independent-shock assumption, not the arrows, is then the operative
   ingredient.
4. **Both close it**: the two mechanisms are confounded with direction in
   this design; the covariance-channel claim is withdrawn to a
   descriptive observation and the contribution rests on the machinery
   and the decomposition design, per the agreed contribution sentence.
5. **A control lands between the anchors** (0 < rho < 0.5 with wide
   intervals, the likely case given effect sizes of 0.01–0.03): the
   honest reading is partial confounding. The report will quote rho per
   window, state that direction-specific content explains at most the
   residual share, and demote the mechanism paragraph from "evidence
   against" to "bounded".

In all cases the numbers go in the report as measured, in the same
subsection that currently raises the objection (the "alternative
explanation" paragraph of the evaluation chapter), and the family-wise
and decomposition results are unaffected mechanically (the controls sit
outside both families).

Ancillary prediction, stated for completeness: D0lw is expected to sit
slightly above D0 at the short window (shrinkage helps most when N ≈ T)
and near D0 at one year; if instead D0lw ≫ D0 everywhere, the study's
use of the plain sample covariance in all Phase-II anchors becomes a
limitation worth its own sentence.

---

## Outcome (recorded after the runs, 2026-08-15, same day)

Everything above this line is unchanged from the pre-run commit.

Net Sharpe at 189/252/378/504 days: D0 0.387/0.403/0.403/0.372;
D1 0.408/0.411/0.415/0.395; D0lw 0.384/0.394/0.383/0.353;
D0df 0.405/0.408/0.416/0.406.

* Shrinkage account: FAILS. D0lw sits below D0 at every window;
  D1 − D0lw = +0.018 to +0.042, p = 0.035 at two years.
* De-factoring account: SUCCEEDS. D0df reproduces the D1 − D0 gap at
  every window; residual D1 − D0df = −0.011 to +0.004, p ≥ 0.51
  throughout. Gap-closure rho ≥ 0.6 at all four windows, ≥ 1 at the
  two longest.

Interpretation 3 applies, as pre-committed: the covariance-channel gain
is not attributed to directional content; the operative ingredient of
Σ_SEM is its independent-shock assumption (market-factor removal),
which needs no graph. The report's mechanism paragraph, headline
summaries, abstract and limitations were revised accordingly. The
ancillary prediction was also wrong in an informative way: Ledoit-Wolf
shrinkage HURTS this allocator at every window, so the sample
covariance was not the weak link.
