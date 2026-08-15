# Pre-committed protocol and interpretations: 2025-26 out-of-sample slice

Committed BEFORE any out-of-sample price was fetched or any out-of-sample
backtest was run. This is the confirmatory-slice discipline the report's
own limitations section calls for ("a deflation correction is not a
substitute for out-of-sample replication").

## Protocol (fixed here, before results)

- Backtest window: 2025-01-02 to the latest available trading day
  (expected ~2026-07/08), giving roughly 19 monthly rebalances. Data
  window starts 2023-01-03 so the 504-day lookback is fully burned in at
  the first rebalance.
- Identical harness to Phase II: same 99-name universe file, same driver
  pool and exclusions, monthly (21-day) rebalancing, 5 bps one-way costs,
  identical allocator code paths, DYNOTEARS graphs refit per rebalance
  with the unchanged Phase-II hyperparameters (lambda_W = lambda_A =
  0.05, threshold 0.01, lag 1). DYNOTEARS only: the in-sample
  orientation effect is DYNOTEARS-specific, so that is the arm under
  test.
- Cells: CORR, D0, D0s, D1, D2, D2s and the de-factored control D0df at
  windows 189/252/378/504.
- Price source: WRDS/CRSP preferred; if CRSP daily data does not yet
  cover the slice, Yahoo Finance for the out-of-sample period, disclosed
  in the report as a source switch. Names delisted since the 2024
  snapshot leave the panel at their last quote, as in-sample.

## What the slice can and cannot show (fixed before results)

Nineteen months of daily returns put the standard error of an annualised
Sharpe at roughly 0.8, and of a within-panel contrast far above the
+0.01 to +0.03 effects under study. THE SLICE CANNOT CONFIRM OR REFUTE
SIGNIFICANCE, and no p-value computed on it will be treated as evidence
in either direction. It is a sign and magnitude check on the point
estimates of the decomposition, made before the data existed in the
study.

## Pre-committed interpretations

Quantities read per window: skeleton = D0 - CORR, orientation = D1 - D0,
total = D1 - CORR, plus D2s - D0 and the D1 - D0df residual.

1. Signs broadly consistent with the in-sample decomposition (orientation
   increment non-negative at the majority of windows, total gain
   non-negative at the majority of windows): reported as directional
   out-of-sample support for the point estimates, explicitly NOT as
   significance, with the SE stated alongside.
2. Signs broadly inconsistent (orientation increment negative at most
   windows, or total gain negative at most windows): reported as an
   out-of-sample warning that weakens the in-sample reading, in the
   limitations section and the conclusion, at the same prominence as
   outcome 1 would have received.
3. Mixed: reported as uninformative, which at this sample length is the
   likeliest outcome; the report will say so.
4. The D1 - D0df residual: if it stays within its in-sample band
   (roughly +/-0.01 in point estimate), the de-factoring mechanism
   reading carries out of sample at point-estimate level; if D1 falls
   materially below D0df, that too is reported.

Whatever the outcome, all cells run are reported; no window, allocator
or contrast will be dropped after results are seen.
