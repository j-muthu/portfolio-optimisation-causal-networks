"""Stage B — conditional greedy refinement on the Stage-A pool.

Algorithm:

1. Start with empty selected set ``S``.
2. For each remaining candidate ``d`` in the pool, fit a small auxiliary
   model: a regularised lagged regression of *each asset's* return on lagged
   values of ``S ∪ {d}``. Score ``d`` by the aggregate held-out log-likelihood
   gain vs the model that uses ``S`` only.
3. Add the argmax candidate to ``S``.
4. Repeat until ``|S| = K`` or the marginal gain falls below the
   permutation-null threshold ``ε``.

Implementation choices:

* Regression: ``sklearn.linear_model.Ridge`` per asset, alpha calibrated
  per call (default 1.0; downstream callers can override).
* Validation: chronological 20 %-tail split of the window. Log-likelihood
  under a Gaussian assumption — equivalent to scoring by negative half MSE
  with a fixed variance, but the additive constants cancel in gain.
* Stopping ``ε``: optional — when supplied, the loop terminates as soon as
  the best marginal gain drops below ``ε`` even if ``|S| < K``. The plan's
  default ``ε`` is the 95th percentile of marginal gains under a
  permutation-null fit; computing that null is the caller's responsibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Auxiliary regression
# ============================================================================
def _make_lagged_design(
    driver_window: pd.DataFrame,
    target_window: pd.DataFrame,
    selected: Sequence[str],
    lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build train/val (X, Y) matrices: lagged-driver predictors → asset returns.

    Drops the first ``lags`` rows (no lag history). Returns ``(X_tr, Y_tr,
    X_va, Y_va)``; validation is the chronological 20 % tail of the post-lag
    sample. ``X`` columns are the stacked lagged copies of the selected
    drivers (shape ``len(selected) * lags``).
    """
    selected = list(selected)
    if not selected:
        # Intercept-only baseline: X is a single column of ones; the regression
        # will fit just the mean and held-out NLL gives the variance bound.
        n = len(target_window) - lags
        X = np.ones((n, 1))
        Y = target_window.iloc[lags:].to_numpy(dtype=float)
        split = int(n * 0.8)
        return X[:split], Y[:split], X[split:], Y[split:]

    cols = []
    for k in range(1, lags + 1):
        cols.append(driver_window[selected].shift(k))
    X_df = pd.concat(cols, axis=1)
    common = X_df.dropna().index.intersection(target_window.index)
    X = X_df.loc[common].to_numpy(dtype=float)
    Y = target_window.loc[common].to_numpy(dtype=float)
    n = len(common)
    split = int(n * 0.8)
    return X[:split], Y[:split], X[split:], Y[split:]


def _ridge_val_score(
    X_tr: np.ndarray, Y_tr: np.ndarray,
    X_va: np.ndarray, Y_va: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Aggregate held-out negative-half MSE across all asset targets.

    Larger is better (the conditional greedy step picks the candidate with
    the largest *increase* in this score over the current ``S``).
    """
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_va)
    mse = np.mean((Y_va - pred) ** 2)
    return -0.5 * float(mse)


# ============================================================================
# Stage B
# ============================================================================
@dataclass
class StageBStep:
    """One iteration of the greedy expansion."""

    step: int
    candidate_added: str | None
    gain: float
    selected: list[str]


@dataclass
class StageBResult:
    """Output of :func:`greedy_select`."""

    selected: list[str]
    steps: list[StageBStep] = field(default_factory=list)
    stopped_by: str = ""  # "K", "epsilon", or "pool_exhausted"

    def gain_log(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "step": [s.step for s in self.steps],
                "added": [s.candidate_added for s in self.steps],
                "gain": [s.gain for s in self.steps],
                "n_selected": [len(s.selected) for s in self.steps],
            }
        )


def greedy_select(
    driver_window: pd.DataFrame,
    asset_window: pd.DataFrame,
    pool: Sequence[str],
    K: int,
    lags: int = 1,
    epsilon: float | None = None,
    alpha: float = 1.0,
) -> StageBResult:
    """Stage B greedy expansion. See module docstring for full algorithm.

    Parameters
    ----------
    driver_window, asset_window:
        Per-window driver/asset frames, *already z-scored*. They must share
        an index.
    pool:
        Stage A's pool of candidate driver names (typically top-2K).
    K:
        Target selected count.
    lags:
        Auxiliary-model lag horizon (e.g. ``1``).
    epsilon:
        Minimum marginal gain to accept a candidate. If a candidate's gain
        falls below ``epsilon``, the loop stops even if ``|S| < K``. ``None``
        disables the early-stop.
    alpha:
        Ridge regularisation for the per-asset auxiliary regression.

    Returns
    -------
    :class:`StageBResult` with the selected drivers (in addition order),
    a per-step gain log, and the termination reason.
    """
    pool = [c for c in pool if c in driver_window.columns]
    selected: list[str] = []
    steps: list[StageBStep] = []
    remaining = list(pool)

    # Baseline (intercept-only) NLL for the first step's gain calculation.
    X_tr, Y_tr, X_va, Y_va = _make_lagged_design(driver_window, asset_window, [], lags)
    baseline = _ridge_val_score(X_tr, Y_tr, X_va, Y_va, alpha=alpha)
    current = baseline

    stopped = "K"
    while len(selected) < K and remaining:
        best_cand: str | None = None
        best_gain = -np.inf
        for cand in remaining:
            X_tr, Y_tr, X_va, Y_va = _make_lagged_design(
                driver_window, asset_window, selected + [cand], lags
            )
            score = _ridge_val_score(X_tr, Y_tr, X_va, Y_va, alpha=alpha)
            gain = score - current
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        if best_cand is None:
            stopped = "pool_exhausted"
            break
        if epsilon is not None and best_gain < epsilon:
            steps.append(
                StageBStep(step=len(steps) + 1, candidate_added=None, gain=best_gain, selected=list(selected))
            )
            stopped = "epsilon"
            break
        selected.append(best_cand)
        remaining.remove(best_cand)
        steps.append(
            StageBStep(
                step=len(steps) + 1,
                candidate_added=best_cand,
                gain=best_gain,
                selected=list(selected),
            )
        )
        # Update the "current" score baseline with the actual fit incorporating
        # the chosen candidate.
        X_tr, Y_tr, X_va, Y_va = _make_lagged_design(driver_window, asset_window, selected, lags)
        current = _ridge_val_score(X_tr, Y_tr, X_va, Y_va, alpha=alpha)

    if not remaining and len(selected) < K:
        stopped = "pool_exhausted"
    logger.info(
        "Stage B greedy: %d drivers selected (stopped by %s) — %s",
        len(selected), stopped, selected,
    )
    return StageBResult(selected=selected, steps=steps, stopped_by=stopped)


__all__ = ["StageBStep", "StageBResult", "greedy_select"]
