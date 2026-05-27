"""Stage A — score every candidate driver by its aggregate outgoing causal
influence on the asset block, and prune to the top-2K survivors.

The score combines edge magnitude with a stability gate:

    score_d = Σ_{p ≥ 1} Σ_{asset i}  |edge_d→i at lag p|  ·  stability_d→i

* DYNOTEARS: ``stability_d→i = 1{|edge_d→i| > τ}``, where ``τ`` is chosen so
  that roughly 10 % of the possible lagged driver→asset edges pass — a robust
  way to set the threshold per window without hand-picking.
* VARLiNGAM: ``stability_d→i = bootstrap_prob(d → i)`` (between 0 and 1).

Only lagged (p ≥ 1) edges contribute: contemporaneous edges are dropped here
because (a) they are where the exogeneity argument is weakest, and (b) drivers
are supposed to be *predictive* of asset moves, which requires temporal
precedence.

Entry points
------------
* :func:`stage_a_score` -- scores; works on any object exposing W, A,
  driver_idx, asset_idx (e.g. ``JointDynotearsWindow`` or
  ``JointVarLingamWindow``).
* :func:`prune_to_pool` -- top-2K survivors; the input pool for Stage B.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Per-edge stability
# ============================================================================
def dynotears_stability_mask(
    A_stacked: np.ndarray,
    target_fraction: float = 0.10,
) -> tuple[np.ndarray, float]:
    """Return a 0/1 stability mask on lagged edges plus the chosen threshold.

    ``A_stacked`` is the stack of lagged matrices ``A[0], A[1], ..., A[p-1]``.
    ``target_fraction`` is the *intended* fraction of non-zero entries to mark
    as stable; the threshold is chosen as the (1 - target_fraction) quantile
    of ``|A_stacked|`` over the non-zero entries. Returns a mask the same
    shape as ``A_stacked``.

    Choosing the threshold from the data (rather than fixing it at, say,
    0.01) keeps Stage A scale-invariant — windows with smaller edge weights
    overall still get a reasonable proportion of "stable" edges.
    """
    flat = np.abs(A_stacked).ravel()
    nonzero = flat[flat > 0]
    if nonzero.size == 0:
        return np.zeros_like(A_stacked, dtype=bool), 0.0
    threshold = float(np.quantile(nonzero, 1.0 - target_fraction))
    mask = np.abs(A_stacked) >= threshold
    return mask, threshold


def varlingam_stability_mask(
    B_lags: list[np.ndarray],
    bootstrap_prob_per_lag: list[np.ndarray] | None,
) -> np.ndarray:
    """Bootstrap-derived stability weights for VARLiNGAM, shape ``(p, d, d)``.

    If ``bootstrap_prob_per_lag`` is ``None`` (no bootstrap run), we fall back
    to a 0/1 mask based on edge presence. The returned array carries
    *probabilities*, not booleans, so the Stage A score multiplies edge
    magnitude by reliability per the plan.
    """
    p = len(B_lags)
    if bootstrap_prob_per_lag is not None and len(bootstrap_prob_per_lag) == p:
        return np.stack(bootstrap_prob_per_lag, axis=0)
    return np.stack([(np.abs(B) > 0).astype(float) for B in B_lags], axis=0)


# ============================================================================
# Stage A score
# ============================================================================
@dataclass
class StageAResult:
    """Outcome of Stage A: per-driver scores plus the kept pool."""

    scores: pd.Series
    threshold: float | None
    pool: list[str]
    method: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"driver": self.scores.index, "score": self.scores.values}
        ).sort_values("score", ascending=False).reset_index(drop=True)


def stage_a_score(
    window: Any,
    method: str = "dynotears",
    target_fraction: float = 0.10,
) -> StageAResult:
    """Compute the Stage A score for every candidate driver in a window.

    Parameters
    ----------
    window:
        Discovery output. Must expose ``driver_idx``, ``asset_idx``,
        ``driver_columns``, and either ``A`` (DYNOTEARS lagged matrices) or
        ``B_lags`` (VARLiNGAM lagged matrices). The full ``W`` / ``B0`` is
        intentionally *not* used — contemporaneous edges are excluded from
        the score.
    method:
        ``"dynotears"`` uses the quantile-threshold stability mask;
        ``"varlingam"`` uses bootstrap probabilities if available, else a
        presence indicator.
    target_fraction:
        Quantile-threshold parameter for DYNOTEARS (ignored for VARLiNGAM).

    Returns
    -------
    :class:`StageAResult` with ``scores`` (one entry per driver name),
    ``threshold`` (the magnitude floor used by DYNOTEARS, ``None`` for
    VARLiNGAM), and ``pool`` (drivers with non-zero score, sorted desc).
    """
    driver_idx = np.asarray(window.driver_idx, dtype=int)
    asset_idx = np.asarray(window.asset_idx, dtype=int)
    driver_columns = list(window.driver_columns)

    if method == "dynotears":
        # window.A is list of (d, d) matrices, one per lag (p ≥ 1).
        A_stacked = np.stack(list(window.A), axis=0)  # (p, d, d)
        # Slice to driver -> asset entries only: shape (p, n_drivers, n_assets).
        d2a = A_stacked[:, driver_idx[:, None], asset_idx[None, :]]
        mask, threshold = dynotears_stability_mask(d2a, target_fraction=target_fraction)
        contributions = np.abs(d2a) * mask.astype(float)
    elif method == "varlingam":
        B_lags = list(window.B_lags)
        boot = None
        # The current JointVarLingamWindow only exposes bootstrap_prob_B0
        # (contemporaneous). Lagged bootstrap probs would need a separate
        # extraction; until then we fall back to presence indicators.
        stab = varlingam_stability_mask(B_lags, bootstrap_prob_per_lag=boot)
        # Slice to driver -> asset entries: shape (p, n_drivers, n_assets).
        d2a = np.stack(
            [B[driver_idx[:, None], asset_idx[None, :]] for B in B_lags], axis=0
        )
        stab_d2a = stab[:, driver_idx[:, None], asset_idx[None, :]]
        contributions = np.abs(d2a) * stab_d2a
        threshold = None
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Sum over lags and asset axis -> one score per driver.
    per_driver = contributions.sum(axis=(0, 2))
    scores = pd.Series(per_driver, index=driver_columns, name="stage_a_score")
    pool = scores[scores > 0].sort_values(ascending=False).index.tolist()
    return StageAResult(
        scores=scores, threshold=threshold, pool=pool, method=method,
    )


# ============================================================================
# Pool reduction
# ============================================================================
def prune_to_pool(
    result: StageAResult,
    K: int,
    pool_multiplier: int = 2,
) -> list[str]:
    """Top-``pool_multiplier * K`` survivors, intersected with the non-zero pool."""
    target = pool_multiplier * K
    return result.pool[:target]


__all__ = [
    "StageAResult",
    "dynotears_stability_mask",
    "varlingam_stability_mask",
    "stage_a_score",
    "prune_to_pool",
]
