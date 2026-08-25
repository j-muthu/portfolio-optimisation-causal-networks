"""Stage A: score each candidate driver by its aggregate outgoing causal
influence on the asset block, then prune to the top-2K survivors.

The score sums |edge| x stability over lagged driver -> asset edges
(quantile-threshold mask for DYNOTEARS, bootstrap probability for VARLiNGAM).
Only lagged edges count: contemporaneous edges lack temporal precedence and
have the weakest exogeneity argument.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Per-edge stability
def dynotears_stability_mask(
    A_stacked: np.ndarray,
    target_fraction: float = 0.10,
) -> tuple[np.ndarray, float]:
    """0/1 stability mask on lagged edges plus the chosen threshold.

    The threshold is the (1 - target_fraction) quantile of the non-zero
    magnitudes, chosen from the data so Stage A stays scale-invariant
    across windows.
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

    Falls back to a 0/1 presence mask when no bootstrap was run.
    """
    p = len(B_lags)
    if bootstrap_prob_per_lag is not None and len(bootstrap_prob_per_lag) == p:
        return np.stack(bootstrap_prob_per_lag, axis=0)
    return np.stack([(np.abs(B) > 0).astype(float) for B in B_lags], axis=0)


# Stage A score
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

    ``window`` must expose ``driver_idx``, ``asset_idx``, ``driver_columns``
    and the lagged matrices (``A`` or ``B_lags``); contemporaneous edges are
    deliberately excluded. Returns a :class:`StageAResult`.
    """
    driver_idx = np.asarray(window.driver_idx, dtype=int)
    asset_idx = np.asarray(window.asset_idx, dtype=int)
    driver_columns = list(window.driver_columns)

    if method == "dynotears":
        A_stacked = np.stack(list(window.A), axis=0)  # (p, d, d)
        # Driver -> asset entries only: shape (p, n_drivers, n_assets).
        d2a = A_stacked[:, driver_idx[:, None], asset_idx[None, :]]
        mask, threshold = dynotears_stability_mask(d2a, target_fraction=target_fraction)
        contributions = np.abs(d2a) * mask.astype(float)
    elif method == "varlingam":
        B_lags = list(window.B_lags)
        boot = None
        # JointVarLingamWindow only exposes contemporaneous bootstrap probs,
        # so lagged stability falls back to presence indicators.
        stab = varlingam_stability_mask(B_lags, bootstrap_prob_per_lag=boot)
        d2a = np.stack(
            [B[driver_idx[:, None], asset_idx[None, :]] for B in B_lags], axis=0
        )
        stab_d2a = stab[:, driver_idx[:, None], asset_idx[None, :]]
        contributions = np.abs(d2a) * stab_d2a
        threshold = None
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # One score per driver: sum over lags and assets.
    per_driver = contributions.sum(axis=(0, 2))
    scores = pd.Series(per_driver, index=driver_columns, name="stage_a_score")
    pool = scores[scores > 0].sort_values(ascending=False).index.tolist()
    return StageAResult(
        scores=scores, threshold=threshold, pool=pool, method=method,
    )


# Pool reduction
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
