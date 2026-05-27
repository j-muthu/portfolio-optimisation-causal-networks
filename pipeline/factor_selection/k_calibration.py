"""One-off calibration of ``K`` (the number of selected drivers) on the burn-in
window.

Two methods are computed in parallel and the more conservative wins:

1. **Kneedle** -- find the elbow in the sorted-Stage-A-score curve.
2. **Permutation null** -- shuffle each driver's time index independently
   (preserves marginal distribution, destroys temporal causal structure),
   refit discovery, collect ``max_d Stage-A-score`` across B=100 shuffles.
   The 95th percentile of those null max-scores is the threshold;
   ``K_perm`` is the count of real candidates with score above it.

``K = max(K_elbow, K_perm)``. The sensitivity sweep range is
``{⌈K/2⌉, K, min(2K, |pool|/2)}`` plus two interpolating values.

This is run **once** at the start of the backtest on the burn-in window. The
chosen K is then fixed for the rest of the run (per the plan), unless the
sensitivity sweep is enabled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT
from pipeline.factor_selection.prune import StageAResult, stage_a_score

logger = logging.getLogger(__name__)

CACHE_DIR = THESIS_ROOT / "cache"


# ============================================================================
# Kneedle (simple implementation; no kneed dependency)
# ============================================================================
def kneedle(scores_desc: np.ndarray) -> int:
    """Knee of a monotone-decreasing curve via the maximum-distance method.

    Standard formulation: normalise ``y`` and ``x`` to ``[0, 1]``, then find
    the ``x`` whose perpendicular distance to the chord from ``(0, 1)`` to
    ``(1, 0)`` (for a descending curve) is maximal. Returns the 1-indexed
    rank of the knee point — i.e. ``K_elbow`` candidates lie at or above the
    knee.

    Returns at least 1 to avoid a degenerate empty selection.
    """
    n = len(scores_desc)
    if n <= 2:
        return n
    y = scores_desc.astype(float)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    x = np.linspace(0.0, 1.0, n)
    # Chord from (0, 1) to (1, 0) has direction (1, -1) / sqrt(2); perpendicular
    # distance of (x_i, y_i) from this chord is |x_i + y_i - 1| / sqrt(2).
    distances = np.abs(x + y - 1.0) / np.sqrt(2.0)
    knee_idx = int(np.argmax(distances))
    return max(1, knee_idx + 1)


# ============================================================================
# Permutation null
# ============================================================================
def permutation_null_threshold(
    fit_score_fn: Callable[[int], np.ndarray],
    n_permutations: int = 100,
    quantile: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray]:
    """Permutation null distribution for the *max* Stage-A score.

    ``fit_score_fn`` takes a seed integer and returns the Stage A score
    vector under one driver-permuted refit. The seed is meant to set the
    refit's RNG so the permutations are reproducible.

    Returns ``(threshold, null_max_scores)`` where ``threshold`` is the
    requested quantile of the per-permutation max scores. Drivers whose
    real Stage A score exceeds the threshold are deemed signal.
    """
    rng = rng or np.random.default_rng(0)
    null_max: list[float] = []
    for b in range(n_permutations):
        seed = int(rng.integers(0, 2**31 - 1))
        scores = fit_score_fn(seed)
        null_max.append(float(np.max(scores)))
    null_max_arr = np.asarray(null_max)
    return float(np.quantile(null_max_arr, quantile)), null_max_arr


# ============================================================================
# K calibration orchestrator
# ============================================================================
@dataclass
class KCalibration:
    """Outcome of the burn-in K-calibration run."""

    K: int
    K_elbow: int
    K_perm: int
    threshold_perm: float
    real_scores_desc: np.ndarray
    real_drivers_desc: list[str]
    pool_size: int
    sensitivity_sweep: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "K": int(self.K),
            "K_elbow": int(self.K_elbow),
            "K_perm": int(self.K_perm),
            "threshold_perm": float(self.threshold_perm),
            "pool_size": int(self.pool_size),
            "sensitivity_sweep": [int(x) for x in self.sensitivity_sweep],
            "real_scores_desc": [float(x) for x in self.real_scores_desc],
            "real_drivers_desc": list(self.real_drivers_desc),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KCalibration":
        d = json.loads(Path(path).read_text())
        return cls(
            K=d["K"], K_elbow=d["K_elbow"], K_perm=d["K_perm"],
            threshold_perm=d["threshold_perm"],
            real_scores_desc=np.asarray(d["real_scores_desc"]),
            real_drivers_desc=list(d["real_drivers_desc"]),
            pool_size=d["pool_size"],
            sensitivity_sweep=list(d.get("sensitivity_sweep", [])),
            metadata=d.get("metadata", {}),
        )


def _build_sweep(K: int, pool_size: int) -> list[int]:
    """``{⌈K/2⌉, K, min(2K, pool/2)}`` plus two interpolating values."""
    half = max(1, int(np.ceil(K / 2)))
    upper = min(2 * K, max(1, pool_size // 2))
    if upper <= half:
        return sorted({half, K})
    # Two interpolating integer values evenly spaced.
    candidates = sorted({half, K, upper, (half + K) // 2, (K + upper) // 2})
    return [int(x) for x in candidates if 1 <= x <= pool_size]


def calibrate_K(
    real_window,
    fit_permuted_score_fn: Callable[[int], np.ndarray],
    method: str = "dynotears",
    target_fraction: float = 0.10,
    n_permutations: int = 100,
    quantile: float = 0.95,
    rng_seed: int = 0,
) -> KCalibration:
    """Run both Kneedle and permutation-null and pick the conservative K.

    Parameters
    ----------
    real_window:
        Discovery output on the *real* burn-in window (already fit).
    fit_permuted_score_fn:
        Callable that takes a seed and returns the Stage-A score vector
        under one driver-permuted refit. Implementation lives upstream
        (Stage 1 orchestration knows how to re-run discovery on permuted
        inputs); we keep the dependency one-way.
    method:
        Pass-through to :func:`stage_a_score` on the real window.
    n_permutations, quantile:
        Permutation-null settings; defaults match the plan (B=100, 95 %).

    Returns
    -------
    :class:`KCalibration`. The final ``K = max(K_elbow, K_perm)`` and a
    sensitivity sweep range is populated.
    """
    real_result = stage_a_score(real_window, method=method, target_fraction=target_fraction)
    sorted_desc = real_result.scores.sort_values(ascending=False)
    pool_size = int((sorted_desc > 0).sum())
    scores_desc = sorted_desc.values
    drivers_desc = sorted_desc.index.tolist()

    K_elbow = kneedle(scores_desc) if pool_size > 0 else 0

    rng = np.random.default_rng(rng_seed)
    threshold, null_max = permutation_null_threshold(
        fit_permuted_score_fn,
        n_permutations=n_permutations,
        quantile=quantile,
        rng=rng,
    )
    K_perm = int(np.sum(scores_desc > threshold))

    K = max(K_elbow, K_perm)
    K = max(1, min(K, pool_size))  # never exceed available signal
    sweep = _build_sweep(K, pool_size)

    logger.info(
        "K calibration: K_elbow=%d, K_perm=%d (threshold=%.4f), chosen K=%d, "
        "pool_size=%d, sweep=%s",
        K_elbow, K_perm, threshold, K, pool_size, sweep,
    )
    return KCalibration(
        K=K, K_elbow=K_elbow, K_perm=K_perm, threshold_perm=threshold,
        real_scores_desc=scores_desc,
        real_drivers_desc=drivers_desc,
        pool_size=pool_size,
        sensitivity_sweep=sweep,
        metadata={
            "method": method,
            "n_permutations": n_permutations,
            "quantile": quantile,
            "target_fraction": target_fraction,
            "null_max_scores": [float(x) for x in null_max],
        },
    )


__all__ = [
    "kneedle",
    "permutation_null_threshold",
    "KCalibration",
    "calibrate_K",
]
