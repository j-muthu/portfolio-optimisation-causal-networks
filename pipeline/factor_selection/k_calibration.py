"""One-off calibration of K (number of selected drivers) on the burn-in window.

Two methods, the more conservative wins: Kneedle on the sorted Stage-A score
curve, and a permutation null with BH-FDR (shuffle driver time indexes, refit,
per-driver one-sided p-values, control FDR at 0.05). BH-FDR replaces the old
max-across-d threshold, which is biased upward at large d and produced the
G.5 ``K_perm = 0`` bug; the legacy statistic is kept as ``K_perm_legacy``.
``K = max(K_elbow, K_perm)``, fixed for the rest of the run.
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


# Kneedle (simple implementation; no kneed dependency)
def kneedle(scores_desc: np.ndarray) -> int:
    """Knee of a monotone-decreasing curve via the maximum-distance method.

    Returns the 1-indexed rank of the knee point, at least 1.
    """
    n = len(scores_desc)
    if n <= 2:
        return n
    y = scores_desc.astype(float)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    x = np.linspace(0.0, 1.0, n)
    # Perpendicular distance from the (0,1)-(1,0) chord is |x + y - 1| / sqrt(2).
    distances = np.abs(x + y - 1.0) / np.sqrt(2.0)
    knee_idx = int(np.argmax(distances))
    return max(1, knee_idx + 1)


# Permutation null
def permutation_null_threshold(
    fit_score_fn: Callable[[int], np.ndarray],
    n_permutations: int = 100,
    quantile: float = 0.95,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Permutation null distribution for Stage-A scores under shuffled drivers.

    ``fit_score_fn`` maps a seed to the Stage-A score vector of one
    driver-permuted refit. Returns ``(threshold, null_max_scores,
    null_per_driver)``: the legacy max-of-d quantile threshold, the (B,)
    per-permutation maxima, and the full (B, d) null matrix consumed by
    :func:`benjamini_hochberg_K_perm`. Seeds are drawn up-front so the
    result is deterministic regardless of ``n_jobs``.
    """
    rng = rng or np.random.default_rng(0)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_permutations)]

    if n_jobs == 1:
        all_scores = [fit_score_fn(s) for s in seeds]
    else:
        from joblib import Parallel, delayed

        all_scores = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(fit_score_fn)(s) for s in seeds
        )

    null_per_driver = np.asarray(all_scores, dtype=float)  # (B, d)
    null_max_arr = null_per_driver.max(axis=1)             # (B,)
    threshold = float(np.quantile(null_max_arr, quantile))
    return threshold, null_max_arr, null_per_driver


# Benjamini-Hochberg FDR control
def benjamini_hochberg_K_perm(
    real_scores: np.ndarray,
    null_per_driver: np.ndarray,
    alpha: float = 0.05,
    method: str = "zscore",
) -> tuple[int, np.ndarray, np.ndarray]:
    """Per-driver p-values plus BH-FDR; returns the significant-driver count.

    ``method="zscore"`` (default) standardises each real score against its
    null column and takes the Gaussian upper tail: continuous p-values, no
    MC discreteness floor. ``method="mc"`` uses the empirical p-value
    ``(#{null >= real} + 1) / (B + 1)``, which has a hard floor at
    ``1/(B+1)``; at typical B that floor sits above BH's threshold, so use
    it only with large B. Returns ``(K_perm, p_values, significant_mask)``.
    """
    real_scores = np.asarray(real_scores, dtype=float)
    null_per_driver = np.asarray(null_per_driver, dtype=float)
    if null_per_driver.ndim != 2 or null_per_driver.shape[1] != real_scores.shape[0]:
        raise ValueError(
            f"null_per_driver shape {null_per_driver.shape} must be (B, d) "
            f"with d == len(real_scores)={real_scores.shape[0]}"
        )

    B, d = null_per_driver.shape

    if method == "zscore":
        from scipy.stats import norm

        null_mean = null_per_driver.mean(axis=0)
        null_std = null_per_driver.std(axis=0, ddof=1)
        null_std = np.maximum(null_std, 1e-12)  # avoid div-by-zero
        z = (real_scores - null_mean) / null_std
        p_values = norm.sf(z)  # one-sided upper tail
    elif method == "mc":
        n_extreme = (null_per_driver >= real_scores[None, :]).sum(axis=0)
        p_values = (n_extreme + 1.0) / (B + 1.0)
    else:
        raise ValueError(f"method must be 'zscore' or 'mc', got {method!r}")

    # BH step-up at level alpha.
    order = np.argsort(p_values, kind="stable")
    p_sorted = p_values[order]
    ranks = np.arange(1, d + 1, dtype=float)
    bh_threshold = ranks * alpha / d
    below = p_sorted <= bh_threshold
    if not below.any():
        k_star = 0
    else:
        k_star = int(np.where(below)[0].max() + 1)

    significant = np.zeros(d, dtype=bool)
    if k_star > 0:
        significant[order[:k_star]] = True
    return int(significant.sum()), p_values, significant


# K calibration orchestrator
@dataclass
class KCalibration:
    """Outcome of the burn-in K-calibration run.

    ``K_perm`` is the BH-FDR count; ``K_perm_legacy`` preserves the biased
    max-across-d statistic for comparison.
    """

    K: int
    K_elbow: int
    K_perm: int
    K_perm_legacy: int
    threshold_perm: float
    p_values: np.ndarray
    significant_drivers_mask: np.ndarray
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
            "K_perm_legacy": int(self.K_perm_legacy),
            "threshold_perm": float(self.threshold_perm),
            "pool_size": int(self.pool_size),
            "sensitivity_sweep": [int(x) for x in self.sensitivity_sweep],
            "real_scores_desc": [float(x) for x in self.real_scores_desc],
            "real_drivers_desc": list(self.real_drivers_desc),
            "p_values": [float(x) for x in self.p_values],
            "significant_drivers_mask": [bool(x) for x in self.significant_drivers_mask],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KCalibration":
        d = json.loads(Path(path).read_text())
        return cls(
            K=d["K"], K_elbow=d["K_elbow"], K_perm=d["K_perm"],
            K_perm_legacy=d.get("K_perm_legacy", -1),
            threshold_perm=d["threshold_perm"],
            real_scores_desc=np.asarray(d["real_scores_desc"]),
            real_drivers_desc=list(d["real_drivers_desc"]),
            p_values=np.asarray(d.get("p_values", []), dtype=float),
            significant_drivers_mask=np.asarray(
                d.get("significant_drivers_mask", []), dtype=bool
            ),
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
    fdr_alpha: float = 0.05,
    n_jobs: int = 1,
) -> KCalibration:
    """Run Kneedle and permutation-null + BH-FDR and pick the conservative K.

    ``fit_permuted_score_fn`` maps a seed to the Stage-A score vector of one
    driver-permuted refit. ``quantile`` only governs the legacy max-of-d
    threshold; the primary ``K_perm`` is FDR-controlled. Returns a
    :class:`KCalibration` with ``K = max(K_elbow, K_perm)`` clipped to
    ``[1, pool_size]``.
    """
    real_result = stage_a_score(real_window, method=method, target_fraction=target_fraction)
    # Keep the unsorted order too: null_per_driver columns align with it.
    real_scores_unsorted = real_result.scores
    sorted_desc = real_scores_unsorted.sort_values(ascending=False)
    pool_size = int((sorted_desc > 0).sum())
    scores_desc = sorted_desc.values
    drivers_desc = sorted_desc.index.tolist()

    K_elbow = kneedle(scores_desc) if pool_size > 0 else 0

    rng = np.random.default_rng(rng_seed)
    threshold, null_max, null_per_driver = permutation_null_threshold(
        fit_permuted_score_fn,
        n_permutations=n_permutations,
        quantile=quantile,
        rng=rng,
        n_jobs=n_jobs,
    )

    # Primary: BH-FDR on per-driver p-values, in the unsorted order so the
    # null columns line up.
    K_perm, p_values, sig_mask = benjamini_hochberg_K_perm(
        real_scores_unsorted.values, null_per_driver, alpha=fdr_alpha,
    )
    # Legacy diagnostic: count of real drivers above the max-of-d threshold.
    K_perm_legacy = int(np.sum(scores_desc > threshold))

    K = max(K_elbow, K_perm)
    K = max(1, min(K, pool_size))  # never exceed available signal
    sweep = _build_sweep(K, pool_size)

    significant_names = [
        drv for drv, ok in zip(real_scores_unsorted.index.tolist(), sig_mask) if ok
    ]
    logger.info(
        "K calibration: K_elbow=%d, K_perm(BH)=%d, K_perm_legacy=%d "
        "(threshold=%.4f), chosen K=%d, pool_size=%d, sweep=%s",
        K_elbow, K_perm, K_perm_legacy, threshold, K, pool_size, sweep,
    )
    logger.info(
        "K calibration: %d driver(s) significant at FDR=%.2f: %s",
        len(significant_names), fdr_alpha, significant_names,
    )
    return KCalibration(
        K=K, K_elbow=K_elbow, K_perm=K_perm, K_perm_legacy=K_perm_legacy,
        threshold_perm=threshold,
        p_values=p_values,
        significant_drivers_mask=sig_mask,
        real_scores_desc=scores_desc,
        real_drivers_desc=drivers_desc,
        pool_size=pool_size,
        sensitivity_sweep=sweep,
        metadata={
            "method": method,
            "n_permutations": n_permutations,
            "quantile": quantile,
            "target_fraction": target_fraction,
            "fdr_alpha": fdr_alpha,
            "n_jobs": n_jobs,
            "null_max_scores": [float(x) for x in null_max],
            "significant_drivers": significant_names,
        },
    )


__all__ = [
    "kneedle",
    "permutation_null_threshold",
    "benjamini_hochberg_K_perm",
    "KCalibration",
    "calibrate_K",
]
