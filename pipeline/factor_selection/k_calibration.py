"""One-off calibration of ``K`` (the number of selected drivers) on the burn-in
window.

Two methods are computed in parallel and the more conservative wins:

1. **Kneedle** -- find the elbow in the sorted-Stage-A-score curve.
2. **Permutation null with BH-FDR** -- shuffle each driver's time index
   independently (preserves marginal distribution, destroys temporal causal
   structure), refit discovery, collect the per-driver Stage-A score
   distribution under the null across B permutations. For each real driver
   compute a one-sided p-value, then control false-discovery rate at α=0.05
   via Benjamini-Hochberg. ``K_perm`` is the count of drivers that survive.

Why BH-FDR rather than the historical "max-across-d" threshold: the
``max_d`` statistic is structurally biased upward when d is large, so a
single real driver's score routinely fails to clear it even under signal.
The G.5 calibration found ``K_perm = 0`` because of exactly this bug. The
legacy max-of-d statistic is still computed and stored as a
side-channel ``K_perm_legacy`` for direct comparison in the methodology
chapter.

Runtime: the permutation loop is embarrassingly parallel across B fits and
each permuted DYNOTEARS fit can be capped at a low ``max_iter`` since
shuffled drivers have no causal structure to converge on. ``n_jobs`` and
the caller's choice of ``permuted_max_iter`` (see ``shakedown.py``) bring
the calibration into the minute-scale regime at thesis-relevant d.

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
    n_jobs: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Permutation null distribution for Stage-A scores under shuffled drivers.

    ``fit_score_fn`` takes a seed integer and returns the Stage A score
    vector under one driver-permuted refit. The seed is meant to set the
    refit's RNG so the permutations are reproducible.

    Returns ``(threshold, null_max_scores, null_per_driver)`` where:

    * ``threshold`` — the requested quantile (default 95th percentile) of
      the per-permutation max scores. Legacy "max-across-d" threshold;
      used to compute ``K_perm_legacy`` for the methodology-chapter
      comparison. Biased upward at large d (multiple-comparisons).
    * ``null_max_scores`` — shape ``(B,)``; the per-permutation max
      scores. Kept for reporting / plotting.
    * ``null_per_driver`` — shape ``(B, d)``; the **full** per-permutation
      per-driver score matrix. Consumed by
      :func:`benjamini_hochberg_K_perm` to compute the FDR-controlled K.
      Memory cost is trivial (B=100, d=35 → 1400 floats).

    ``n_jobs > 1`` parallelises the permutation loop via joblib
    (embarrassingly parallel — each permutation is an independent fit).
    All seeds are drawn up-front from ``rng`` so the result is
    deterministic regardless of n_jobs.
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


# ============================================================================
# Benjamini-Hochberg FDR control
# ============================================================================
def benjamini_hochberg_K_perm(
    real_scores: np.ndarray,
    null_per_driver: np.ndarray,
    alpha: float = 0.05,
    method: str = "zscore",
) -> tuple[int, np.ndarray, np.ndarray]:
    """Per-driver p-values + BH-FDR → number of significant drivers.

    Two p-value formulations:

    * ``method="zscore"`` (default, recommended): per-driver one-sided
      p-value via the standardised score
      ``z_d = (real_d − mean(null_{·,d})) / std(null_{·,d})``,
      converted via ``1 − Φ(z)``. Assumes the per-driver null is
      approximately Gaussian — typical for Stage A scores because they
      aggregate over many lagged-edge magnitudes (CLT regime).
      Continuous p-values, not limited by the MC discreteness floor.
    * ``method="mc"`` (non-parametric backup): empirical one-sided
      p-value ``p_d = (#{b : null[b, d] ≥ real_d} + 1) / (B + 1)``.
      Conservative but with a hard floor at ``1 / (B + 1)`` — at
      typical thesis ``(B=100, d=135, α=0.05)`` this floor is far above
      BH's threshold so nothing can clear FDR. Use only when ``B`` is
      large enough that the MC floor is well below ``α / d``.

    Then Benjamini-Hochberg at level ``alpha``: sort p-values ascending,
    find the largest k with ``p_(k) ≤ (k / m) · α``, declare drivers
    with rank ≤ k as significant.

    Parameters
    ----------
    real_scores:
        Per-driver real Stage-A scores; shape ``(d,)``.
    null_per_driver:
        Permutation null matrix from :func:`permutation_null_threshold`;
        shape ``(B, d)`` matching ``real_scores`` columnwise.
    alpha:
        Target FDR level. ``0.05`` is the canonical academic default.
    method:
        ``"zscore"`` (parametric, default) or ``"mc"`` (non-parametric).

    Returns
    -------
    ``(K_perm, p_values, significant_mask)``:

    * ``K_perm`` -- count of significant drivers (the BH-FDR estimate of
      the number of true positives).
    * ``p_values`` -- shape ``(d,)``; raw per-driver p-values.
    * ``significant_mask`` -- shape ``(d,)``, bool; True for drivers
      surviving BH at level ``alpha``.
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
        null_std = np.maximum(null_std, 1e-12)  # floor to avoid div-by-zero
        z = (real_scores - null_mean) / null_std
        p_values = norm.sf(z)  # one-sided upper tail
    elif method == "mc":
        n_extreme = (null_per_driver >= real_scores[None, :]).sum(axis=0)
        p_values = (n_extreme + 1.0) / (B + 1.0)
    else:
        raise ValueError(f"method must be 'zscore' or 'mc', got {method!r}")

    # Benjamini-Hochberg step-up procedure at level alpha.
    order = np.argsort(p_values, kind="stable")
    p_sorted = p_values[order]
    ranks = np.arange(1, d + 1, dtype=float)
    bh_threshold = ranks * alpha / d
    below = p_sorted <= bh_threshold
    if not below.any():
        k_star = 0
    else:
        # Largest rank k with p_(k) <= (k/d)*alpha.
        k_star = int(np.where(below)[0].max() + 1)

    significant = np.zeros(d, dtype=bool)
    if k_star > 0:
        significant[order[:k_star]] = True
    return int(significant.sum()), p_values, significant


# ============================================================================
# K calibration orchestrator
# ============================================================================
@dataclass
class KCalibration:
    """Outcome of the burn-in K-calibration run.

    ``K_perm`` is the FDR-controlled count (Benjamini-Hochberg). The
    historical "max-across-d" statistic is preserved in
    ``K_perm_legacy`` for the methodology-chapter comparison — it's
    structurally biased low at large d and was the source of G.5's
    ``K_perm = 0`` finding.
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
    """Run both Kneedle and permutation-null + BH-FDR and pick the conservative K.

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
        ``quantile`` only governs the **legacy** max-of-d threshold; the
        primary ``K_perm`` is FDR-controlled.
    fdr_alpha:
        Benjamini-Hochberg target false-discovery rate for the
        per-driver p-values (default 0.05 — academic canonical).
    n_jobs:
        Parallelism across the B permutations. ``-1`` uses all cores.
        Embarrassingly parallel; speedup is near-linear up to the number
        of physical cores.

    Returns
    -------
    :class:`KCalibration` with ``K = max(K_elbow, K_perm)`` (clipped to
    ``[1, pool_size]``), the per-driver p-values + significance mask, and
    the legacy max-of-d ``K_perm_legacy`` for direct comparison.
    """
    real_result = stage_a_score(real_window, method=method, target_fraction=target_fraction)
    # Keep the original (unsorted) order alongside the sorted view so the
    # null-per-driver column alignment is unambiguous.
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

    # Primary: BH-FDR on per-driver p-values. Uses the unsorted order so
    # ``null_per_driver`` columns line up with ``real_scores_unsorted``
    # (the closure returns scores in the unsorted driver-name order).
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
