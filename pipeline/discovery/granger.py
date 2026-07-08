"""Ridge-VAR(1) directed comparator ("GRANGER") for Phase II.

Ce Guo's Option 2 folded in as a discovery row of the Phase-II matrix: a
deterministic, closed-form, low-variance directed graph — no ICA, no
L-BFGS-B, no seeds. If the cheap directed graph allocates as well as the
fancy one, the expensive discovery step is not where the value lives.

Per window: z-score the asset panel, regress each asset's return on every
asset's lag-1 return with a ridge penalty (closed form), and take
``M[i, j] = |coef(x_{i,t-1} → x_{j,t})|`` — the repo's ``i → j`` convention.
Edges are thresholded to a target density (typically matched to the paired
DYNOTEARS window's asset-block density, so the DYNO-vs-GRANGER comparison
is fair). Two honest caveats, recorded on the window: the matrix is
**lagged** (not contemporaneous) and **not guaranteed acyclic** — callers
branch on ``is_dag`` (truncated-Neumann total effects; feedback-arc removal
before topological ordering).

The interface mirrors ``JointDynotearsWindow`` exactly where the Phase-II
chokepoint (``pipeline.discovery.asset_graph``) needs it: ``asset_columns``,
``asset_idx``, ``zscore_mean``/``zscore_std`` (full-length over ``columns``),
``asset_to_asset_block(0)`` — so it slots into ``load_or_compute_discovery``
with ``method="granger_ridge"`` and into the closed-loop asset-only path
unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from pipeline.discovery.asset_graph import is_dag_matrix

logger = logging.getLogger(__name__)


@dataclass
class JointGrangerWindow:
    """Ridge-VAR(1) output for one window of the joint ``[D | A]`` panel.

    Only the asset block is fitted (drivers are outside the GRANGER
    comparator's scope), but the dataclass carries the full column layout so
    it duck-types with the DYNOTEARS/VARLiNGAM joint windows downstream.
    ``M1[i, j]`` = |ridge coefficient of asset i's lag-1 in asset j's
    equation|, thresholded, zero outside the asset block.
    """

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    columns: list[str]
    driver_columns: list[str]
    asset_columns: list[str]
    driver_idx: np.ndarray
    asset_idx: np.ndarray
    M1: np.ndarray
    lambda_ridge: float
    threshold: float
    target_density: float | None
    achieved_density: float
    is_dag: bool
    zscore_mean: np.ndarray
    zscore_std: np.ndarray

    def asset_to_asset_block(self, lag: int) -> np.ndarray:
        """The lag-1 asset block (the only block GRANGER fits). The ``lag``
        argument exists for interface parity with the joint windows; GRANGER's
        directed structure is lagged by construction (documented caveat)."""
        return self.M1[np.ix_(self.asset_idx, self.asset_idx)]

    def driver_to_asset_block(self, lag: int) -> np.ndarray:
        return self.M1[np.ix_(self.driver_idx, self.asset_idx)]

    def asset_to_driver_block(self, lag: int) -> np.ndarray:
        return self.M1[np.ix_(self.asset_idx, self.driver_idx)]


def _density_threshold(A: np.ndarray, target_density: float) -> float:
    """Magnitude cut-off giving ≈``target_density`` off-diagonal edges."""
    N = A.shape[0]
    off = np.abs(A[~np.eye(N, dtype=bool)])
    k = int(round(target_density * N * (N - 1)))
    if k <= 0:
        return float(np.inf)
    if k >= off.size:
        return 0.0
    # Keep the k largest magnitudes: threshold at the k-th largest value.
    return float(np.partition(off, off.size - k)[off.size - k])


def run_granger_joint_window(
    joint_window: pd.DataFrame,
    driver_columns: Sequence[str],
    asset_columns: Sequence[str],
    lambda_ridge: float = 1e-2,
    target_density: float | None = None,
    threshold: float = 0.0,
) -> JointGrangerWindow:
    """Fit the ridge-VAR(1) asset graph on one window of the joint panel.

    ``lambda_ridge`` scales with the sample size (the penalty is
    ``λ · n · I`` on the Gram matrix) so the regularisation strength is
    window-length invariant. Exactly one of ``target_density`` /
    ``threshold`` drives the sparsification: with ``target_density`` the
    cut-off is chosen per window to hit that edge count (density matching
    against the paired DYNOTEARS window); otherwise the fixed magnitude
    ``threshold`` applies.
    """
    columns = list(joint_window.columns)
    driver_columns = list(driver_columns)
    asset_columns = list(asset_columns)
    if set(columns) != set(driver_columns) | set(asset_columns):
        raise ValueError(
            "joint_window columns must equal the union of driver_columns and "
            "asset_columns"
        )

    # Per-window z-score over the FULL joint panel (stats stored full-length,
    # mirroring run_dynotears_joint_window, so the chokepoint slices them
    # identically for every method).
    mean = joint_window.mean(axis=0)
    std = joint_window.std(axis=0, ddof=0).where(lambda s: s > 1e-12, 1e-12)

    X_assets = ((joint_window[asset_columns] - mean[asset_columns])
                / std[asset_columns]).to_numpy(dtype=float)
    n = X_assets.shape[0] - 1
    N = len(asset_columns)
    X_lag, Y = X_assets[:-1], X_assets[1:]

    # Closed-form ridge: coef = (XᵀX + λ n I)⁻¹ Xᵀ Y ; coef[i, j] is the
    # effect of x_{i,t-1} on x_{j,t} — already the i → j convention.
    gram = X_lag.T @ X_lag + lambda_ridge * n * np.eye(N)
    coef = np.linalg.solve(gram, X_lag.T @ Y)

    A = np.abs(coef)
    np.fill_diagonal(A, 0.0)  # own-lag AR terms are not cross-asset edges

    if target_density is not None:
        thr = _density_threshold(A, target_density)
    else:
        thr = threshold
    A[A < thr] = 0.0
    achieved = float((A != 0.0).sum()) / max(N * (N - 1), 1)

    # Embed the asset block into the full joint layout (driver blocks zero).
    d = len(columns)
    M1 = np.zeros((d, d))
    driver_idx = np.array([columns.index(c) for c in driver_columns], dtype=int)
    asset_idx = np.array([columns.index(c) for c in asset_columns], dtype=int)
    M1[np.ix_(asset_idx, asset_idx)] = A

    dag = is_dag_matrix(A)
    logger.debug(
        "granger window %s..%s: λ=%g, thr=%.4g, density=%.3f (target %s), dag=%s",
        joint_window.index.min().date(), joint_window.index.max().date(),
        lambda_ridge, thr, achieved, target_density, dag,
    )

    return JointGrangerWindow(
        start_date=pd.Timestamp(joint_window.index.min()),
        end_date=pd.Timestamp(joint_window.index.max()),
        columns=columns,
        driver_columns=driver_columns,
        asset_columns=asset_columns,
        driver_idx=driver_idx,
        asset_idx=asset_idx,
        M1=M1,
        lambda_ridge=lambda_ridge,
        threshold=float(thr),
        target_density=target_density,
        achieved_density=achieved,
        is_dag=dag,
        zscore_mean=mean.to_numpy(),
        zscore_std=std.to_numpy(),
    )


__all__ = ["JointGrangerWindow", "run_granger_joint_window"]
