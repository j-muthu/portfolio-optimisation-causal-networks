"""Ridge-VAR(1) directed comparator ("GRANGER") for Phase II.

Per window: z-score, closed-form ridge regression of each asset on every
asset's lag-1 return, take ``M[i, j] = |coef|`` (i -> j), threshold to a
target density. Caveats recorded on the window: the matrix is lagged, not
contemporaneous, and not guaranteed acyclic (callers branch on ``is_dag``).
The interface duck-types with ``JointDynotearsWindow`` where the asset_graph
chokepoint needs it.
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

    Only the asset block is fitted; the full column layout is carried so it
    duck-types with the other joint windows. ``M1`` is thresholded and zero
    outside the asset block.
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
        """The lag-1 asset block. ``lag`` exists only for interface parity;
        GRANGER's structure is lagged by construction."""
        return self.M1[np.ix_(self.asset_idx, self.asset_idx)]

    def driver_to_asset_block(self, lag: int) -> np.ndarray:
        return self.M1[np.ix_(self.driver_idx, self.asset_idx)]

    def asset_to_driver_block(self, lag: int) -> np.ndarray:
        return self.M1[np.ix_(self.asset_idx, self.driver_idx)]


def _density_threshold(A: np.ndarray, target_density: float) -> float:
    """Magnitude cut-off giving roughly ``target_density`` off-diagonal edges."""
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

    The ridge penalty scales with sample size, so its strength is
    window-length invariant. ``target_density`` picks a per-window cut-off
    to hit that edge count; otherwise the fixed ``threshold`` applies.
    """
    columns = list(joint_window.columns)
    driver_columns = list(driver_columns)
    asset_columns = list(asset_columns)
    if set(columns) != set(driver_columns) | set(asset_columns):
        raise ValueError(
            "joint_window columns must equal the union of driver_columns and "
            "asset_columns"
        )

    # Z-score stats over the full joint panel, stored full-length so the
    # chokepoint slices them identically for every method.
    mean = joint_window.mean(axis=0)
    std = joint_window.std(axis=0, ddof=0).where(lambda s: s > 1e-12, 1e-12)

    X_assets = ((joint_window[asset_columns] - mean[asset_columns])
                / std[asset_columns]).to_numpy(dtype=float)
    n = X_assets.shape[0] - 1
    N = len(asset_columns)
    X_lag, Y = X_assets[:-1], X_assets[1:]

    # Closed-form ridge; coef[i, j] is already in the i -> j convention.
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
