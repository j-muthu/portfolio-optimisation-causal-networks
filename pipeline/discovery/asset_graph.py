"""Per-window asset-asset directed graph, extracted from a fitted discovery
window. The single chokepoint every direction-aware allocator goes through.

Conventions: ``M[i, j]`` is i -> j (VARLiNGAM's B0 is already transposed at
fit time, do not re-transpose). The tau threshold and universe slicing happen
here, and residual variances are computed on the fit window's z-scored data
so they match the graph the allocator sees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# DAG check (Kahn's algorithm on the binarised adjacency)
def is_dag_matrix(M: np.ndarray) -> bool:
    """True iff the non-zero pattern of ``M`` (i → j) is acyclic."""
    adj = (M != 0.0).astype(np.int64)
    np.fill_diagonal(adj, 0)
    in_deg = adj.sum(axis=0).astype(np.int64)
    stack = list(np.flatnonzero(in_deg == 0))
    seen = 0
    while stack:
        i = stack.pop()
        seen += 1
        children = np.flatnonzero(adj[i])
        adj[i, children] = 0
        in_deg[children] -= 1
        stack.extend(int(c) for c in children if in_deg[c] == 0)
    return seen == M.shape[0]


# The one type every allocator consumes
@dataclass(frozen=True)
class AssetGraphWindow:
    """Asset-asset directed graph for one rebalance window.

    ``M`` is ``(N, N)``, i -> j, tau-thresholded, sliced to ``asset_names``.
    ``resid_var_z`` is ``None`` only when no fit window was supplied (tests).
    """

    end_date: pd.Timestamp
    asset_names: list[str]
    M: np.ndarray
    zscore_std: np.ndarray
    resid_var_z: np.ndarray | None
    method: str
    tau: float
    is_dag: bool
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        N = len(self.asset_names)
        if self.M.shape != (N, N):
            raise ValueError(
                f"M must be ({N}, {N}) on asset_names; got {self.M.shape}"
            )
        if self.zscore_std.shape != (N,):
            raise ValueError(
                f"zscore_std must be ({N},); got {self.zscore_std.shape}"
            )
        if self.resid_var_z is not None and self.resid_var_z.shape != (N,):
            raise ValueError(
                f"resid_var_z must be ({N},); got {self.resid_var_z.shape}"
            )

    @property
    def n_assets(self) -> int:
        return len(self.asset_names)


# Extraction from a fitted discovery window
def asset_graph_from_discovery(
    disc,
    joint_window: pd.DataFrame | None,
    *,
    method: str,
    tau: float = 0.0,
    universe: Sequence[str] | None = None,
) -> AssetGraphWindow:
    """Extract the sliced, thresholded asset-asset graph from a fit window.

    ``joint_window`` is the exact data window the fit saw; it is needed for
    the residual variances (pass ``None`` only in tests). ``tau`` is applied
    before residuals are computed, so they match the graph the allocator
    uses. ``universe`` restricts to the eligible names; ``None`` keeps all.
    """
    disc_assets = list(disc.asset_columns)
    if universe is None:
        names = disc_assets
    else:
        names = [a for a in universe if a in disc_assets]
    if len(names) < 2:
        raise ValueError(
            f"asset graph needs ≥2 assets after universe slicing; got {names}"
        )
    pos = [disc_assets.index(a) for a in names]

    # Rows and columns sliced together so M stays square on `names`.
    M = np.asarray(disc.asset_to_asset_block(0), dtype=float)[np.ix_(pos, pos)].copy()
    if tau > 0.0:
        M[np.abs(M) < tau] = 0.0
    np.fill_diagonal(M, 0.0)

    # z-stats are stored full-length over disc.columns; slice to the asset block.
    asset_idx = np.asarray(disc.asset_idx, dtype=int)
    mean_a = np.asarray(disc.zscore_mean, dtype=float)[asset_idx][pos]
    std_a = np.asarray(disc.zscore_std, dtype=float)[asset_idx][pos]

    # Residuals in z-space with the stored z-stats and thresholded M:
    # x = x M + eps (row form) so E = X_z (I - M).
    resid_var_z: np.ndarray | None = None
    if joint_window is not None:
        missing = [a for a in names if a not in joint_window.columns]
        if missing:
            raise ValueError(f"joint_window is missing asset columns: {missing}")
        X = joint_window[names].to_numpy(dtype=float)
        X_z = (X - mean_a) / np.where(std_a > 1e-12, std_a, 1e-12)
        E = X_z @ (np.eye(len(names)) - M)
        resid_var_z = E.var(axis=0, ddof=0)

    dag = is_dag_matrix(M)
    if not dag:
        logger.debug(
            "asset graph (%s, %s) is not a DAG after τ=%g", method,
            getattr(disc, "end_date", "?"), tau,
        )

    return AssetGraphWindow(
        end_date=pd.Timestamp(getattr(disc, "end_date", pd.NaT)),
        asset_names=names,
        M=M,
        zscore_std=std_a,
        resid_var_z=resid_var_z,
        method=method,
        tau=tau,
        is_dag=dag,
        meta={
            "n_assets_discovery": len(disc_assets),
            "n_edges": int(np.count_nonzero(M)),
        },
    )


__all__ = ["AssetGraphWindow", "asset_graph_from_discovery", "is_dag_matrix"]
