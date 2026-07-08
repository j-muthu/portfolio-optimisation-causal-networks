"""Single chokepoint: the per-window asset–asset directed graph (Phase II).

Every direction-aware allocator (``pipeline.portfolio.directed`` /
``pipeline.portfolio.topological``) consumes exactly one type,
:class:`AssetGraphWindow`, extracted here from a fitted discovery window
(``JointDynotearsWindow``, ``JointVarLingamWindow`` or
``JointGrangerWindow``). Allocators never touch the discovery dataclasses
directly — the Phase-II fixed-graph ablation depends on every allocator
seeing the byte-identical ``M`` for a given (method, window, τ), and this
module is where that is guaranteed.

Conventions encoded once, here:

* ``M[i, j]`` is the effect of asset ``i`` on asset ``j`` (the repo-wide
  ``i → j`` convention; VARLiNGAM's ``B0`` is *already* transposed to this
  convention at fit time — do not re-transpose).
* The magnitude threshold ``tau`` is applied here, so the E3 sparsity sweep
  is a single extraction-level flag.
* Universe restriction drops rows *and* columns of ``M`` together with the
  matching entries of ``asset_names`` / ``zscore_std`` — late-inception
  names excluded by the eligibility mask can never desynchronise the graph
  from the returns panel.
* Structural residual variances (the diagonal ``Σ_ε`` needed by
  ``structural_covariance_v2``) are computed here, on the *fit window's*
  z-scored data using the z-stats stored on the discovery window — so they
  are consistent with the data the graph was fitted on, restricted to the
  same sliced asset set as ``M``. Note the asset-block SEM marginalises
  over drivers: ``ε`` absorbs both idiosyncratic noise and driver-explained
  variance. Diagonality of ``Σ_ε`` is therefore an approximation (common
  driver shocks correlate residuals across assets); this is a documented
  modelling choice, not an estimation shortcut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# DAG check (cheap Kahn attempt on the binarised adjacency)
# ============================================================================
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


# ============================================================================
# The one type every allocator consumes
# ============================================================================
@dataclass(frozen=True)
class AssetGraphWindow:
    """Asset–asset directed graph for one rebalance window.

    ``M`` is ``(N, N)`` in the ``i → j`` convention, τ-thresholded, sliced
    to ``asset_names`` (rows and columns in the same order). ``zscore_std``
    maps z-units back to return units for de-standardising the structural
    covariance; ``resid_var_z`` is the per-asset structural residual
    variance in z-units (``None`` when the fit window was not supplied —
    only acceptable in unit tests).
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


# ============================================================================
# Extraction from a fitted discovery window
# ============================================================================
def asset_graph_from_discovery(
    disc,
    joint_window: pd.DataFrame | None,
    *,
    method: str,
    tau: float = 0.0,
    universe: Sequence[str] | None = None,
) -> AssetGraphWindow:
    """Extract the (sliced, thresholded) asset–asset graph from a fit window.

    Parameters
    ----------
    disc:
        A ``JointDynotearsWindow`` / ``JointVarLingamWindow`` /
        ``JointGrangerWindow`` — anything exposing ``asset_columns``,
        ``asset_idx``, ``asset_to_asset_block(0)``, ``zscore_mean`` and
        ``zscore_std`` (full-length over ``columns``).
    joint_window:
        The exact data window the fit saw (rows = trading days, columns ⊇
        asset columns). Required to compute the structural residual
        variances; pass ``None`` only in tests (``resid_var_z`` will be
        ``None`` and Σ_struct falls back to unit shocks with a warning).
    tau:
        Magnitude threshold: entries with ``|M| < tau`` are zeroed *before*
        residuals are computed, so the residuals correspond to the graph the
        allocator actually uses.
    universe:
        Eligible asset names at this rebalance, in the caller's order
        (typically ``universe_at(t)`` ∩ discovery columns). ``None`` keeps
        every discovery asset.
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

    # Rows AND columns sliced together — M stays square on `names`.
    M = np.asarray(disc.asset_to_asset_block(0), dtype=float)[np.ix_(pos, pos)].copy()
    if tau > 0.0:
        M[np.abs(M) < tau] = 0.0
    np.fill_diagonal(M, 0.0)

    # z-stats: stored full-length over disc.columns; asset_idx → asset block.
    asset_idx = np.asarray(disc.asset_idx, dtype=int)
    mean_a = np.asarray(disc.zscore_mean, dtype=float)[asset_idx][pos]
    std_a = np.asarray(disc.zscore_std, dtype=float)[asset_idx][pos]

    # Structural residuals on the fit window, in z-space, with the *stored*
    # z-stats (so X_z is exactly what the fit saw) and the *thresholded* M:
    #   x = x M + ε  (row form, i → j)   ⇒   E = X_z (I − M)
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
