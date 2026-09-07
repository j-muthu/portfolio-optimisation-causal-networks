"""Direction-aware allocators (Phase II): the D-variant family.

Edge direction enters through the total-effect matrix ``B = (I - Mᵀ)⁻¹`` of
the structural model ``(I - Mᵀ) x = ε``. Every allocator takes
``(AssetGraphWindow, returns_window)`` and returns a name-indexed long-only
weight Series that sums to 1. All of them are deterministic and use no seed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.discovery.asset_graph import AssetGraphWindow
from pipeline.portfolio._old_v123 import (
    causal_embedding_distance,
    correlation_distance,
    nearest_psd,
    symmetrise_distance,
)
from pipeline.portfolio.hrp import herc_weights, hrp_weights
from pipeline.portfolio.hsp import (
    defactored_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
)

logger = logging.getLogger(__name__)


# Total-effect matrix B and the structural covariance
def total_effect_matrix(
    M: np.ndarray,
    is_dag: bool = True,
    k_trunc: int = 10,
    spectral_target: float = 0.95,
) -> np.ndarray:
    """Return the total-effect matrix ``B = (I - Mᵀ)⁻¹``. ``B[i, j]`` is the
    effect of a unit shock at ``j`` on ``i`` over all directed paths.

    I solve exactly for DAGs. Inputs that are not DAGs (GRANGER) get a
    truncated Neumann series, and I rescale M to the spectral radius
    ``spectral_target`` if the spectral radius of M is 1 or more.
    """
    N = M.shape[0]
    if is_dag:
        return np.linalg.solve(np.eye(N) - M.T, np.eye(N))
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    Mt = M.T if rho < 1.0 else (M.T * (spectral_target / rho))
    if rho >= 1.0:
        logger.warning(
            "total_effect_matrix: ρ(M)=%.3f ≥ 1 — rescaled to %.2f before the "
            "truncated Neumann series", rho, spectral_target,
        )
    B = np.eye(N)
    term = np.eye(N)
    for _ in range(k_trunc):
        term = term @ Mt
        B = B + term
    return B


def structural_covariance_v2(
    graph: AssetGraphWindow,
    ridge: float = 1e-6,
    k_trunc: int = 10,
) -> pd.DataFrame:
    """Return the SEM-implied covariance in return units,
    ``Σ = D_σ (B Σ_ε Bᵀ) D_σ``.

    Σ_ε is diagonal and comes from the fit window's structural residuals. A
    dense Σ_ε would bring sample correlation back in. D_σ undoes the z-scoring
    of the discovery units, because skipping it would give every asset the
    same volatility. I project the result to the nearest PSD matrix and add a
    ridge so that it is not singular.
    """
    N = graph.n_assets
    if graph.resid_var_z is not None:
        resid = np.maximum(np.asarray(graph.resid_var_z, dtype=float), 1e-12)
    else:
        logger.warning(
            "structural_covariance_v2: no residual variances on the graph — "
            "falling back to unit shocks (tests only; real runs must supply "
            "the fit window at extraction)"
        )
        resid = np.ones(N)

    B = total_effect_matrix(graph.M, is_dag=graph.is_dag, k_trunc=k_trunc)
    cov_z = (B * resid[None, :]) @ B.T          # B diag(σ_ε²) Bᵀ
    sigma = np.asarray(graph.zscore_std, dtype=float)
    cov = cov_z * np.outer(sigma, sigma)         # D_σ Σ_z D_σ
    cov = nearest_psd(cov)
    cov = cov + (ridge * np.trace(cov) / N) * np.eye(N)
    cond = float(np.linalg.cond(cov))
    if cond > 1e10:
        logger.warning(
            "structural_covariance_v2 (%s): condition number %.2e after ridge",
            graph.end_date, cond,
        )
    return pd.DataFrame(cov, index=graph.asset_names, columns=graph.asset_names)


# Long-only equal-risk-contribution
class ERCConvergenceError(RuntimeError):
    """The ERC coordinate descent failed to reach risk-contribution parity."""


def erc_weights(
    cov: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    rc_tol: float = 1e-8,
) -> np.ndarray:
    """Return long-only ERC weights by cyclical coordinate descent on the
    log-barrier form ``½ wᵀΣw − λ Σᵢ ln wᵢ`` (Spinu 2013). The result is
    deterministic because the initial point and the sweep order are fixed.
    """
    cov = np.asarray(cov, dtype=float)
    N = cov.shape[0]
    if np.any(np.diag(cov) <= 0):
        raise ValueError("ERC needs a strictly positive covariance diagonal")
    lam = float(np.trace(cov)) / (N * N)  # barrier weight matched to the scale of cov
    w = np.full(N, 1.0 / N)
    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(N):
            b = float(cov[i] @ w - cov[i, i] * w[i])
            a = float(cov[i, i])
            w[i] = (-b + np.sqrt(b * b + 4.0 * a * lam)) / (2.0 * a)
        if float(np.max(np.abs(w - w_prev))) < tol:
            break
    w = w / w.sum()
    rc = w * (cov @ w)
    spread = float(rc.max() - rc.min()) / max(float(rc.mean()), 1e-300)
    if spread > rc_tol:
        raise ERCConvergenceError(
            f"ERC risk-contribution spread {spread:.2e} > {rc_tol:.0e} after "
            f"{max_iter} sweeps"
        )
    return w


# Shared helpers
def _sample_cov(graph: AssetGraphWindow, returns_window: pd.DataFrame) -> pd.DataFrame:
    """Return the sample covariance on the graph's asset set (the same recipe
    as V0')."""
    return sample_covariance(returns_window[list(graph.asset_names)].dropna())


def _hrp_from_distance(
    dist_arr: np.ndarray,
    graph: AssetGraphWindow,
    covariance: pd.DataFrame,
    linkage_method: str,
    psd_project_distance: bool = False,
) -> pd.Series:
    """Run HRP on the given clustering distance.

    ``psd_project_distance=True`` reproduces the Phase I behaviour of
    projecting the distance to the nearest PSD matrix before linkage. That
    projection is a bug for clustering. A Euclidean distance matrix has
    exactly 1 positive eigenvalue, so clipping the rest gives a rank-1 matrix
    and the single-linkage dendrogram becomes a chain ordered by its top
    eigenvector. I leave it off by default and keep it only to replay the
    committed Phase I result.
    """
    if psd_project_distance:
        dist_arr = nearest_psd(dist_arr)
    D = pd.DataFrame(dist_arr, index=graph.asset_names, columns=graph.asset_names)
    return hrp_weights(D, covariance, linkage_method=linkage_method)


# The D-variant allocators
def corr_hrp_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """CORR: plain correlation-distance HRP (López de Prado 2016). This is the
    baseline that uses no graph. Everything after the distance is identical
    to the D-variants, and ``graph`` supplies only the asset universe.
    """
    rets = returns_window[list(graph.asset_names)].dropna()
    corr = rets.corr().to_numpy()
    return _hrp_from_distance(
        correlation_distance(corr), graph, sample_covariance(rets), linkage_method,
    )


def d0_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D0: embedding distance and sample covariance. The maths is identical
    to V0'."""
    return _hrp_from_distance(
        causal_embedding_distance(graph.M), graph,
        _sample_cov(graph, returns_window), linkage_method,
    )


def d0s_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D0s: ``(|M|+|Mᵀ|)/2`` distance and sample covariance (the second
    symmetrisation)."""
    return _hrp_from_distance(
        symmetrise_distance(graph.M), graph,
        _sample_cov(graph, returns_window), linkage_method,
    )


def d0lw_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D0lw: D0's clustering with a Ledoit-Wolf covariance. This is the
    shrinkage control that uses no edge directions
    (PREDICTIONS_COVARIANCE_CONTROLS.md)."""
    rets = returns_window[list(graph.asset_names)].dropna()
    return _hrp_from_distance(
        causal_embedding_distance(graph.M), graph,
        ledoit_wolf_covariance(rets), linkage_method,
    )


def d0df_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D0df: D0's clustering with a single-factor residual covariance. This
    is the de-factoring control that uses no edge directions."""
    rets = returns_window[list(graph.asset_names)].dropna()
    return _hrp_from_distance(
        causal_embedding_distance(graph.M), graph,
        defactored_covariance(rets), linkage_method,
    )


def _herc_from_distance(
    dist_arr: np.ndarray,
    graph: AssetGraphWindow,
    covariance: pd.DataFrame,
    linkage_method: str,
    psd_project_distance: bool = False,
) -> pd.Series:
    """Run HERC on the given clustering distance. The flag means the same as
    in :func:`_hrp_from_distance`."""
    if psd_project_distance:
        dist_arr = nearest_psd(dist_arr)
    D = pd.DataFrame(dist_arr, index=graph.asset_names, columns=graph.asset_names)
    return herc_weights(D, covariance, linkage_method=linkage_method)


def hercc_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """HERCC: correlation-distance HERC. This is the HERC baseline that uses
    no graph (PREDICTIONS_HERC.md)."""
    rets = returns_window[list(graph.asset_names)].dropna()
    corr = rets.corr().to_numpy()
    return _herc_from_distance(
        correlation_distance(corr), graph, sample_covariance(rets), linkage_method,
    )


def herc0_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """HERC0: embedding-distance HERC with sample covariance (D0's distance
    and HERC's tree-reading rule)."""
    return _herc_from_distance(
        causal_embedding_distance(graph.M), graph,
        _sample_cov(graph, returns_window), linkage_method,
    )


def herc1_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """HERC1: embedding-distance HERC on Σ_struct (D1's covariance and HERC's
    tree-reading rule)."""
    return _herc_from_distance(
        causal_embedding_distance(graph.M), graph,
        structural_covariance_v2(graph), linkage_method,
    )


def d0pc_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D0pc: the skeleton control that uses no graph
    (PREDICTIONS_SKELETON_CONTROL.md).

    This is D0 with the discovered skeleton replaced by a thresholded
    partial-correlation matrix. I match its density to the number of nonzero
    cells in the paired graph. No causal discovery is used anywhere."""
    rets = returns_window[list(graph.asset_names)].dropna()
    theta = np.linalg.inv(ledoit_wolf_covariance(rets).to_numpy())
    d = np.sqrt(np.diag(theta))
    pc = -theta / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    nnz = int(np.count_nonzero(graph.M))
    flat = np.sort(np.abs(pc).ravel())
    if 0 < nnz < flat.size:
        kth = flat[-nnz]
        pc = np.where(np.abs(pc) >= kth, pc, 0.0)
    return _hrp_from_distance(
        causal_embedding_distance(pc), graph,
        _sample_cov(graph, returns_window), linkage_method,
    )


def ew_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """EW: 1/N over the graph's asset set. The graph and the window are unused."""
    names = list(graph.asset_names)
    return pd.Series(1.0 / len(names), index=names)


def ivp_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """IVP: inverse-variance weights on the window's sample variances. The
    graph is unused."""
    rets = returns_window[list(graph.asset_names)].dropna()
    iv = 1.0 / rets.var().to_numpy(dtype=float)
    return pd.Series(iv / iv.sum(), index=list(graph.asset_names))


def d1_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D1: D0's distance with allocation on Σ_struct, so that edge direction
    enters recursive bisection through ``B``."""
    return _hrp_from_distance(
        causal_embedding_distance(graph.M), graph,
        structural_covariance_v2(graph), linkage_method,
    )


def d3_srp_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",  # unused, kept so that dispatch has 1 signature
) -> pd.Series:
    """D3: structural-shock risk parity, i.e. long-only ERC on Σ_struct with
    no hierarchy."""
    cov = structural_covariance_v2(graph)
    w = erc_weights(cov.to_numpy())
    return pd.Series(w, index=graph.asset_names, name="weight")


def d4_coancestry_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D4: co-ancestry clustering. I take the similarity ``S = B̃ B̃ᵀ`` on the
    row-normalised ``B``, the distance ``√(2(1−S))`` and then standard HRP on
    the sample covariance."""
    B = total_effect_matrix(graph.M, is_dag=graph.is_dag)
    norms = np.linalg.norm(B, axis=1, keepdims=True)
    B_t = B / np.maximum(norms, 1e-12)
    S = np.clip(B_t @ B_t.T, -1.0, 1.0)
    dist = np.sqrt(np.clip(2.0 * (1.0 - S), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return _hrp_from_distance(
        dist, graph, _sample_cov(graph, returns_window), linkage_method,
    )


# Dispatch
ALLOCATORS = ("CORR", "D0", "D0s", "D0lw", "D0df", "D0pc", "EW", "IVP",
              "HERCC", "HERC0", "HERC1",
              "D1", "D2", "D2s", "D3", "D4")


def dispatch_allocator(
    name: str,
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Call the weight function for an allocator tag. All of them have the
    same signature."""
    if name in ("D2", "D2s"):
        # Import lazily because topological.py imports from this module.
        from pipeline.portfolio.topological import d2_weights

        return d2_weights(
            graph, returns_window,
            covariance="structural" if name == "D2s" else "sample",
        )
    fn = {
        "CORR": corr_hrp_weights,
        "D0": d0_weights,
        "D0s": d0s_weights,
        "D0lw": d0lw_weights,
        "D0df": d0df_weights,
        "D0pc": d0pc_weights,
        "EW": ew_weights,
        "IVP": ivp_weights,
        "HERCC": hercc_weights,
        "HERC0": herc0_weights,
        "HERC1": herc1_weights,
        "D1": d1_weights,
        "D3": d3_srp_weights,
        "D4": d4_coancestry_weights,
    }.get(name)
    if fn is None:
        raise ValueError(f"unknown allocator {name!r}; expected one of {ALLOCATORS}")
    return fn(graph, returns_window, linkage_method=linkage_method)


__all__ = [
    "ALLOCATORS",
    "ERCConvergenceError",
    "corr_hrp_weights",
    "dispatch_allocator",
    "d0_weights",
    "d0s_weights",
    "d0lw_weights",
    "d0df_weights",
    "d1_weights",
    "d3_srp_weights",
    "d4_coancestry_weights",
    "erc_weights",
    "structural_covariance_v2",
    "total_effect_matrix",
]
