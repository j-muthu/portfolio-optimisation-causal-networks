"""Direction-aware allocators (Phase II): the D-variant family.

Edge direction enters through the total-effect matrix ``B = (I - Mᵀ)⁻¹`` of
the structural model ``(I - Mᵀ) x = ε``. Every allocator takes
``(AssetGraphWindow, returns_window)`` and returns a name-indexed long-only
weight Series summing to 1. All deterministic and seed-free.
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
    """Total-effect matrix ``B = (I - Mᵀ)⁻¹``; ``B[i, j]`` is the effect of a
    unit shock at ``j`` on ``i`` over all directed paths.

    Exact solve for DAGs. Non-DAG inputs (GRANGER) get a truncated Neumann
    series, rescaling M to spectral radius ``spectral_target`` if ρ(M) ≥ 1.
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
    """SEM-implied covariance in return units: ``Σ = D_σ (B Σ_ε Bᵀ) D_σ``.

    Σ_ε is diagonal, from the fit window's structural residuals (a dense Σ_ε
    would smuggle sample correlation back in). D_σ de-standardises from the
    z-scored discovery units; skipping it would equalise asset vols. The
    result is nearest-PSD projected and ridge-loaded to avoid singularity.
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
    """Long-only ERC via cyclical coordinate descent on the log-barrier form
    ``½ wᵀΣw − λ Σᵢ ln wᵢ`` (Spinu 2013). Deterministic: fixed init and sweep order.
    """
    cov = np.asarray(cov, dtype=float)
    N = cov.shape[0]
    if np.any(np.diag(cov) <= 0):
        raise ValueError("ERC needs a strictly positive covariance diagonal")
    lam = float(np.trace(cov)) / (N * N)  # scale-matched barrier weight
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
    """Sample covariance on the graph's asset set (the exact V0' recipe)."""
    return sample_covariance(returns_window[list(graph.asset_names)].dropna())


def _hrp_from_distance(
    dist_arr: np.ndarray,
    graph: AssetGraphWindow,
    covariance: pd.DataFrame,
    linkage_method: str,
) -> pd.Series:
    """Nearest-PSD the distance, then HRP (mirrors V0')."""
    dist_arr = nearest_psd(dist_arr)
    D = pd.DataFrame(dist_arr, index=graph.asset_names, columns=graph.asset_names)
    return hrp_weights(D, covariance, linkage_method=linkage_method)


# The D-variant allocators
def corr_hrp_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """CORR: plain correlation-distance HRP (López de Prado 2016), the
    graph-blind control. Everything downstream of the distance is identical
    to the D-variants; ``graph`` supplies only the asset universe.
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
    """D0: embedding distance + sample cov. Identical math to V0'."""
    return _hrp_from_distance(
        causal_embedding_distance(graph.M), graph,
        _sample_cov(graph, returns_window), linkage_method,
    )


def d0s_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D0s: ``(|M|+|Mᵀ|)/2`` distance + sample cov (2nd symmetrisation)."""
    return _hrp_from_distance(
        symmetrise_distance(graph.M), graph,
        _sample_cov(graph, returns_window), linkage_method,
    )


def d0lw_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D0lw: D0's clustering with a Ledoit-Wolf covariance. Direction-free
    shrinkage control (PREDICTIONS_COVARIANCE_CONTROLS.md)."""
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
    """D0df: D0's clustering with a single-factor residual covariance.
    Direction-free de-factoring control."""
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
) -> pd.Series:
    """Nearest-PSD the distance, then HERC (mirrors ``_hrp_from_distance``)."""
    dist_arr = nearest_psd(dist_arr)
    D = pd.DataFrame(dist_arr, index=graph.asset_names, columns=graph.asset_names)
    return herc_weights(D, covariance, linkage_method=linkage_method)


def hercc_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """HERCC: correlation-distance HERC, the graph-blind HERC control
    (PREDICTIONS_HERC.md)."""
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
    """HERC0: embedding-distance HERC + sample cov (D0's distance, HERC's
    tree-reading rule)."""
    return _herc_from_distance(
        causal_embedding_distance(graph.M), graph,
        _sample_cov(graph, returns_window), linkage_method,
    )


def herc1_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """HERC1: embedding-distance HERC on Σ_struct (D1's covariance, HERC's
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
    """D0pc: graph-free skeleton control (PREDICTIONS_SKELETON_CONTROL.md).

    D0 with the discovered skeleton replaced by a thresholded
    partial-correlation matrix, density-matched to the paired graph's
    nonzero-cell count. No causal discovery anywhere."""
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
    """EW: 1/N over the graph's asset set (naive anchor; graph and window unused)."""
    names = list(graph.asset_names)
    return pd.Series(1.0 / len(names), index=names)


def ivp_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """IVP: inverse-variance weights on the window's sample variances (graph unused)."""
    rets = returns_window[list(graph.asset_names)].dropna()
    iv = 1.0 / rets.var().to_numpy(dtype=float)
    return pd.Series(iv / iv.sum(), index=list(graph.asset_names))


def d1_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D1: D0's distance but allocation on Σ_struct, so direction enters
    recursive bisection through ``B``."""
    return _hrp_from_distance(
        causal_embedding_distance(graph.M), graph,
        structural_covariance_v2(graph), linkage_method,
    )


def d3_srp_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",  # unused; kept for the dispatch contract
) -> pd.Series:
    """D3: structural-shock risk parity, long-only ERC on Σ_struct (no hierarchy)."""
    cov = structural_covariance_v2(graph)
    w = erc_weights(cov.to_numpy())
    return pd.Series(w, index=graph.asset_names, name="weight")


def d4_coancestry_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D4: co-ancestry clustering. Similarity ``S = B̃ B̃ᵀ`` on row-normalised
    ``B``, distance ``√(2(1−S))``, then standard HRP on sample cov."""
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
    """Route an allocator tag to its weight function (same contract for all)."""
    if name in ("D2", "D2s"):
        # Lazy import: topological.py imports from this module.
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
