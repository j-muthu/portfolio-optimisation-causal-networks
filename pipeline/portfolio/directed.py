"""Direction-aware allocation from asset–asset causal graphs (Phase II).

Phase I's V0′ symmetrises the discovered asset–asset block
(``causal_embedding_distance``) before clustering — it discards exactly the
directional information that motivated causal discovery. This module keeps
it. The structural model (both DYNOTEARS ``W`` and VARLiNGAM ``B0`` in the
repo's ``i → j`` convention) is

    (I − Mᵀ) x = ε        ⇒        x = (I − Mᵀ)⁻¹ ε  =:  B ε

``M`` restricted to the asset block is a DAG (DYNOTEARS enforces acyclicity;
LiNGAM's causal order makes ``B0`` permutation-triangular), so the adjacency
is nilpotent and ``B = Σₖ (Mᵀ)ᵏ`` terminates exactly — no spectral-radius
assumption. ``B`` propagates influence through *all* directed paths; its
asymmetry is where edge direction enters allocation natively.

The allocator family (every function takes ``(AssetGraphWindow,
returns_window)`` and returns a name-indexed long-only weight Series summing
to 1 — the same contract as ``v0prime_asset_only_causal_hrp``):

* **D0** — embedding distance + sample cov (= V0′; the replication control).
* **D0s** — ``(|M|+|Mᵀ|)/2`` distance + sample cov (second symmetrisation).
* **D1** — embedding distance + **structural covariance** (direction enters
  the allocation step through ``B``).
* **D2/D2s** — topological-order bisection (``pipeline.portfolio.topological``).
* **D3** — no hierarchy: long-only equal-risk-contribution on Σ_struct.
  Portfolio risk decomposes as ``wᵀΣw = ‖Σ_ε^{1/2} Bᵀ w‖²`` — parity of
  contributions on Σ_struct is parity over *structural shock origins*
  rather than over correlated returns. Exact per-shock parity generally
  requires shorting; the long-only ERC on Σ_struct is the practical
  projection (stated, not clipped).
* **D4** — co-ancestry distance from ``B̃ B̃ᵀ`` (row-normalised ``B``): two
  assets are close iff they inherit shocks from the same upstream sources,
  even with no direct edge.

All are deterministic and seed-free by construction — no FFNN anywhere.
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
from pipeline.portfolio.hrp import hrp_weights
from pipeline.portfolio.hsp import (
    defactored_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Total-effect matrix B and the structural covariance (v2)
# ============================================================================
def total_effect_matrix(
    M: np.ndarray,
    is_dag: bool = True,
    k_trunc: int = 10,
    spectral_target: float = 0.95,
) -> np.ndarray:
    """``B = (I − Mᵀ)⁻¹`` — the total-effect (Leontief-inverse) matrix.

    ``B[i, j] = ∂x_i / ∂ε_j``: the total effect of a unit structural shock at
    asset ``j`` on asset ``i``, summed over all directed paths. Exact (via
    ``solve``) when the graph is a DAG. For non-DAG graphs (the GRANGER
    comparator's lagged matrix is not guaranteed acyclic) a truncated Neumann
    series ``Σ_{k≤k_trunc} (Mᵀ)ᵏ`` is used, with ``M`` rescaled to spectral
    radius ``spectral_target`` if ρ(M) ≥ 1 — a documented approximation.
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
    """SEM-implied covariance in *return units*: ``Σ = D_σ (B Σ_ε Bᵀ) D_σ``.

    Two upgrades over the legacy ``_old_v123.structural_covariance``:

    1. ``Σ_ε = diag(var(E))`` is estimated from the fit window's structural
       residuals (``graph.resid_var_z``), not defaulted to identity.
       Diagonality is the SEM's own independent-shocks assumption — a dense
       Σ_ε would smuggle sample correlation back in and destroy the
       fixed-graph ablation.
    2. **De-standardisation.** Discovery is fit on per-window z-scored data,
       so ``B Σ_ε Bᵀ`` lives in z-units. Allocation compares cluster
       variances, so skipping the rescale would silently equalise asset
       vols: ``Σ = D_σ Σ_z D_σ`` with ``D_σ = diag(zscore_std)``.

    The result is nearest-PSD projected and ridge-loaded
    (``+ ridge · tr(Σ)/N · I``) so recursive bisection / ERC never see a
    singular matrix in dense-graph stress windows.
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


# ============================================================================
# Long-only equal-risk-contribution (Spinu-style cyclical coordinate descent)
# ============================================================================
class ERCConvergenceError(RuntimeError):
    """The ERC coordinate descent failed to reach risk-contribution parity."""


def erc_weights(
    cov: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    rc_tol: float = 1e-8,
) -> np.ndarray:
    """Long-only ERC via cyclical coordinate descent on the log-barrier form.

    Minimises ``½ wᵀΣw − λ Σᵢ ln wᵢ`` (Spinu 2013); each coordinate update is
    the positive root of ``Σᵢᵢ wᵢ² + (Σ_{j≠i} Σᵢⱼ wⱼ) wᵢ − λ = 0``, which
    exists whenever the diagonal is positive. Deterministic: init ``w = 1/N``,
    fixed sweep order. The result is normalised to sum 1 (risk contributions
    are scale-invariant, so parity is preserved).
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


# ============================================================================
# Shared helpers
# ============================================================================
def _sample_cov(graph: AssetGraphWindow, returns_window: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance on the graph's asset set — the exact V0′ recipe."""
    return sample_covariance(returns_window[list(graph.asset_names)].dropna())


def _hrp_from_distance(
    dist_arr: np.ndarray,
    graph: AssetGraphWindow,
    covariance: pd.DataFrame,
    linkage_method: str,
) -> pd.Series:
    """House pattern: nearest-PSD the distance, then HRP (mirrors V0′)."""
    dist_arr = nearest_psd(dist_arr)
    D = pd.DataFrame(dist_arr, index=graph.asset_names, columns=graph.asset_names)
    return hrp_weights(D, covariance, linkage_method=linkage_method)


# ============================================================================
# The D-variant allocators
# ============================================================================
def corr_hrp_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """CORR — plain correlation-distance HRP (López de Prado 2016), the
    like-for-like control for the skeleton-vs-orientation decomposition.

    Ignores the causal graph entirely: the distance is ``√(½(1−ρ))`` on the
    sample correlation of the lookback window, everything downstream (the
    nearest-PSD house pattern, sample covariance, linkage, recursive
    bisection) is byte-identical to the D-variants — so D0 − CORR isolates
    exactly the replacement of the correlation matrix by the graph skeleton.
    The ``graph`` argument supplies only the asset universe.
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
    """D0 — embedding distance + sample cov. Identical math to V0′ (the
    replication gate asserts this against the Phase-I bundle)."""
    return _hrp_from_distance(
        causal_embedding_distance(graph.M), graph,
        _sample_cov(graph, returns_window), linkage_method,
    )


def d0s_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D0s — ``(|M|+|Mᵀ|)/2`` distance + sample cov (2nd symmetrisation)."""
    return _hrp_from_distance(
        symmetrise_distance(graph.M), graph,
        _sample_cov(graph, returns_window), linkage_method,
    )


def d0lw_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D0lw — D0's clustering with a Ledoit-Wolf shrunk covariance. A
    direction-free mechanism control (PREDICTIONS_COVARIANCE_CONTROLS.md):
    if generic shrinkage explains the D1 − D0 gap, this closes it with no
    directional content. Outside both SPA families by construction."""
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
    """D0df — D0's clustering with a single-factor residual (de-factored)
    covariance. The second mechanism control: if stripping the market factor
    explains the D1 − D0 gap, this closes it with no directional content."""
    rets = returns_window[list(graph.asset_names)].dropna()
    return _hrp_from_distance(
        causal_embedding_distance(graph.M), graph,
        defactored_covariance(rets), linkage_method,
    )


def d1_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D1 — embedding distance (as D0) but allocation on Σ_struct: direction
    enters the recursive-bisection step through ``B``."""
    return _hrp_from_distance(
        causal_embedding_distance(graph.M), graph,
        structural_covariance_v2(graph), linkage_method,
    )


def d3_srp_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",  # unused; kept for the dispatch contract
) -> pd.Series:
    """D3 — structural-shock risk parity: long-only ERC on Σ_struct, no
    hierarchy at all."""
    cov = structural_covariance_v2(graph)
    w = erc_weights(cov.to_numpy())
    return pd.Series(w, index=graph.asset_names, name="weight")


def d4_coancestry_weights(
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """D4 — co-ancestry clustering: similarity ``S = B̃ B̃ᵀ`` on row-normalised
    ``B`` (each row = an asset's shock-inheritance profile), distance
    ``√(2(1−S))``, then the unchanged symmetric HRP pipeline on sample cov."""
    B = total_effect_matrix(graph.M, is_dag=graph.is_dag)
    norms = np.linalg.norm(B, axis=1, keepdims=True)
    B_t = B / np.maximum(norms, 1e-12)
    S = np.clip(B_t @ B_t.T, -1.0, 1.0)
    dist = np.sqrt(np.clip(2.0 * (1.0 - S), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return _hrp_from_distance(
        dist, graph, _sample_cov(graph, returns_window), linkage_method,
    )


# ============================================================================
# Dispatch (the runner's single entry point)
# ============================================================================
ALLOCATORS = ("CORR", "D0", "D0s", "D0lw", "D0df", "D1", "D2", "D2s", "D3", "D4")


def dispatch_allocator(
    name: str,
    graph: AssetGraphWindow,
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Route an allocator tag to its weight function (same contract for all)."""
    if name in ("D2", "D2s"):
        # Local import: topological.py needs Σ_struct from this module, so the
        # dependency is one-way at import time and lazy here.
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
