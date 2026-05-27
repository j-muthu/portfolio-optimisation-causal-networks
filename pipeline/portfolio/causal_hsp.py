"""Strategy variants V0 / V0' / V1 / V2 (per ``Closed-Loop Causal-HSP Portfolio.md``).

The four strategies share most of the machinery and only differ in *which
distance matrix* enters HRP's clustering stage:

* **V0 — Vanilla HSP** -- cumulative-correlation driver selection, then
  sensitivity-space distance. Caller supplies ``S`` computed from the
  Rodriguez-Dominguez cum-corr selection.
* **V0' — Asset-only Causal-HRP** -- no drivers; uses the asset-asset causal
  embedding distance (``causal_embedding_distance`` from the legacy
  ``portfolio._old_v123``).
* **V1 — Causal-HSP open-loop** -- causal-discovery + greedy selection
  produces the K drivers, FFNN gives ``S``, then HSP clustering. ``α=1``.
* **V2 — Causal-HSP closed-loop** -- same as V1 but the driver-selection
  step blends causal evidence with the historical driver utility ``U[t-1]``.
  Caller controls α via the selector.

Each variant is a thin wrapper that delegates the actual clustering /
allocation math to :mod:`pipeline.portfolio.hrp` and :mod:`pipeline.portfolio.hsp`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.portfolio._old_v123 import causal_embedding_distance, nearest_psd
from pipeline.portfolio.hrp import hrp_weights
from pipeline.portfolio.hsp import hsp_weights_from_S, sample_covariance

logger = logging.getLogger(__name__)


# ============================================================================
# V0 — Vanilla HSP
# ============================================================================
def v0_vanilla_hsp(
    S: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Vanilla HSP: caller supplies the cum-corr-derived sensitivity matrix."""
    return hsp_weights_from_S(
        S=S, asset_names=asset_names, returns_window=returns_window,
        linkage_method=linkage_method,
    )


# ============================================================================
# V0' — Asset-only Causal-HRP
# ============================================================================
def v0prime_asset_only_causal_hrp(
    causal_W: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Asset-only Causal-HRP using the (asset_idx, asset_idx) block of W.

    Tests whether exogenous drivers add value over asset-only causal structure
    (the natural question raised by the supervisor pivot).
    """
    if causal_W.shape != (len(asset_names), len(asset_names)):
        raise ValueError(
            f"causal_W must be NxN on asset_names; got {causal_W.shape} "
            f"vs N={len(asset_names)}"
        )
    dist_arr = causal_embedding_distance(causal_W)
    dist_arr = nearest_psd(dist_arr)
    D = pd.DataFrame(dist_arr, index=asset_names, columns=asset_names)
    cov = sample_covariance(returns_window[asset_names].dropna())
    return hrp_weights(D, cov, linkage_method=linkage_method)


# ============================================================================
# V1 — Causal-HSP open-loop
# ============================================================================
def v1_causal_hsp_open_loop(
    S: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Identical math to V0; the contribution is upstream (causal selection)."""
    return v0_vanilla_hsp(
        S=S, asset_names=asset_names, returns_window=returns_window,
        linkage_method=linkage_method,
    )


# ============================================================================
# V2 — Causal-HSP closed-loop
# ============================================================================
def v2_causal_hsp_closed_loop(
    S: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """V2 weights from S; the feedback effect enters the *selection* upstream.

    The clustering / allocation step is identical to V1 — the closed-loop
    contribution lives in the driver selection (``α < 1`` blends causal and
    utility), which produces a different ``S`` to V1 in general. This wrapper
    keeps the variant grid clean.
    """
    return v0_vanilla_hsp(
        S=S, asset_names=asset_names, returns_window=returns_window,
        linkage_method=linkage_method,
    )


__all__ = [
    "v0_vanilla_hsp",
    "v0prime_asset_only_causal_hrp",
    "v1_causal_hsp_open_loop",
    "v2_causal_hsp_closed_loop",
]
