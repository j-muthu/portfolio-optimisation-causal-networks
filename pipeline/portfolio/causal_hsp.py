"""Strategy variants V0, V0', V1 and V2 (see ``Closed-Loop Causal-HSP Portfolio.md``).

They differ only in which distance matrix goes into HRP's clustering step.
Each one is a thin wrapper over :mod:`pipeline.portfolio.hrp` and
:mod:`pipeline.portfolio.hsp`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.portfolio._old_v123 import causal_embedding_distance, nearest_psd
from pipeline.portfolio.hrp import hrp_weights
from pipeline.portfolio.hsp import hsp_weights_from_S, sample_covariance

logger = logging.getLogger(__name__)


# V0 - Vanilla HSP
def v0_vanilla_hsp(
    S: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Vanilla HSP. The caller supplies the sensitivity matrix from
    cumulative-correlation selection."""
    return hsp_weights_from_S(
        S=S, asset_names=asset_names, returns_window=returns_window,
        linkage_method=linkage_method,
    )


# V0' - Asset-only Causal-HRP
def v0prime_asset_only_causal_hrp(
    causal_W: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
    psd_project_distance: bool = False,
) -> pd.Series:
    """Asset-only Causal-HRP on the (asset, asset) block of W.

    ``psd_project_distance=True`` replays the original Phase I run, which
    projected the distance to the nearest PSD matrix before linkage. This was
    a bug because the projection collapses a Euclidean distance matrix to
    rank 1. I leave it off by default.
    """
    if causal_W.shape != (len(asset_names), len(asset_names)):
        raise ValueError(
            f"causal_W must be NxN on asset_names; got {causal_W.shape} "
            f"vs N={len(asset_names)}"
        )
    dist_arr = causal_embedding_distance(causal_W)
    if psd_project_distance:
        dist_arr = nearest_psd(dist_arr)
    D = pd.DataFrame(dist_arr, index=asset_names, columns=asset_names)
    cov = sample_covariance(returns_window[asset_names].dropna())
    return hrp_weights(D, cov, linkage_method=linkage_method)


# V1 - Causal-HSP open-loop
def v1_causal_hsp_open_loop(
    S: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Same maths as V0. The difference is upstream, in the causal driver
    selection."""
    return v0_vanilla_hsp(
        S=S, asset_names=asset_names, returns_window=returns_window,
        linkage_method=linkage_method,
    )


# V2 - Causal-HSP closed-loop
def v2_causal_hsp_closed_loop(
    S: np.ndarray,
    asset_names: list[str],
    returns_window: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """V2 weights from S. The closed-loop feedback changes the driver
    selection upstream, so clustering and allocation are identical to V1.
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
