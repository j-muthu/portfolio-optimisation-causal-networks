"""NTS-NOTEARS joint-window wrapper (non-linear discovery probe).

Wraps the vendored NTS-NOTEARS behind the DYNOTEARS interface; each edge is
summarised by the L2-norm of its CNN kernel. The asset -> driver prior is
enforced via NTS's native bound-dicts. A reduced-scope probe, not a full
backtest path: fits are too slow to run at every rebalance.
"""

from __future__ import annotations

import logging
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT
from pipeline.discovery.dynotears import JointDynotearsWindow

logger = logging.getLogger(__name__)

_NTS_DIR = THESIS_ROOT / "nts-notears" / "notears"


def _import_nts():
    """Lazy import of the vendored NTS-NOTEARS; its bare relative imports need
    the notears/ dir on sys.path, and this defers the torch import."""
    if str(_NTS_DIR) not in sys.path:
        sys.path.insert(0, str(_NTS_DIR))
    # source code available at: https://github.com/xiangyu-sun-789/NTS-NOTEARS
    import nts_notears as _nts  # noqa: E402
    return _nts


def _asset_to_driver_prior(driver_columns, asset_columns, p):
    """Bound-dicts forbidding every asset->driver edge at lags 0..p."""
    rules = []
    for lag in range(p + 1):
        for a in asset_columns:
            for d in driver_columns:
                rules.append({"from_node": a, "from_lag": lag, "to_node": d,
                              "lower_bound": 0.0, "upper_bound": 0.0})
    return rules


def run_nts_notears_joint_window(
    joint_window: pd.DataFrame,
    driver_columns: Sequence[str],
    asset_columns: Sequence[str],
    p: int = 1,
    hidden: int = 8,
    lambda1: float = 0.02,
    lambda2: float = 0.01,
    w_threshold: float = 0.1,
    max_iter: int = 20,
    h_tol: float = 1e-6,
    enforce_prior: bool = True,
) -> JointDynotearsWindow:
    """Fit NTS-NOTEARS on one joint ``[D | A]`` window; returns a
    ``JointDynotearsWindow`` of kernel-norm magnitudes (``W[i, j]`` = i->j)."""
    nts = _import_nts()
    # source code available at: https://github.com/pytorch/pytorch
    import torch

    columns = list(joint_window.columns)
    d = len(columns)
    driver_columns = list(driver_columns)
    asset_columns = list(asset_columns)
    driver_idx = np.array([columns.index(c) for c in driver_columns])
    asset_idx = np.array([columns.index(c) for c in asset_columns])

    # Per-window z-score (NTS expects standardised input).
    X = joint_window.to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    Xn = (X - mean) / std_safe

    prior = _asset_to_driver_prior(driver_columns, asset_columns, p) if enforce_prior else None

    device = torch.device("cpu")
    model = nts.NTS_NOTEARS(
        dims=[d, hidden, 1], bias=True, number_of_lags=p,
        prior_knowledge=prior, variable_names_no_time=columns,
    ).to(device)

    W_full = nts.train_NTS_NOTEARS(
        model, Xn.astype(np.float32), device, lambda1=lambda1, lambda2=lambda2,
        w_threshold=w_threshold, max_iter=max_iter, h_tol=h_tol, verbose=0,
    )
    # W_full is (d*(p+1), d*(p+1)); only the last d columns (contemporaneous
    # targets) are populated. Rows are stacked [lag_oldest..lag_1, instantaneous].
    tgt = W_full[:, -d:]
    A = [np.ascontiguousarray(tgt[i * d:(i + 1) * d, :]) for i in range(p)]  # lag blocks
    W = np.ascontiguousarray(tgt[p * d:(p + 1) * d, :])                      # instantaneous

    return JointDynotearsWindow(
        index=0, start_row=0, end_row=len(joint_window),
        start_date=pd.Timestamp(joint_window.index[0]),
        end_date=pd.Timestamp(joint_window.index[-1]),
        columns=columns, driver_columns=driver_columns, asset_columns=asset_columns,
        driver_idx=driver_idx, asset_idx=asset_idx,
        W=W, A=A, p=p,
        lambda_w=lambda1, lambda_a=lambda2,
        converged=True, acyclic_edges_removed=0,
        zscore_mean=mean, zscore_std=std,
        fit_loss=float("nan"), tabu_enforced=enforce_prior,
    )


__all__ = ["run_nts_notears_joint_window"]
