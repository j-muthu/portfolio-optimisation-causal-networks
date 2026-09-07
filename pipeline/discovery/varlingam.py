"""Rolling-window VARLiNGAM on S&P 500 log-returns.

VARLiNGAM has 2 stages. It fits a VAR, then runs DirectLiNGAM on the
residuals to get the contemporaneous structure. I transpose every matrix
exposed here from lingam's raw ``j -> i`` output into the ``i -> j``
convention used across the repository. For large d the OLS VAR is
underdetermined, so :func:`estimate_var_coefs_masked` provides a ridge
alternative that is passed in through ``ar_coefs``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from pipeline._parallel import execute_windows
from pipeline._vendored import VARLiNGAM
from pipeline.discovery.dynotears import rolling_windows

logger = logging.getLogger(__name__)

Criterion = Literal["aic", "bic", "hqic", "fpe"]


# Stage 1 joint-matrix path: drivers and assets with the asset -> driver mask.
# In lingam's prior_knowledge convention, -1 means no prior, 0 means no edge
# j -> i and 1 means an edge j -> i. Forbidding asset -> driver therefore
# means prior_knowledge[driver_j, asset_i] = 0.
def make_prior_knowledge_asset_to_driver(
    driver_idx: np.ndarray,
    asset_idx: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Return the DirectLiNGAM prior_knowledge matrix that forbids
    asset -> driver edges.

    All entries are -1 (no prior) except ``pk[driver_j, asset_i] = 0``.
    """
    pk = np.full((n_features, n_features), -1, dtype=int)
    for dj in driver_idx:
        for ai in asset_idx:
            pk[int(dj), int(ai)] = 0
    return pk


def estimate_var_coefs_masked(
    X: np.ndarray,
    lags: int,
    driver_idx: np.ndarray,
    asset_idx: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit a ridge VAR with the asset -> driver lag mask enforced row by row.

    Driver equations regress only on lagged drivers. Asset equations are
    unconstrained. The result is ``(lags, d, d)`` in the lingam convention
    (``M[tau, i, j]`` is the effect of lagged j on i), and the masked
    entries are exactly zero.
    """
    from sklearn.linear_model import Ridge

    X = np.asarray(X, dtype=float)
    n, d = X.shape
    design = np.concatenate(
        [X[lags - k - 1 : n - k - 1] for k in range(lags)], axis=1
    )  # shape (n - lags, lags*d)
    target = X[lags:]                                                     # shape (n - lags, d)

    driver_set = set(int(i) for i in driver_idx)
    # Indices in the design matrix of the lagged drivers across all lags.
    driver_design_cols = np.array(
        [k * d + j for k in range(lags) for j in range(d) if j in driver_set],
        dtype=int,
    )

    coef_T = np.zeros((d, lags * d), dtype=float)  # (n_targets, n_features)
    for i in range(d):
        is_driver = i in driver_set
        cols = driver_design_cols if is_driver else np.arange(lags * d)
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(design[:, cols], target[:, i])
        coef_T[i, cols] = model.coef_

    # coef_T[i, k*d+j] is the coefficient of x_{t-k-1}[j] in equation i, so
    # M[k, i, j] in the same convention is coef_T[i, k*d+j].
    return np.stack([coef_T[:, k * d : (k + 1) * d] for k in range(lags)], axis=0)


@dataclass
class JointVarLingamWindow:
    """VARLiNGAM output for 1 window of the joint ``[D | A]`` panel.

    It has the same layout as :class:`JointDynotearsWindow`. ``B0[i, j]``
    is i -> j.
    """

    index: int
    start_row: int
    end_row: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    columns: list[str]
    driver_columns: list[str]
    asset_columns: list[str]
    driver_idx: np.ndarray
    asset_idx: np.ndarray
    B0: np.ndarray
    B_lags: list[np.ndarray]
    causal_order: list[int]
    selected_lags: int
    zscore_mean: np.ndarray
    zscore_std: np.ndarray
    bootstrap_prob_B0: np.ndarray | None = None
    prior_knowledge_enforced: bool = True
    error_indep_pvalues: np.ndarray | None = None

    def driver_to_asset_block(self, lag: int) -> np.ndarray:
        mat = self.B0 if lag == 0 else self.B_lags[lag - 1]
        return mat[np.ix_(self.driver_idx, self.asset_idx)]

    def asset_to_driver_block(self, lag: int) -> np.ndarray:
        mat = self.B0 if lag == 0 else self.B_lags[lag - 1]
        return mat[np.ix_(self.asset_idx, self.driver_idx)]

    def asset_to_asset_block(self, lag: int) -> np.ndarray:
        """Return ``M[a, a]``, the asset-only causal block."""
        mat = self.B0 if lag == 0 else self.B_lags[lag - 1]
        return mat[np.ix_(self.asset_idx, self.asset_idx)]


@dataclass
class RollingJointVarLingamResult:
    windows: list[JointVarLingamWindow]
    columns: list[str]
    driver_columns: list[str]
    asset_columns: list[str]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([w.end_date for w in self.windows])


def run_varlingam_joint_window(
    joint_window: pd.DataFrame,
    driver_columns,
    asset_columns,
    lags: int = 1,
    criterion: Criterion | None = "bic",
    prune: bool = True,
    random_state: int = 42,
    ridge_alpha: float = 1.0,
    enforce_prior_knowledge: bool = True,
    n_bootstrap: int = 0,
    bootstrap_min_effect: float = 0.01,
    compute_error_independence: bool = False,
) -> JointVarLingamWindow:
    """Fit VARLiNGAM on 1 joint-matrix window with the asset -> driver mask.

    The lagged coefficients come from :func:`estimate_var_coefs_masked` and
    the contemporaneous B0 comes from DirectLiNGAM with a prior_knowledge
    mask. ``criterion`` is ignored when the mask is enforced, because I fit
    the VAR myself with fixed ``lags``. ``compute_error_independence`` runs
    the HSIC misspecification check, which needs O(d^2) tests, so use it as
    a spot check only.
    """
    columns = list(joint_window.columns)
    driver_columns = list(driver_columns)
    asset_columns = list(asset_columns)
    driver_idx = np.array([columns.index(c) for c in driver_columns], dtype=int)
    asset_idx = np.array([columns.index(c) for c in asset_columns], dtype=int)
    d = len(columns)

    # Z-score the window.
    mean = joint_window.mean(axis=0)
    std = joint_window.std(axis=0, ddof=0).where(lambda s: s > 1e-12, 1e-12)
    normalised = (joint_window - mean) / std
    X = normalised.to_numpy(dtype=float)

    # Pre-compute the masked VAR coefficients. This skips VARLiNGAM's own VAR step.
    if enforce_prior_knowledge:
        ar_coefs = estimate_var_coefs_masked(
            X, lags=lags, driver_idx=driver_idx, asset_idx=asset_idx, alpha=ridge_alpha
        )
        # Construct DirectLiNGAM with prior_knowledge.
        # source code available at: https://github.com/cdt15/lingam
        from lingam.direct_lingam import DirectLiNGAM

        pk = make_prior_knowledge_asset_to_driver(driver_idx, asset_idx, d)
        lingam_model = DirectLiNGAM(prior_knowledge=pk)
        effective_criterion = None  # ar_coefs is supplied, so the VAR step is skipped
    else:
        ar_coefs = None
        lingam_model = None
        effective_criterion = criterion

    model = VARLiNGAM(
        lags=lags,
        criterion=effective_criterion,
        prune=prune,
        ar_coefs=ar_coefs,
        lingam_model=lingam_model,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)

    am = model.adjacency_matrices_  # (lags + 1, d, d), lingam raw j -> i
    B0 = am[0].T.copy()
    B_lags = [am[k].T.copy() for k in range(1, len(am))]
    causal_order = [int(i) for i in model.causal_order_]

    # VARLiNGAM's pruning refits the lagged blocks without prior_knowledge, so
    # some mass ends up in B_tau[asset, driver]. I zero it explicitly. B0 is
    # already exactly enforced by DirectLiNGAM.
    if enforce_prior_knowledge:
        for B in (B0, *B_lags):
            B[np.ix_(asset_idx, driver_idx)] = 0.0

    bootstrap_prob_B0: np.ndarray | None = None
    if n_bootstrap > 0:
        np.random.seed(random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.bootstrap(X, n_sampling=n_bootstrap)
            probs = result.get_probabilities(min_causal_effect=bootstrap_min_effect)
        bootstrap_prob_B0 = np.asarray(probs)[:, :d].T.copy()

    error_indep_pvalues: np.ndarray | None = None
    if compute_error_independence:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            error_indep_pvalues = np.asarray(model.get_error_independence_p_values())
        # Fraction of off-diagonal p-values below 0.05. Roughly 5% is expected under the null.
        triu = np.triu_indices_from(error_indep_pvalues, k=1)
        rejection_rate = float(np.mean(error_indep_pvalues[triu] < 0.05))
        log_fn = logger.warning if rejection_rate > 0.20 else logger.info
        log_fn(
            "HSIC error-independence: %d pairs, rejection_rate@5%%=%.3f%s",
            len(triu[0]), rejection_rate,
            "  [LIKELY MISSPECIFIED]" if rejection_rate > 0.20 else "",
        )

    return JointVarLingamWindow(
        index=-1,
        start_row=-1,
        end_row=-1,
        start_date=pd.Timestamp(joint_window.index.min()),
        end_date=pd.Timestamp(joint_window.index.max()),
        columns=columns,
        driver_columns=driver_columns,
        asset_columns=asset_columns,
        driver_idx=driver_idx,
        asset_idx=asset_idx,
        B0=B0,
        B_lags=B_lags,
        causal_order=causal_order,
        selected_lags=len(B_lags),
        zscore_mean=mean.to_numpy(),
        zscore_std=std.to_numpy(),
        bootstrap_prob_B0=bootstrap_prob_B0,
        prior_knowledge_enforced=enforce_prior_knowledge,
        error_indep_pvalues=error_indep_pvalues,
    )


def run_rolling_varlingam_joint(
    joint,
    window: int = 504,
    step: int = 21,
    lags: int = 1,
    criterion: Criterion | None = "bic",
    prune: bool = True,
    random_state: int = 42,
    ridge_alpha: float = 1.0,
    enforce_prior_knowledge: bool = True,
    n_bootstrap: int = 0,
    error_independence_every_n_windows: int = 0,
    n_jobs: int = 1,
    checkpoint_dir: str | Path | None = None,
) -> RollingJointVarLingamResult:
    """Slide VARLiNGAM over the joint ``[D | A]`` matrix with the asset mask.

    ``error_independence_every_n_windows > 0`` runs the HSIC test on every
    n-th window, because it is too slow to run on all of them. 0 disables it.
    """
    from pipeline.discovery.dynotears import rolling_windows

    frame = joint.frame
    if frame.shape[0] < window:
        raise ValueError(f"window={window} exceeds joint-matrix rows ({frame.shape[0]})")
    dates = pd.DatetimeIndex(frame.index)
    driver_columns = list(joint.driver_columns)
    asset_columns = list(joint.asset_columns)
    jobs = [(i, s, e) for i, (s, e) in enumerate(rolling_windows(frame.shape[0], window, step))]
    logger.info(
        "Rolling VARLiNGAM (joint): %d windows of %d rows (step %d), "
        "drivers=%d, assets=%d, lags=%d, prior_knowledge=%s, "
        "error_indep_every_n=%d",
        len(jobs), window, step, len(driver_columns), len(asset_columns),
        lags, enforce_prior_knowledge, error_independence_every_n_windows,
    )

    def _call(job):
        idx, start, end = job
        sub = frame.iloc[start:end]
        do_hsic = (
            error_independence_every_n_windows > 0
            and idx % error_independence_every_n_windows == 0
        )
        win = run_varlingam_joint_window(
            sub,
            driver_columns=driver_columns,
            asset_columns=asset_columns,
            lags=lags,
            criterion=criterion,
            prune=prune,
            random_state=random_state,
            ridge_alpha=ridge_alpha,
            enforce_prior_knowledge=enforce_prior_knowledge,
            n_bootstrap=n_bootstrap,
            compute_error_independence=do_hsic,
        )
        win.index = idx
        win.start_row = start
        win.end_row = end
        win.start_date = dates[start]
        win.end_date = dates[end - 1]
        return win

    windows = execute_windows(
        jobs, _call, n_jobs, "varlingam-joint", checkpoint_dir=checkpoint_dir
    )
    return RollingJointVarLingamResult(
        windows=windows,
        columns=list(frame.columns),
        driver_columns=driver_columns,
        asset_columns=asset_columns,
        meta={
            "method": "varlingam-joint",
            "window": window,
            "step": step,
            "lags": lags,
            "prior_knowledge_enforced": enforce_prior_knowledge,
            "ridge_alpha": ridge_alpha,
            "n_bootstrap": n_bootstrap,
            "error_independence_every_n_windows": error_independence_every_n_windows,
            **(joint.meta or {}),
        },
    )
