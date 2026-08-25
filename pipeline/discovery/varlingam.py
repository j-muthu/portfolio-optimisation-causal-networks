"""Rolling-window VARLiNGAM on S&P 500 log-returns.

Two-stage: fit a VAR, then DirectLiNGAM on the residuals for the
contemporaneous structure. Every matrix exposed here is transposed from
lingam's raw ``j -> i`` output into the repo-wide ``i -> j`` convention.
For large d the OLS VAR is underdetermined; :func:`estimate_var_coefs`
provides a ridge alternative fed in via ``ar_coefs``.
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
from pipeline.data import Dataset
from pipeline.discovery.dynotears import rolling_windows

logger = logging.getLogger(__name__)

Criterion = Literal["aic", "bic", "hqic", "fpe"]


# Result containers
@dataclass
class VarLingamWindow:
    """Causal model learned from a single rolling window.

    Convention: ``B0[i, j]`` / ``B_lags[k][i, j]`` is i -> j (transposed
    from lingam's raw output).
    """

    index: int
    start_row: int
    end_row: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    columns: list[str]
    B0: np.ndarray
    B_lags: list[np.ndarray]
    causal_order: list[int]
    selected_lags: int
    error_indep_pvalues: np.ndarray | None = None
    bootstrap_prob_B0: np.ndarray | None = None

    @property
    def n_contemp_edges(self) -> int:
        return int(np.count_nonzero(self.B0))

    @property
    def n_lagged_edges(self) -> int:
        return int(sum(np.count_nonzero(b) for b in self.B_lags))

    @property
    def causal_order_tickers(self) -> list[str]:
        """The causal order expressed as ticker symbols (upstream first)."""
        return [self.columns[i] for i in self.causal_order]


@dataclass
class RollingVarLingamResult:
    """Sequence of per-window VARLiNGAM models plus run metadata."""

    windows: list[VarLingamWindow]
    columns: list[str]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """End date of each window -- the natural timestamp for its graph."""
        return pd.DatetimeIndex([w.end_date for w in self.windows])

    def b0_stack(self) -> np.ndarray:
        """All contemporaneous matrices stacked: shape ``(n_windows, d, d)``."""
        return np.stack([w.B0 for w in self.windows])

    def to_frame(self) -> pd.DataFrame:
        """One row per window summarising edge counts and selected lags."""
        return pd.DataFrame(
            {
                "start_date": [w.start_date for w in self.windows],
                "end_date": [w.end_date for w in self.windows],
                "n_contemp_edges": [w.n_contemp_edges for w in self.windows],
                "n_lagged_edges": [w.n_lagged_edges for w in self.windows],
                "selected_lags": [w.selected_lags for w in self.windows],
            }
        )


# Stage-1 VAR coefficient estimation
def estimate_var_coefs(
    X: np.ndarray,
    lags: int,
    method: Literal["ols", "ridge"] = "ridge",
    alpha: float = 1.0,
) -> np.ndarray:
    """Estimate VAR(``lags``) coefficients, optionally ridge-regularised.

    Returns shape ``(lags, d, d)``, the layout VARLiNGAM's ``ar_coefs``
    expects. No intercept: returns are mean-centred upstream.
    """
    from sklearn.linear_model import Ridge

    X = np.asarray(X, dtype=float)
    n, d = X.shape
    # Design: each row t (>= lags) regresses on [X_{t-1} | X_{t-2} | ...].
    design = np.concatenate([X[lags - k - 1 : n - k - 1] for k in range(lags)], axis=1)
    target = X[lags:]

    if method == "ols":
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)  # (lags*d, d)
    elif method == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(design, target)
        coef = model.coef_.T  # sklearn gives (d_targets, lags*d) -> transpose
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"unknown method: {method!r}")

    # coef rows are ordered [lag1 block | lag2 block | ...]; M_tau[i, j] must be
    # the effect of X_{t-tau}[j] on X_t[i], hence the transpose of each block.
    return np.stack([coef[k * d : (k + 1) * d].T for k in range(lags)])


# Single-window fit
def run_varlingam_window(
    window_df: pd.DataFrame,
    lags: int = 1,
    criterion: Criterion | None = "bic",
    prune: bool = True,
    random_state: int = 42,
    ar_coefs: np.ndarray | None = None,
    compute_error_independence: bool = False,
) -> VarLingamWindow:
    """Fit VARLiNGAM on one window and return a :class:`VarLingamWindow`.

    ``ar_coefs`` supplies pre-computed VAR coefficients and skips the
    internal VAR step. ``compute_error_independence`` runs the HSIC test,
    which is O(d^2) and only practical for small d. Row/index fields are
    placeholders filled by the rolling driver.
    """
    columns = list(window_df.columns)
    X = window_df.to_numpy(dtype=float)

    model = VARLiNGAM(
        lags=lags,
        criterion=None if ar_coefs is not None else criterion,
        prune=prune,
        ar_coefs=ar_coefs,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)

    am = model.adjacency_matrices_  # (selected_lags + 1, d, d), raw j -> i
    B0 = am[0].T.copy()  # transpose into i -> j
    B_lags = [am[k].T.copy() for k in range(1, len(am))]
    causal_order = [int(i) for i in model.causal_order_]
    selected_lags = len(am) - 1

    error_pvals = None
    if compute_error_independence:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            error_pvals = model.get_error_independence_p_values()

    return VarLingamWindow(
        index=-1,
        start_row=-1,
        end_row=-1,
        start_date=pd.NaT,
        end_date=pd.NaT,
        columns=columns,
        B0=B0,
        B_lags=B_lags,
        causal_order=causal_order,
        selected_lags=selected_lags,
        error_indep_pvalues=error_pvals,
    )


def bootstrap_window(
    window_df: pd.DataFrame,
    n_sampling: int = 100,
    lags: int = 1,
    random_state: int = 42,
    min_causal_effect: float = 0.01,
) -> np.ndarray:
    """Bootstrap edge probabilities for the contemporaneous matrix ``B0``.

    Each entry is the fraction of resamples in which the edge appeared with
    ``|effect| > min_causal_effect``. Returned in the ``i -> j`` convention.
    """
    X = window_df.to_numpy(dtype=float)
    d = X.shape[1]
    model = VARLiNGAM(lags=lags, criterion=None, prune=True, random_state=random_state)
    # lingam's bootstrap resamples via sklearn.utils.resample without a seed, so
    # it draws from the global NumPy RNG -- seed it for reproducible probabilities.
    np.random.seed(random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.bootstrap(X, n_sampling=n_sampling)
        probs = result.get_probabilities(min_causal_effect=min_causal_effect)
    # probs is (d, d*(1+lags)) for VAR; block 0 is the B0 probabilities (j -> i).
    return np.asarray(probs)[:, :d].T.copy()


# Rolling driver
def _fit_one(
    args: tuple[int, int, int],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lags: int,
    criterion: Criterion | None,
    prune: bool,
    random_state: int,
    var_method: Literal["builtin", "ols", "ridge"],
    ridge_alpha: float,
    compute_error_independence: bool,
    n_bootstrap: int,
    bootstrap_min_effect: float,
) -> VarLingamWindow:
    """Fit VARLiNGAM for a single window (top-level so joblib can pickle it)."""
    idx, start, end = args
    window_df = returns.iloc[start:end]

    ar_coefs = None
    if var_method in ("ols", "ridge"):
        ar_coefs = estimate_var_coefs(
            window_df.to_numpy(dtype=float), lags=lags,
            method=var_method, alpha=ridge_alpha,
        )

    win = run_varlingam_window(
        window_df, lags=lags, criterion=criterion, prune=prune,
        random_state=random_state, ar_coefs=ar_coefs,
        compute_error_independence=compute_error_independence,
    )
    win.index = idx
    win.start_row = start
    win.end_row = end
    win.start_date = dates[start]
    win.end_date = dates[end - 1]

    if n_bootstrap > 0:
        win.bootstrap_prob_B0 = bootstrap_window(
            window_df, n_sampling=n_bootstrap, lags=win.selected_lags,
            random_state=random_state, min_causal_effect=bootstrap_min_effect,
        )

    logger.info(
        "VARLiNGAM window %d (%s..%s): %d contemp / %d lagged edges, lags=%d",
        idx, win.start_date.date(), win.end_date.date(),
        win.n_contemp_edges, win.n_lagged_edges, win.selected_lags,
    )
    return win


def run_rolling_varlingam(
    dataset: Dataset,
    window: int = 504,
    step: int = 21,
    lags: int = 1,
    criterion: Criterion | None = "bic",
    prune: bool = True,
    random_state: int = 42,
    var_method: Literal["builtin", "ols", "ridge"] = "builtin",
    ridge_alpha: float = 1.0,
    compute_error_independence: bool = False,
    n_bootstrap: int = 0,
    bootstrap_min_effect: float = 0.01,
    n_jobs: int = 1,
    checkpoint_dir: str | Path | None = None,
) -> RollingVarLingamResult:
    """Slide VARLiNGAM across a :class:`Dataset`.

    ``var_method="ridge"``/``"ols"`` pre-estimates the VAR via
    :func:`estimate_var_coefs`; use ridge for large d. ``checkpoint_dir``
    enables resume, keyed by window index only, so use a fresh directory
    when parameters change.
    """
    returns = dataset.returns
    dates = dataset.dates
    n = len(returns)
    if window > n:
        raise ValueError(f"window={window} exceeds available rows ({n})")

    jobs = [(i, s, e) for i, (s, e) in enumerate(rolling_windows(n, window, step))]
    logger.info(
        "Rolling VARLiNGAM: %d windows of %d rows (step %d), d=%d, var_method=%s",
        len(jobs), window, step, returns.shape[1], var_method,
    )

    def _call(job: tuple[int, int, int]) -> VarLingamWindow:
        return _fit_one(
            job, returns, dates, lags, criterion, prune, random_state,
            var_method, ridge_alpha, compute_error_independence,
            n_bootstrap, bootstrap_min_effect,
        )

    windows = execute_windows(
        jobs, _call, n_jobs, "varlingam", checkpoint_dir=checkpoint_dir
    )
    return RollingVarLingamResult(
        windows=windows,
        columns=list(returns.columns),
        meta={
            "method": "varlingam",
            "window": window,
            "step": step,
            "lags": lags,
            "criterion": criterion,
            "prune": prune,
            "var_method": var_method,
            "n_bootstrap": n_bootstrap,
            **dataset.meta,
        },
    )


# Stage 1 joint-matrix path: drivers + assets with asset -> driver mask
# lingam's prior_knowledge convention: -1 no prior, 0 no edge j -> i,
# 1 edge j -> i. Forbidding asset -> driver therefore means
# prior_knowledge[driver_j, asset_i] = 0.
def make_prior_knowledge_asset_to_driver(
    driver_idx: np.ndarray,
    asset_idx: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """DirectLiNGAM prior_knowledge matrix forbidding asset -> driver edges.

    All entries -1 (no prior) except ``pk[driver_j, asset_i] = 0``.
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
    """Ridge VAR with the asset -> driver lag mask enforced row-by-row.

    Driver equations regress only on lagged drivers; asset equations are
    unconstrained. Returns ``(lags, d, d)`` in the lingam convention
    (``M[tau, i, j]`` = effect of lagged j on i), masked entries exactly zero.
    """
    from sklearn.linear_model import Ridge

    X = np.asarray(X, dtype=float)
    n, d = X.shape
    design = np.concatenate(
        [X[lags - k - 1 : n - k - 1] for k in range(lags)], axis=1
    )  # shape (n - lags, lags*d)
    target = X[lags:]                                                     # shape (n - lags, d)

    driver_set = set(int(i) for i in driver_idx)
    # Indices in the design matrix corresponding to lagged drivers across all lags:
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

    # Reshape: coef_T[i, k*d+j] is the coefficient of x_{t-k-1}[j] in equation i.
    # M[k, i, j] in the same convention is therefore coef_T[i, k*d+j].
    return np.stack([coef_T[:, k * d : (k + 1) * d] for k in range(lags)], axis=0)


@dataclass
class JointVarLingamWindow:
    """VARLiNGAM output for one window of the joint ``[D | A]`` panel.

    Mirrors :class:`JointDynotearsWindow`; ``B0[i, j]`` is i -> j.
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
        """``M[a, a]``: the asset-only causal block."""
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
    """Fit VARLiNGAM on one joint-matrix window with the asset -> driver mask.

    Lagged coefficients come from :func:`estimate_var_coefs_masked`; the
    contemporaneous B0 from DirectLiNGAM with a prior_knowledge mask.
    ``criterion`` is ignored when the mask is enforced, since the VAR is
    hand-rolled with fixed ``lags``. ``compute_error_independence`` runs the
    HSIC misspecification check (O(d^2) tests, so spot-check only).
    """
    columns = list(joint_window.columns)
    driver_columns = list(driver_columns)
    asset_columns = list(asset_columns)
    driver_idx = np.array([columns.index(c) for c in driver_columns], dtype=int)
    asset_idx = np.array([columns.index(c) for c in asset_columns], dtype=int)
    d = len(columns)

    # Per-window z-score.
    mean = joint_window.mean(axis=0)
    std = joint_window.std(axis=0, ddof=0).where(lambda s: s > 1e-12, 1e-12)
    normalised = (joint_window - mean) / std
    X = normalised.to_numpy(dtype=float)

    # Pre-compute masked VAR coefficients (skips VARLiNGAM's own VAR step).
    if enforce_prior_knowledge:
        ar_coefs = estimate_var_coefs_masked(
            X, lags=lags, driver_idx=driver_idx, asset_idx=asset_idx, alpha=ridge_alpha
        )
        # Construct DirectLiNGAM with prior_knowledge.
        # source code available at: https://github.com/cdt15/lingam
        from lingam.direct_lingam import DirectLiNGAM

        pk = make_prior_knowledge_asset_to_driver(driver_idx, asset_idx, d)
        lingam_model = DirectLiNGAM(prior_knowledge=pk)
        effective_criterion = None  # ar_coefs supplied, so VAR step is skipped
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

    # Post-fit projection: VARLiNGAM's pruning refits the lagged blocks
    # without prior_knowledge, so residual mass leaks into B_tau[asset, driver];
    # zero it explicitly. B0 is already exactly enforced by DirectLiNGAM.
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
        # Fraction of off-diagonal p < 0.05; ~5% expected under the null.
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
    n-th window (it is too slow for all of them); 0 disables it.
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
