"""Plan B -- rolling-window VARLiNGAM on S&P 500 log-returns.

VARLiNGAM is the head-to-head comparison partner for DYNOTEARS.  It is a
two-stage algorithm: fit a VAR, then run DirectLiNGAM on the residuals to
recover the contemporaneous structure, exploiting the non-Gaussianity of
returns to *uniquely* identify the DAG.

Like DYNOTEARS it assumes stationarity, so we again slide a window across the
series (the plan's Step 1) and learn one set of matrices per window:

* ``B0`` -- ``d x d`` contemporaneous adjacency.
* ``B_lags`` -- one ``d x d`` matrix per lag.
* ``causal_order`` -- the discovered causal ordering of the assets (unique to
  VARLiNGAM; tracking how it drifts across windows is a regime-change signal).

Convention
----------
``lingam`` returns ``B0[i, j] = effect of j on i``.  For consistency with
:mod:`pipeline.rolling_dynotears` every matrix exposed by this module is
**transposed into the ``i -> j`` convention**: ``B0[i, j]`` is the effect of
asset ``i`` on asset ``j``.

Scalability
-----------
Stage 1 fits a VAR with ``d^2`` coefficients per lag; for ``d=500`` on a
~504-row window this is badly underdetermined.  :func:`estimate_var_coefs`
offers a ridge-regularised VAR whose coefficients can be passed straight to
VARLiNGAM via ``ar_coefs`` -- the plan's recommended mitigation.
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


# ============================================================================
# Result containers
# ============================================================================
@dataclass
class VarLingamWindow:
    """Causal model learned from a single rolling window.

    Matrix convention: ``B0[i, j]`` / ``B_lags[k][i, j]`` is the causal effect
    of asset ``i`` on asset ``j`` (``i -> j``) -- transposed from lingam's raw
    ``j -> i`` output to match :mod:`pipeline.rolling_dynotears`.
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


# ============================================================================
# Stage-1 VAR coefficient estimation (scalability mitigation)
# ============================================================================
def estimate_var_coefs(
    X: np.ndarray,
    lags: int,
    method: Literal["ols", "ridge"] = "ridge",
    alpha: float = 1.0,
) -> np.ndarray:
    """Estimate VAR(``lags``) coefficients, optionally ridge-regularised.

    For ``d=500`` the ordinary-least-squares VAR is severely underdetermined
    (``d^2`` coefficients per lag, ~504 observations).  Ridge shrinkage keeps
    Stage 1 well-posed.  The returned array has shape ``(lags, d, d)`` -- the
    layout VARLiNGAM's ``ar_coefs`` argument expects, so the result can be fed
    straight in to skip VARLiNGAM's own VAR step.

    The model is ``X_t = sum_{tau=1..lags} M_tau X_{t-tau} + e_t`` with no
    intercept (returns are mean-centred upstream).
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


# ============================================================================
# Single-window fit
# ============================================================================
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

    Parameters
    ----------
    lags:
        Maximum VAR lag.  With ``criterion`` set, BIC/AIC picks the best lag in
        ``1..lags``; the plan caps this at 5.
    criterion:
        Lag-order selection criterion, or ``None`` to force ``lags`` exactly.
    ar_coefs:
        Pre-computed VAR coefficients ``(lags, d, d)`` (e.g. from
        :func:`estimate_var_coefs`).  Skips VARLiNGAM's internal VAR step --
        the scalability mitigation for large ``d``.
    compute_error_independence:
        If ``True``, also run the HSIC error-independence test.  This is
        ``O(d^2)`` HSIC tests and only practical for small ``d``.

    Note
    ----
    ``index``/``start_row``/``end_row``/dates are placeholders here (filled by
    :func:`run_rolling_varlingam`); call sites that fit a lone window can ignore
    them.
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

    VARLiNGAM has a built-in ``bootstrap`` (DYNOTEARS does not).  An entry of
    the returned ``d x d`` matrix is the fraction of the ``n_sampling`` resamples
    in which that edge appeared with ``|effect| > min_causal_effect``.  Per the
    plan, edges present in 90%+ of samples are reliable; <50% are noise.

    Returned matrix is in the ``i -> j`` convention.
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


# ============================================================================
# Rolling driver
# ============================================================================
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
    """Slide VARLiNGAM across a :class:`Dataset` (the plan's Step 1).

    Parameters
    ----------
    window, step:
        Window length and stride in trading days (~2 years, 1 month).
    lags, criterion:
        Max VAR lag and the lag-selection criterion (the plan caps ``lags`` at
        5 and lets BIC choose).
    var_method:
        ``"builtin"`` uses VARLiNGAM's own OLS VAR.  ``"ridge"``/``"ols"``
        pre-estimate the VAR with :func:`estimate_var_coefs` and pass it via
        ``ar_coefs`` -- use ``"ridge"`` for large ``d`` where OLS is
        underdetermined.
    compute_error_independence:
        Run the HSIC error-independence assumption check per window
        (``O(d^2)``; only practical for small ``d``).
    n_bootstrap:
        Bootstrap resamples per window for edge reliability (0 = skip).
    n_jobs:
        Process-level parallelism (``joblib``); windows are independent.
    checkpoint_dir:
        If set, each completed window is pickled there and an interrupted run
        resumes from the checkpoints instead of recomputing.  Keyed by window
        index only -- use a fresh directory when parameters change.

    Returns
    -------
    RollingVarLingamResult
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


# ============================================================================
# Stage 1 joint-matrix path: drivers + assets with asset → driver mask
# ============================================================================
# lingam's prior_knowledge convention (DirectLiNGAM, see lingam/base.py):
#   prior_knowledge[i, j]:
#     -1 = no prior
#      0 = no edge j → i  (equivalently: B0_raw[i, j] = 0, where B0_raw[i, j] is
#           the effect of variable j on variable i in lingam's native form)
#      1 = edge j → i
# Forbidding the arrow "asset → driver" means: no effect of asset on driver,
# i.e. raw B0[driver, asset] = 0, i.e. prior_knowledge[driver_j, asset_i] = 0.
def make_prior_knowledge_asset_to_driver(
    driver_idx: np.ndarray,
    asset_idx: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Build a DirectLiNGAM prior_knowledge matrix forbidding asset → driver edges.

    Shape: ``(n_features, n_features)``. All entries default to ``-1`` (no
    prior); ``pk[driver_j, asset_i] = 0`` for every (asset, driver) pair —
    this encodes "no effect of asset i on driver j" in lingam's raw indexing
    convention.
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
    """Ridge VAR with the asset → driver lag mask enforced row-by-row.

    For each row ``i`` of the VAR coefficient matrix M_τ (the equation for
    variable i), we regress X_t[i] on the lagged regressors:

    * If i is a driver: only lagged drivers are allowed as predictors
      (asset lag coefficients are explicitly zero).
    * If i is an asset: all lagged variables are predictors (unconstrained).

    Returns M with shape ``(lags, d, d)`` in the lingam convention
    (``M[τ, i, j] = effect of x_{t-τ-1}[j] on x_t[i]``). Pre-zero entries are
    set to exactly zero on output.
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

    Mirrors :class:`pipeline.discovery.dynotears.JointDynotearsWindow`. Matrix
    convention follows the legacy :class:`VarLingamWindow`: ``B0[i, j]`` is
    the effect of variable ``i`` on variable ``j`` (transposed from lingam's
    raw ``j -> i``).
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
    """Fit VARLiNGAM on one joint-matrix window with asset → driver constraint.

    Constraint mechanism:

    1. **Lagged VAR coefficients** (``M_τ``) are estimated with
       :func:`estimate_var_coefs_masked` — driver equations regress only on
       lagged drivers, asset equations are unconstrained.
    2. **Contemporaneous structure** (``B_0``) is recovered by
       ``DirectLiNGAM`` initialised with the prior_knowledge mask from
       :func:`make_prior_knowledge_asset_to_driver`.
    3. Together these enforce ``B_τ[asset, driver] = 0`` for all lags
       (verified by smoke test).

    Per-window z-score normalisation is applied first; the lingam
    ``criterion`` argument is ignored when ``enforce_prior_knowledge`` is set
    because we hand-roll the VAR with a fixed ``lags`` value.

    ``compute_error_independence`` enables the **HSIC residual-independence
    test** — VARLiNGAM's misspecification check. Returns a ``(d, d)`` matrix
    of pairwise p-values on the recovered residuals; if many off-diagonal
    p-values are < 0.05 the LiNGAM assumption is violated for this window
    and the recovered causal order shouldn't be trusted. **Cost**: O(d²)
    HSIC tests each O(n²); prohibitive on every window for d ≈ 135.
    See :func:`run_rolling_varlingam_joint` for the spot-check cadence.
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

    # Post-fit projection for the lagged mask. VARLiNGAM's `_pruning` step
    # refits each B_τ via least-squares without applying prior_knowledge to the
    # lagged blocks, so a small residual mass leaks into B_τ[asset, driver]
    # even with masked ar_coefs + DirectLiNGAM prior_knowledge on B_0. We zero
    # those entries explicitly. (B_0 is already exactly enforced by DirectLiNGAM.)
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
        # One-line misspecification summary: fraction of off-diagonal p < 0.05.
        # Expected ~5 % under the LiNGAM null; substantially higher → misspecified.
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
    """Slide VARLiNGAM over the joint ``[D | A]`` matrix with the asset-mask.

    Parameters
    ----------
    error_independence_every_n_windows:
        If > 0, run the HSIC residual-independence test on windows where
        ``window_index % n == 0`` (e.g. ``n=12`` gives ~annual spot checks
        across a monthly-rebalanced backtest). Test is O(d²) HSIC each
        O(n²) so per-window cost is in minutes at d≈135 — running on every
        window is prohibitive. ``0`` disables the test entirely
        (default). Stored as ``error_indep_pvalues`` on each tested window;
        non-tested windows have ``None``.
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
