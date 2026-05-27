"""Plan A -- rolling-window DYNOTEARS on S&P 500 log-returns.

DYNOTEARS assumes the causal structure is *fixed* over its input window.  For
financial data that only holds locally, so we slide a window across the series
and learn one causal graph per window (the plan's Step 1).

Each window yields:

* ``W`` -- the ``d x d`` contemporaneous (intra-slice) weighted adjacency
  matrix.  ``W[i, j]`` is the same-day causal effect of asset ``i`` on asset
  ``j``.
* ``A`` -- one ``d x d`` lagged (inter-slice) matrix per lag.  ``A[k][i, j]`` is
  the effect of asset ``i`` at lag ``k+1`` on asset ``j`` today.

Per Howard et al., the lagged weights are nearly always ~0 for daily returns,
so the contemporaneous ``W`` carries the signal -- but we extract both.

Entry points
------------
* :func:`run_dynotears_window` -- fit one window.
* :func:`select_lambdas` -- cross-validate the L1 penalties on a held-out tail.
* :func:`run_rolling_dynotears` -- slide the window across a :class:`Dataset`.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from pipeline._parallel import execute_windows
from pipeline._vendored import from_pandas_dynamic
from pipeline.data import Dataset

logger = logging.getLogger(__name__)


# ============================================================================
# Result containers
# ============================================================================
@dataclass
class DynotearsWindow:
    """Causal graph learned from a single rolling window.

    Matrix convention: ``W[i, j]`` / ``A[k][i, j]`` is the causal effect of the
    asset at column index ``i`` on the asset at column index ``j`` (``i -> j``).
    """

    index: int
    start_row: int
    end_row: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    columns: list[str]
    W: np.ndarray
    A: list[np.ndarray]
    p: int
    lambda_w: float
    lambda_a: float
    converged: bool
    acyclic_edges_removed: int = 0

    @property
    def n_intra_edges(self) -> int:
        return int(np.count_nonzero(self.W))

    @property
    def n_inter_edges(self) -> int:
        return int(sum(np.count_nonzero(a) for a in self.A))


@dataclass
class RollingDynotearsResult:
    """Sequence of per-window DYNOTEARS graphs plus run metadata."""

    windows: list[DynotearsWindow]
    columns: list[str]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """End date of each window -- the natural timestamp for its graph."""
        return pd.DatetimeIndex([w.end_date for w in self.windows])

    def w_stack(self) -> np.ndarray:
        """All contemporaneous matrices stacked: shape ``(n_windows, d, d)``."""
        return np.stack([w.W for w in self.windows])

    def to_frame(self) -> pd.DataFrame:
        """One row per window summarising edge counts and convergence."""
        return pd.DataFrame(
            {
                "start_date": [w.start_date for w in self.windows],
                "end_date": [w.end_date for w in self.windows],
                "n_intra_edges": [w.n_intra_edges for w in self.windows],
                "n_inter_edges": [w.n_inter_edges for w in self.windows],
                "acyclic_removed": [w.acyclic_edges_removed for w in self.windows],
                "lambda_w": [w.lambda_w for w in self.windows],
                "lambda_a": [w.lambda_a for w in self.windows],
                "converged": [w.converged for w in self.windows],
            }
        )


# ============================================================================
# Windowing
# ============================================================================
def rolling_windows(n_rows: int, window: int, step: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` row-index pairs for each rolling window.

    ``end`` is exclusive, so a window has exactly ``window`` rows.  The last
    partial window (fewer than ``window`` rows) is skipped.
    """
    start = 0
    while start + window <= n_rows:
        yield start, start + window
        start += step


# ============================================================================
# StructureModel -> matrix extraction
# ============================================================================
def _split_node(name: str) -> tuple[str, int]:
    """``"AAPL_lag1"`` -> ``("AAPL", 1)``.  Splits on the *last* ``_lag``."""
    var, lag = name.rsplit("_lag", 1)
    return var, int(lag)


def structure_model_to_matrices(
    sm, columns: Sequence[str], p: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Convert a DYNOTEARS ``StructureModel`` to ``W`` and ``A`` matrices.

    DYNOTEARS edges always point into a ``lag0`` node.  ``lag0 -> lag0`` edges
    populate ``W``; ``lagk -> lag0`` edges populate ``A[k-1]``.
    """
    d = len(columns)
    col_idx = {c: i for i, c in enumerate(columns)}
    W = np.zeros((d, d))
    A = [np.zeros((d, d)) for _ in range(p)]

    for u, v, weight in sm.edges(data="weight"):
        var_u, lag_u = _split_node(u)
        var_v, _lag_v = _split_node(v)
        i, j = col_idx[var_u], col_idx[var_v]
        if lag_u == 0:
            W[i, j] = weight
        else:
            A[lag_u - 1][i, j] = weight
    return W, A


def enforce_dag(W: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop the weakest edge on each cycle until the matrix is acyclic.

    DYNOTEARS's continuous acyclicity constraint is only satisfied up to
    ``h_tol``, so a thresholded ``W`` can retain tiny residual cycles (usually
    weak 2-cycles where ``W[i, j]`` and ``W[j, i]`` both survive the
    threshold).  This greedily removes the lowest-magnitude edge lying on a
    detected cycle until the contemporaneous graph is a genuine DAG -- the
    matrix-level analogue of causalnex's ``StructureModel.threshold_till_dag``.

    Returns the acyclic matrix and the number of edges removed.
    """
    W = W.copy()
    removed = 0
    while True:
        graph = nx.DiGraph()
        rows, cols = np.nonzero(W)
        graph.add_edges_from(zip(rows.tolist(), cols.tolist()))
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            return W, removed
        i, j = min(cycle, key=lambda e: abs(W[e[0], e[1]]))
        W[i, j] = 0.0
        removed += 1


# ============================================================================
# Single-window fit
# ============================================================================
def run_dynotears_window(
    window_df: pd.DataFrame,
    p: int = 1,
    lambda_w: float = 0.05,
    lambda_a: float = 0.05,
    w_threshold: float = 0.01,
    max_iter: int = 100,
    enforce_acyclic: bool = True,
    tabu_edges: list[tuple[int, str, str]] | None = None,
) -> tuple[np.ndarray, list[np.ndarray], bool, int]:
    """Fit DYNOTEARS on one window.

    The window is re-indexed to a sequential ``RangeIndex`` (the transformer
    requires integer, gap-free indices).

    Parameters
    ----------
    enforce_acyclic:
        If ``True`` (default), post-process ``W`` with :func:`enforce_dag` so
        the contemporaneous graph is a genuine DAG.  The lagged matrices ``A``
        are inter-slice and may legitimately contain cycles, so they are left
        untouched.
    tabu_edges:
        Optional list of ``(lag, from_col_name, to_col_name)`` tuples to
        forbid. ``lag == 0`` is an intra-slice (W) constraint; ``lag >= 1`` is
        an inter-slice (A[lag-1]) constraint. ``causalnex.from_pandas_dynamic``
        translates these into ``(0, 0)`` L-BFGS-B bounds on the corresponding
        cells, giving an exact hard constraint with no optimiser surgery
        (see :mod:`pipeline.discovery.dynotears_joint`).

    Returns
    -------
    ``(W, A, converged, edges_removed)`` -- see :class:`DynotearsWindow` for the
    matrix conventions.  ``converged`` is ``False`` if the optimiser hit
    ``max_iter`` without the acyclicity constraint reaching tolerance;
    ``edges_removed`` is how many edges :func:`enforce_dag` had to drop.
    """
    df = window_df.reset_index(drop=True)
    columns = list(df.columns)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sm = from_pandas_dynamic(
            df,
            p=p,
            lambda_w=lambda_w,
            lambda_a=lambda_a,
            max_iter=max_iter,
            w_threshold=w_threshold,
            tabu_edges=tabu_edges,
        )
    converged = not any("converge" in str(w.message).lower() for w in caught)
    W, A = structure_model_to_matrices(sm, columns, p)
    removed = 0
    if enforce_acyclic:
        W, removed = enforce_dag(W)
    return W, A, converged, removed


# ============================================================================
# Hyper-parameter selection (cross-validation)
# ============================================================================
def _make_x_xlags(values: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the DYNOTEARS design matrices from a contiguous data block.

    Mirrors ``causalnex``'s ``DynamicDataTransformer``: ``X`` is the data from
    row ``p`` onward, ``Xlags`` stacks the ``p`` lagged copies horizontally
    as ``[shift(1) | shift(2) | ... | shift(p)]``.
    """
    X = values[p:]
    lags = [values[p - i - 1 : len(values) - i - 1] for i in range(p)]
    Xlags = np.concatenate(lags, axis=1)
    return X, Xlags


def reconstruction_error(
    values: np.ndarray, p: int, W: np.ndarray, A: list[np.ndarray]
) -> float:
    """Frobenius norm of the DYNOTEARS residual ``X(I - W) - Xlags A``.

    This is the (unregularised) fit term of the DYNOTEARS objective; the plan
    recommends scoring held-out data with exactly this quantity.
    """
    X, Xlags = _make_x_xlags(values, p)
    d = W.shape[0]
    A_stacked = np.vstack(A) if A else np.zeros((0, d))
    residual = X @ (np.eye(d) - W) - Xlags @ A_stacked
    return float(np.linalg.norm(residual, "fro"))


def select_lambdas(
    window_df: pd.DataFrame,
    lambda_grid: Sequence[float],
    p: int = 1,
    val_frac: float = 0.2,
    w_threshold: float = 0.01,
    max_iter: int = 100,
) -> tuple[float, float, pd.DataFrame]:
    """Grid-search ``(lambda_w, lambda_a)`` on a held-out tail of the window.

    The window is split chronologically into a training head and a validation
    tail (``val_frac``).  For each grid pair we fit on the head and score the
    tail with :func:`reconstruction_error`.  The pair with the lowest validation
    error wins.  ``lambda_w`` and ``lambda_a`` share the same grid.

    Returns ``(best_lambda_w, best_lambda_a, scores_df)``.
    """
    df = window_df.reset_index(drop=True)
    n = len(df)
    split = int(n * (1 - val_frac))
    train, val = df.iloc[:split], df.iloc[split:].to_numpy()

    rows = []
    best = (np.inf, lambda_grid[0], lambda_grid[0])
    for lw in lambda_grid:
        for la in lambda_grid:
            W, A, _, _ = run_dynotears_window(
                train, p=p, lambda_w=lw, lambda_a=la,
                w_threshold=w_threshold, max_iter=max_iter,
            )
            err = reconstruction_error(val, p, W, A)
            rows.append({"lambda_w": lw, "lambda_a": la, "val_error": err})
            if err < best[0]:
                best = (err, lw, la)

    scores = pd.DataFrame(rows)
    logger.info("CV selected lambda_w=%.4g lambda_a=%.4g (val_error=%.4g)", best[1], best[2], best[0])
    return best[1], best[2], scores


# ============================================================================
# Rolling driver
# ============================================================================
def _fit_one(
    args: tuple[int, int, int],
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    p: int,
    lambda_w: float,
    lambda_a: float,
    w_threshold: float,
    max_iter: int,
    lambda_grid: Sequence[float] | None,
    cv_val_frac: float,
) -> DynotearsWindow:
    """Fit DYNOTEARS for a single window (top-level so joblib can pickle it)."""
    idx, start, end = args
    window_df = returns.iloc[start:end]

    lw, la = lambda_w, lambda_a
    if lambda_grid is not None:
        lw, la, _ = select_lambdas(
            window_df, lambda_grid, p=p, val_frac=cv_val_frac,
            w_threshold=w_threshold, max_iter=max_iter,
        )

    W, A, converged, removed = run_dynotears_window(
        window_df, p=p, lambda_w=lw, lambda_a=la,
        w_threshold=w_threshold, max_iter=max_iter,
    )
    win = DynotearsWindow(
        index=idx,
        start_row=start,
        end_row=end,
        start_date=dates[start],
        end_date=dates[end - 1],
        columns=list(returns.columns),
        W=W,
        A=A,
        p=p,
        lambda_w=lw,
        lambda_a=la,
        converged=converged,
        acyclic_edges_removed=removed,
    )
    logger.info(
        "DYNOTEARS window %d (%s..%s): %d intra / %d inter edges "
        "(%d removed for acyclicity)%s",
        idx, win.start_date.date(), win.end_date.date(),
        win.n_intra_edges, win.n_inter_edges, removed,
        "" if converged else " [NOT CONVERGED]",
    )
    return win


def run_rolling_dynotears(
    dataset: Dataset,
    window: int = 504,
    step: int = 21,
    p: int = 1,
    lambda_w: float = 0.05,
    lambda_a: float = 0.05,
    w_threshold: float = 0.01,
    max_iter: int = 100,
    lambda_grid: Sequence[float] | None = None,
    cv_val_frac: float = 0.2,
    n_jobs: int = 1,
    checkpoint_dir: str | Path | None = None,
) -> RollingDynotearsResult:
    """Slide DYNOTEARS across a :class:`Dataset` (the plan's Step 1).

    Parameters
    ----------
    window, step:
        Window length and stride in trading days.  Defaults: ~2 years, 1 month.
    p:
        Lag order.  Howard et al. find ``p=1`` sufficient for daily returns.
    lambda_w, lambda_a:
        Fixed L1 penalties, used when ``lambda_grid`` is ``None``.
    lambda_grid:
        If given, each window cross-validates its penalties over this grid via
        :func:`select_lambdas` instead of using the fixed values.
    n_jobs:
        Process-level parallelism (``joblib``).  Each window is independent.
    checkpoint_dir:
        If set, each completed window is pickled there and an interrupted run
        resumes from the checkpoints instead of recomputing.  Keyed by window
        index only -- use a fresh directory when parameters change.

    Returns
    -------
    RollingDynotearsResult
    """
    returns = dataset.returns
    dates = dataset.dates
    n = len(returns)
    if window > n:
        raise ValueError(f"window={window} exceeds available rows ({n})")

    jobs = [(i, s, e) for i, (s, e) in enumerate(rolling_windows(n, window, step))]
    logger.info(
        "Rolling DYNOTEARS: %d windows of %d rows (step %d), d=%d, p=%d",
        len(jobs), window, step, returns.shape[1], p,
    )

    def _call(job: tuple[int, int, int]) -> DynotearsWindow:
        return _fit_one(
            job, returns, dates, p, lambda_w, lambda_a,
            w_threshold, max_iter, lambda_grid, cv_val_frac,
        )

    windows = execute_windows(
        jobs, _call, n_jobs, "dynotears", checkpoint_dir=checkpoint_dir
    )
    return RollingDynotearsResult(
        windows=windows,
        columns=list(returns.columns),
        meta={
            "method": "dynotears",
            "window": window,
            "step": step,
            "p": p,
            "w_threshold": w_threshold,
            "max_iter": max_iter,
            "cross_validated": lambda_grid is not None,
            **dataset.meta,
        },
    )


# ============================================================================
# Stage 1 joint-matrix path: drivers + assets with asset→driver tabu_edges
# ============================================================================
def make_tabu_edges_asset_to_driver(
    driver_columns: Sequence[str],
    asset_columns: Sequence[str],
    p: int,
) -> list[tuple[int, str, str]]:
    """Enumerate the (lag, from, to) tuples forbidding asset → driver edges.

    Encodes the directional hypothesis that within the analysis timescale
    drivers cause assets, not vice versa. Returns ~|assets| × |drivers| × (p+1)
    entries (e.g. 100 × 50 × 2 ≈ 10 k for the primary backtest configuration);
    the cost is paid once per window and the optimiser only sees ``(0, 0)``
    bounds on the masked cells.
    """
    out: list[tuple[int, str, str]] = []
    for lag in range(p + 1):
        for asset in asset_columns:
            for driver in driver_columns:
                out.append((lag, asset, driver))
    return out


@dataclass
class JointDynotearsWindow:
    """DYNOTEARS output for one window of the joint ``[D | A]`` panel.

    Stage 1's per-window output, persisted to Parquet downstream. The matrix
    convention matches :class:`DynotearsWindow` (``W[i, j]`` is ``i -> j``),
    but here columns include drivers and assets; ``driver_idx`` /
    ``asset_idx`` carry the block layout.
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
    W: np.ndarray
    A: list[np.ndarray]
    p: int
    lambda_w: float
    lambda_a: float
    converged: bool
    acyclic_edges_removed: int
    zscore_mean: np.ndarray
    zscore_std: np.ndarray
    fit_loss: float
    tabu_enforced: bool

    @property
    def n_drivers(self) -> int:
        return len(self.driver_columns)

    @property
    def n_assets(self) -> int:
        return len(self.asset_columns)

    def driver_to_asset_block(self, lag: int) -> np.ndarray:
        """``M[d, a]`` for ``M ∈ {W, A[lag-1]}`` — the block that should be non-trivial."""
        mat = self.W if lag == 0 else self.A[lag - 1]
        return mat[np.ix_(self.driver_idx, self.asset_idx)]

    def asset_to_driver_block(self, lag: int) -> np.ndarray:
        """``M[a, d]`` — the block masked to zero by the tabu_edges constraint."""
        mat = self.W if lag == 0 else self.A[lag - 1]
        return mat[np.ix_(self.asset_idx, self.driver_idx)]

    def n_intra_edges_driver_to_asset(self) -> int:
        return int(np.count_nonzero(self.driver_to_asset_block(0)))


@dataclass
class RollingJointDynotearsResult:
    """Per-window sequence of :class:`JointDynotearsWindow` plus meta."""

    windows: list[JointDynotearsWindow]
    columns: list[str]
    driver_columns: list[str]
    asset_columns: list[str]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([w.end_date for w in self.windows])

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            d2a = w.driver_to_asset_block(0)
            a2d = w.asset_to_driver_block(0)
            rows.append(
                {
                    "start_date": w.start_date,
                    "end_date": w.end_date,
                    "intra_edges_total": int(np.count_nonzero(w.W)),
                    "intra_edges_driver_to_asset": int(np.count_nonzero(d2a)),
                    "intra_edges_asset_to_driver": int(np.count_nonzero(a2d)),
                    "lambda_w": w.lambda_w,
                    "lambda_a": w.lambda_a,
                    "converged": w.converged,
                    "fit_loss": w.fit_loss,
                    "tabu_enforced": w.tabu_enforced,
                }
            )
        return pd.DataFrame(rows)


def run_dynotears_joint_window(
    joint_window: pd.DataFrame,
    driver_columns: Sequence[str],
    asset_columns: Sequence[str],
    p: int = 1,
    lambda_w: float = 0.05,
    lambda_a: float = 0.05,
    w_threshold: float = 0.01,
    max_iter: int = 100,
    enforce_acyclic: bool = True,
    enforce_tabu: bool = True,
) -> JointDynotearsWindow:
    """Fit DYNOTEARS on one window of the joint ``[D | A]`` panel.

    Steps:

    1. Per-window z-score normalisation (mean and std stored on the output).
    2. Build the asset → driver tabu mask if ``enforce_tabu`` is set.
    3. Call :func:`run_dynotears_window` with the mask.
    4. Compute the residual fit loss (Frobenius of the DYNOTEARS objective on
       the normalised window) for diagnostics.

    ``index`` / ``start_row`` / ``end_row`` / dates are filled with placeholder
    values; the rolling driver overwrites them.
    """
    columns = list(joint_window.columns)
    driver_columns = list(driver_columns)
    asset_columns = list(asset_columns)
    if set(columns) != set(driver_columns) | set(asset_columns):
        raise ValueError(
            "joint_window columns must equal the union of driver_columns and "
            "asset_columns"
        )

    # 1. Per-window z-score.
    mean = joint_window.mean(axis=0)
    std = joint_window.std(axis=0, ddof=0).where(lambda s: s > 1e-12, 1e-12)
    normalised = (joint_window - mean) / std

    # 2. Tabu mask.
    tabu = (
        make_tabu_edges_asset_to_driver(driver_columns, asset_columns, p)
        if enforce_tabu
        else None
    )

    # 3. Fit.
    W, A, converged, removed = run_dynotears_window(
        normalised,
        p=p,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        w_threshold=w_threshold,
        max_iter=max_iter,
        enforce_acyclic=enforce_acyclic,
        tabu_edges=tabu,
    )

    # 4. Fit loss (Frobenius residual on the normalised window).
    fit_loss = reconstruction_error(normalised.to_numpy(), p, W, A)

    driver_idx = np.array([columns.index(c) for c in driver_columns], dtype=int)
    asset_idx = np.array([columns.index(c) for c in asset_columns], dtype=int)

    return JointDynotearsWindow(
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
        W=W,
        A=A,
        p=p,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        converged=converged,
        acyclic_edges_removed=removed,
        zscore_mean=mean.to_numpy(),
        zscore_std=std.to_numpy(),
        fit_loss=fit_loss,
        tabu_enforced=enforce_tabu,
    )


def run_rolling_dynotears_joint(
    joint,  # JointMatrix from pipeline.data.alignment (avoid import cycle)
    window: int = 504,
    step: int = 21,
    p: int = 1,
    lambda_w: float = 0.05,
    lambda_a: float = 0.05,
    w_threshold: float = 0.01,
    max_iter: int = 100,
    enforce_tabu: bool = True,
    n_jobs: int = 1,
    checkpoint_dir: str | Path | None = None,
) -> RollingJointDynotearsResult:
    """Slide DYNOTEARS over the joint ``[D | A]`` matrix.

    Parameters
    ----------
    joint:
        A ``JointMatrix`` (from :func:`pipeline.data.alignment.build_joint_matrix`)
        carrying the column-role indices and the trading-day panel.
    window, step:
        Window length and stride in trading days. Defaults match the discovery
        plan: 504 (≈ 2y) and 21 (1 month).
    enforce_tabu:
        If ``True`` (default), forbid asset → driver edges. Set ``False`` for
        the prior-knowledge verification step (refit without the constraint
        and compare the driver → asset block).

    Returns
    -------
    RollingJointDynotearsResult
    """
    frame = joint.frame
    if frame.shape[0] < window:
        raise ValueError(
            f"window={window} exceeds joint-matrix rows ({frame.shape[0]})"
        )

    dates = pd.DatetimeIndex(frame.index)
    driver_columns = list(joint.driver_columns)
    asset_columns = list(joint.asset_columns)
    jobs = [(i, s, e) for i, (s, e) in enumerate(rolling_windows(frame.shape[0], window, step))]
    logger.info(
        "Rolling DYNOTEARS (joint): %d windows of %d rows (step %d), "
        "drivers=%d, assets=%d, p=%d, tabu=%s",
        len(jobs), window, step, len(driver_columns), len(asset_columns), p, enforce_tabu,
    )

    def _call(job: tuple[int, int, int]) -> JointDynotearsWindow:
        idx, start, end = job
        sub = frame.iloc[start:end]
        win = run_dynotears_joint_window(
            sub,
            driver_columns=driver_columns,
            asset_columns=asset_columns,
            p=p,
            lambda_w=lambda_w,
            lambda_a=lambda_a,
            w_threshold=w_threshold,
            max_iter=max_iter,
            enforce_tabu=enforce_tabu,
        )
        win.index = idx
        win.start_row = start
        win.end_row = end
        win.start_date = dates[start]
        win.end_date = dates[end - 1]
        logger.info(
            "joint window %d (%s..%s): %d intra edges, %d d->a, %d a->d (should=0), "
            "fit_loss=%.4f%s",
            idx, win.start_date.date(), win.end_date.date(),
            int(np.count_nonzero(win.W)),
            win.n_intra_edges_driver_to_asset(),
            int(np.count_nonzero(win.asset_to_driver_block(0))),
            win.fit_loss,
            "" if win.converged else " [NOT CONVERGED]",
        )
        return win

    windows = execute_windows(
        jobs, _call, n_jobs, "dynotears-joint", checkpoint_dir=checkpoint_dir
    )
    return RollingJointDynotearsResult(
        windows=windows,
        columns=list(frame.columns),
        driver_columns=driver_columns,
        asset_columns=asset_columns,
        meta={
            "method": "dynotears-joint",
            "window": window,
            "step": step,
            "p": p,
            "w_threshold": w_threshold,
            "max_iter": max_iter,
            "tabu_enforced": enforce_tabu,
            "lambda_w": lambda_w,
            "lambda_a": lambda_a,
            **(joint.meta or {}),
        },
    )
