"""Stage 1 orchestration: data → discovery → selection → sensitivities → Parquet.

Drives the Causal Factor Discovery pipeline (per
``Causal Factor Discovery Pipeline.md``) over a configured backtest period.

For each rebalance date ``t``, this module produces and persists:

* ``W[t]``, ``A_p[t]``, fit diagnostics — from DYNOTEARS joint-matrix discovery
  with asset → driver ``tabu_edges`` enforcement.
* ``selected_drivers[t]`` — ordered list of ≤ K driver names from
  :func:`pipeline.factor_selection.select_drivers` (Stage A prune + Stage B
  greedy + optional α-blend with ``U[t-1]``).
* ``sensitivities[t]`` — per-asset sensitivity matrix from
  :func:`pipeline.sensitivities.fit_sensitivities_window` (PyTorch multi-head
  FFNN + autograd Jacobian).

Outputs are written to ``results/<tag>/stage1/`` as Parquet (one file per
artefact type), keyed by rebalance date.

V2 (closed-loop) is enabled by passing a ``UtilityStore`` for both
``selector_utility_lookup`` and ``feedback_store`` — Stage 2 wires this up
explicitly, see :func:`pipeline.stage2_pipeline.run_stage2`.

Entry point: :func:`run_stage1`.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT
from pipeline.data.alignment import build_joint_matrix, trading_calendar, zscore_window
from pipeline.discovery.dynotears import (
    JointDynotearsWindow,
    run_dynotears_joint_window,
)
from pipeline.discovery.varlingam import (
    JointVarLingamWindow,
    run_varlingam_joint_window,
)
from pipeline.factor_selection import SelectionResult, select_drivers
from pipeline.factor_selection.correlation_selector import (
    CorrelationSelectionResult,
    select_top_k_corr,
)
from pipeline.sensitivities import SensitivityWindow, fit_sensitivities_window

logger = logging.getLogger(__name__)

RESULTS_ROOT = THESIS_ROOT / "results"


# ============================================================================
# Per-rebalance result
# ============================================================================
@dataclass
class Stage1Rebalance:
    """Stage 1 output for one rebalance date — the §Interface contract.

    ``discovery`` is ``None`` on the V0 path (cum-corr selection skips the
    causal-graph fitting step entirely). ``selection`` is either a
    ``SelectionResult`` (causal-greedy) or a ``CorrelationSelectionResult``
    (V0); both expose ``.selected`` and ``.K``.
    """

    rebalance_date: pd.Timestamp
    discovery: JointDynotearsWindow | JointVarLingamWindow | None
    selection: SelectionResult | CorrelationSelectionResult
    sensitivities: SensitivityWindow


@dataclass
class Stage1Output:
    """Sequence of Stage 1 rebalances + run metadata."""

    rebalances: list[Stage1Rebalance]
    tag: str
    config: dict = field(default_factory=dict)

    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([r.rebalance_date for r in self.rebalances])

    def selected_drivers_frame(self) -> pd.DataFrame:
        """Tidy long-form table of (date, position, driver)."""
        rows = []
        for r in self.rebalances:
            for pos, d in enumerate(r.selection.selected):
                rows.append(
                    {"rebalance_date": r.rebalance_date, "position": pos, "driver": d}
                )
        return pd.DataFrame(rows)


# ============================================================================
# Helpers
# ============================================================================
def derive_rebalance_dates(
    calendar: pd.DatetimeIndex,
    burn_in_days: int,
    rebalance_step: int = 21,
) -> pd.DatetimeIndex:
    """Rebalance every ``rebalance_step`` trading days starting after burn-in.

    The default 21 trading days ≈ one calendar month and matches the
    closed-loop plan's monthly cadence.
    """
    if len(calendar) <= burn_in_days:
        raise ValueError(
            f"calendar has {len(calendar)} trading days but burn_in_days={burn_in_days}"
        )
    return calendar[burn_in_days::rebalance_step]


# ============================================================================
# Single-rebalance orchestrator
# ============================================================================
def fit_stage1_rebalance(
    rebalance_idx: int,
    rebalance_date: pd.Timestamp,
    joint_window: pd.DataFrame,
    driver_columns: list[str],
    asset_columns: list[str],
    K: int,
    discovery_kwargs: dict,
    selector_kwargs: dict,
    sensitivities_kwargs: dict,
    utility_lookup,
    selection_method: str = "causal_greedy",
    discovery_method: str | None = "dynotears",
    correlation_kwargs: dict | None = None,
) -> Stage1Rebalance:
    """Run discovery → selection → sensitivities on a single window.

    Public entry-point for one-rebalance Stage 1 work. Composed by both
    :func:`run_stage1` (batch over all rebalances) and
    :func:`pipeline.closed_loop.run_closed_loop` (just-in-time, per rebalance,
    inside the V2 closed loop).

    Parameters
    ----------
    selection_method:
        ``"causal_greedy"`` (default; V1/V2 path — Stage A + Stage B + utility
        blend per :func:`select_drivers`) or ``"correlation"`` (V0 path —
        cumulative-correlation top-K per :func:`select_top_k_corr`). The
        latter skips discovery entirely.
    discovery_method:
        ``"dynotears"`` (default) or ``"varlingam"``. Ignored (with a debug
        log) when ``selection_method == "correlation"``.
    correlation_kwargs:
        Forwarded to ``select_top_k_corr`` (e.g. ``{"lags": (0, 1)}``).
        Ignored on the causal-greedy path.
    """
    # Per-window z-score for both discovery (already done internally by
    # run_dynotears_joint_window) and the selection / sensitivity steps.
    zs, _, _ = zscore_window(joint_window)
    dw = zs[driver_columns]
    aw = zs[asset_columns]

    if selection_method not in ("causal_greedy", "correlation"):
        raise ValueError(
            f"selection_method must be 'causal_greedy' or 'correlation', "
            f"got {selection_method!r}"
        )

    disc = None
    sel: SelectionResult | CorrelationSelectionResult

    if selection_method == "correlation":
        # V0 path: skip discovery entirely; just rank drivers by cum-corr
        # with the asset block and take the top K.
        if discovery_method is not None:
            logger.debug(
                "selection_method='correlation': ignoring discovery_method=%r "
                "(V0 doesn't use a causal graph)", discovery_method,
            )
        corr_kw = dict(correlation_kwargs or {})
        sel = select_top_k_corr(
            driver_window=dw, asset_window=aw, K=K,
            rebalance_date=rebalance_date, **corr_kw,
        )
    else:
        # V1/V2 path: discovery → Stage A + Stage B + utility blend.
        if discovery_method == "varlingam":
            disc = run_varlingam_joint_window(
                joint_window, driver_columns=driver_columns,
                asset_columns=asset_columns, **discovery_kwargs,
            )
        elif discovery_method == "dynotears":
            disc = run_dynotears_joint_window(
                joint_window, driver_columns=driver_columns,
                asset_columns=asset_columns, **discovery_kwargs,
            )
        else:
            raise ValueError(
                f"discovery_method must be 'dynotears' or 'varlingam' for "
                f"selection_method='causal_greedy', got {discovery_method!r}"
            )

        # Thread the method choice through so Stage A applies the right
        # stability mask (DYNOTEARS magnitude threshold vs VARLiNGAM
        # bootstrap probabilities, when available).
        sel_kw = dict(selector_kwargs)
        sel_kw.setdefault("method", discovery_method)
        sel = select_drivers(
            rebalance_date=rebalance_date,
            discovery_window=disc,
            driver_window=dw,
            asset_window=aw,
            K=K,
            utility_lookup=utility_lookup,
            rebalance_index=rebalance_idx,
            **sel_kw,
        )

    # Sensitivities on the selected drivers (shared across V0/V1/V2).
    if not sel.selected:
        # No drivers selected (e.g. ε early-stop or empty pool); return an
        # empty placeholder so the loop can carry on.
        N = len(asset_columns)
        sens = SensitivityWindow(
            rebalance_date=rebalance_date,
            selected_drivers=[],
            asset_names=list(asset_columns),
            S=np.zeros((N, 0), dtype=float),
            arch={"depth": 0, "width": 0},
            val_rmse=float("nan"),
            n_train=0, n_val=0,
            metadata={"empty_selection": True},
        )
    else:
        sens = fit_sensitivities_window(
            drivers=dw, assets=aw,
            selected_drivers=sel.selected,
            rebalance_date=rebalance_date,
            **sensitivities_kwargs,
        )

    return Stage1Rebalance(
        rebalance_date=pd.Timestamp(rebalance_date),
        discovery=disc, selection=sel, sensitivities=sens,
    )


# ============================================================================
# Top-level orchestrator
# ============================================================================
def run_stage1(
    joint_frame: pd.DataFrame,
    driver_columns: list[str],
    asset_columns: list[str],
    rebalance_dates: pd.DatetimeIndex,
    window_size: int = 504,
    K: int = 10,
    tag: str = "stage1",
    selection_method: str = "causal_greedy",
    discovery_method: str | None = "dynotears",
    discovery_kwargs: dict | None = None,
    selector_kwargs: dict | None = None,
    sensitivities_kwargs: dict | None = None,
    correlation_kwargs: dict | None = None,
    utility_lookup: Callable | None = None,
    output_dir: Path | None = None,
    progress_log_every: int = 6,
) -> Stage1Output:
    """Drive Stage 1 over a sequence of rebalance dates.

    Parameters
    ----------
    joint_frame:
        Aligned trading-day panel (rows = trading days, columns = drivers +
        assets) from :func:`pipeline.data.alignment.build_joint_matrix`.
    driver_columns, asset_columns:
        The two column subsets. The intersection of their union with
        ``joint_frame.columns`` must equal ``joint_frame.columns``.
    rebalance_dates:
        Sequence of rebalance timestamps; each gets ``window_size`` lookback
        days ending at that timestamp.
    window_size:
        Lookback length in trading days (default 504 ≈ 2 years).
    K:
        Selected-driver count target.
    discovery_kwargs:
        Forwarded to :func:`run_dynotears_joint_window`. Typical values:
        ``{"p": 1, "lambda_w": 0.05, "lambda_a": 0.05, "w_threshold": 0.01}``.
    selector_kwargs:
        Forwarded to :func:`select_drivers`. Typical:
        ``{"alpha": 0.6, "burn_in_rebalances": 6}``.
    sensitivities_kwargs:
        Forwarded to :func:`fit_sensitivities_window`.
    utility_lookup:
        Lookahead-safe callable from
        :func:`pipeline.feedback.UtilityStore.as_lookup`. ``None`` (or the
        burn-in handling inside the selector) yields V1 open-loop behaviour.
    output_dir:
        If provided, the run is pickled to ``output_dir / stage1_<tag>.pkl``.
        Default writes to ``results/<tag>/stage1/``.

    Returns
    -------
    :class:`Stage1Output`. The caller (Stage 2) consumes the per-rebalance
    triple of (W, selected_drivers, S).
    """
    discovery_kwargs = dict(discovery_kwargs or {})
    selector_kwargs = dict(selector_kwargs or {})
    sensitivities_kwargs = dict(sensitivities_kwargs or {})
    correlation_kwargs = dict(correlation_kwargs or {})

    # Embed the method choices in the output dir so V0 / V1-DYNO / V1-VAR
    # runs with the same tag don't clobber each other.
    method_suffix = (
        "v0_corr" if selection_method == "correlation"
        else f"causal_{discovery_method}"
    )
    if output_dir is None:
        output_dir = RESULTS_ROOT / tag / f"stage1__{method_suffix}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cal = pd.DatetimeIndex(joint_frame.index)
    rebs: list[Stage1Rebalance] = []
    for i, t in enumerate(rebalance_dates):
        end_pos = cal.searchsorted(t, side="right")
        start_pos = max(0, end_pos - window_size)
        if end_pos - start_pos < window_size:
            logger.warning(
                "Rebalance %s: only %d rows available (window_size=%d); skipping",
                t.date(), end_pos - start_pos, window_size,
            )
            continue
        window_df = joint_frame.iloc[start_pos:end_pos]
        try:
            reb = fit_stage1_rebalance(
                rebalance_idx=i,
                rebalance_date=pd.Timestamp(t),
                joint_window=window_df,
                driver_columns=driver_columns,
                asset_columns=asset_columns,
                K=K,
                discovery_kwargs=discovery_kwargs,
                selector_kwargs=selector_kwargs,
                sensitivities_kwargs=sensitivities_kwargs,
                utility_lookup=utility_lookup,
                selection_method=selection_method,
                discovery_method=discovery_method,
                correlation_kwargs=correlation_kwargs,
            )
        except Exception as exc:
            logger.exception("Rebalance %s failed: %s", t.date(), exc)
            continue
        rebs.append(reb)
        if (i + 1) % progress_log_every == 0 or i == len(rebalance_dates) - 1:
            logger.info(
                "stage1 [%d/%d] t=%s, K_sel=%d, drivers=%s, val_rmse=%.4f",
                i + 1, len(rebalance_dates), t.date(),
                len(reb.selection.selected), reb.selection.selected,
                reb.sensitivities.val_rmse,
            )

    out = Stage1Output(
        rebalances=rebs, tag=tag,
        config={
            "window_size": window_size, "K": K,
            "selection_method": selection_method,
            "discovery_method": discovery_method,
            "discovery_kwargs": discovery_kwargs,
            "selector_kwargs": selector_kwargs,
            "sensitivities_kwargs": sensitivities_kwargs,
            "correlation_kwargs": correlation_kwargs,
            "n_rebalances": len(rebs),
        },
    )
    pickle_path = output_dir / f"stage1_{tag}.pkl"
    with open(pickle_path, "wb") as fh:
        pickle.dump(out, fh)
    out.selected_drivers_frame().to_parquet(output_dir / f"selected_drivers_{tag}.parquet")
    logger.info("stage1 done: %d rebalances → %s", len(rebs), output_dir)
    return out


__all__ = [
    "Stage1Rebalance",
    "Stage1Output",
    "derive_rebalance_dates",
    "run_stage1",
]
