"""Stage 1 orchestration: discovery, driver selection and sensitivities per
rebalance date, persisted under ``results/<tag>/stage1/``.

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
from pipeline.discovery.cache import load_or_compute_discovery
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


# Per-rebalance result
@dataclass
class Stage1Rebalance:
    """Stage 1 output for one rebalance date.

    ``discovery`` is ``None`` on the V0 path. ``selection`` is either kind of
    selection result; both expose ``.selected`` and ``.K``.
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


# Helpers
def derive_rebalance_dates(
    calendar: pd.DatetimeIndex,
    burn_in_days: int,
    rebalance_step: int = 21,
) -> pd.DatetimeIndex:
    """Rebalance every ``rebalance_step`` trading days starting after burn-in."""
    if len(calendar) <= burn_in_days:
        raise ValueError(
            f"calendar has {len(calendar)} trading days but burn_in_days={burn_in_days}"
        )
    return calendar[burn_in_days::rebalance_step]


# Single-rebalance orchestrator
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
    discovery_cache: bool = False,
) -> Stage1Rebalance:
    """Run discovery, selection and sensitivities on a single window.

    Used by both :func:`run_stage1` and the V2 closed loop.
    ``selection_method="correlation"`` (V0) skips discovery entirely and
    ignores ``discovery_method``.
    """
    # Per-window z-score for selection and sensitivities (discovery z-scores
    # internally).
    zs, _, _ = zscore_window(joint_window)
    dw = zs[driver_columns]
    aw = zs[asset_columns]

    if selection_method not in ("causal_greedy", "correlation", "asset_only"):
        raise ValueError(
            f"selection_method must be 'causal_greedy', 'correlation' or "
            f"'asset_only', got {selection_method!r}"
        )

    disc = None
    sel: SelectionResult | CorrelationSelectionResult

    if selection_method == "asset_only":
        # V0' path: same joint discovery as V1, but only the asset-asset block
        # is used downstream. Empty selection yields an empty
        # SensitivityWindow; the closed loop reads asset_to_asset_block.
        # Default stays "dynotears" so Phase-I V0' reproductions are
        # bit-identical.
        if discovery_method in (None, "dynotears"):
            disc = load_or_compute_discovery(
                lambda: run_dynotears_joint_window(
                    joint_window, driver_columns=driver_columns,
                    asset_columns=asset_columns, **discovery_kwargs,
                ),
                joint_window=joint_window, driver_columns=driver_columns,
                asset_columns=asset_columns, method="dynotears",
                discovery_kwargs=discovery_kwargs, use_cache=discovery_cache,
            )
        elif discovery_method == "varlingam":
            disc = load_or_compute_discovery(
                lambda: run_varlingam_joint_window(
                    joint_window, driver_columns=driver_columns,
                    asset_columns=asset_columns, **discovery_kwargs,
                ),
                joint_window=joint_window, driver_columns=driver_columns,
                asset_columns=asset_columns, method="varlingam",
                discovery_kwargs=discovery_kwargs, use_cache=discovery_cache,
            )
        elif discovery_method == "granger":
            from pipeline.discovery.granger import run_granger_joint_window

            # Density matching: the paired DYNOTEARS window supplies the
            # asset-block edge density so the granger graph is compared at
            # like-for-like sparsity. The resolved density enters the cache
            # key.
            g_kwargs = dict(discovery_kwargs)
            if g_kwargs.pop("density_match_dynotears", False):
                dyno = load_or_compute_discovery(
                    lambda: run_dynotears_joint_window(
                        joint_window, driver_columns=driver_columns,
                        asset_columns=asset_columns,
                    ),
                    joint_window=joint_window, driver_columns=driver_columns,
                    asset_columns=asset_columns, method="dynotears",
                    discovery_kwargs={}, use_cache=discovery_cache,
                )
                A = dyno.asset_to_asset_block(0)
                n_a = A.shape[0]
                off = A[~np.eye(n_a, dtype=bool)]
                g_kwargs["target_density"] = (
                    float(np.count_nonzero(off)) / max(n_a * (n_a - 1), 1)
                )
            disc = load_or_compute_discovery(
                lambda: run_granger_joint_window(
                    joint_window, driver_columns=driver_columns,
                    asset_columns=asset_columns, **g_kwargs,
                ),
                joint_window=joint_window, driver_columns=driver_columns,
                asset_columns=asset_columns, method="granger_ridge",
                discovery_kwargs=g_kwargs, use_cache=discovery_cache,
            )
        else:
            raise ValueError(
                f"discovery_method must be 'dynotears', 'varlingam' or "
                f"'granger' for selection_method='asset_only', got "
                f"{discovery_method!r}"
            )
        sel = CorrelationSelectionResult(
            rebalance_date=pd.Timestamp(rebalance_date),
            selected=[], scores=pd.Series(dtype=float), K=0, lags=(),
        )
    elif selection_method == "correlation":
        # V0 path: rank drivers by cum-corr with the asset block, take top K.
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
        # V1/V2 path: discovery then Stage A + Stage B + utility blend.
        if discovery_method == "varlingam":
            disc = load_or_compute_discovery(
                lambda: run_varlingam_joint_window(
                    joint_window, driver_columns=driver_columns,
                    asset_columns=asset_columns, **discovery_kwargs,
                ),
                joint_window=joint_window, driver_columns=driver_columns,
                asset_columns=asset_columns, method="varlingam",
                discovery_kwargs=discovery_kwargs, use_cache=discovery_cache,
            )
        elif discovery_method == "dynotears":
            disc = load_or_compute_discovery(
                lambda: run_dynotears_joint_window(
                    joint_window, driver_columns=driver_columns,
                    asset_columns=asset_columns, **discovery_kwargs,
                ),
                joint_window=joint_window, driver_columns=driver_columns,
                asset_columns=asset_columns, method="dynotears",
                discovery_kwargs=discovery_kwargs, use_cache=discovery_cache,
            )
        else:
            raise ValueError(
                f"discovery_method must be 'dynotears' or 'varlingam' for "
                f"selection_method='causal_greedy', got {discovery_method!r}"
            )

        # Thread the method through so Stage A applies the right stability
        # mask.
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
        # Empty placeholder so the loop can carry on.
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


# Top-level orchestrator
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
    """Drive Stage 1 over a sequence of rebalance dates; returns Stage1Output.

    Each rebalance gets ``window_size`` lookback days ending at its date.
    ``utility_lookup`` is a lookahead-safe callable from
    ``UtilityStore.as_lookup``; ``None`` gives V1 open-loop behaviour.
    """
    discovery_kwargs = dict(discovery_kwargs or {})
    selector_kwargs = dict(selector_kwargs or {})
    sensitivities_kwargs = dict(sensitivities_kwargs or {})
    correlation_kwargs = dict(correlation_kwargs or {})

    # Embed the method in the output dir so runs with the same tag don't
    # clobber each other.
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
