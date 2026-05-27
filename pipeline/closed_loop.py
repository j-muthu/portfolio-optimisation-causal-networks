"""V2 closed-loop driver — interleaves Stage 1 + backtest + feedback per rebalance.

This is the canonical entry point for the **V2 Causal-HSP closed-loop**
variant from ``Closed-Loop Causal-HSP Portfolio.md``. The conventional
``run_stage1 → run_stage2`` batch path is correct for V0prime / V1
(open-loop) but produces a measurement artefact for V2: it pre-computes all
of Stage 1's selections before the backtest loop, so the utility update at
rebalance ``t`` never feeds back into the selector at ``t+1``.

This module fixes that. At each rebalance:

1. The strategy callable runs **one** rebalance's Stage 1 with the *live*
   ``UtilityStore.as_lookup`` — so U[t-1] (keyed by holding-period-end
   ≤ t - 21d, per the lookahead-safe schema) is actually read.
2. ``run_backtest`` simulates the holding period and produces a
   :class:`RebalanceRecord` with the realised reward.
3. The post-rebalance hook (fired before the loop advances) computes the
   sensitivity-weighted credit, EMA-updates U, and appends the row to the
   store keyed by the holding-period-end.
4. The next iteration's strategy callable sees the freshly-written row.

The whole composition reuses existing building blocks — no algorithm logic
lives here, only the orchestration.

Entry point: :func:`run_closed_loop`.
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
from pipeline.feedback import (
    CreditAttribution,
    UtilityStore,
    ema_update,
    sensitivity_weighted_credit,
)
from pipeline.portfolio import (
    BacktestResult,
    equal_weight,
    run_backtest,
    v2_causal_hsp_closed_loop,
)
from pipeline.stage1_pipeline import Stage1Rebalance, fit_stage1_rebalance

logger = logging.getLogger(__name__)

RESULTS_ROOT = THESIS_ROOT / "results"


# ============================================================================
# Result container
# ============================================================================
@dataclass
class ClosedLoopResult:
    """Output of :func:`run_closed_loop`."""

    backtest: BacktestResult
    utility_store: UtilityStore
    stage1_cache: dict[pd.Timestamp, Stage1Rebalance]
    credit_history: list[CreditAttribution] = field(default_factory=list)
    tag: str = "v2_closed_loop"
    config: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """One row per rebalance with the key V2 diagnostics."""
        rows = []
        for rec, attr in zip(self.backtest.rebalances, self.credit_history):
            rows.append(
                {
                    "rebalance_date": rec.rebalance_date,
                    "holding_end": rec.holding_end,
                    "holding_reward": rec.holding_reward,
                    "turnover": rec.turnover,
                    "n_selected": len(attr.credits) if attr else 0,
                    "total_credit": float(attr.credits.sum()) if attr else float("nan"),
                }
            )
        return pd.DataFrame(rows)


# ============================================================================
# Closed-loop driver
# ============================================================================
def run_closed_loop(
    joint_frame: pd.DataFrame,
    asset_returns: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    universe_at: Callable[[pd.Timestamp], list[str]],
    *,
    driver_columns: list[str],
    asset_columns: list[str],
    K: int = 10,
    linkage_method: str = "single",
    window_size: int = 504,
    lookback_days: int = 504,
    holding_days: int = 21,
    transaction_cost_bps: float = 5.0,
    gamma_ema: float = 0.3,
    selection_method: str = "causal_greedy",
    discovery_method: str | None = "dynotears",
    discovery_kwargs: dict | None = None,
    selector_kwargs: dict | None = None,
    sensitivities_kwargs: dict | None = None,
    correlation_kwargs: dict | None = None,
    utility_store: UtilityStore | None = None,
    utility_lookup: Callable[
        [pd.Timestamp], tuple[pd.Series, pd.Timestamp | None]
    ] | None = None,
    asset_eligibility: pd.DataFrame | None = None,
    tag: str = "v2_closed_loop",
    output_dir: Path | None = None,
) -> ClosedLoopResult:
    """Run V2 closed-loop end-to-end with genuine per-rebalance feedback.

    Parameters
    ----------
    joint_frame:
        Aligned trading-day panel (rows = trading days, columns = drivers +
        assets) from :func:`pipeline.data.alignment.build_joint_matrix`.
    asset_returns:
        Wide daily-returns panel (index = trading days, columns = tickers)
        consumed by the backtest engine to compute holding-period P&L.
    rebalance_dates:
        Sequence of trading-day timestamps at which Stage 1 is re-fit and
        the portfolio is rebuilt.
    universe_at:
        Callable returning eligible assets at each rebalance date
        (e.g. top-100-by-mcap intersected with available prices).
    driver_columns, asset_columns:
        Column subsets of ``joint_frame`` — full pool feeds discovery, then
        the selector picks K of them.
    K:
        Selected-driver count target (Stage 1's ``select_drivers``).
    window_size:
        Stage 1 lookback in trading days (default 504 ≈ 2 years).
    lookback_days:
        Backtest sample-covariance lookback for the HSP allocation step.
    holding_days:
        Trading-day holding period before the next rebalance. Default 21
        (monthly). Must match ``MIN_LOOKAHEAD_GAP_DAYS`` semantics in
        ``UtilityStore.lookup_utility``.
    gamma_ema:
        EMA decay applied to the per-rebalance credit attribution.
    discovery_kwargs, selector_kwargs, sensitivities_kwargs:
        Forwarded to :func:`fit_stage1_rebalance`.
    utility_store:
        Pre-built or empty store. If ``None``, a new one is loaded from
        ``output_dir / "utility.parquet"`` (typically empty on first run).
    utility_lookup:
        Optional override for the U-lookup callable. Default
        ``utility_store.as_lookup()``. **Tests** can inject
        ``pipeline.feedback.leak_canary.make_leaky_lookup(store)`` to verify
        the lookahead guard makes a measurable difference.
    tag, output_dir:
        Provenance for persisted results.

    Returns
    -------
    :class:`ClosedLoopResult` containing the backtest, the (now-populated)
    utility store, every per-rebalance ``Stage1Rebalance``, and the
    per-rebalance ``CreditAttribution`` history.
    """
    discovery_kwargs = dict(discovery_kwargs or {})
    selector_kwargs = dict(selector_kwargs or {})
    sensitivities_kwargs = dict(sensitivities_kwargs or {})
    correlation_kwargs = dict(correlation_kwargs or {})

    if output_dir is None:
        output_dir = RESULTS_ROOT / tag
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if utility_store is None:
        utility_store = UtilityStore.load_or_empty(output_dir / "utility.parquet")
    if utility_lookup is None:
        utility_lookup = utility_store.as_lookup()

    rebalance_dates = list(rebalance_dates)
    date_to_index: dict[pd.Timestamp, int] = {
        pd.Timestamp(t): i for i, t in enumerate(rebalance_dates)
    }
    stage1_cache: dict[pd.Timestamp, Stage1Rebalance] = {}
    credit_history: list[CreditAttribution] = []

    # ------------------------------------------------------------------
    # Strategy: just-in-time Stage 1 at this t, then HSP allocation
    # ------------------------------------------------------------------
    def strategy(t: pd.Timestamp, asset_names: list[str]) -> pd.Series:
        # 1. Slice the Stage 1 lookback window ending at t (exclusive).
        end_pos = joint_frame.index.searchsorted(t, side="right")
        start_pos = max(0, end_pos - window_size)
        if end_pos - start_pos < window_size:
            logger.warning(
                "Closed-loop strategy at %s: insufficient lookback "
                "(have %d days, need %d) — using equal-weight",
                pd.Timestamp(t).date(), end_pos - start_pos, window_size,
            )
            return equal_weight(asset_names)
        joint_window = joint_frame.iloc[start_pos:end_pos]

        # If an asset-eligibility mask was supplied (drivers_only mode), slice
        # it to this window and inject into sensitivities_kwargs so the FFNN
        # masks pre-inception zero-fills out of its per-asset loss.
        if asset_eligibility is not None:
            window_dates = joint_window.index
            eligibility_window = asset_eligibility.reindex(window_dates).reindex(
                columns=asset_columns
            )
            sens_kwargs_this_call = {
                **sensitivities_kwargs,
                "asset_eligibility": eligibility_window,
            }
        else:
            sens_kwargs_this_call = sensitivities_kwargs

        # 2. Fit Stage 1 for this single rebalance with the live U lookup.
        s1 = fit_stage1_rebalance(
            rebalance_idx=date_to_index[pd.Timestamp(t)],
            rebalance_date=pd.Timestamp(t),
            joint_window=joint_window,
            driver_columns=driver_columns,
            asset_columns=asset_columns,
            K=K,
            discovery_kwargs=discovery_kwargs,
            selector_kwargs=selector_kwargs,
            sensitivities_kwargs=sens_kwargs_this_call,
            utility_lookup=utility_lookup,
            selection_method=selection_method,
            discovery_method=discovery_method,
            correlation_kwargs=correlation_kwargs,
        )
        stage1_cache[pd.Timestamp(t)] = s1

        # 3. Build the V2 portfolio. If selection or sensitivities are
        #    empty (selector early-stop, etc.), fall back to equal-weight.
        S = s1.sensitivities.S
        sens_assets = list(s1.sensitivities.asset_names)
        common = [a for a in asset_names if a in sens_assets]
        if not common or S.size == 0:
            logger.warning(
                "Closed-loop %s: empty sensitivities, falling back to equal-weight",
                pd.Timestamp(t).date(),
            )
            return equal_weight(asset_names)
        S_sub = S[[sens_assets.index(a) for a in common], :]
        end_pos_ret = asset_returns.index.searchsorted(t, side="right")
        start_pos_ret = max(0, end_pos_ret - lookback_days)
        ret_window = asset_returns.iloc[start_pos_ret:end_pos_ret][common]
        w = v2_causal_hsp_closed_loop(
            S_sub, common, ret_window, linkage_method=linkage_method
        )
        # Pad to full universe (zero on assets with no signal).
        padded = w.reindex(asset_names).fillna(0.0)
        total = padded.sum()
        if total < 1e-12:
            return equal_weight(asset_names)
        return padded / total

    # ------------------------------------------------------------------
    # Hook: credit attribution + EMA update + store.append (between rebalances)
    # ------------------------------------------------------------------
    def on_rebalance_complete(rec) -> None:
        t = pd.Timestamp(rec.rebalance_date)
        s1 = stage1_cache.get(t)
        if s1 is None or not s1.selection.selected:
            credit_history.append(None)  # type: ignore[arg-type]
            return
        sens_df = pd.DataFrame(
            s1.sensitivities.S,
            index=s1.sensitivities.asset_names,
            columns=s1.sensitivities.selected_drivers,
        )
        attr = sensitivity_weighted_credit(
            rec.weights, sens_df, rec.holding_reward,
            rec.rebalance_date, rec.holding_end,
        )
        # Prior U: explicit ignore-strict on the read here; the strict guard
        # is enforced where the *selector* reads U (inside fit_stage1_rebalance),
        # not on the bookkeeping path.
        if utility_store.frame.empty:
            prior = pd.Series(dtype=float, name="utility")
        else:
            prior, _ = utility_store.lookup_utility(
                rec.rebalance_date, require_strict=False,
            )
        updated = ema_update(prior, attr.credits, gamma_ema, selected=s1.selection.selected)
        utility_store.append(rec.rebalance_date, rec.holding_end, updated, rec.holding_reward)
        credit_history.append(attr)

    # ------------------------------------------------------------------
    # Drive the backtest with the hook
    # ------------------------------------------------------------------
    prices = (1.0 + asset_returns.fillna(0.0)).cumprod()
    bt = run_backtest(
        rebalance_dates=rebalance_dates,
        universe_at=universe_at,
        strategy=strategy,
        prices=prices,
        holding_days=holding_days,
        transaction_cost_bps=transaction_cost_bps,
        on_rebalance_complete=on_rebalance_complete,
    )

    # Persist the store (idempotent — only writes if a parquet_path is set).
    try:
        utility_store.save()
    except Exception as exc:  # pragma: no cover - best-effort persistence
        logger.warning("Could not persist UtilityStore to %s: %s",
                       utility_store.parquet_path, exc)

    config = {
        "tag": tag,
        "K": K,
        "linkage_method": linkage_method,
        "window_size": window_size,
        "lookback_days": lookback_days,
        "holding_days": holding_days,
        "transaction_cost_bps": transaction_cost_bps,
        "gamma_ema": gamma_ema,
        "selection_method": selection_method,
        "discovery_method": discovery_method,
        "discovery_kwargs": discovery_kwargs,
        "selector_kwargs": selector_kwargs,
        "sensitivities_kwargs": sensitivities_kwargs,
        "correlation_kwargs": correlation_kwargs,
        "n_rebalances": len(rebalance_dates),
    }
    # Cheap pickle for downstream analysis.
    try:
        with (output_dir / "closed_loop.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "config": config,
                    "backtest": bt,
                    "credit_history": credit_history,
                    "utility_frame": utility_store.frame,
                }, fh,
            )
    except Exception as exc:  # pragma: no cover - best-effort persistence
        logger.warning("Could not persist closed_loop.pkl: %s", exc)

    return ClosedLoopResult(
        backtest=bt,
        utility_store=utility_store,
        stage1_cache=stage1_cache,
        credit_history=credit_history,
        tag=tag,
        config=config,
    )


__all__ = ["ClosedLoopResult", "run_closed_loop"]
