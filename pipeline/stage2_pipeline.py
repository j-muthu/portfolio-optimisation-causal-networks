"""Stage 2 orchestration: Stage 1 outputs → backtest → metrics.

For each variant V0 / V0' / V1 / V2:

1. Construct a per-rebalance strategy function ``strategy(t, asset_names) →
   weights`` that delegates into the V0/V1/V2 wrapper using the cached
   Stage 1 sensitivity (or asset-only W for V0').
2. Hand the strategy + universe + price panel to
   :func:`pipeline.portfolio.backtest.run_backtest`.
3. For V2, on each rebalance compute the realised reward + credit attribution
   + EMA update, append to the :class:`UtilityStore` so the *next* Stage 1
   selection sees ``U[t-1]``. Storage is keyed by holding-period-end with the
   lookahead-safe lookup.

V2 requires *interleaving* Stage 1 and Stage 2 — Stage 1's selection at
rebalance ``t+1`` depends on the utility produced by Stage 2's holding-period
ending around ``t+21d``. This module sequences that interleaving correctly.

Entry point: :func:`run_stage2`.
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
from pipeline.evaluation import (
    SharpeDiffCI,
    annualised_sharpe,
    performance_summary,
    sharpe_difference_ci,
)
from pipeline.feedback import UtilityStore, ema_update, sensitivity_weighted_credit
from pipeline.portfolio import (
    BacktestResult,
    equal_weight,
    run_backtest,
    v0prime_asset_only_causal_hrp,
    v1_causal_hsp_open_loop,
    v2_causal_hsp_closed_loop,
)
from pipeline.stage1_pipeline import Stage1Output

logger = logging.getLogger(__name__)

RESULTS_ROOT = THESIS_ROOT / "results"


@dataclass
class Stage2VariantResult:
    name: str
    backtest: BacktestResult
    weights_history: list[pd.Series] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


@dataclass
class Stage2Output:
    variants: dict[str, Stage2VariantResult]
    sharpe_ci_vs_v0: dict[str, SharpeDiffCI]
    tag: str
    config: dict = field(default_factory=dict)


# ============================================================================
# Strategy adapters — turn a Stage 1 cache into a backtest strategy callable
# ============================================================================
def _build_strategy_v1_or_v2(
    stage1_by_date: dict[pd.Timestamp, "Stage1Rebalance"],
    variant: str,
    linkage_method: str,
    returns_frame: pd.DataFrame,
    lookback_days: int,
) -> Callable[[pd.Timestamp, list[str]], pd.Series]:
    """Build a strategy callable for V1 or V2 that pulls S from the Stage 1 cache."""
    wrapper = v1_causal_hsp_open_loop if variant == "V1" else v2_causal_hsp_closed_loop

    def strategy(t, asset_names):
        if t not in stage1_by_date:
            # No cached output (e.g. dropped rebalance) — fall back to equal weight.
            logger.warning("No Stage 1 output for %s; using equal-weight fallback", t.date())
            return equal_weight(asset_names)
        s1 = stage1_by_date[t]
        S = s1.sensitivities.S
        sens_assets = list(s1.sensitivities.asset_names)
        # Restrict to assets present in both Stage 1 and today's universe.
        common = [a for a in asset_names if a in sens_assets]
        if not common or S.size == 0:
            return equal_weight(asset_names)
        S_sub = S[[sens_assets.index(a) for a in common], :]
        end_pos = returns_frame.index.searchsorted(t, side="right")
        start_pos = max(0, end_pos - lookback_days)
        ret_window = returns_frame.iloc[start_pos:end_pos][common]
        w = wrapper(S_sub, common, ret_window, linkage_method=linkage_method)
        # Pad to full universe (zero weight for assets we have no signal on).
        return w.reindex(asset_names).fillna(0.0).pipe(lambda s: s / s.sum() if s.sum() > 0 else s)

    return strategy


def _build_strategy_v0prime(
    stage1_by_date: dict[pd.Timestamp, "Stage1Rebalance"],
    linkage_method: str,
    returns_frame: pd.DataFrame,
    lookback_days: int,
) -> Callable[[pd.Timestamp, list[str]], pd.Series]:
    """Asset-only Causal-HRP using the (asset, asset) block of Stage 1's W."""

    def strategy(t, asset_names):
        if t not in stage1_by_date:
            return equal_weight(asset_names)
        s1 = stage1_by_date[t]
        disc = s1.discovery
        common = [a for a in asset_names if a in disc.asset_columns]
        if not common:
            return equal_weight(asset_names)
        idx = [disc.asset_columns.index(a) for a in common]
        asset_block = disc.W[np.ix_(disc.asset_idx, disc.asset_idx)][np.ix_(idx, idx)]
        end_pos = returns_frame.index.searchsorted(t, side="right")
        start_pos = max(0, end_pos - lookback_days)
        ret_window = returns_frame.iloc[start_pos:end_pos][common]
        w = v0prime_asset_only_causal_hrp(asset_block, common, ret_window, linkage_method=linkage_method)
        return w.reindex(asset_names).fillna(0.0).pipe(lambda s: s / s.sum() if s.sum() > 0 else s)

    return strategy


# ============================================================================
# Top-level orchestrator
# ============================================================================
def run_stage2(
    stage1: Stage1Output,
    asset_returns: pd.DataFrame,
    universe_at: Callable[[pd.Timestamp], list[str]],
    variants: Sequence[str] = ("V0prime", "V1", "V2"),
    linkage_method: str = "single",
    lookback_days: int = 504,
    holding_days: int = 21,
    transaction_cost_bps: float = 5.0,
    gamma_ema: float = 0.3,
    bootstrap_resamples: int = 1000,
    tag: str | None = None,
    output_dir: Path | None = None,
) -> Stage2Output:
    """Run the Stage 2 backtest for the requested variants.

    Parameters
    ----------
    stage1:
        Output of :func:`pipeline.stage1_pipeline.run_stage1`. V2 requires
        Stage 1 to have been run *with* a ``UtilityStore.as_lookup`` callable
        — the closed-loop discipline is enforced by Stage 1's selector.
    asset_returns:
        Wide daily returns panel (index = trading days, columns = tickers).
    universe_at:
        Callable returning eligible assets at a given rebalance date.
    variants:
        Subset of ``{"V0prime", "V1", "V2"}`` to run. V0 (vanilla HSP) requires
        cum-corr-derived S which isn't generated by this pipeline — add later.
    gamma_ema:
        EMA decay for the V2 utility update.
    bootstrap_resamples:
        Resamples for Sharpe-difference CIs (vs V1 as the "base of comparison"
        for the closed-loop ablation since V0 isn't run here).

    Returns
    -------
    :class:`Stage2Output` with one :class:`Stage2VariantResult` per variant.
    """
    tag = tag or stage1.tag
    if output_dir is None:
        output_dir = RESULTS_ROOT / tag / "stage2"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_by_date = {r.rebalance_date: r for r in stage1.rebalances}
    rebalance_dates = pd.DatetimeIndex(sorted(stage1_by_date))

    variant_results: dict[str, Stage2VariantResult] = {}

    for variant in variants:
        if variant == "V0prime":
            strategy = _build_strategy_v0prime(
                stage1_by_date, linkage_method, asset_returns, lookback_days,
            )
            store = None
        elif variant == "V1":
            strategy = _build_strategy_v1_or_v2(
                stage1_by_date, "V1", linkage_method, asset_returns, lookback_days,
            )
            store = None
        elif variant == "V2":
            strategy = _build_strategy_v1_or_v2(
                stage1_by_date, "V2", linkage_method, asset_returns, lookback_days,
            )
            store = UtilityStore.load_or_empty(output_dir / "utility.parquet")
        else:
            raise ValueError(f"Unknown variant {variant!r}")

        bt = run_backtest(
            rebalance_dates=rebalance_dates,
            universe_at=universe_at,
            strategy=strategy,
            prices=(1.0 + asset_returns.fillna(0.0)).cumprod(),
            holding_days=holding_days,
            transaction_cost_bps=transaction_cost_bps,
        )

        # V2: write the credit attribution + EMA update per rebalance.
        if variant == "V2":
            assert store is not None
            for rec in bt.rebalances:
                if rec.rebalance_date not in stage1_by_date:
                    continue
                s1 = stage1_by_date[rec.rebalance_date]
                if not s1.selection.selected:
                    continue
                sens_df = pd.DataFrame(
                    s1.sensitivities.S,
                    index=s1.sensitivities.asset_names,
                    columns=s1.sensitivities.selected_drivers,
                )
                attr = sensitivity_weighted_credit(
                    rec.weights, sens_df, rec.holding_reward,
                    rec.rebalance_date, rec.holding_end,
                )
                prior, _ = store.lookup_utility(
                    rec.rebalance_date, require_strict=False,
                ) if not store.frame.empty else (pd.Series(dtype=float), None)
                updated = ema_update(prior, attr.credits, gamma_ema, selected=s1.selection.selected)
                store.append(rec.rebalance_date, rec.holding_end, updated, rec.holding_reward)
            store.save()

        # Gross daily returns of the backtest for summary purposes.
        daily_ret = bt.nav_gross.pct_change().dropna()
        summary = performance_summary(
            daily_ret,
            weights_history=[r.weights for r in bt.rebalances],
            rebalances_per_year=int(252 / holding_days),
        )
        variant_results[variant] = Stage2VariantResult(
            name=variant, backtest=bt,
            weights_history=[r.weights for r in bt.rebalances],
            summary=summary,
        )
        logger.info("variant %s: Sharpe=%.4f, MDD=%.4f", variant, summary["annualised_sharpe"], summary["max_drawdown"])

    # Sharpe-diff CIs vs V1 (the open-loop reference). V0 needs cum-corr S
    # which isn't built here.
    sharpe_cis: dict[str, SharpeDiffCI] = {}
    if "V1" in variant_results and bootstrap_resamples > 0:
        ref_ret = variant_results["V1"].backtest.nav_gross.pct_change().dropna()
        for name, res in variant_results.items():
            if name == "V1":
                continue
            r = res.backtest.nav_gross.pct_change().dropna()
            sharpe_cis[name] = sharpe_difference_ci(
                r, ref_ret, n_resamples=bootstrap_resamples,
            )

    out = Stage2Output(
        variants=variant_results, sharpe_ci_vs_v0=sharpe_cis, tag=tag,
        config={
            "variants": list(variants), "linkage_method": linkage_method,
            "lookback_days": lookback_days, "holding_days": holding_days,
            "transaction_cost_bps": transaction_cost_bps, "gamma_ema": gamma_ema,
            "bootstrap_resamples": bootstrap_resamples,
        },
    )
    with open(output_dir / f"stage2_{tag}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    summary_df = pd.DataFrame({n: v.summary for n, v in variant_results.items()}).T
    summary_df.to_parquet(output_dir / f"stage2_summary_{tag}.parquet")
    logger.info("stage2 done: variants=%s → %s", list(variant_results), output_dir)
    return out


__all__ = [
    "Stage2VariantResult",
    "Stage2Output",
    "run_stage2",
]
