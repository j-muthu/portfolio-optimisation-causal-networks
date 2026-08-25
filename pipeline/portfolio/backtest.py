"""Walk-forward backtest simulator, strategy-agnostic.

Charges transaction costs on one-way turnover at rebalance and records the
excess Sharpe vs equal-weight over each holding period (the V2 reward).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Result containers
@dataclass
class RebalanceRecord:
    """One rebalance event."""

    index: int
    rebalance_date: pd.Timestamp
    holding_end: pd.Timestamp
    asset_names: list[str]
    weights: pd.Series
    turnover: float            # one-way: 0.5 * Σ |w[t] - w[t-1]| over the union
    holding_returns: pd.Series  # gross daily returns of the held portfolio
    holding_returns_net: pd.Series  # after applying tx-cost at rebalance day
    holding_reward: float       # annualised Sharpe excess vs 1/N over holding window


@dataclass
class BacktestResult:
    """Walk-forward backtest output."""

    rebalances: list[RebalanceRecord]
    nav_gross: pd.Series        # net-asset-value over the full backtest, gross
    nav_net: pd.Series          # ... net of tx costs
    meta: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "rebalance_date": r.rebalance_date,
                "holding_end": r.holding_end,
                "turnover": r.turnover,
                "holding_reward_excess_sharpe": r.holding_reward,
                "n_assets": len(r.asset_names),
            }
            for r in self.rebalances
        ]
        return pd.DataFrame(rows)


# Helpers
def _annualised_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    sigma = r.std(ddof=0)
    if sigma < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / sigma)


def _one_way_turnover(prev: pd.Series, new: pd.Series) -> float:
    """``0.5 · Σ |w_new - w_prev|`` across the union of asset names."""
    union = sorted(set(prev.index) | set(new.index))
    p = prev.reindex(union).fillna(0.0)
    n = new.reindex(union).fillna(0.0)
    return float(0.5 * (n - p).abs().sum())


# Walk-forward driver
def run_backtest(
    rebalance_dates: Sequence[pd.Timestamp],
    universe_at: Callable[[pd.Timestamp], list[str]],
    strategy: Callable[[pd.Timestamp, list[str]], pd.Series],
    prices: pd.DataFrame,
    holding_days: int = 21,
    transaction_cost_bps: float = 5.0,
    log_every: int = 12,
    on_rebalance_complete: Callable[["RebalanceRecord"], None] | None = None,
) -> BacktestResult:
    """Walk-forward backtest; returns per-rebalance records plus gross/net NAV.

    ``prices`` must cover through ``last_rebalance + holding_days``.
    Transaction costs hit the net track only. ``on_rebalance_complete`` fires
    after each holding period so a closed-loop driver can update state (e.g.
    the V2 UtilityStore) before the next ``strategy`` call.
    """
    rebalances: list[RebalanceRecord] = []
    prev_weights = pd.Series(dtype=float)
    nav_gross = [1.0]
    nav_net = [1.0]
    nav_index: list[pd.Timestamp] = []

    prices = prices.sort_index()
    daily_returns = prices.pct_change()

    for i, t in enumerate(rebalance_dates):
        universe = universe_at(t)
        weights = strategy(t, universe)
        # Defensive normalisation.
        weights = weights.reindex(universe).fillna(0.0)
        total = weights.sum()
        if total < 1e-12:
            raise ValueError(
                f"strategy returned all-zero weights at {t.date()}"
            )
        weights = weights / total

        # Holding-period returns.
        end_idx = prices.index.searchsorted(t) + holding_days
        end_idx = min(end_idx, len(prices.index) - 1)
        holding_idx = prices.index[prices.index.searchsorted(t):end_idx + 1]
        holding_idx = holding_idx[holding_idx > t]  # strictly after rebalance day
        held_returns = daily_returns.loc[holding_idx, universe].fillna(0.0)
        portfolio_returns = held_returns @ weights

        turnover = _one_way_turnover(prev_weights, weights) if not prev_weights.empty else weights.abs().sum() / 2.0
        tx_cost = turnover * (transaction_cost_bps / 10_000)

        # Costs hit the first day of the holding period.
        portfolio_returns_net = portfolio_returns.copy()
        if len(portfolio_returns_net) > 0:
            portfolio_returns_net.iloc[0] = portfolio_returns_net.iloc[0] - tx_cost

        # Reward: excess annualised Sharpe vs 1/N.
        equal_returns = held_returns.mean(axis=1)
        reward = _annualised_sharpe(portfolio_returns) - _annualised_sharpe(equal_returns)

        # NAV path.
        nav_index.extend(holding_idx)
        for r in portfolio_returns:
            nav_gross.append(nav_gross[-1] * (1.0 + r))
        for r in portfolio_returns_net:
            nav_net.append(nav_net[-1] * (1.0 + r))

        record = RebalanceRecord(
            index=i,
            rebalance_date=pd.Timestamp(t),
            holding_end=pd.Timestamp(holding_idx[-1]) if len(holding_idx) else pd.Timestamp(t),
            asset_names=list(universe),
            weights=weights,
            turnover=turnover,
            holding_returns=portfolio_returns,
            holding_returns_net=portfolio_returns_net,
            holding_reward=reward,
        )
        rebalances.append(record)
        prev_weights = weights

        # Fire before advancing so a closed-loop driver can update state
        # that the next strategy() call reads.
        if on_rebalance_complete is not None:
            on_rebalance_complete(record)

        if (i + 1) % log_every == 0 or i == len(rebalance_dates) - 1:
            logger.info(
                "backtest %d/%d: t=%s, n=%d, turnover=%.3f, holding_reward=%.4f, NAV=%.4f / %.4f",
                i + 1, len(rebalance_dates), t.date(), len(universe), turnover,
                reward, nav_gross[-1], nav_net[-1],
            )

    nav_index = pd.DatetimeIndex([rebalance_dates[0]] + nav_index[:len(nav_gross) - 1])
    nav_gross_s = pd.Series(nav_gross, index=nav_index, name="nav_gross")
    nav_net_s = pd.Series(nav_net, index=nav_index, name="nav_net")
    return BacktestResult(
        rebalances=rebalances,
        nav_gross=nav_gross_s,
        nav_net=nav_net_s,
        meta={
            "holding_days": holding_days,
            "transaction_cost_bps": transaction_cost_bps,
            "n_rebalances": len(rebalance_dates),
        },
    )


__all__ = ["RebalanceRecord", "BacktestResult", "run_backtest"]
