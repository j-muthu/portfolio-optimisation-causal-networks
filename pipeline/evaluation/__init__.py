"""Backtest evaluation: performance metrics, regime classification, and
stationary block bootstrap for Sharpe-difference CIs."""

from __future__ import annotations

from pipeline.evaluation.bootstrap import (
    SharpeDiffCI,
    bootstrap_statistic,
    sharpe_difference_ci,
    stationary_block_indices,
)
from pipeline.evaluation.metrics import (
    annualised_return,
    annualised_sharpe,
    annualised_sortino,
    annualised_volatility,
    calmar_ratio,
    certainty_equivalent_return,
    downside_deviation,
    effective_n,
    herfindahl_index,
    max_drawdown,
    max_weight,
    one_way_annualised_turnover,
    performance_summary,
    time_underwater,
)
from pipeline.evaluation.regime import (
    nber_recession_dates,
    network_density_regimes,
    regime_conditional_summary,
    vix_regime_masks,
)

__all__ = [
    "annualised_sharpe",
    "annualised_sortino",
    "calmar_ratio",
    "max_drawdown",
    "time_underwater",
    "annualised_return",
    "annualised_volatility",
    "downside_deviation",
    "herfindahl_index",
    "effective_n",
    "max_weight",
    "one_way_annualised_turnover",
    "certainty_equivalent_return",
    "performance_summary",
    "nber_recession_dates",
    "vix_regime_masks",
    "network_density_regimes",
    "regime_conditional_summary",
    "stationary_block_indices",
    "bootstrap_statistic",
    "SharpeDiffCI",
    "sharpe_difference_ci",
]
