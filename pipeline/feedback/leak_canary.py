"""Deliberately leaky utility lookup, used as a leak-detection canary.

A backtest using this should show visibly inflated Sharpe vs the
lookahead-safe lookup; if not, the feedback signal is too weak or the leak
detection is broken.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from pipeline.feedback.storage import UtilityStore

logger = logging.getLogger(__name__)


def leaky_lookup(
    store: UtilityStore,
    rebalance_date: pd.Timestamp,
    peek_ahead_days: int = 21,
) -> tuple[pd.Series, pd.Timestamp | None]:
    """U row from min(t + peek_ahead_days, latest); leaky on purpose. The
    default 21 days peeks exactly one rebalance ahead."""
    t = pd.Timestamp(rebalance_date).normalize()
    if store.frame.empty:
        return pd.Series(dtype=float, name="utility"), None
    peek = t + pd.Timedelta(days=peek_ahead_days)
    eligible = store.frame.index[store.frame.index <= peek]
    if len(eligible) == 0:
        return pd.Series(dtype=float, name="utility"), None
    latest = eligible[-1]
    row = store.frame.loc[latest]
    utility = row.drop(labels=list(store.aux_columns), errors="ignore").astype(float)
    utility.name = "utility"
    return utility, latest


def make_leaky_lookup(
    store: UtilityStore, peek_ahead_days: int = 21,
) -> Callable[[pd.Timestamp], tuple[pd.Series, pd.Timestamp | None]]:
    """Leaky-lookup callable matching selector.utility_lookup's signature."""
    return lambda t: leaky_lookup(store, t, peek_ahead_days=peek_ahead_days)


__all__ = ["leaky_lookup", "make_leaky_lookup"]
