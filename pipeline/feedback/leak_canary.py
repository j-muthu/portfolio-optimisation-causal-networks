"""Deliberately-broken utility lookup — leak-detection canary.

Used periodically as a sanity check that the lookahead-safe path
(:class:`pipeline.feedback.storage.UtilityStore.lookup_utility`) actually
makes a measurable difference. The Sharpe of a V2 backtest using this leaky
lookup should be **visibly inflated** vs the lookahead-safe lookup; if not,
either the feedback signal is too weak to matter, or the leak detection is
itself broken.

Run this monthly during long backtests (per the closed-loop plan) and alarm
if the gap closes.
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
    """Return the U row from ``min(t + peek_ahead_days, latest)`` — leaky on purpose.

    Default ``peek_ahead_days=21`` corresponds to looking ahead exactly one
    rebalance — the natural "one cycle of cheating" that should be easy to
    detect downstream via inflated Sharpe.
    """
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
    """Build a leaky-lookup callable matching ``selector.utility_lookup``'s signature."""
    return lambda t: leaky_lookup(store, t, peek_ahead_days=peek_ahead_days)


__all__ = ["leaky_lookup", "make_leaky_lookup"]
