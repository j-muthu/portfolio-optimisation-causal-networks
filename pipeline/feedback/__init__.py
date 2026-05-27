"""Closed-loop feedback.

* :mod:`pipeline.feedback.utility` -- sensitivity-weighted credit attribution
  and EMA driver-utility update.
* :mod:`pipeline.feedback.storage` -- lookahead-safe persistence, keyed by
  holding-period-end date; the ``lookup_utility`` API asserts the lookahead
  gap to detect leakage.
* :mod:`pipeline.feedback.leak_canary` -- deliberately-broken U lookup that
  peeks one rebalance ahead; used as a periodic sanity check that the
  feedback signal is strong enough to matter.
"""

from __future__ import annotations

from pipeline.feedback.leak_canary import leaky_lookup, make_leaky_lookup
from pipeline.feedback.storage import MIN_LOOKAHEAD_GAP_DAYS, UtilityStore
from pipeline.feedback.utility import CreditAttribution, ema_update, sensitivity_weighted_credit

__all__ = [
    "CreditAttribution",
    "sensitivity_weighted_credit",
    "ema_update",
    "UtilityStore",
    "MIN_LOOKAHEAD_GAP_DAYS",
    "leaky_lookup",
    "make_leaky_lookup",
]
