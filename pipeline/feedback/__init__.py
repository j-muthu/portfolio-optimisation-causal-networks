"""Closed-loop feedback: credit attribution and EMA utility update (utility),
lookahead-safe persistence (storage), and a deliberately leaky lookup used as
a leak-detection canary (leak_canary)."""

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
