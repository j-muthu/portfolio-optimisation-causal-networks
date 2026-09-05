"""Benchmark portfolios. Only the equal-weight (1/N) benchmark survives; the
min-variance, mean-variance and cap-weighted benchmarks were never part of
the reported grid and were removed as dead code.

Returns a name-indexed pd.Series of weights summing to 1, matching the
HRP/HSP signature.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight(asset_names: list[str]) -> pd.Series:
    """``w_i = 1/N`` for every asset."""
    n = len(asset_names)
    if n == 0:
        raise ValueError("empty asset list")
    return pd.Series([1.0 / n] * n, index=asset_names, name="weight")


__all__ = ["equal_weight"]
