"""Benchmark portfolios.

I keep only the equal-weight (1/N) benchmark. I removed the min-variance,
mean-variance and cap-weighted benchmarks because they were never part of
the reported grid.

The function returns a name-indexed pd.Series of weights that sum to 1,
matching the HRP and HSP signature.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight(asset_names: list[str]) -> pd.Series:
    """Give every asset the weight 1/N."""
    n = len(asset_names)
    if n == 0:
        raise ValueError("empty asset list")
    return pd.Series([1.0 / n] * n, index=asset_names, name="weight")


__all__ = ["equal_weight"]
