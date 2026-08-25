"""Data layer: universe membership, prices (WRDS/CRSP with yfinance fallback),
drivers, and calendar alignment.

Re-exports the legacy ``Dataset`` / ``build_dataset`` API so old scripts keep
working.
"""

from __future__ import annotations

from pipeline.data.legacy import Dataset, build_dataset

__all__ = ["Dataset", "build_dataset"]
