"""Data layer: universe membership, prices (WRDS/CRSP with a yfinance
fallback), drivers and calendar alignment.

I re-export the legacy ``Dataset`` and ``build_dataset`` API so that old
scripts keep working.
"""

from __future__ import annotations

from pipeline.data.legacy import Dataset, build_dataset

__all__ = ["Dataset", "build_dataset"]
