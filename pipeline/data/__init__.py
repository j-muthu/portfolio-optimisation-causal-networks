"""Data layer for the Causal-HSP pipeline.

This package will host:

* ``universe`` -- point-in-time S&P 500 membership via the ``fja05680/sp500``
  GitHub repo, plus a top-N-by-market-cap-at-date selector.
* ``assets`` -- price fetcher with **WRDS / CRSP** as the primary backend
  (survivorship-bias-free, historical shares-outstanding) and ``yfinance``
  as a thin fallback for the most recent ~2 trading days CRSP hasn't yet
  ingested. WRDS is gated on the ``wrds`` library being installed and
  credentials being set up in ``~/.pgpass``; without them the cascade
  silently runs on yfinance only.
* ``drivers`` -- ~50-series exogenous driver pool from FRED and Yahoo.
* ``alignment`` -- NYSE-calendar alignment, joint matrix construction, per-window
  z-score normalisation, and per-series stationarity flags.

For continuity with the legacy asset-only DYNOTEARS/VARLiNGAM pipeline, the
``Dataset`` / ``build_dataset`` API is re-exported here from
``pipeline.data.legacy`` so existing scripts keep working until they are ported
to the new joint-matrix API.
"""

from __future__ import annotations

from pipeline.data.legacy import Dataset, build_dataset

__all__ = ["Dataset", "build_dataset"]
