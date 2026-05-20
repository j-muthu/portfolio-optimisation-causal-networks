"""Causal-discovery pipelines for the S&P 500 thesis.

Two interchangeable causal-discovery methods over a shared data pipeline:

* Plan A -- :mod:`pipeline.rolling_dynotears` (score-based, DYNOTEARS).
* Plan B -- :mod:`pipeline.rolling_varlingam` (ICA-based, VARLiNGAM).

Both consume a :class:`pipeline.data.Dataset` of standardised daily log-returns
and emit rolling sequences of weighted causal adjacency matrices, analysed by
:mod:`pipeline.graph_analysis` and fed into :mod:`pipeline.portfolio`.

The DYNOTEARS and VARLiNGAM implementations are the vendored ``causalnex`` and
``lingam`` source trees in this repo; :mod:`pipeline._vendored` bridges to them.
"""

from __future__ import annotations

from pipeline.data import Dataset, build_dataset
from pipeline.graph_analysis import (
    analyse_rolling,
    causal_order_drift,
    compare_rolling,
    detect_regime_changes,
    sector_flow,
)
from pipeline.rolling_dynotears import run_dynotears_window, run_rolling_dynotears
from pipeline.rolling_varlingam import run_rolling_varlingam, run_varlingam_window

__all__ = [
    "Dataset",
    "build_dataset",
    "run_rolling_dynotears",
    "run_dynotears_window",
    "run_rolling_varlingam",
    "run_varlingam_window",
    "analyse_rolling",
    "detect_regime_changes",
    "sector_flow",
    "causal_order_drift",
    "compare_rolling",
]
