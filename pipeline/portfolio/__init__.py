"""Portfolio construction (HRP, HSP, Causal-HSP variants).

Until the new HRP / HSP / Causal-HSP modules land in Phase 6, this package
re-exports the legacy helpers from ``_old_v123`` (V1/V2/V3 sketches from the
prior asset-only plan) so existing scripts keep working. Only the genuinely
reusable helpers (``nearest_psd``, ``symmetrise``, ``causal_embedding_distance``)
will survive the Phase 6 rewrite — they remain useful for the V0' asset-only
Causal-HRP variant.
"""

from __future__ import annotations

from pipeline.portfolio._old_v123 import (
    causal_embedding_distance,
    compare_hrp,
    nearest_psd,
    symmetrise,
)
from pipeline.portfolio.backtest import BacktestResult, RebalanceRecord, run_backtest
from pipeline.portfolio.benchmarks import (
    cap_weighted,
    equal_weight,
    mean_variance,
    min_variance,
)
from pipeline.portfolio.causal_hsp import (
    v0_vanilla_hsp,
    v0prime_asset_only_causal_hrp,
    v1_causal_hsp_open_loop,
    v2_causal_hsp_closed_loop,
)
from pipeline.portfolio.hrp import (
    hrp_weights,
    quasi_diagonal_order,
    recursive_bisection,
)
from pipeline.portfolio.hsp import (
    hsp_weights_from_S,
    hsp_weights_from_window,
    ledoit_wolf_covariance,
    sample_covariance,
)

__all__ = [
    # Legacy retained helpers
    "nearest_psd",
    "symmetrise",
    "causal_embedding_distance",
    "compare_hrp",
    # HRP
    "hrp_weights",
    "quasi_diagonal_order",
    "recursive_bisection",
    # HSP
    "hsp_weights_from_S",
    "hsp_weights_from_window",
    "sample_covariance",
    "ledoit_wolf_covariance",
    # Strategy variants
    "v0_vanilla_hsp",
    "v0prime_asset_only_causal_hrp",
    "v1_causal_hsp_open_loop",
    "v2_causal_hsp_closed_loop",
    # Benchmarks
    "equal_weight",
    "min_variance",
    "mean_variance",
    "cap_weighted",
    # Backtest
    "RebalanceRecord",
    "BacktestResult",
    "run_backtest",
]
