"""Verification harness: V0 (vanilla HSP) on Rodriguez-Dominguez's data.

Loads the reference panels under ``thesis/HSP/`` and runs the V0 path so
the weights can be cross-checked against the published notebook. No-op if
the Excel files are absent (not under version control).

Not a pytest test; run as ``.venv/bin/python -m tests.test_hsp_reference``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
HSP_DATA_DIR = REPO_ROOT / "HSP"
ASSETS_PATH = HSP_DATA_DIR / "Assets_SPX.xlsx"
DRIVERS_PATH = HSP_DATA_DIR / "Drivers_no_SB_Sectors.xlsx"

logger = logging.getLogger(__name__)


# Loader
class ReferenceData(NamedTuple):
    """Cleaned panels: daily log-returns on a common trading-day calendar."""

    asset_returns: pd.DataFrame
    driver_returns: pd.DataFrame
    asset_prices: pd.DataFrame
    driver_prices: pd.DataFrame


def load_hsp_reference_data() -> ReferenceData:
    """Load both Excel files, compute log-returns, intersect calendars."""
    if not ASSETS_PATH.exists() or not DRIVERS_PATH.exists():
        raise FileNotFoundError(
            f"Reference data missing — expected {ASSETS_PATH} and {DRIVERS_PATH}."
        )

    # Assets sheet: prices, Date already a Timestamp.
    asset_prices = pd.read_excel(ASSETS_PATH, sheet_name="Daily_Equities_Hypo")
    asset_prices = asset_prices.set_index("Date").sort_index()
    asset_prices.index = pd.DatetimeIndex(asset_prices.index).normalize()

    # Drivers sheet: values are already returns; Date is an integer YYYYMMDD.
    drivers_raw = pd.read_excel(DRIVERS_PATH, sheet_name="Sheet1")
    drivers_raw["Date"] = pd.to_datetime(
        drivers_raw["Date"].astype(int).astype(str), format="%Y%m%d"
    )
    driver_returns = drivers_raw.set_index("Date").sort_index()
    driver_returns.index = pd.DatetimeIndex(driver_returns.index).normalize()
    # Driver "prices" don't exist (values are returns); placeholder only.
    driver_prices = driver_returns.cumsum()

    # Intersect calendars (assets 2011-2024, drivers 2015-2022).
    common = asset_prices.index.intersection(driver_returns.index)
    if common.empty:
        raise RuntimeError(
            "Asset and driver calendars don't overlap after parsing — check the "
            "Excel files (asset: %s..%s; driver: %s..%s)" % (
                asset_prices.index.min(), asset_prices.index.max(),
                driver_returns.index.min(), driver_returns.index.max(),
            )
        )
    asset_prices = asset_prices.loc[common]
    driver_returns = driver_returns.loc[common]
    driver_prices = driver_prices.loc[common]

    asset_returns = np.log(asset_prices / asset_prices.shift(1)).dropna(how="all")

    # Drop drivers with materially missing data.
    coverage = driver_returns.notna().mean()
    keep = coverage[coverage > 0.90].index
    dropped = sorted(set(driver_returns.columns) - set(keep))
    if dropped:
        logger.info("Dropped %d drivers with > 10%% NaN coverage", len(dropped))
    driver_returns = driver_returns[keep]

    # Final intersection after the diff drops rows.
    common = asset_returns.index.intersection(driver_returns.index)
    return ReferenceData(
        asset_returns=asset_returns.loc[common],
        driver_returns=driver_returns.loc[common],
        asset_prices=asset_prices.loc[common],
        driver_prices=driver_prices.loc[common],
    )


# V0 path on a single rebalance window
def run_v0_on_reference_window(
    ref: ReferenceData,
    rebalance_date: pd.Timestamp | str,
    lookback_days: int = 252,
    K: int = 10,
    lags: tuple[int, ...] = (0, 1),
    linkage_method: str = "single",
    ffnn_epochs: int = 100,
) -> pd.Series:
    """One V0 rebalance: cum-corr selection, FFNN sensitivities, HRP."""
    from pipeline.factor_selection import select_top_k_corr
    from pipeline.portfolio import v0_vanilla_hsp
    from pipeline.sensitivities import fit_sensitivities_window

    rdate = pd.Timestamp(rebalance_date).normalize()
    cal = ref.asset_returns.index
    end_pos = cal.searchsorted(rdate, side="right")
    start_pos = max(0, end_pos - lookback_days)
    window_cal = cal[start_pos:end_pos]
    if len(window_cal) < lookback_days:
        raise ValueError(
            f"Insufficient lookback at {rdate.date()}: have {len(window_cal)} "
            f"trading days, need {lookback_days}"
        )

    drivers_w = ref.driver_returns.loc[window_cal].dropna(axis=1, how="any")
    assets_w = ref.asset_returns.loc[window_cal].dropna(axis=1, how="any")

    sel = select_top_k_corr(drivers_w, assets_w, K=K, rebalance_date=rdate, lags=lags)
    logger.info("V0 cum-corr selected: %s", sel.selected)

    # Z-score per window before the FFNN.
    zs_d = (drivers_w - drivers_w.mean()) / drivers_w.std(ddof=0).replace(0, 1)
    zs_a = (assets_w - assets_w.mean()) / assets_w.std(ddof=0).replace(0, 1)
    sens = fit_sensitivities_window(
        drivers=zs_d, assets=zs_a, selected_drivers=sel.selected,
        rebalance_date=rdate, lags=1, epochs=ffnn_epochs, use_cache=False,
    )

    weights = v0_vanilla_hsp(
        S=sens.S, asset_names=sens.asset_names,
        returns_window=assets_w, linkage_method=linkage_method,
    )
    return weights


# CLI: print one rebalance's weights for eyeball cross-check
def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not ASSETS_PATH.exists():
        print(f"SKIP: reference data missing ({ASSETS_PATH})")
        return 0

    ref = load_hsp_reference_data()
    print(f"Reference data loaded:")
    print(f"  assets:  {ref.asset_returns.shape} ({list(ref.asset_returns.columns)[:3]}, ...)")
    print(f"  drivers: {ref.driver_returns.shape}")
    print(f"  date range: {ref.asset_returns.index.min().date()} .. {ref.asset_returns.index.max().date()}")

    rdate = ref.asset_returns.index[-1]
    weights = run_v0_on_reference_window(ref, rdate, K=10, ffnn_epochs=80)
    print(f"\nV0 weights @ {rdate.date()} (K=10, lookback=252):")
    print(weights.round(4).sort_values(ascending=False).to_string())
    print(f"sum: {weights.sum():.6f}  (should be 1.000)")
    print(f"min: {weights.min():.6f}  max: {weights.max():.6f}")

    assert abs(weights.sum() - 1.0) < 1e-6, "weights must sum to 1"
    assert (weights >= -1e-8).all(), "weights must be non-negative (long-only HSP)"
    print("\nPASS: V0 weights on reference data are valid (sum=1, non-negative)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
