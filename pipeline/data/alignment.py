"""NYSE calendar alignment, joint [D | A] matrix construction, per-window
z-scoring and stationarity flags.

zscore_window must be called inside the rolling loop and never on the full
panel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Trading calendar
def trading_calendar(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    use_market_calendars: bool = True,
) -> pd.DatetimeIndex:
    """Return the NYSE trading days for [start, end] inclusive.

    I use pandas_market_calendars if it is installed. Otherwise I fall back to
    SPY's trading dates from yfinance, which are equivalent but start in 1993.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if use_market_calendars:
        try:
            import pandas_market_calendars as mcal  # type: ignore

            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=start_ts, end_date=end_ts)
            return pd.DatetimeIndex(schedule.index).normalize()
        except ImportError:
            logger.info(
                "pandas_market_calendars not installed; using SPY-derived NYSE "
                "calendar fallback (install pandas_market_calendars for the "
                "canonical path)"
            )

    # Otherwise derive the calendar from SPY.
    import warnings

    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            "SPY",
            start=str(start_ts.date()),
            end=str((end_ts + pd.Timedelta(days=1)).date()),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    if raw is None or raw.empty:
        raise RuntimeError(
            "SPY-based NYSE calendar fallback returned no data — install "
            "pandas_market_calendars or fix the data fetch."
        )
    return pd.DatetimeIndex(raw.index).normalize().unique().sort_values()


# Joint matrix
@dataclass
class JointMatrix:
    """The [D | A] panel ready for discovery. The columns are always the
    drivers first, then the assets."""

    frame: pd.DataFrame
    driver_columns: list[str]
    asset_columns: list[str]
    driver_idx: np.ndarray
    asset_idx: np.ndarray
    rows_dropped: int = 0
    meta: dict = field(default_factory=dict)
    # Mask of real data per (row, asset), filled only under
    # drop_na='drivers_only'. False means the cell was NaN and was zero-filled.
    asset_eligibility: pd.DataFrame | None = None

    @property
    def n(self) -> int:
        return self.frame.shape[0]

    @property
    def d(self) -> int:
        return self.frame.shape[1]

    def driver_block(self) -> pd.DataFrame:
        return self.frame[self.driver_columns]

    def asset_block(self) -> pd.DataFrame:
        return self.frame[self.asset_columns]

    def assets_eligible_in_window(
        self, start: pd.Timestamp, end: pd.Timestamp,
    ) -> list[str]:
        """Return the assets with real (not filled) data on every trading day
        in the window."""
        if self.asset_eligibility is None:
            return list(self.asset_columns)
        mask = self.asset_eligibility.loc[start:end]
        if mask.empty:
            return []
        fully_eligible = mask.all(axis=0)
        return [a for a in self.asset_columns if bool(fully_eligible.get(a, False))]


def build_joint_matrix(
    drivers: pd.DataFrame,
    assets: pd.DataFrame,
    calendar: pd.DatetimeIndex | None = None,
    drop_na: bool | str = True,
) -> JointMatrix:
    """Combine the driver and asset frames into X = [D | A] on a shared
    calendar.

    drop_na=True or "any" drops any row with a NaN. "drivers_only" drops rows
    with driver NaNs but keeps rows with asset NaNs, zero-fills them and
    records the fills in asset_eligibility (callers must not trust the
    zeros). False leaves the NaNs in place.
    """
    drivers = drivers.copy()
    assets = assets.copy()
    drivers.index = pd.DatetimeIndex(drivers.index).normalize()
    assets.index = pd.DatetimeIndex(assets.index).normalize()

    if calendar is None:
        calendar = drivers.index.intersection(assets.index)
    calendar = pd.DatetimeIndex(calendar).normalize()

    aligned_d = drivers.reindex(calendar)
    aligned_a = assets.reindex(calendar)

    # Refuse to let an asset name silently shadow a driver name.
    overlap = set(aligned_d.columns) & set(aligned_a.columns)
    if overlap:
        raise ValueError(f"Driver/asset column overlap: {sorted(overlap)}")

    driver_columns = list(aligned_d.columns)
    asset_columns = list(aligned_a.columns)
    joint = pd.concat([aligned_d, aligned_a], axis=1)

    rows_dropped = 0
    asset_eligibility: pd.DataFrame | None = None

    if drop_na in (True, "any"):
        before = len(joint)
        joint = joint.dropna(how="any")
        rows_dropped = before - len(joint)
        if rows_dropped:
            logger.debug(
                "build_joint_matrix: dropped %d/%d rows with NaN", rows_dropped, before
            )
    elif drop_na == "drivers_only":
        # Drop rows with driver NaNs. Zero-fill asset NaNs and record them in
        # the eligibility mask.
        before = len(joint)
        driver_nan_rows = joint[driver_columns].isna().any(axis=1)
        joint = joint.loc[~driver_nan_rows]
        rows_dropped = before - len(joint)
        if rows_dropped:
            logger.debug(
                "build_joint_matrix[drivers_only]: dropped %d/%d rows where a "
                "driver was NaN; %d rows retained.",
                rows_dropped, before, len(joint),
            )

        asset_block = joint[asset_columns]
        asset_eligibility = asset_block.notna().copy()
        if asset_block.isna().any().any():
            n_zero_filled = int(asset_block.isna().sum().sum())
            joint.loc[:, asset_columns] = asset_block.fillna(0.0)
            logger.debug(
                "build_joint_matrix[drivers_only]: zero-filled %d (row, asset) cells; "
                "asset_eligibility mask populated.", n_zero_filled,
            )
    elif drop_na is False:
        pass
    else:
        raise ValueError(
            f"drop_na must be bool or 'any'/'drivers_only', got {drop_na!r}"
        )

    driver_idx = np.arange(0, len(driver_columns), dtype=int)
    asset_idx = np.arange(
        len(driver_columns), len(driver_columns) + len(asset_columns), dtype=int
    )
    return JointMatrix(
        frame=joint,
        driver_columns=driver_columns,
        asset_columns=asset_columns,
        driver_idx=driver_idx,
        asset_idx=asset_idx,
        rows_dropped=rows_dropped,
        meta={"calendar_len": len(calendar), "drop_na_mode": drop_na},
        asset_eligibility=asset_eligibility,
    )


# Per-window z-score
def zscore_window(
    frame: pd.DataFrame,
    ddof: int = 0,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score each column with this window's mean and standard deviation.

    Return (normalised, mean, std). eps is the minimum standard deviation, so
    that constant columns do not divide by zero.
    """
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=ddof)
    std_floored = std.where(std > eps, eps)
    zeros = std[std <= eps].index.tolist()
    if zeros:
        logger.debug("zscore_window: %d constant columns floored: %s", len(zeros), zeros)
    return (frame - mean) / std_floored, mean, std_floored


# Stationarity flags
@dataclass
class StationarityFlags:
    """Per-column ADF and KPSS outcomes. Note that the nulls point opposite
    ways. A low p means stationary for ADF and a high p means stationary for
    KPSS. Flagged series are kept rather than dropped."""

    adf_pvalues: pd.Series
    kpss_pvalues: pd.Series
    adf_stationary: pd.Series
    kpss_stationary: pd.Series
    both_fail: list[str]

    def summary(self) -> str:
        return (
            f"ADF stationary: {self.adf_stationary.sum()}/{len(self.adf_stationary)}; "
            f"KPSS stationary: {self.kpss_stationary.sum()}/{len(self.kpss_stationary)}; "
            f"both-fail (potentially non-stationary): {len(self.both_fail)}"
        )


def stationarity_flags(
    frame: pd.DataFrame,
    alpha: float = 0.05,
    skip_columns: Iterable[str] = (),
) -> StationarityFlags:
    """Run ADF and KPSS on every column. Return the flags. This never drops
    columns."""
    import warnings

    from statsmodels.tsa.stattools import adfuller, kpss

    skip = set(skip_columns)
    adf_p, kpss_p = {}, {}
    for col in frame.columns:
        if col in skip:
            adf_p[col] = np.nan
            kpss_p[col] = np.nan
            continue
        series = frame[col].dropna()
        if len(series) < 20:
            adf_p[col] = np.nan
            kpss_p[col] = np.nan
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_p[col] = adfuller(series, autolag="AIC")[1]
        except Exception as exc:
            logger.debug("ADF failed on %s: %s", col, exc)
            adf_p[col] = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_p[col] = kpss(series, regression="c", nlags="auto")[1]
        except Exception as exc:
            logger.debug("KPSS failed on %s: %s", col, exc)
            kpss_p[col] = np.nan

    adf_s = pd.Series(adf_p, name="adf_p")
    kpss_s = pd.Series(kpss_p, name="kpss_p")
    adf_stat = adf_s < alpha
    kpss_stat = kpss_s > alpha
    both_fail = sorted(
        set(adf_s[adf_s.isna() | ~adf_stat].index)
        & set(kpss_s[kpss_s.isna() | ~kpss_stat].index)
    )
    return StationarityFlags(
        adf_pvalues=adf_s,
        kpss_pvalues=kpss_s,
        adf_stationary=adf_stat,
        kpss_stationary=kpss_stat,
        both_fail=both_fail,
    )


__all__ = [
    "trading_calendar",
    "JointMatrix",
    "build_joint_matrix",
    "zscore_window",
    "StationarityFlags",
    "stationarity_flags",
]
