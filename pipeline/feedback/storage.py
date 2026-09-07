"""Lookahead-safe storage of the driver-utility table U[t].

I key the rows by the holding-period end date rather than the rebalance
date, because the credit for a rebalance is not known until the holding
period ends. This makes "what was known at t" a single lookup.
lookup_utility asserts that end_date <= t - 21 days to guard against
same-day leaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum gap in calendar days between a queried rebalance and the latest
# visible utility row (1 monthly rebalance).
MIN_LOOKAHEAD_GAP_DAYS = 21


# Storage
@dataclass
class UtilityStore:
    """Driver-utility table indexed by end_date, with hard lookahead
    assertions. aux_columns are the non-utility columns, which are excluded
    from lookups."""

    parquet_path: Path
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    aux_columns: tuple[str, ...] = ("rebalance_date", "reward")

    # IO
    @classmethod
    def load_or_empty(cls, path: Path | str) -> "UtilityStore":
        path = Path(path)
        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.DatetimeIndex(df.index)
            df = df.sort_index()
            logger.info("UtilityStore loaded %d rows from %s", len(df), path)
            return cls(parquet_path=path, frame=df)
        return cls(parquet_path=path, frame=pd.DataFrame())

    def save(self) -> None:
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.frame.copy()
        df.index.name = "end_date"
        df.to_parquet(self.parquet_path)
        logger.debug("UtilityStore saved %d rows to %s", len(df), self.parquet_path)

    # Append
    def append(
        self,
        rebalance_date: pd.Timestamp,
        holding_end: pd.Timestamp,
        updated_utility: pd.Series,
        reward: float,
    ) -> None:
        """Append the row keyed by holding_end, or replace it."""
        end_ts = pd.Timestamp(holding_end).normalize()
        row = updated_utility.copy()
        row["rebalance_date"] = pd.Timestamp(rebalance_date)
        row["reward"] = float(reward)
        if not isinstance(self.frame, pd.DataFrame) or self.frame.empty:
            df = pd.DataFrame([row], index=[end_ts])
        else:
            df = self.frame.copy()
            df.loc[end_ts] = row
        df = df.sort_index()
        # Keep the last write per end_date.
        df = df.loc[~df.index.duplicated(keep="last")]
        self.frame = df

    # Lookup with lookahead assertion
    def lookup_utility(
        self,
        rebalance_date: pd.Timestamp,
        min_gap_days: int = MIN_LOOKAHEAD_GAP_DAYS,
        require_strict: bool = True,
    ) -> tuple[pd.Series, pd.Timestamp | None]:
        """Return the (U, end_date) valid at rebalance t. This is the latest
        row with end_date <= t - min_gap_days, or (empty, None) during
        burn-in. require_strict asserts the gap, and only the leak check
        bypasses it."""
        t = pd.Timestamp(rebalance_date).normalize()
        if self.frame.empty:
            return pd.Series(dtype=float, name="utility"), None
        cutoff = t - pd.Timedelta(days=min_gap_days)
        eligible = self.frame.index[self.frame.index <= cutoff]
        if len(eligible) == 0:
            return pd.Series(dtype=float, name="utility"), None
        latest = eligible[-1]
        if require_strict:
            assert latest <= t - pd.Timedelta(days=min_gap_days), (
                f"lookahead leak: utility row {latest.date()} not at least "
                f"{min_gap_days} days before rebalance {t.date()}"
            )
        row = self.frame.loc[latest]
        utility = row.drop(labels=list(self.aux_columns), errors="ignore")
        utility = utility.astype(float)
        utility.name = "utility"
        return utility, latest

    def as_lookup(self) -> Callable[[pd.Timestamp], tuple[pd.Series, pd.Timestamp | None]]:
        """Return a callable with the signature that selector.select_drivers
        expects."""
        return lambda t: self.lookup_utility(t)


__all__ = ["UtilityStore", "MIN_LOOKAHEAD_GAP_DAYS"]
