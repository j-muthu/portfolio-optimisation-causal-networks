"""Lookahead-safe persistence of the driver-utility table ``U[t]``.

Per the locked implementation choice (``Closed-Loop Causal-HSP Portfolio.md``):

* Rows are keyed by **holding-period-end date**, not rebalance date. The
  credit for the rebalance at ``t`` cannot be computed until ``t + 21d`` when
  the holding period ends; storing rows by holding-period-end makes
  "what was known at trading-day ``t'``" a single explicit lookup.
* :func:`UtilityStore.lookup_utility` returns the latest row whose
  ``end_date ≤ t'``, and **asserts** ``end_date ≤ t' - 21d`` to guard against
  any same-day-as-rebalance leak. This is hardest-edge protection in the
  pipeline.

Schema of the persisted Parquet (``results/<run>/utility.parquet``):

* Index: ``end_date`` (pd.Timestamp, the holding-period-end).
* Columns: one per driver name, dtype float; reward ``R`` and
  ``rebalance_date`` are persisted as auxiliary columns.

Updates append a row each rebalance, deduplicated by ``end_date``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum gap (in trading days, approximated as calendar days × 1.45) that the
# storage layer requires between a queried rebalance and the latest visible
# utility row. With monthly rebalancing this is 21 trading days ≈ 30 calendar
# days; we use the latter to keep the assertion calendar-aware.
MIN_LOOKAHEAD_GAP_DAYS = 21


# ============================================================================
# Storage
# ============================================================================
@dataclass
class UtilityStore:
    """Time-indexed driver-utility table with hard lookahead assertions.

    Attributes
    ----------
    parquet_path:
        Filesystem location. Loaded on construction; updates write back.
    frame:
        In-memory representation. Sorted by ``end_date``; ``end_date`` is the
        index. Auxiliary columns ``rebalance_date`` and ``reward`` are stored
        alongside the per-driver utility floats.
    aux_columns:
        Column names that are *not* driver utilities — excluded from
        ``lookup_utility`` returns.
    """

    parquet_path: Path
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    aux_columns: tuple[str, ...] = ("rebalance_date", "reward")

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Append a rebalance's credit attribution → updated U row
    # ------------------------------------------------------------------
    def append(
        self,
        rebalance_date: pd.Timestamp,
        holding_end: pd.Timestamp,
        updated_utility: pd.Series,
        reward: float,
    ) -> None:
        """Append (or replace) the row keyed by ``holding_end``."""
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
        # Dedup: keep last write per end_date.
        df = df.loc[~df.index.duplicated(keep="last")]
        self.frame = df

    # ------------------------------------------------------------------
    # Lookup with lookahead assertion
    # ------------------------------------------------------------------
    def lookup_utility(
        self,
        rebalance_date: pd.Timestamp,
        min_gap_days: int = MIN_LOOKAHEAD_GAP_DAYS,
        require_strict: bool = True,
    ) -> tuple[pd.Series, pd.Timestamp | None]:
        """Return ``(U, end_date)`` valid at the rebalance ``t``.

        Reads the latest row whose ``end_date ≤ t - min_gap_days`` (calendar
        days). If no such row exists yet (e.g. first 6 rebalances during
        burn-in), returns an empty utility ``pd.Series`` and ``None``.

        ``require_strict=True`` (the default) **asserts** the lookahead gap
        — this fires only on a programming error. The leak canary
        (:mod:`pipeline.feedback.leak_canary`) deliberately bypasses this.
        """
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

    # ------------------------------------------------------------------
    # Convenience: make a Callable for selector.utility_lookup
    # ------------------------------------------------------------------
    def as_lookup(self) -> Callable[[pd.Timestamp], tuple[pd.Series, pd.Timestamp | None]]:
        """Return a function with the signature ``selector.select_drivers`` expects."""
        return lambda t: self.lookup_utility(t)


__all__ = ["UtilityStore", "MIN_LOOKAHEAD_GAP_DAYS"]
