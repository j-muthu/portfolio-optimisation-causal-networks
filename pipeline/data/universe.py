"""Point-in-time S&P 500 membership (fja05680/sp500 GitHub CSV) and the
S&P-100 approximation: top 100 by market cap from the S&P 500 at date.

Prices and shares are supplied by the caller; this module fetches neither.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT

logger = logging.getLogger(__name__)

CACHE_DIR = THESIS_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

FJA05680_REPO_API = "https://api.github.com/repos/fja05680/sp500/contents/"
FJA05680_RAW_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"
_HTTP_HEADERS = {"User-Agent": "thesis-causal-hsp/1.0 (academic research)"}

# The fja05680 file name carries a date stamp that changes on update.
_FJA_NAME_RE = re.compile(
    r"S&P 500 Historical Components & Changes\((\d{2}-\d{2}-\d{4})\)\.csv"
)


# Membership table
@dataclass
class SP500History:
    """S&P 500 membership snapshots: one row per change date with the active
    ticker set. Membership at a date = the latest row <= that date."""

    frame: pd.DataFrame
    source_filename: str
    fetched_at: pd.Timestamp

    @property
    def first_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame["date"].iloc[0])

    @property
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame["date"].iloc[-1])


def _normalise_ticker(symbol: str) -> str:
    """Yahoo's convention: '-' not '.' for share-class suffixes."""
    return symbol.strip().upper().replace(".", "-")


def _resolve_latest_fja_filename(use_cache: bool = True) -> str:
    """Pick the most recent date-stamped CSV in the repo. Cached 24 h to spare
    the GitHub API."""
    cache = CACHE_DIR / "fja05680_filename.txt"
    if use_cache and cache.exists():
        age = pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")
        if age < pd.Timedelta(hours=24):
            return cache.read_text().strip()

    import requests

    resp = requests.get(FJA05680_REPO_API, headers=_HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    entries = resp.json()

    matches: list[tuple[datetime, str]] = []
    for entry in entries:
        name = entry.get("name", "")
        m = _FJA_NAME_RE.fullmatch(name)
        if not m:
            continue
        stamp = datetime.strptime(m.group(1), "%m-%d-%Y")
        matches.append((stamp, name))
    if not matches:
        raise RuntimeError(
            "Could not find a 'S&P 500 Historical Components & Changes' CSV in "
            "fja05680/sp500 — repo layout may have changed."
        )
    latest = max(matches, key=lambda t: t[0])[1]
    cache.write_text(latest)
    logger.info("fja05680/sp500 latest file: %s", latest)
    return latest


def fetch_fja05680(use_cache: bool = True) -> SP500History:
    """Download (or load from cache) the fja05680/sp500 historical CSV."""
    cache = CACHE_DIR / "fja05680_sp500_history.parquet"
    meta = CACHE_DIR / "fja05680_sp500_history.meta.json"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        if meta.exists():
            import json

            m = json.loads(meta.read_text())
            return SP500History(
                frame=df,
                source_filename=m.get("source_filename", "(unknown)"),
                fetched_at=pd.Timestamp(m.get("fetched_at", "1970-01-01")),
            )
        return SP500History(frame=df, source_filename="(cached)", fetched_at=pd.Timestamp.now())

    import requests

    filename = _resolve_latest_fja_filename(use_cache=use_cache)
    # The raw URL needs spaces and & ( ) URL-encoded.
    encoded = (
        filename.replace("&", "%26")
        .replace(" ", "%20")
        .replace("(", "%28")
        .replace(")", "%29")
    )
    url = FJA05680_RAW_BASE + encoded
    logger.info("Fetching fja05680/sp500 from %s", url)
    resp = requests.get(url, headers=_HTTP_HEADERS, timeout=60)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))

    # Expected columns: 'date', 'tickers' (comma-separated symbols).
    cols = {c.lower(): c for c in raw.columns}
    if "date" not in cols or "tickers" not in cols:
        raise RuntimeError(
            f"Unexpected fja05680 CSV columns: {list(raw.columns)} — "
            "expected ['date', 'tickers']."
        )
    date_col, tickers_col = cols["date"], cols["tickers"]
    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "tickers": raw[tickers_col].astype(str).map(
                lambda s: tuple(sorted({_normalise_ticker(t) for t in s.split(",") if t.strip()}))
            ),
        }
    )
    parsed = parsed.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    parsed.to_parquet(cache)
    import json

    meta.write_text(
        json.dumps(
            {
                "source_filename": filename,
                "fetched_at": pd.Timestamp.now().isoformat(),
            }
        )
    )
    logger.info(
        "Stored %d S&P 500 membership snapshots (%s..%s)",
        len(parsed), parsed["date"].iloc[0].date(), parsed["date"].iloc[-1].date(),
    )
    return SP500History(
        frame=parsed,
        source_filename=filename,
        fetched_at=pd.Timestamp.now(),
    )


# Query helpers
def membership_at(
    date: str | pd.Timestamp,
    history: SP500History | None = None,
    use_cache: bool = True,
) -> frozenset[str]:
    """S&P 500 constituents active on date (latest snapshot <= date). Raises
    ValueError before the start of the table."""
    if history is None:
        history = fetch_fja05680(use_cache=use_cache)
    ts = pd.Timestamp(date).normalize()
    frame = history.frame
    idx = frame["date"].searchsorted(ts, side="right") - 1
    if idx < 0:
        raise ValueError(
            f"date {ts.date()} is before the earliest fja05680 snapshot "
            f"({history.first_date.date()})"
        )
    return frozenset(frame["tickers"].iloc[int(idx)])


def all_tickers_ever(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    history: SP500History | None = None,
    use_cache: bool = True,
) -> list[str]:
    """Every ticker that was a member at any point in [start, end], including
    since-delisted names. Used to pre-fetch the price universe."""
    if history is None:
        history = fetch_fja05680(use_cache=use_cache)
    frame = history.frame
    if start is not None:
        s = pd.Timestamp(start).normalize()
        # Include the snapshot active at start (latest <= start).
        first_idx = max(0, int(frame["date"].searchsorted(s, side="right")) - 1)
        frame = frame.iloc[first_idx:]
    if end is not None:
        e = pd.Timestamp(end).normalize()
        frame = frame[frame["date"] <= e]
    seen: set[str] = set()
    for tickers in frame["tickers"]:
        seen.update(tickers)
    return sorted(seen)


# S&P 100 approximation: top-N by market cap at date
def top_n_by_mcap_at(
    date: str | pd.Timestamp,
    n: int,
    prices: pd.DataFrame,
    shares_outstanding: pd.Series,
    history: SP500History | None = None,
    use_cache: bool = True,
    min_price_lookback_days: int = 5,
) -> list[str]:
    """Top-n S&P 500 constituents by market cap as of date, descending.

    Uses the most recent close <= date, tolerating up to
    min_price_lookback_days of staleness. shares_outstanding is one value per
    ticker (a constant proxy on the non-WRDS path). Members missing price or
    shares data are skipped and logged.
    """
    ts = pd.Timestamp(date).normalize()
    members = membership_at(ts, history=history, use_cache=use_cache)
    available = sorted(set(members) & set(prices.columns) & set(shares_outstanding.index))
    missing = sorted(members - set(available))
    if missing:
        logger.debug(
            "top_n_by_mcap_at %s: %d/%d members missing price or shares data",
            ts.date(), len(missing), len(members),
        )

    # Most recent close at or before date, within the lookback window.
    window = prices.loc[:ts].tail(min_price_lookback_days)
    if window.empty:
        raise ValueError(f"No prices available at or before {ts.date()}")
    latest_close = window.ffill().iloc[-1]

    caps = pd.Series(
        {
            tkr: float(latest_close.get(tkr, np.nan) * shares_outstanding.get(tkr, np.nan))
            for tkr in available
        }
    ).dropna()
    if len(caps) < n:
        logger.warning(
            "top_n_by_mcap_at %s: only %d/%d members had usable mcap (requested %d)",
            ts.date(), len(caps), len(members), n,
        )
    return caps.sort_values(ascending=False).head(n).index.tolist()


def rolling_top_n_universe(
    rebalance_dates: Sequence[pd.Timestamp],
    n: int,
    prices: pd.DataFrame,
    shares_outstanding: pd.Series,
    history: SP500History | None = None,
    use_cache: bool = True,
) -> dict[pd.Timestamp, list[str]]:
    """Top-N selection per rebalance date in one call."""
    if history is None:
        history = fetch_fja05680(use_cache=use_cache)
    return {
        pd.Timestamp(d): top_n_by_mcap_at(
            d, n, prices, shares_outstanding, history=history, use_cache=use_cache
        )
        for d in rebalance_dates
    }


def union_of_universes(
    universes: Iterable[Sequence[str]] | dict[pd.Timestamp, Sequence[str]],
) -> list[str]:
    """Sorted union of all tickers appearing in any rebalance's top-N; trims
    price downloads to the relevant subset."""
    iterable = universes.values() if isinstance(universes, dict) else universes
    seen: set[str] = set()
    for u in iterable:
        seen.update(u)
    return sorted(seen)


__all__ = [
    "SP500History",
    "fetch_fja05680",
    "membership_at",
    "all_tickers_ever",
    "top_n_by_mcap_at",
    "rolling_top_n_universe",
    "union_of_universes",
]
