"""Survivorship-bias-aware price fetcher for the S&P-100 backtest.

The backtest spans 2007-2024, including the GFC, during which ~30-40 S&P 500
constituents (mostly financials) were delisted. ``yfinance`` silently returns
no data for many of these — if we relied on yfinance alone the resulting
universe would be biased toward survivors. The intended primary source is
CRSP via WRDS, which is survivorship-bias-free and also exposes
*historical* shares-outstanding (so the S&P-100 market-cap approximation
becomes point-in-time rather than today's shares × historical price).

Strategy
--------
Two-backend cascade per ticker, in priority order:

1. **WRDS / CRSP** (primary) -- :mod:`pipeline.data.wrds_backend`. Requires
   the ``wrds`` Python library + WRDS credentials configured in
   ``~/.pgpass`` (the convention the ``wrds`` library expects). When
   credentials are missing or the library is not installed the cascade
   silently falls through to yfinance — so the pipeline still runs (with
   the survivorship-bias caveat) before WRDS approval lands.
2. **yfinance** -- live-data fallback for the most recent ~2 trading days
   that CRSP typically hasn't yet ingested. Also the only backend before
   WRDS approval.

Every requested ticker is logged with its resolved source. At call sites
that build a rebalance universe, an alarm fires if joint coverage falls
below 95 %.

Per-ticker caching to ``cache/prices/<ticker>.parquet`` makes the
multi-hour download a one-time cost.

Shares-outstanding for the market-cap-at-date selection comes from CRSP
``shrout`` when WRDS is available (true point-in-time historical values),
otherwise falls back to yfinance ``Ticker.fast_info.shares`` as a constant
proxy. The fallback is a known approximation — flagged in the methodology
chapter and removed automatically once WRDS is wired in.

Entry points
------------
* :func:`fetch_prices` -- price panel for an explicit ticker list.
* :func:`fetch_shares_outstanding` -- shares panel/snapshot for the same tickers.
* :func:`coverage_report` -- joint backend coverage stats per rebalance date.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency).

    Reads ``KEY=value`` lines, ignores comments and blanks, only sets env vars
    that are not already defined (lets shell-level overrides win).
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv(THESIS_ROOT / ".env")

logger = logging.getLogger(__name__)

CACHE_DIR = THESIS_ROOT / "cache"
PRICES_DIR = CACHE_DIR / "prices"
SHARES_DIR = CACHE_DIR / "shares"
PRICES_DIR.mkdir(parents=True, exist_ok=True)
SHARES_DIR.mkdir(parents=True, exist_ok=True)

_HTTP_HEADERS = {"User-Agent": "thesis-causal-hsp/1.0 (academic research)"}

Source = Literal["wrds", "yfinance", "cache", "missing"]


# ============================================================================
# Result containers
# ============================================================================
@dataclass
class PricePanel:
    """Adjusted-close panel for a set of tickers over a date range.

    Attributes
    ----------
    prices:
        ``DataFrame`` indexed by date, columns = tickers actually resolved.
    sources:
        Mapping ``ticker -> source`` describing where each column came from
        (``"wrds"``, ``"yfinance"``, ``"cache"``). Tickers that could not be
        resolved at all are recorded as ``"missing"`` and omitted from
        ``prices``.
    coverage:
        Per-ticker fraction of trading days with data in the requested range.
    """

    prices: pd.DataFrame
    sources: dict[str, Source] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)

    @property
    def resolved(self) -> list[str]:
        return [t for t, s in self.sources.items() if s != "missing"]

    @property
    def missing(self) -> list[str]:
        return [t for t, s in self.sources.items() if s == "missing"]


# ============================================================================
# Ticker normalisation
# ============================================================================
def _normalise_ticker(symbol: str) -> str:
    """Yahoo's convention (``-`` for share-class)."""
    return symbol.strip().upper().replace(".", "-")


# ============================================================================
# WRDS / CRSP backend (primary)
# ============================================================================
def fetch_from_wrds(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series | None:
    """Return CRSP's split-and-dividend-adjusted close as a ``Series``.

    Thin shim over :mod:`pipeline.data.wrds_backend`. Returns ``None`` if the
    ``wrds`` library isn't installed, credentials are missing, or the ticker
    has no CRSP coverage in the date range — the cascade then falls through
    to yfinance.
    """
    try:
        from pipeline.data.wrds_backend import fetch_crsp_prices
    except ImportError:
        # `wrds_backend` module unavailable — shouldn't happen since it ships
        # with the package, but stay defensive.
        return None
    try:
        return fetch_crsp_prices(ticker, start, end)
    except (ImportError, ModuleNotFoundError):
        # `wrds` library not installed — silent fall-through.
        return None
    except Exception as exc:
        logger.debug("WRDS fetch failed for %s: %s", ticker, exc)
        return None


# ============================================================================
# yfinance backend (fallback)
# ============================================================================
def fetch_from_yfinance(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series | None:
    """Return auto-adjusted close from yfinance or ``None`` on empty/missing."""
    import warnings

    import yfinance as yf

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                ticker,
                start=str(start.date()),
                end=str((end + pd.Timedelta(days=1)).date()),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
    except Exception as exc:
        logger.debug("yfinance fetch raised for %s: %s", ticker, exc)
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if ("Close", ticker) in raw.columns:
            close = raw[("Close", ticker)]
        else:
            close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]
    close = close.dropna()
    if close.empty:
        return None
    close.index = pd.to_datetime(close.index)
    series = close.astype(float)
    series.name = ticker
    return series


# ============================================================================
# Per-ticker cached fetch (wrds → yfinance cascade)
# ============================================================================
def _cache_path(ticker: str) -> Path:
    # Yahoo's ``BRK-B`` is fine for filenames; uppercase for consistency.
    return PRICES_DIR / f"{ticker.upper()}.parquet"


def _read_cache(ticker: str) -> tuple[pd.Series, Source] | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    src: Source = df.attrs.get("source", "cache") if hasattr(df, "attrs") else "cache"
    # Parquet round-trips the column name; pull it out as a Series.
    series = df.iloc[:, 0]
    series.name = ticker
    return series, src


def _write_cache(ticker: str, series: pd.Series, source: Source) -> None:
    path = _cache_path(ticker)
    df = series.to_frame(name=ticker)
    df.attrs["source"] = source
    df.to_parquet(path)


def fetch_one(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_cache: bool = True,
    prefer: tuple[str, ...] = ("wrds", "yfinance"),
) -> tuple[pd.Series | None, Source]:
    """Fetch one ticker's close prices through the configured backend cascade.

    Returns ``(series, source)``; ``series`` is ``None`` if every backend
    failed (the ticker is recorded as ``"missing"``).
    """
    ticker = _normalise_ticker(ticker)
    if use_cache:
        cached = _read_cache(ticker)
        if cached is not None:
            series, src = cached
            if series.index.min() <= start and series.index.max() >= end:
                # Cached range fully covers the requested window.
                return series.loc[start:end], src

    for backend in prefer:
        if backend == "wrds":
            series = fetch_from_wrds(ticker, start, end)
        elif backend == "yfinance":
            series = fetch_from_yfinance(ticker, start, end)
        else:
            raise ValueError(f"unknown backend: {backend!r}")
        if series is not None and not series.empty:
            _write_cache(ticker, series, backend)  # type: ignore[arg-type]
            return series.loc[start:end], backend  # type: ignore[return-value]
    return None, "missing"


# ============================================================================
# Top-level: panel for a ticker list
# ============================================================================
def fetch_prices(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    use_cache: bool = True,
    prefer: tuple[str, ...] = ("wrds", "yfinance"),
) -> PricePanel:
    """Build a price panel for ``tickers`` over ``[start, end]``.

    Per-ticker caching means re-running the same call is cheap. Tickers that
    cannot be resolved by any backend are logged and omitted from the panel.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    tickers = sorted({_normalise_ticker(t) for t in tickers})

    series_map: dict[str, pd.Series] = {}
    sources: dict[str, Source] = {}
    for tkr in tickers:
        series, src = fetch_one(tkr, start_ts, end_ts, use_cache=use_cache, prefer=prefer)
        sources[tkr] = src
        if series is not None:
            series_map[tkr] = series

    if not series_map:
        prices = pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
    else:
        prices = pd.concat(series_map.values(), axis=1).sort_index()
        prices = prices.loc[start_ts:end_ts]

    n_resolved = sum(1 for s in sources.values() if s != "missing")
    n_wrds = sum(1 for s in sources.values() if s == "wrds")
    n_yf = sum(1 for s in sources.values() if s == "yfinance")
    n_cache = sum(1 for s in sources.values() if s == "cache")
    n_missing = sum(1 for s in sources.values() if s == "missing")
    logger.info(
        "fetch_prices: %d/%d resolved (wrds=%d, yfinance=%d, cache=%d, missing=%d)",
        n_resolved, len(tickers), n_wrds, n_yf, n_cache, n_missing,
    )
    if n_missing:
        logger.warning("Missing tickers (no backend resolved): %s", sorted(
            t for t, s in sources.items() if s == "missing"
        ))

    coverage = {}
    if not prices.empty:
        nonnull = prices.notna().sum()
        denom = max(len(prices), 1)
        coverage = {t: float(nonnull.get(t, 0)) / denom for t in series_map}

    return PricePanel(prices=prices, sources=sources, coverage=coverage)


# ============================================================================
# Shares outstanding
# ============================================================================
def fetch_shares_outstanding(
    tickers: Iterable[str],
    as_of: pd.Timestamp | str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """Shares-outstanding per ticker for the market-cap-at-date selection.

    Strategy:

    1. Prefer CRSP via :func:`pipeline.data.wrds_backend.fetch_crsp_shares_outstanding`
       when WRDS is available. CRSP exposes *historical* ``shrout`` so this
       returns the value snapped to the most recent observation at or before
       ``as_of``; the universe builder gets true point-in-time market cap.
    2. Fall back to yfinance ``Ticker.fast_info.shares`` (current value as a
       constant proxy) when WRDS isn't available. This is the standing
       approximation flagged in the methodology chapter until WRDS lands.

    ``as_of`` is ignored on the yfinance fallback path (no historical
    coverage); on the CRSP path it determines which snapshot is returned.

    Cached as ``cache/shares/shares_outstanding.parquet`` (yfinance path only;
    the WRDS backend has its own per-query cache).
    """
    requested = sorted({_normalise_ticker(t) for t in tickers})

    # --- WRDS path -----------------------------------------------------------
    try:
        from pipeline.data.wrds_backend import fetch_crsp_shares_outstanding
    except ImportError:
        fetch_crsp_shares_outstanding = None  # type: ignore[assignment]
    if fetch_crsp_shares_outstanding is not None:
        try:
            ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
            return fetch_crsp_shares_outstanding(requested, ts)
        except Exception as exc:
            logger.debug("WRDS shrout lookup failed (%s); falling back to yfinance proxy", exc)

    # --- yfinance fallback (current shares as proxy) ------------------------
    cache_path = SHARES_DIR / "shares_outstanding.parquet"
    cached: pd.Series
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path).iloc[:, 0]
    else:
        cached = pd.Series(dtype=float)

    missing = [t for t in requested if t not in cached.index]
    if missing:
        import yfinance as yf

        logger.info("Fetching shares-outstanding for %d new tickers", len(missing))
        new = {}
        for tkr in missing:
            try:
                fast = yf.Ticker(tkr).fast_info
                shares = float(fast.get("shares") or float("nan"))
            except Exception as exc:
                logger.debug("shares lookup failed for %s: %s", tkr, exc)
                shares = float("nan")
            new[tkr] = shares
        cached = pd.concat([cached, pd.Series(new)]).sort_index()
        cached.name = "shares_outstanding"
        cached.to_frame().to_parquet(cache_path)

    return cached.loc[requested]


# ============================================================================
# Coverage check
# ============================================================================
def coverage_report(
    panel: PricePanel,
    rebalance_universes: dict[pd.Timestamp, Sequence[str]],
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Per-rebalance coverage: fraction of intended top-N actually resolved.

    Returns a ``DataFrame`` (one row per rebalance) with columns
    ``intended``, ``resolved``, ``coverage``, ``missing``. The Stage 2
    backtest treats a row whose coverage falls below ``min_coverage`` as a
    backtest-invalidating event (raise, do not silently down-sample).
    """
    resolved = set(panel.resolved)
    rows = []
    for ts, universe in rebalance_universes.items():
        universe_set = {_normalise_ticker(t) for t in universe}
        present = universe_set & resolved
        coverage = len(present) / max(len(universe_set), 1)
        rows.append(
            {
                "date": pd.Timestamp(ts),
                "intended": len(universe_set),
                "resolved": len(present),
                "coverage": coverage,
                "missing": sorted(universe_set - resolved),
            }
        )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    below = df[df["coverage"] < min_coverage]
    if not below.empty:
        logger.warning(
            "%d rebalance dates have <%.0f%% coverage; first: %s (missing=%d)",
            len(below), 100 * min_coverage, below.iloc[0]["date"].date(),
            len(below.iloc[0]["missing"]),
        )
    return df


__all__ = [
    "PricePanel",
    "fetch_from_wrds",
    "fetch_from_yfinance",
    "fetch_one",
    "fetch_prices",
    "fetch_shares_outstanding",
    "coverage_report",
]
