"""WRDS / CRSP backend for survivorship-bias-free prices and shares.

Talks straight to the WRDS Postgres endpoint via SQLAlchemy + psycopg (the
official wrds library pins old pandas/numpy). Auth comes from ~/.pgpass
(host wrds-pgdata.wharton.upenn.edu:9737, db wrds, chmod 600); WRDS_USERNAME
env var optionally overrides the username. Queries are cached under
cache/wrds/ and retried on transient drops.

Tickers are resolved to PERMNOs (stable; tickers get reused, PERMNOs don't).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT

logger = logging.getLogger(__name__)

CACHE_DIR = THESIS_ROOT / "cache" / "wrds"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DATABASE = "wrds"
RETRY_BACKOFF_S = (2.0, 5.0, 10.0)

_ENGINE = None  # singleton sqlalchemy.Engine


# Connection management
def _username_from_pgpass() -> str | None:
    """Username from the matching ~/.pgpass line
    (host:port:database:username:password), or None."""
    path = Path(os.environ.get("PGPASSFILE", Path.home() / ".pgpass"))
    if not path.exists():
        return None
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 5:
                continue
            host, port, db, user, _pw = parts[0], parts[1], parts[2], parts[3], ":".join(parts[4:])
            if host in (WRDS_HOST, "*") and port in (str(WRDS_PORT), "*") and db in (WRDS_DATABASE, "*"):
                return user
    except OSError as exc:
        logger.debug("Could not read %s: %s", path, exc)
    return None


def _get_engine():
    """Lazy-init the singleton WRDS engine. Username: WRDS_USERNAME env var,
    then ~/.pgpass; password comes from libpq at connect time."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    username = os.environ.get("WRDS_USERNAME", "").strip() or _username_from_pgpass()
    if not username:
        raise RuntimeError(
            f"Could not resolve WRDS username. Either set ``WRDS_USERNAME`` or "
            f"add a line to ~/.pgpass of the form "
            f"``{WRDS_HOST}:{WRDS_PORT}:{WRDS_DATABASE}:USERNAME:PASSWORD``."
        )

    from sqlalchemy import create_engine

    url = (
        f"postgresql+psycopg://{username}@{WRDS_HOST}:{WRDS_PORT}/{WRDS_DATABASE}"
        f"?sslmode=require"
    )
    logger.info("Opening WRDS engine for user %s @ %s:%d", username, WRDS_HOST, WRDS_PORT)
    _ENGINE = create_engine(url, pool_pre_ping=True, pool_recycle=1800)
    return _ENGINE


def _retry_query(query, params: dict | None = None) -> pd.DataFrame:
    """Run a query with retry on transient drops. Terminal failures (missing
    driver or username) raise immediately so the assets cascade falls through
    to yfinance."""
    last_exc: Exception | None = None
    for attempt, wait in enumerate([0.0, *RETRY_BACKOFF_S]):
        if wait:
            time.sleep(wait)
        try:
            engine = _get_engine()
            return pd.read_sql(query, engine, params=params or {})
        except (ImportError, ModuleNotFoundError):
            raise  # sqlalchemy/psycopg not installed
        except RuntimeError as exc:
            # Missing username is also terminal.
            if "WRDS_USERNAME" in str(exc):
                raise
            last_exc = exc
        except Exception as exc:
            last_exc = exc
            logger.debug("WRDS query attempt %d failed: %s", attempt + 1, exc)
            # Rebuild the engine next try; the pool may be in a bad state.
            global _ENGINE
            _ENGINE = None
    raise RuntimeError(f"WRDS query failed after retries: {last_exc}") from last_exc


def verify_connection() -> bool:
    """Smoke-test the WRDS connection; True on success."""
    from sqlalchemy import text

    try:
        df = _retry_query(text("SELECT current_user, version() AS pg_version"))
    except Exception as exc:
        logger.error("WRDS verify_connection failed: %s", exc)
        return False
    print(df.to_string(index=False))
    return True


# Cache helpers
def _cache_key(*parts: str) -> Path:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return CACHE_DIR / f"{h}.parquet"


# Ticker -> PERMNO resolution
def _resolve_permnos(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    """All PERMNOs the ticker mapped to in [start, end]. One ticker can map
    to several PERMNOs over time (delisting + reuse)."""
    cache = _cache_key("permnos", ticker.upper(), start.isoformat(), end.isoformat())
    if cache.exists():
        return pd.read_parquet(cache)["permno"].astype(int).tolist()

    from sqlalchemy import text

    query = text(
        """
        SELECT DISTINCT permno
        FROM crsp.stocknames
        WHERE ticker = :ticker
          AND namedt <= :end
          AND nameenddt >= :start
        """
    )
    df = _retry_query(
        query,
        params={"ticker": ticker.upper(), "start": start.date(), "end": end.date()},
    )
    permnos = sorted(int(p) for p in df["permno"].tolist())
    pd.DataFrame({"permno": permnos}).to_parquet(cache)
    return permnos


def _resolve_permnos_batch(
    tickers: Sequence[str], start: pd.Timestamp, end: pd.Timestamp,
) -> dict[str, list[int]]:
    """Batch ticker -> PERMNO mapping in one SQL query. Tickers with no CRSP
    coverage in the window are absent from the result."""
    tickers_up = sorted({t.upper() for t in tickers})
    if not tickers_up:
        return {}
    cache = _cache_key(
        "permnos_batch",
        "|".join(tickers_up),
        start.isoformat(), end.isoformat(),
    )
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        from sqlalchemy import bindparam, text

        query = text(
            """
            SELECT DISTINCT ticker, permno
            FROM crsp.stocknames
            WHERE ticker IN :tickers
              AND namedt <= :end
              AND nameenddt >= :start
            """
        ).bindparams(bindparam("tickers", expanding=True))
        df = _retry_query(
            query,
            params={"tickers": tickers_up, "start": start.date(), "end": end.date()},
        )
        df.to_parquet(cache)

    out: dict[str, list[int]] = {}
    for ticker, group in df.groupby("ticker"):
        out[str(ticker).upper()] = sorted(int(p) for p in group["permno"].tolist())
    return out


# Bulk market cap at a snapshot date
def fetch_crsp_mcap_at_snapshot(
    tickers: Sequence[str],
    as_of: pd.Timestamp,
    lookback_days: int = 5,
    use_cache: bool = True,
) -> pd.Series:
    """Market cap (USD) per ticker at as_of in two SQL round-trips.

    mcap = |prc| * shrout * 1000 / cfacshr (shrout is in thousands; cfacshr
    undoes splits). Takes the latest observation per PERMNO within
    lookback_days of as_of. Tickers without CRSP coverage are absent from
    the result.
    """
    tickers_up = sorted({t.upper() for t in tickers})
    if not tickers_up:
        return pd.Series(dtype=float, name="mcap")

    cache = _cache_key(
        "mcap_snapshot",
        "|".join(tickers_up),
        as_of.isoformat(),
        str(lookback_days),
    )
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        return df.iloc[:, 0]

    start_window = as_of - pd.Timedelta(days=lookback_days)
    # Resolve PERMNOs over a 1y window to catch ticker changes near as_of.
    resolve_start = as_of - pd.Timedelta(days=365)
    permno_map = _resolve_permnos_batch(tickers_up, resolve_start, as_of)
    if not permno_map:
        empty = pd.Series(dtype=float, name="mcap")
        empty.to_frame().to_parquet(cache)
        return empty

    permno_to_ticker: dict[int, str] = {}
    all_permnos: set[int] = set()
    for ticker, permnos in permno_map.items():
        for p in permnos:
            permno_to_ticker[p] = ticker
            all_permnos.add(p)

    from sqlalchemy import bindparam, text

    query = text(
        """
        SELECT permno, date,
               ABS(prc) * shrout * 1000.0 / NULLIF(cfacshr, 0) AS mcap
        FROM crsp.dsf
        WHERE permno IN :permnos
          AND date BETWEEN :start AND :end
          AND prc IS NOT NULL
          AND shrout IS NOT NULL
        """
    ).bindparams(bindparam("permnos", expanding=True))
    df = _retry_query(
        query,
        params={
            "permnos": sorted(all_permnos),
            "start": start_window.date(),
            "end": as_of.date(),
        },
    )
    if df.empty:
        empty = pd.Series(dtype=float, name="mcap")
        empty.to_frame().to_parquet(cache)
        return empty

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["permno"].map(permno_to_ticker)
    # Latest row per PERMNO, then sum across PERMNOs per ticker (rare).
    latest_per_permno = (
        df.sort_values(["permno", "date"])
        .drop_duplicates("permno", keep="last")
    )
    mcap_per_ticker = (
        latest_per_permno.groupby("ticker")["mcap"].sum().astype(float)
    )
    mcap_per_ticker.name = "mcap"
    mcap_per_ticker = mcap_per_ticker.sort_index()
    mcap_per_ticker.to_frame().to_parquet(cache)
    logger.info(
        "WRDS mcap snapshot @ %s: %d/%d tickers resolved (lookback=%dd)",
        as_of.date(), len(mcap_per_ticker), len(tickers_up), lookback_days,
    )
    return mcap_per_ticker


# Prices
def fetch_crsp_prices(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.Series | None:
    """Daily CRSP split-adjusted close for ticker over [start, end], or None
    if no PERMNO matched. Cached per (ticker, start, end)."""
    cache = _cache_key("prices", ticker.upper(), start.isoformat(), end.isoformat())
    if cache.exists():
        return pd.read_parquet(cache).iloc[:, 0]

    permnos = _resolve_permnos(ticker, start, end)
    if not permnos:
        logger.debug("No PERMNO for %s in [%s, %s]", ticker, start.date(), end.date())
        return None

    from sqlalchemy import bindparam, text

    # prc is negative for bid-ask midpoints (no trade), so take abs();
    # adjusted_close = abs(prc) / cfacpr.
    query = text(
        """
        SELECT date, permno, ABS(prc) / NULLIF(cfacpr, 0) AS adj_close
        FROM crsp.dsf
        WHERE permno IN :permnos
          AND date BETWEEN :start AND :end
          AND prc IS NOT NULL
        ORDER BY date
        """
    ).bindparams(bindparam("permnos", expanding=True))
    df = _retry_query(
        query,
        params={"permnos": permnos, "start": start.date(), "end": end.date()},
    )
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    # PERMNOs for one ticker cover disjoint dates, so per-date last is safe.
    series = (
        df.sort_values(["date", "permno"])
        .drop_duplicates("date", keep="last")
        .set_index("date")["adj_close"]
        .astype(float)
    )
    series.name = ticker
    series.to_frame().to_parquet(cache)
    logger.info(
        "WRDS prices for %s: %d obs (%s..%s) across %d PERMNO(s)",
        ticker, len(series), series.index.min().date(),
        series.index.max().date(), len(permnos),
    )
    return series


# Shares outstanding
def fetch_crsp_shares_outstanding(
    tickers: Sequence[str], as_of: pd.Timestamp,
) -> pd.Series:
    """Shares outstanding per ticker at as_of, in units (CRSP shrout is in
    thousands; we multiply by 1000). Most recent observation <= as_of;
    tickers without coverage are missing from the result."""
    cache = _cache_key(
        "shrout_panel",
        "|".join(sorted(t.upper() for t in tickers)),
        as_of.isoformat(),
    )
    if cache.exists():
        return pd.read_parquet(cache).iloc[:, 0]

    from sqlalchemy import bindparam, text

    out: dict[str, float] = {}
    for ticker in tickers:
        permnos = _resolve_permnos(ticker, pd.Timestamp("1990-01-01"), as_of)
        if not permnos:
            continue
        query = text(
            """
            SELECT shrout
            FROM crsp.dsf
            WHERE permno IN :permnos
              AND date <= :asof
              AND shrout IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
            """
        ).bindparams(bindparam("permnos", expanding=True))
        df = _retry_query(query, params={"permnos": permnos, "asof": as_of.date()})
        if df.empty:
            continue
        out[ticker.upper()] = float(df["shrout"].iloc[0]) * 1000.0  # thousands to units

    series = pd.Series(out, name="shares_outstanding").sort_index()
    series.to_frame().to_parquet(cache)
    logger.info(
        "WRDS shrout @ %s: %d/%d tickers resolved",
        as_of.date(), len(series), len(tickers),
    )
    return series


__all__ = [
    "fetch_crsp_prices",
    "fetch_crsp_shares_outstanding",
    "fetch_crsp_mcap_at_snapshot",
    "verify_connection",
]
