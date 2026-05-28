"""WRDS / CRSP backend for survivorship-bias-free US-equity prices + shares.

Talks directly to WRDS's PostgreSQL endpoint via SQLAlchemy + psycopg 3.
The official ``wrds`` Python library is a thin wrapper that pins old pandas
(<2.3) and numpy (<1.27), which conflicts with this project's pandas 3 /
numpy 2 stack — direct SQL through libpq bypasses that constraint while
honouring the same ``~/.pgpass`` auth convention the wrds library uses
internally.

Setup (one-time, on the developer machine — not in version control)::

    # 1. Create ~/.pgpass with your WRDS credentials
    cat >> ~/.pgpass <<'EOF'
    wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_WRDS_USERNAME:YOUR_WRDS_PASSWORD
    EOF

    # 2. Lock down permissions (libpq refuses to read it otherwise)
    chmod 600 ~/.pgpass

That's it. No env var is needed — :func:`_username_from_pgpass` parses the
username out of the same ``.pgpass`` line. ``WRDS_USERNAME`` env var is
honoured if set (lets you point at a different account without editing
``.pgpass``), but it's optional.

Then :func:`verify_connection` confirms everything is wired correctly.

Two entry points, both used by :mod:`pipeline.data.assets`:

* :func:`fetch_crsp_prices` -- daily CRSP split-and-dividend-adjusted close
  for one ticker over ``[start, end]``. Resolves ticker → PERMNO at the
  date range (CRSP's stable identifier; tickers get reused across delisted
  and new companies, PERMNOs do not).
* :func:`fetch_crsp_shares_outstanding` -- shares-outstanding for a list of
  tickers, snapped to the most-recent observation at or before ``as_of``.

Implementation notes
--------------------
* Singleton SQLAlchemy engine with lazy init. The engine itself pools
  connections; we cache it at module level so per-call overhead is just
  borrowing from the pool.
* All queries cached to ``cache/wrds/<query_hash>.parquet`` so re-runs are
  effectively free.
* Polite retry (3 attempts, exponential backoff) on transient connection
  drops — WRDS occasionally resets idle SSL connections.

CRSP table reference
--------------------
* ``crsp.dsf`` -- daily stock file. Columns we use: ``permno``, ``date``,
  ``prc`` (close; negative if bid-ask midpoint), ``cfacpr`` (cumulative
  price-adjustment factor for splits & spin-offs), ``cfacshr`` (cumulative
  share-adjustment factor), ``shrout`` (shares outstanding, thousands).
* ``crsp.stocknames`` -- ticker ↔ PERMNO mapping with effective-date ranges.
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


# ============================================================================
# Connection management
# ============================================================================
def _username_from_pgpass() -> str | None:
    """Parse the username field out of ``~/.pgpass`` for the WRDS host.

    libpq uses ``.pgpass`` for password lookup; we additionally extract the
    matching username so the caller doesn't have to duplicate it in an env
    var. Returns ``None`` if no matching line is found.

    ``.pgpass`` lines are ``host:port:database:username:password`` (``*`` is
    a wildcard, ``\\:`` escapes a literal colon, but we don't need to handle
    either for the standard WRDS entry).
    """
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
    """Lazy-init a SQLAlchemy engine pointing at the WRDS Postgres endpoint.

    The username is resolved in this order:
      1. ``WRDS_USERNAME`` env var (explicit override)
      2. The ``username`` field of the matching ``~/.pgpass`` line
      3. (fail with ``RuntimeError``)
    The password is supplied by libpq from ``~/.pgpass`` at connect time,
    so nothing sensitive lives in this process or its env.
    """
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
    """Run a SQLAlchemy ``text()`` query with retry on transient drops.

    Non-transient failures (``ImportError`` from a missing SQL driver,
    missing ``WRDS_USERNAME``) raise immediately so the cascade in
    :mod:`pipeline.data.assets` falls through to yfinance instantly.
    """
    last_exc: Exception | None = None
    for attempt, wait in enumerate([0.0, *RETRY_BACKOFF_S]):
        if wait:
            time.sleep(wait)
        try:
            engine = _get_engine()
            return pd.read_sql(query, engine, params=params or {})
        except (ImportError, ModuleNotFoundError):
            raise  # terminal — sqlalchemy/psycopg not installed
        except RuntimeError as exc:
            # WRDS_USERNAME missing — also terminal.
            if "WRDS_USERNAME" in str(exc):
                raise
            last_exc = exc
        except Exception as exc:
            last_exc = exc
            logger.debug("WRDS query attempt %d failed: %s", attempt + 1, exc)
            # Force engine rebuild on the next try (pool may be in a bad state).
            global _ENGINE
            _ENGINE = None
    raise RuntimeError(f"WRDS query failed after retries: {last_exc}") from last_exc


def verify_connection() -> bool:
    """Smoke-test the WRDS connection. Returns True on success.

    Run interactively the first time you set up ``.pgpass`` to confirm
    everything is wired correctly. Prints diagnostic context on failure.
    """
    from sqlalchemy import text

    try:
        df = _retry_query(text("SELECT current_user, version() AS pg_version"))
    except Exception as exc:
        logger.error("WRDS verify_connection failed: %s", exc)
        return False
    print(df.to_string(index=False))
    return True


# ============================================================================
# Cache helpers
# ============================================================================
def _cache_key(*parts: str) -> Path:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return CACHE_DIR / f"{h}.parquet"


# ============================================================================
# Ticker → PERMNO resolution
# ============================================================================
def _resolve_permnos(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    """Return all PERMNOs that the ticker mapped to within ``[start, end]``.

    A single ticker can map to multiple PERMNOs over time (delisting + reuse).
    The caller takes the union and joins the resulting price series in date
    order; CRSP's ``dsf`` is the source of truth for which PERMNO was active
    on which date.
    """
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
    """Batch ticker → PERMNO mapping in a single SQL query.

    Returns ``{ticker: [permnos that mapped to this ticker in [start, end]]}``
    for every input ticker. Tickers with no CRSP coverage in the window
    are absent from the result (caller should treat as missing).

    Used by :func:`fetch_crsp_mcap_at_snapshot` to avoid the per-ticker
    round-trip storm in the universe-builder hot path.
    """
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


# ============================================================================
# Bulk market cap at a snapshot date (G.6)
# ============================================================================
def fetch_crsp_mcap_at_snapshot(
    tickers: Sequence[str],
    as_of: pd.Timestamp,
    lookback_days: int = 5,
    use_cache: bool = True,
) -> pd.Series:
    """Market cap (USD) per ticker at ``as_of`` in one SQL round-trip.

    The bulk replacement for the universe builder's per-ticker
    ``fetch_prices`` + ``fetch_shares_outstanding`` calls. Steps:

    1. Batch-resolve every input ticker to its PERMNO(s) in
       a date window around ``as_of`` (one SQL).
    2. Issue a single ``crsp.dsf`` query for all PERMNOs over a small
       trailing window ending at ``as_of`` (one SQL).
    3. Compute split-adjusted market cap per row in pandas:
       ``mcap = |prc| × shrout × 1000 / cfacshr`` (CRSP ``shrout``
       is in thousands; dividing by the cumulative share-adjustment
       factor undoes splits to give shares-on-that-date).
    4. Per PERMNO take the latest row in the window; then collapse
       multiple PERMNOs per ticker (rare; usually rename events).

    Result is a ``pd.Series`` indexed by uppercase ticker; tickers with
    no CRSP coverage in the window are absent from the result.

    Cached at ``cache/wrds/<hash>.parquet`` keyed by ``(tickers, as_of,
    lookback_days)`` so repeated calls are free.

    Parameters
    ----------
    tickers:
        S&P 500 (or any) member tickers. Order ignored; case-normalised.
    as_of:
        Snapshot date. The function returns the latest available mcap
        for each PERMNO at or before this date within ``lookback_days``.
    lookback_days:
        Trailing calendar-day window to find the most-recent observation
        per PERMNO. Default 5 catches weekend / holiday gaps.
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
    # Resolve PERMNOs over a wide-enough window (1y) to handle ticker
    # changes / corporate actions near the snapshot date.
    resolve_start = as_of - pd.Timedelta(days=365)
    permno_map = _resolve_permnos_batch(tickers_up, resolve_start, as_of)
    if not permno_map:
        empty = pd.Series(dtype=float, name="mcap")
        empty.to_frame().to_parquet(cache)
        return empty

    # Flatten + dedupe the PERMNO universe; track reverse mapping.
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
    # Per PERMNO, take the latest row in the lookback window.
    latest_per_permno = (
        df.sort_values(["permno", "date"])
        .drop_duplicates("permno", keep="last")
    )
    # Per ticker, sum mcaps across PERMNOs (rare multi-PERMNO case).
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


# ============================================================================
# Prices
# ============================================================================
def fetch_crsp_prices(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.Series | None:
    """Daily CRSP split-adjusted close for ``ticker`` over ``[start, end]``.

    Returns a ``pd.Series`` indexed by date, named ``ticker``. ``None`` if no
    PERMNO matched the ticker in the requested window (e.g. typo, or ticker
    that never existed in CRSP's coverage).

    Cached per ``(ticker, start, end)`` triple.
    """
    cache = _cache_key("prices", ticker.upper(), start.isoformat(), end.isoformat())
    if cache.exists():
        return pd.read_parquet(cache).iloc[:, 0]

    permnos = _resolve_permnos(ticker, start, end)
    if not permnos:
        logger.debug("No PERMNO for %s in [%s, %s]", ticker, start.date(), end.date())
        return None

    from sqlalchemy import bindparam, text

    # CRSP `prc` is negative when it represents a bid-ask midpoint (no trade);
    # we take abs() to recover the magnitude. `cfacpr` is the cumulative
    # price-adjustment factor: adjusted_close = abs(prc) / cfacpr.
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
    # Multiple PERMNOs over disjoint dates: take the per-date last value (one
    # row per date by construction since PERMNOs don't overlap for the same
    # ticker).
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


# ============================================================================
# Shares outstanding
# ============================================================================
def fetch_crsp_shares_outstanding(
    tickers: Sequence[str], as_of: pd.Timestamp,
) -> pd.Series:
    """Shares-outstanding (in *units*, not thousands) per ticker at ``as_of``.

    Snaps to the most recent CRSP observation at or before ``as_of`` for each
    ticker. CRSP's ``shrout`` is in thousands; we multiply by 1000 to return
    actual share counts (matches the yfinance convention).

    Returns a Series indexed by ticker (uppercase); tickers with no CRSP
    coverage are missing from the result (caller should ``.reindex()`` and
    decide whether to drop or fall back).
    """
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
        out[ticker.upper()] = float(df["shrout"].iloc[0]) * 1000.0  # thousands → units

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
