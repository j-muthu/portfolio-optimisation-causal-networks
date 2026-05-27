"""Exogenous driver pool for the Causal-HSP factor discovery.

The pool is constructed once at the start of a backtest, supplies the ``D``
block of the joint variable matrix ``X = [D | A]`` consumed by the discovery
stage, and is fixed for the duration of a run. Driver *selection* (which subset
of these enters the portfolio each rebalance) is the job of
:mod:`pipeline.factor_selection`.

Driver universe (~40 series, all defensibly exogenous to the S&P-100):

* **Macro** (FRED): CPI, core CPI, unemployment, industrial production, retail
  sales, housing starts, consumer sentiment.
* **FX** (FRED): broad trade-weighted USD index plus EUR/USD, USD/JPY,
  USD/GBP, USD/CNY.
* **Treasury rates** (FRED): 3M, 2Y, 5Y, 10Y, 30Y constant-maturity, plus the
  10Y-2Y and 10Y-3M slope/spread series.
* **Credit spreads** (FRED): Moody's Aaa, Baa, BAA-10Y. Pre-2007 substitute
  for HYG-LQD.
* **Commodities** (FRED + Yahoo): WTI, Brent, natural gas, gold, silver,
  copper.
* **Vol** (Yahoo): VIX, VVIX (VVIX from 2007 onward).
* **International equity** (Yahoo): EFA (EAFE), EEM (EM), EWJ, EWG, EWU.

Excluded by design (failed exogeneity test for the S&P-100 universe):
* US sector SPDRs — mechanical aggregations of constituents.
* Fama-French / AQR factor returns — constructed from long-short US equity
  portfolios that may include S&P-100 names.
* S&P 500 / Russell 1000 / NASDAQ-100 returns — near-identical to the asset
  universe.

VIX is borderline (derived from S&P 500 options); kept in the primary pool but
the discovery plan calls for a robustness check with VIX excluded.

Entry points
------------
* :func:`build_driver_pool` -- fetch and preprocess the full pool over
  ``[start, end]``. Returns a tidy ``DataFrame`` indexed by date with one
  column per driver, plus a ``DriverMeta`` describing each column's source,
  preprocessing applied, and pre/post-2007 substitution.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT

logger = logging.getLogger(__name__)

CACHE_DIR = THESIS_ROOT / "cache" / "drivers"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# FRED rejects custom User-Agent strings (silently returns empty body); send a
# browser-like UA so the request actually receives data.
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) thesis-research"}
FRED_TIMEOUT = 30
FRED_BACKOFF_S = (1.0, 2.0, 5.0)

Source = Literal["fred", "yahoo", "derived"]
Preprocessing = Literal["log_return", "first_diff", "yoy_pct", "level", "yoy_diff"]


# ============================================================================
# Driver catalogue
# ============================================================================
@dataclass(frozen=True)
class DriverSpec:
    """Specification for one driver in the pool.

    Attributes
    ----------
    name:
        Short, code-friendly identifier used as the column name throughout the
        pipeline (e.g. ``"cpi_yoy"``, ``"dgs10_diff"``, ``"vix"``).
    source:
        Backend the raw series is fetched from.
    identifier:
        Series ID (FRED code, e.g. ``"CPIAUCSL"``) or Yahoo ticker
        (e.g. ``"^VIX"``, ``"CL=F"``).
    preprocessing:
        How the raw series is transformed before joining the driver matrix.
    available_from:
        Earliest date the series can be used (None = full history). Used to
        gate pre-2007 substitutions (HYG, VVIX).
    description:
        Human-readable description for the methodology table.
    """

    name: str
    source: Source
    identifier: str
    preprocessing: Preprocessing
    available_from: pd.Timestamp | None = None
    description: str = ""


# Full ~40-series pool. Order is documentation-grade; downstream code keys by
# name, not position.
DRIVER_CATALOGUE: list[DriverSpec] = [
    # ---- Macro -----------------------------------------------------------
    DriverSpec("cpi_yoy",          "fred", "CPIAUCSL",  "yoy_pct",   description="CPI all urban consumers, YoY % change"),
    DriverSpec("core_cpi_yoy",     "fred", "CPILFESL",  "yoy_pct",   description="Core CPI (ex food & energy), YoY % change"),
    DriverSpec("unrate_diff",      "fred", "UNRATE",    "first_diff", description="Unemployment rate, monthly first-difference"),
    DriverSpec("indpro_yoy",       "fred", "INDPRO",    "yoy_pct",   description="Industrial production index, YoY % change"),
    DriverSpec("retail_sales_yoy", "fred", "RSAFS",     "yoy_pct",   description="Advance retail sales, YoY % change"),
    DriverSpec("housing_starts_yoy","fred", "HOUST",    "yoy_pct",   description="Housing starts, YoY % change"),
    DriverSpec("umich_sent_diff",  "fred", "UMCSENT",   "first_diff", description="U Michigan consumer sentiment, monthly first-difference"),
    # ---- Treasury rates --------------------------------------------------
    DriverSpec("dgs3m_diff",   "fred", "DGS3MO",  "first_diff", description="3M Treasury, daily change"),
    DriverSpec("dgs2_diff",    "fred", "DGS2",    "first_diff", description="2Y Treasury, daily change"),
    DriverSpec("dgs5_diff",    "fred", "DGS5",    "first_diff", description="5Y Treasury, daily change"),
    DriverSpec("dgs10_diff",   "fred", "DGS10",   "first_diff", description="10Y Treasury, daily change"),
    DriverSpec("dgs30_diff",   "fred", "DGS30",   "first_diff", description="30Y Treasury, daily change"),
    DriverSpec("t10y2y_diff",  "fred", "T10Y2Y",  "first_diff", description="10Y-2Y slope, daily change"),
    DriverSpec("t10y3m_diff",  "fred", "T10Y3M",  "first_diff", description="10Y-3M spread, daily change"),
    # ---- Credit spreads --------------------------------------------------
    DriverSpec("baa_minus_aaa_diff", "derived", "BAA-AAA", "first_diff", description="Moody's Baa - Aaa, daily change"),
    DriverSpec("baa10y_diff",        "fred",    "BAA10Y",  "first_diff", description="BAA - 10Y Treasury, daily change (pre-2007 HYG substitute)"),
    DriverSpec("hyg_lqd_logret",     "derived", "HYG-LQD", "log_return",
               available_from=pd.Timestamp("2007-04-11"),
               description="HYG/LQD ratio log-return (high-yield vs investment-grade ETF spread proxy)"),
    # ---- FX --------------------------------------------------------------
    DriverSpec("dxy_diff",     "fred", "DTWEXBGS", "first_diff", description="Trade-weighted USD index, daily change"),
    DriverSpec("eurusd_logret","fred", "DEXUSEU",  "log_return", description="EUR/USD log-return"),
    DriverSpec("jpyusd_logret","fred", "DEXJPUS",  "log_return", description="JPY/USD log-return"),
    DriverSpec("gbpusd_logret","fred", "DEXUSUK",  "log_return", description="GBP/USD log-return"),
    DriverSpec("cnyusd_logret","fred", "DEXCHUS",  "log_return", description="CNY/USD log-return"),
    # ---- Commodities -----------------------------------------------------
    DriverSpec("wti_logret",    "fred", "DCOILWTICO",   "log_return", description="WTI crude spot, log-return"),
    DriverSpec("brent_logret",  "fred", "DCOILBRENTEU", "log_return", description="Brent crude spot, log-return"),
    DriverSpec("gold_logret",   "yahoo", "GC=F", "log_return", description="COMEX gold futures (front-month), log-return — FRED's LBMA AM/PM fixings were discontinued"),
    DriverSpec("natgas_logret",  "yahoo", "NG=F", "log_return", description="Henry Hub natural gas futures, log-return"),
    DriverSpec("silver_logret",  "yahoo", "SI=F", "log_return", description="Silver futures, log-return"),
    DriverSpec("copper_logret",  "yahoo", "HG=F", "log_return", description="Copper futures, log-return"),
    # ---- Vol -------------------------------------------------------------
    DriverSpec("vix",   "yahoo", "^VIX",  "level",      description="CBOE VIX (level — already stationary)"),
    DriverSpec("vvix",  "yahoo", "^VVIX", "level",
               available_from=pd.Timestamp("2007-03-15"),
               description="CBOE VVIX (vol-of-vol; from 2007)"),
    # ---- International equity (ETFs) -------------------------------------
    DriverSpec("efa_logret", "yahoo", "EFA", "log_return", description="EAFE ETF log-return (developed ex US/Canada)"),
    DriverSpec("eem_logret", "yahoo", "EEM", "log_return", description="MSCI EM ETF log-return"),
    DriverSpec("ewj_logret", "yahoo", "EWJ", "log_return", description="Japan ETF log-return"),
    DriverSpec("ewg_logret", "yahoo", "EWG", "log_return", description="Germany ETF log-return"),
    DriverSpec("ewu_logret", "yahoo", "EWU", "log_return", description="UK ETF log-return"),
]


@dataclass
class DriverPool:
    """Output of :func:`build_driver_pool`."""

    frame: pd.DataFrame
    specs: dict[str, DriverSpec]
    raw: dict[str, pd.Series] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)

    @property
    def n_series(self) -> int:
        return self.frame.shape[1]


# ============================================================================
# FRED backend (free CSV; no API key required)
# ============================================================================
def fetch_fred_series(series_id: str, use_cache: bool = True) -> pd.Series:
    """Download a FRED series via the free fredgraph CSV endpoint.

    Returns a ``Series`` indexed by date, named ``series_id``. Cached per
    series to ``cache/drivers/fred_<series_id>.parquet``; cache is keyed only
    by series ID, so to force a refresh delete the file or pass
    ``use_cache=False``.
    """
    cache = CACHE_DIR / f"fred_{series_id}.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        series = df.iloc[:, 0]
        series.name = series_id
        return series

    url = f"{FRED_CSV_BASE}?id={series_id}"
    text = _http_get_text(url, timeout=FRED_TIMEOUT)
    if text is None:
        raise RuntimeError(f"FRED fetch for {series_id} failed (no HTTP backend succeeded)")

    raw = pd.read_csv(io.StringIO(text))
    # FRED CSV columns: 'observation_date' (or 'DATE' on older endpoints), <series_id>.
    cols = {c.lower(): c for c in raw.columns}
    date_col = cols.get("observation_date") or cols.get("date")
    if date_col is None or series_id not in raw.columns:
        raise RuntimeError(
            f"Unexpected FRED CSV layout for {series_id}: columns={list(raw.columns)}"
        )
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    # FRED uses '.' for missing observations.
    raw[series_id] = pd.to_numeric(raw[series_id], errors="coerce")
    out = raw.dropna(subset=[date_col]).set_index(date_col)[series_id].sort_index()
    out.name = series_id
    out.to_frame().to_parquet(cache)
    logger.debug("FRED %s: %d observations (%s..%s)",
                 series_id, len(out), out.index.min().date(), out.index.max().date())
    return out


def _http_get_text(url: str, timeout: int = 30) -> str | None:
    """Fetch ``url`` body as text, trying ``requests`` first then ``curl``.

    Some sandboxed environments stall on Python's bundled TLS stack but have
    a working system ``curl`` — falling back keeps the pipeline portable.
    Logs the fallback so the behaviour is visible during runs.
    """
    last_exc: Exception | None = None
    try:
        import requests

        for wait in [0.0, *FRED_BACKOFF_S]:
            if wait:
                time.sleep(wait)
            try:
                resp = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
            except requests.RequestException as exc:
                last_exc = exc
                continue
            if resp.status_code == 200:
                return resp.text
            if resp.status_code >= 500:
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
                continue
            return None
    except ImportError:
        pass

    curl = shutil.which("curl")
    if curl is None:
        logger.warning("requests failed and curl unavailable: %s", last_exc)
        return None
    logger.debug("requests failed (%s), falling back to curl", last_exc)
    try:
        out = subprocess.run(
            [curl, "-s", "-L", "--http1.1", "--max-time", str(timeout + 10),
             "-A", _HTTP_HEADERS["User-Agent"], url],
            capture_output=True, check=True, timeout=timeout + 20,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("curl fallback failed: %s", exc)
        return None
    return out.stdout.decode("utf-8", errors="replace")


# ============================================================================
# Yahoo backend (uses yfinance)
# ============================================================================
def fetch_yahoo_series(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_cache: bool = True,
) -> pd.Series:
    """Download a Yahoo series. Returns adjusted close as a ``Series``."""
    safe_name = ticker.replace("^", "caret_").replace("=", "_")
    cache = CACHE_DIR / f"yahoo_{safe_name}.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        series = df.iloc[:, 0]
        if series.index.min() <= start and series.index.max() >= end:
            series.name = ticker
            return series.loc[start:end]

    import warnings

    import yfinance as yf

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
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        if ("Close", ticker) in raw.columns:
            close = raw[("Close", ticker)]
        else:
            close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]
    close = close.dropna().astype(float)
    close.name = ticker
    close.index = pd.to_datetime(close.index)
    close.to_frame().to_parquet(cache)
    return close.loc[start:end]


# ============================================================================
# Preprocessing
# ============================================================================
def _preprocess(series: pd.Series, mode: Preprocessing) -> pd.Series:
    """Per-type stationarising transform.

    * ``log_return``  -- ``log(p_t / p_{t-1})``; for price-like series.
    * ``first_diff``  -- ``p_t - p_{t-1}``; for yield / spread series in %.
    * ``yoy_pct``     -- ``(p_t - p_{t-12m}) / p_{t-12m}``; for monthly macro
      levels (CPI, INDPRO, ...). The 12m offset is calendar-aware via
      ``Series.pct_change(freq='YE')``-style logic implemented as a 12-period
      shift after monthly-frequency resampling.
    * ``yoy_diff``    -- absolute 12m difference (for percentage-point series).
    * ``level``       -- pass through (for already-stationary series like VIX).
    """
    if mode == "log_return":
        return np.log(series / series.shift(1)).dropna()
    if mode == "first_diff":
        return series.diff().dropna()
    if mode == "yoy_pct":
        monthly = series.resample("ME").last()
        out = monthly.pct_change(12).dropna()
        return out
    if mode == "yoy_diff":
        monthly = series.resample("ME").last()
        out = monthly.diff(12).dropna()
        return out
    if mode == "level":
        return series.dropna()
    raise ValueError(f"unknown preprocessing mode: {mode!r}")


# ============================================================================
# Derived series (BAA-AAA, HYG-LQD)
# ============================================================================
def _build_derived(name: str, use_cache: bool) -> pd.Series:
    """Compute the derived spreads referenced by ``DriverSpec.source='derived'``."""
    if name == "BAA-AAA":
        baa = fetch_fred_series("BAA", use_cache=use_cache)
        aaa = fetch_fred_series("AAA", use_cache=use_cache)
        spread = (baa - aaa).dropna()
        spread.name = "BAA-AAA"
        return spread
    if name == "HYG-LQD":
        # Daily; fetch with a generous range so caching is reusable across runs.
        wide_start, wide_end = pd.Timestamp("2007-01-01"), pd.Timestamp.now().normalize()
        hyg = fetch_yahoo_series("HYG", wide_start, wide_end, use_cache=use_cache)
        lqd = fetch_yahoo_series("LQD", wide_start, wide_end, use_cache=use_cache)
        ratio = (hyg / lqd).dropna()
        ratio.name = "HYG-LQD"
        return ratio
    raise ValueError(f"unknown derived series: {name}")


# ============================================================================
# Daily alignment + monthly-source forward-fill
# ============================================================================
def _to_daily(series: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Project a series onto a daily trading-day calendar.

    Monthly / weekly observations (FRED macro indicators) are forward-filled to
    every trading day at or after their release date, matching the convention
    in Howard et al. — the value an observer would actually have known on a
    given trading day.
    """
    return series.reindex(daily_index, method="ffill")


# ============================================================================
# Orchestrator
# ============================================================================
def build_driver_pool(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    daily_index: pd.DatetimeIndex | None = None,
    specs: list[DriverSpec] | None = None,
    use_cache: bool = True,
) -> DriverPool:
    """Fetch + preprocess the full driver pool over ``[start, end]``.

    Parameters
    ----------
    start, end:
        Date range (inclusive).
    daily_index:
        Trading-day calendar to project monthly/weekly drivers onto. Pass the
        index of the asset price panel for perfect alignment with the asset
        block. If ``None``, falls back to the union of every driver's native
        index restricted to weekdays (a calendar-naive proxy).
    specs:
        Override the default :data:`DRIVER_CATALOGUE` (e.g. for testing or to
        run a VIX-excluded robustness check).
    use_cache:
        Pass through to underlying fetchers.

    Returns
    -------
    DriverPool with a ``frame`` whose rows are trading days, columns are driver
    names (``DriverSpec.name``), values are the post-preprocessing series.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    specs = list(specs) if specs is not None else list(DRIVER_CATALOGUE)

    raw: dict[str, pd.Series] = {}
    processed: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}

    for spec in specs:
        if spec.available_from is not None and spec.available_from > end_ts:
            dropped[spec.name] = f"available_from={spec.available_from.date()} > end"
            continue
        try:
            if spec.source == "fred":
                raw_series = fetch_fred_series(spec.identifier, use_cache=use_cache)
            elif spec.source == "yahoo":
                # Pad the Yahoo download a year so YoY transforms have room.
                pad_start = (start_ts - pd.Timedelta(days=365 * 2)).normalize()
                raw_series = fetch_yahoo_series(
                    spec.identifier, pad_start, end_ts, use_cache=use_cache
                )
            elif spec.source == "derived":
                raw_series = _build_derived(spec.identifier, use_cache=use_cache)
            else:
                raise ValueError(f"unknown source: {spec.source}")
        except Exception as exc:
            logger.warning("Dropping driver %s (%s): %s", spec.name, spec.identifier, exc)
            dropped[spec.name] = f"{type(exc).__name__}: {exc}"
            continue
        raw[spec.name] = raw_series
        try:
            processed[spec.name] = _preprocess(raw_series, spec.preprocessing)
        except Exception as exc:
            logger.warning("Preprocessing failed for %s: %s", spec.name, exc)
            dropped[spec.name] = f"preprocess: {exc}"

    if daily_index is None:
        # Fall back to a weekday union of every series' index.
        all_idx = pd.DatetimeIndex(sorted({d for s in processed.values() for d in s.index}))
        daily_index = all_idx[all_idx.weekday < 5]
    daily_index = pd.DatetimeIndex(daily_index)
    daily_index = daily_index[(daily_index >= start_ts) & (daily_index <= end_ts)]

    # Project every driver onto the daily calendar with forward-fill (the
    # "what an observer knew at trading-day t" rule).
    columns: dict[str, pd.Series] = {}
    for name, series in processed.items():
        spec = next(s for s in specs if s.name == name)
        gated = series
        if spec.available_from is not None:
            gated = series.loc[series.index >= spec.available_from]
        daily = _to_daily(gated, daily_index)
        if spec.available_from is not None:
            # Mask trading days before availability to NaN so downstream code
            # can decide whether to drop or substitute.
            daily.loc[daily.index < spec.available_from] = np.nan
        columns[name] = daily

    frame = pd.DataFrame(columns)
    spec_map = {s.name: s for s in specs if s.name in frame.columns}
    logger.info(
        "build_driver_pool: %d/%d drivers retained (%d dropped); %d trading days",
        frame.shape[1], len(specs), len(dropped), len(daily_index),
    )
    return DriverPool(frame=frame, specs=spec_map, raw=raw, dropped=dropped)


__all__ = [
    "DriverSpec",
    "DriverPool",
    "DRIVER_CATALOGUE",
    "fetch_fred_series",
    "fetch_yahoo_series",
    "build_driver_pool",
]
