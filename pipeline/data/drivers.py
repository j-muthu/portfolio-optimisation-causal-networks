"""Exogenous driver pool (~40 FRED/Yahoo series) supplying the D block of
X = [D | A].

Sector SPDRs, Fama-French factors and broad US index returns are excluded as
non-exogenous to the S&P-100. VIX is borderline (S&P 500 options) but kept,
with a VIX-excluded robustness check planned.
"""

from __future__ import annotations

import io
import json
import logging
import os
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
# FRED silently returns an empty body for non-browser User-Agents.
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) thesis-research"}
FRED_TIMEOUT = 30
FRED_BACKOFF_S = (1.0, 2.0, 5.0)

Source = Literal["fred", "yahoo", "derived"]
Preprocessing = Literal["log_return", "first_diff", "yoy_pct", "level", "yoy_diff"]


# Driver catalogue
@dataclass(frozen=True)
class DriverSpec:
    """One driver: column name, source backend, series ID / ticker,
    stationarising transform, and optional availability gate (used for the
    pre-2007 HYG/VVIX substitutions)."""

    name: str
    source: Source
    identifier: str
    preprocessing: Preprocessing
    available_from: pd.Timestamp | None = None
    description: str = ""


# Downstream code keys by name, not position.
DRIVER_CATALOGUE: list[DriverSpec] = [
    # Macro
    DriverSpec("cpi_yoy",          "fred", "CPIAUCSL",  "yoy_pct",   description="CPI all urban consumers, YoY % change"),
    DriverSpec("core_cpi_yoy",     "fred", "CPILFESL",  "yoy_pct",   description="Core CPI (ex food & energy), YoY % change"),
    DriverSpec("unrate_diff",      "fred", "UNRATE",    "first_diff", description="Unemployment rate, monthly first-difference"),
    DriverSpec("indpro_yoy",       "fred", "INDPRO",    "yoy_pct",   description="Industrial production index, YoY % change"),
    DriverSpec("retail_sales_yoy", "fred", "RSAFS",     "yoy_pct",   description="Advance retail sales, YoY % change"),
    DriverSpec("housing_starts_yoy","fred", "HOUST",    "yoy_pct",   description="Housing starts, YoY % change"),
    DriverSpec("umich_sent_diff",  "fred", "UMCSENT",   "first_diff", description="U Michigan consumer sentiment, monthly first-difference"),
    # Treasury rates
    DriverSpec("dgs3m_diff",   "fred", "DGS3MO",  "first_diff", description="3M Treasury, daily change"),
    DriverSpec("dgs2_diff",    "fred", "DGS2",    "first_diff", description="2Y Treasury, daily change"),
    DriverSpec("dgs5_diff",    "fred", "DGS5",    "first_diff", description="5Y Treasury, daily change"),
    DriverSpec("dgs10_diff",   "fred", "DGS10",   "first_diff", description="10Y Treasury, daily change"),
    DriverSpec("dgs30_diff",   "fred", "DGS30",   "first_diff", description="30Y Treasury, daily change"),
    DriverSpec("t10y2y_diff",  "fred", "T10Y2Y",  "first_diff", description="10Y-2Y slope, daily change"),
    DriverSpec("t10y3m_diff",  "fred", "T10Y3M",  "first_diff", description="10Y-3M spread, daily change"),
    # Credit spreads
    DriverSpec("baa_minus_aaa_diff", "derived", "BAA-AAA", "first_diff", description="Moody's Baa - Aaa, daily change"),
    DriverSpec("baa10y_diff",        "fred",    "BAA10Y",  "first_diff", description="BAA - 10Y Treasury, daily change (pre-2007 HYG substitute)"),
    DriverSpec("hyg_lqd_logret",     "derived", "HYG-LQD", "log_return",
               available_from=pd.Timestamp("2007-04-11"),
               description="HYG/LQD ratio log-return (high-yield vs investment-grade ETF spread proxy)"),
    # FX
    DriverSpec("dxy_diff",     "fred", "DTWEXBGS", "first_diff", description="Trade-weighted USD index, daily change"),
    DriverSpec("eurusd_logret","fred", "DEXUSEU",  "log_return", description="EUR/USD log-return"),
    DriverSpec("jpyusd_logret","fred", "DEXJPUS",  "log_return", description="JPY/USD log-return"),
    DriverSpec("gbpusd_logret","fred", "DEXUSUK",  "log_return", description="GBP/USD log-return"),
    DriverSpec("cnyusd_logret","fred", "DEXCHUS",  "log_return", description="CNY/USD log-return"),
    # Commodities
    DriverSpec("wti_logret",    "fred", "DCOILWTICO",   "log_return", description="WTI crude spot, log-return"),
    DriverSpec("brent_logret",  "fred", "DCOILBRENTEU", "log_return", description="Brent crude spot, log-return"),
    DriverSpec("gold_logret",   "yahoo", "GC=F", "log_return", description="COMEX gold futures (front-month), log-return — FRED's LBMA AM/PM fixings were discontinued"),
    DriverSpec("natgas_logret",  "yahoo", "NG=F", "log_return", description="Henry Hub natural gas futures, log-return"),
    DriverSpec("silver_logret",  "yahoo", "SI=F", "log_return", description="Silver futures, log-return"),
    DriverSpec("copper_logret",  "yahoo", "HG=F", "log_return", description="Copper futures, log-return"),
    # Vol
    DriverSpec("vix",   "yahoo", "^VIX",  "level",      description="CBOE VIX (level — already stationary)"),
    DriverSpec("vvix",  "yahoo", "^VVIX", "level",
               available_from=pd.Timestamp("2007-03-15"),
               description="CBOE VVIX (vol-of-vol; from 2007)"),
    # International equity ETFs
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


# FRED backend (free CSV; no API key required)
def fetch_fred_series(series_id: str, use_cache: bool = True) -> pd.Series:
    """Download a FRED series via the fredgraph CSV endpoint.

    Cache is keyed by series ID only; delete the file or pass use_cache=False
    to refresh.
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
    # Date column is 'observation_date', or 'DATE' on older endpoints.
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
    """GET url as text via requests, falling back to system curl (some
    sandboxes stall on Python's TLS stack)."""
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


# Yahoo backend (uses yfinance)
def fetch_yahoo_series(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_cache: bool = True,
) -> pd.Series:
    """Download a Yahoo series' adjusted close."""
    safe_name = ticker.replace("^", "caret_").replace("=", "_")
    cache = CACHE_DIR / f"yahoo_{safe_name}.parquet"
    meta_path = cache.with_suffix(".parquet.meta")
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        series = df.iloc[:, 0]
        # Accept the cache if the data spans [start, end], or if a previous
        # fetch requested a containing span (sidecar .meta). The second route
        # matters for late-inception series (e.g. EEM): their data never
        # reaches a padded start, and re-fetching live each run gives
        # nondeterministic auto-adjusted values that perturb DYNOTEARS.
        cov_ok = series.index.min() <= start and series.index.max() >= end
        req_ok = False
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
                req_ok = (pd.Timestamp(m["req_start"]) <= start
                          and pd.Timestamp(m["req_end"]) >= end)
            except Exception:
                req_ok = False
        if cov_ok or req_ok:
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
    # Atomic writes so parallel first-fetches can't tear the parquet/meta.
    # The meta records the requested span for the cache-acceptance check above.
    tmp = cache.with_suffix(f".parquet.{os.getpid()}.tmp")
    close.to_frame().to_parquet(tmp)
    os.replace(tmp, cache)
    meta_tmp = meta_path.with_suffix(f".meta.{os.getpid()}.tmp")
    meta_tmp.write_text(json.dumps({"req_start": str(pd.Timestamp(start).date()),
                                    "req_end": str(pd.Timestamp(end).date())}))
    os.replace(meta_tmp, meta_path)
    return close.loc[start:end]


# Preprocessing
def _preprocess(series: pd.Series, mode: Preprocessing) -> pd.Series:
    """Stationarising transform per mode: log_return, first_diff, yoy_pct /
    yoy_diff (monthly resample then 12-period change), or level."""
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


# Derived series (BAA-AAA, HYG-LQD)
def _build_derived(name: str, use_cache: bool) -> pd.Series:
    """Compute the spreads referenced by DriverSpec.source='derived'."""
    if name == "BAA-AAA":
        baa = fetch_fred_series("BAA", use_cache=use_cache)
        aaa = fetch_fred_series("AAA", use_cache=use_cache)
        spread = (baa - aaa).dropna()
        spread.name = "BAA-AAA"
        return spread
    if name == "HYG-LQD":
        # Wide range so the cache is reusable across runs.
        wide_start, wide_end = pd.Timestamp("2007-01-01"), pd.Timestamp.now().normalize()
        hyg = fetch_yahoo_series("HYG", wide_start, wide_end, use_cache=use_cache)
        lqd = fetch_yahoo_series("LQD", wide_start, wide_end, use_cache=use_cache)
        ratio = (hyg / lqd).dropna()
        ratio.name = "HYG-LQD"
        return ratio
    raise ValueError(f"unknown derived series: {name}")


# Daily alignment
def _to_daily(series: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill onto the trading-day calendar: the value an observer
    would actually have known on day t."""
    return series.reindex(daily_index, method="ffill")


# Orchestrator
def build_driver_pool(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    daily_index: pd.DatetimeIndex | None = None,
    specs: list[DriverSpec] | None = None,
    use_cache: bool = True,
) -> DriverPool:
    """Fetch and preprocess the full driver pool over [start, end].

    daily_index should be the asset panel's index; if None a weekday union of
    the drivers' own indices is used. specs overrides DRIVER_CATALOGUE (e.g.
    the VIX-excluded robustness check).
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
                # Pad the download so YoY transforms have room.
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
        all_idx = pd.DatetimeIndex(sorted({d for s in processed.values() for d in s.index}))
        daily_index = all_idx[all_idx.weekday < 5]
    daily_index = pd.DatetimeIndex(daily_index)
    daily_index = daily_index[(daily_index >= start_ts) & (daily_index <= end_ts)]

    # Forward-fill onto the daily calendar (what an observer knew at day t).
    columns: dict[str, pd.Series] = {}
    for name, series in processed.items():
        spec = next(s for s in specs if s.name == name)
        gated = series
        if spec.available_from is not None:
            gated = series.loc[series.index >= spec.available_from]
        daily = _to_daily(gated, daily_index)
        if spec.available_from is not None:
            # NaN before availability so downstream can drop or substitute.
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
