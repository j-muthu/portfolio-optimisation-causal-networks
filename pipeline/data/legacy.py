"""Shared data pipeline for the S&P 500 causal-discovery experiments.

Both DYNOTEARS (Plan A) and VARLiNGAM (Plan B) consume *identical* input: a
DataFrame of daily log-returns, rows = trading days, columns = assets.  This
module builds that DataFrame once.

Pipeline stages
---------------
1. Resolve the asset universe (see ``UniverseApproach``).
2. Download daily auto-adjusted close prices via ``yfinance``.
3. Handle missing data (drop sparse assets, forward-fill small gaps, align).
4. Compute log-returns ``log(P_t / P_{t-1})``.
5. Stationarity check: ADF test per asset, drop non-stationary series.
6. Standardise to zero mean / unit variance.

The headline entry point is :func:`build_dataset`, which returns a
:class:`Dataset` bundling the returns matrix with provenance metadata.

Universe approaches (see the plan's "Handling S&P 500 Constituent Changes")
---------------------------------------------------------------------------
* ``"fixed"`` (default) -- today's S&P 500 constituents, full history.  Simple,
  comparable adjacency matrices across windows, but carries survivorship bias.
* ``"intersection"`` -- only tickers that were in the index for the *entire*
  study period.  Reduces (does not eliminate) survivorship bias; still a fixed
  ``d``.  Intended for the thesis robustness-check section.

Results are cached under ``thesis/cache/`` so re-runs are cheap.
"""

from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT

logger = logging.getLogger(__name__)

UniverseApproach = Literal["fixed", "intersection"]

CACHE_DIR = THESIS_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HTTP_HEADERS = {"User-Agent": "thesis-causal-discovery/1.0 (academic research)"}


# ============================================================================
# Result container
# ============================================================================
@dataclass
class Dataset:
    """Output of :func:`build_dataset`.

    Attributes
    ----------
    returns:
        The model-ready matrix -- standardised log-returns, shape ``(n, d)``.
        ``n`` = trading days, ``d`` = assets.  Index is a sequential
        ``RangeIndex`` (required by DYNOTEARS); ``dates`` holds the real
        calendar dates aligned row-for-row.
    dates:
        ``DatetimeIndex`` of length ``n`` -- the trading day of each row.
    prices:
        Cleaned adjusted-close prices the returns were derived from.
    sectors:
        Mapping ``ticker -> GICS sector`` (empty if the universe was supplied
        explicitly and sector lookup was skipped).
    adf_pvalues:
        ADF-test p-value per *retained* asset.
    dropped:
        Mapping ``reason -> [tickers]`` for every asset removed during cleaning.
    meta:
        Free-form provenance (universe approach, date range, parameters).
    """

    returns: pd.DataFrame
    dates: pd.DatetimeIndex
    prices: pd.DataFrame
    sectors: dict[str, str] = field(default_factory=dict)
    adf_pvalues: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    dropped: dict[str, list[str]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of time-series observations (rows)."""
        return self.returns.shape[0]

    @property
    def d(self) -> int:
        """Number of variables / assets (columns)."""
        return self.returns.shape[1]

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(n={self.n}, d={self.d}, "
            f"approach={self.meta.get('approach')!r}, "
            f"range={self.meta.get('start')}..{self.meta.get('end')})"
        )


# ============================================================================
# Ticker symbol hygiene
# ============================================================================
def normalise_ticker(symbol: str) -> str:
    """Convert a Wikipedia/exchange symbol to the form ``yfinance`` expects.

    Yahoo uses ``-`` where many listings use ``.`` for share classes
    (``BRK.B`` -> ``BRK-B``, ``BF.B`` -> ``BF-B``).
    """
    return symbol.strip().upper().replace(".", "-")


# ============================================================================
# S&P 500 constituents (Approach 1: fixed universe)
# ============================================================================
def _fetch_wikipedia_tables() -> list[pd.DataFrame]:
    """Fetch and parse the tables on the Wikipedia S&P 500 page.

    Table 0 = current constituents, table 1 = historical changes.  We fetch the
    HTML ourselves (with a polite User-Agent) and hand it to ``read_html`` so
    behaviour does not depend on pandas' optional URL support.
    """
    import requests

    resp = requests.get(WIKI_SP500_URL, headers=_HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_current_constituents(use_cache: bool = True) -> pd.DataFrame:
    """Return today's S&P 500 constituents (Approach 1).

    Returns
    -------
    DataFrame indexed by normalised ticker, with at least a ``sector`` column.
    """
    cache = CACHE_DIR / "sp500_constituents.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    tables = _fetch_wikipedia_tables()
    raw = tables[0]
    # Column names on the page: "Symbol", "Security", "GICS Sector", ...
    df = pd.DataFrame(
        {
            "ticker": raw["Symbol"].map(normalise_ticker),
            "security": raw["Security"].astype(str),
            "sector": raw["GICS Sector"].astype(str),
        }
    ).set_index("ticker")
    df = df[~df.index.duplicated(keep="first")]
    df.to_parquet(cache)
    logger.info("Fetched %d current S&P 500 constituents", len(df))
    return df


# ============================================================================
# S&P 500 historical changes (Approach 3: intersection universe)
# ============================================================================
def get_constituent_changes(use_cache: bool = True) -> pd.DataFrame:
    """Return the S&P 500 add/remove history from Wikipedia (table 1).

    Returns
    -------
    DataFrame with columns ``date`` (datetime), ``added`` (ticker or ""),
    ``removed`` (ticker or ""), sorted oldest-first.  Rows with an unparseable
    date are dropped -- the Wikipedia table is known to have gaps.
    """
    cache = CACHE_DIR / "sp500_changes.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    tables = _fetch_wikipedia_tables()
    raw = tables[1].copy()
    # The changes table has a 2-level header: ("Date",""), ("Added","Ticker"),
    # ("Added","Security"), ("Removed","Ticker"), ... -- flatten it.
    raw.columns = ["_".join(str(x) for x in col).strip("_") for col in raw.columns]

    def _col(*candidates: str) -> pd.Series:
        for name in raw.columns:
            low = name.lower()
            if all(c in low for c in candidates):
                return raw[name]
        return pd.Series([""] * len(raw), index=raw.index)

    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(_col("date"), errors="coerce"),
            "added": _col("added", "ticker").fillna("").astype(str).map(_clean_symbol),
            "removed": _col("removed", "ticker").fillna("").astype(str).map(_clean_symbol),
        }
    )
    changes = changes.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    changes.to_parquet(cache)
    logger.info("Fetched %d S&P 500 constituent changes", len(changes))
    return changes


def _clean_symbol(raw: str) -> str:
    """Best-effort ticker extraction from a free-text Wikipedia cell."""
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return ""
    # Cells are occasionally "AAPL[1]" or "AAPL Apple Inc." -- take the first token.
    token = text.split()[0].split("[")[0]
    return normalise_ticker(token)


def membership_at(date: str | pd.Timestamp, use_cache: bool = True) -> set[str]:
    """Reconstruct S&P 500 membership as of ``date`` by replaying changes.

    Starts from today's constituents and *undoes* every change that happened
    after ``date``: a ticker added after ``date`` is removed from the set, a
    ticker removed after ``date`` is added back.
    """
    date = pd.Timestamp(date)
    members = set(get_current_constituents(use_cache).index)
    changes = get_constituent_changes(use_cache)
    future = changes[changes["date"] > date].sort_values("date", ascending=False)
    for _, row in future.iterrows():
        if row["added"]:
            members.discard(row["added"])  # was not a member before it was added
        if row["removed"]:
            members.add(row["removed"])  # was a member before it was removed
    return members


def intersection_universe(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    use_cache: bool = True,
) -> list[str]:
    """Tickers in the S&P 500 for the *entire* ``[start, end]`` window (Approach 3).

    A ticker qualifies if it was a member at ``start`` and was not removed by
    any change inside ``(start, end]``.  This is an approximation -- it ignores
    ticker renames and remove-then-readd cases -- but matches the plan's
    "practical compromise" and needs no paid data source.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    at_start = membership_at(start, use_cache)
    changes = get_constituent_changes(use_cache)
    window = changes[(changes["date"] > start) & (changes["date"] <= end)]
    removed_during = set(window["removed"]) - {""}
    universe = sorted(at_start - removed_during)
    logger.info(
        "Intersection universe %s..%s: %d tickers (%d members at start, "
        "%d removed during window)",
        start.date(), end.date(), len(universe), len(at_start), len(removed_during),
    )
    return universe


# ============================================================================
# Price download
# ============================================================================
def download_prices(
    tickers: Sequence[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    use_cache: bool = True,
    cache_key: str | None = None,
) -> pd.DataFrame:
    """Download daily auto-adjusted close prices via ``yfinance``.

    Parameters
    ----------
    tickers:
        Symbols to download (already normalised).
    cache_key:
        If given, results are cached at ``cache/prices_<cache_key>.parquet``.

    Returns
    -------
    DataFrame indexed by trading date, one column per *successfully downloaded*
    ticker.  Tickers that yfinance could not return (delisted, bad symbol) are
    logged and omitted -- the plan calls for try/except around delisted names.
    """
    import yfinance as yf

    tickers = [normalise_ticker(t) for t in tickers]
    if cache_key:
        cache = CACHE_DIR / f"prices_{cache_key}.parquet"
        if use_cache and cache.exists():
            logger.info("Loaded cached prices: %s", cache.name)
            return pd.read_parquet(cache)

    logger.info("Downloading %d tickers %s..%s", len(tickers), start, end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )

    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data for the requested tickers")

    # With multiple tickers the columns are a (field, ticker) MultiIndex.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:  # single ticker -> flat columns
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Drop tickers yfinance could not resolve at all (all-NaN columns).
    all_nan = [c for c in prices.columns if prices[c].isna().all()]
    if all_nan:
        logger.warning("No data for %d tickers: %s", len(all_nan), sorted(all_nan))
        prices = prices.drop(columns=all_nan)

    if cache_key:
        prices.to_parquet(CACHE_DIR / f"prices_{cache_key}.parquet")
    return prices


# ============================================================================
# Cleaning stages
# ============================================================================
def handle_missing(
    prices: pd.DataFrame,
    max_missing: float = 0.05,
    ffill_limit: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop sparse assets, forward-fill small gaps, align to common dates.

    Returns the cleaned price frame and the list of dropped tickers.
    """
    missing_frac = prices.isna().mean()
    too_sparse = missing_frac[missing_frac > max_missing].index.tolist()
    clean = prices.drop(columns=too_sparse)
    if too_sparse:
        logger.info(
            "Dropped %d assets with >%.0f%% missing days", len(too_sparse), 100 * max_missing
        )
    # Forward-fill short gaps (e.g. a single untraded day); leaves leading NaNs.
    clean = clean.ffill(limit=ffill_limit)
    # Align every asset to a common set of trading days.
    clean = clean.dropna(axis=0, how="any")
    return clean, too_sparse


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log-returns ``log(P_t / P_{t-1})``; first (NaN) row dropped."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def adf_pvalues(returns: pd.DataFrame) -> pd.Series:
    """Augmented Dickey-Fuller p-value per asset (low p => stationary)."""
    from statsmodels.tsa.stattools import adfuller

    pvals = {}
    for col in returns.columns:
        series = returns[col].dropna()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pvals[col] = adfuller(series, autolag="AIC")[1]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ADF test failed for %s: %s", col, exc)
            pvals[col] = np.nan
    return pd.Series(pvals, name="adf_pvalue")


def filter_stationary(
    returns: pd.DataFrame, alpha: float = 0.01
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Drop assets whose returns fail the ADF stationarity test.

    An asset is kept if ADF p-value ``< alpha`` (reject the unit-root null).
    Log-returns of equities are almost always stationary, so this typically
    drops nothing -- it is a guard, as the plan requires.
    """
    pvals = adf_pvalues(returns)
    non_stationary = pvals[(pvals >= alpha) | pvals.isna()].index.tolist()
    kept = returns.drop(columns=non_stationary)
    if non_stationary:
        logger.info("Dropped %d non-stationary assets (ADF p >= %.3f)", len(non_stationary), alpha)
    return kept, pvals, non_stationary


def standardise(returns: pd.DataFrame) -> pd.DataFrame:
    """Zero mean, unit variance per column."""
    return (returns - returns.mean()) / returns.std(ddof=0)


# ============================================================================
# Orchestrator
# ============================================================================
def build_dataset(
    start: str = "2014-01-01",
    end: str = "2024-12-31",
    approach: UniverseApproach = "fixed",
    tickers: Iterable[str] | None = None,
    standardise_returns: bool = True,
    adf_alpha: float = 0.01,
    max_missing: float = 0.05,
    max_assets: int | None = None,
    use_cache: bool = True,
) -> Dataset:
    """Build the model-ready log-return dataset shared by both methods.

    Parameters
    ----------
    start, end:
        Study period (``YYYY-MM-DD``).  The plan's default is ~10 years.
    approach:
        ``"fixed"`` (Approach 1, default) or ``"intersection"`` (Approach 3).
        Ignored when ``tickers`` is supplied explicitly.
    tickers:
        Explicit universe.  Bypasses Wikipedia entirely -- handy for the
        small-subset smoke test.
    max_assets:
        If set, keep only the first ``max_assets`` tickers of the resolved
        universe.  Intended for the plan's scaling test (d=100, 200, 500).
    standardise_returns:
        Standardise to zero mean / unit variance (stage 6).  Set ``False`` to
        keep raw log-returns (e.g. for variance-sensitive diagnostics).
    adf_alpha:
        Significance level for the ADF stationarity filter.
    max_missing:
        Drop an asset if more than this fraction of days are missing.

    Returns
    -------
    Dataset
    """
    # --- Stage 1: universe ---------------------------------------------------
    sectors: dict[str, str] = {}
    if tickers is not None:
        universe = [normalise_ticker(t) for t in tickers]
        approach_label = "explicit"
    elif approach == "fixed":
        constituents = get_current_constituents(use_cache)
        universe = list(constituents.index)
        sectors = constituents["sector"].to_dict()
        approach_label = "fixed"
    elif approach == "intersection":
        universe = intersection_universe(start, end, use_cache)
        constituents = get_current_constituents(use_cache)
        sectors = {t: constituents["sector"].get(t, "Unknown") for t in universe}
        approach_label = "intersection"
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"unknown approach: {approach!r}")

    if not universe:
        raise ValueError("resolved an empty asset universe")

    if max_assets is not None and len(universe) > max_assets:
        logger.info("Capping universe at %d of %d tickers", max_assets, len(universe))
        universe = universe[:max_assets]

    # --- Stage 2: prices -----------------------------------------------------
    cache_key = f"{approach_label}_{start}_{end}_{len(universe)}"
    prices = download_prices(universe, start, end, use_cache, cache_key=cache_key)

    # --- Stage 3: missing data ----------------------------------------------
    prices, dropped_sparse = handle_missing(prices, max_missing=max_missing)

    # --- Stage 4: log-returns -----------------------------------------------
    returns = compute_log_returns(prices)

    # --- Stage 5: stationarity ----------------------------------------------
    returns, pvals, dropped_nonstat = filter_stationary(returns, alpha=adf_alpha)
    prices = prices[returns.columns]

    # --- Stage 6: standardise ------------------------------------------------
    model_returns = standardise(returns) if standardise_returns else returns.copy()

    # DYNOTEARS requires a sequential integer index; keep the real dates aside.
    dates = pd.DatetimeIndex(model_returns.index)
    model_returns = model_returns.reset_index(drop=True)

    dropped = {
        "missing_data": dropped_sparse,
        "non_stationary": dropped_nonstat,
    }
    meta = {
        "approach": approach_label,
        "start": start,
        "end": end,
        "universe_size": len(universe),
        "standardised": standardise_returns,
        "adf_alpha": adf_alpha,
        "max_missing": max_missing,
    }
    ds = Dataset(
        returns=model_returns,
        dates=dates,
        prices=prices,
        sectors={t: sectors.get(t, "Unknown") for t in model_returns.columns},
        adf_pvalues=pvals[model_returns.columns],
        dropped=dropped,
        meta=meta,
    )
    logger.info("Built %r", ds)
    return ds


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demo = build_dataset(
        start="2021-01-01",
        end="2023-01-01",
        tickers=["AAPL", "MSFT", "AMZN", "GOOGL", "JPM"],
    )
    print(demo)
    print(demo.returns.head())
