"""Legacy asset-only data pipeline: builds the standardised log-return matrix
shared by DYNOTEARS and VARLiNGAM.

Universe approaches: "fixed" (today's constituents; survivorship bias) or
"intersection" (members for the whole period; reduces but does not eliminate
it). Cached under thesis/cache/.
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


# Result container
@dataclass
class Dataset:
    """Output of build_dataset. returns has a sequential RangeIndex (DYNOTEARS
    requires it); dates holds the real trading days row-for-row."""

    returns: pd.DataFrame
    dates: pd.DatetimeIndex
    prices: pd.DataFrame
    sectors: dict[str, str] = field(default_factory=dict)
    adf_pvalues: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    dropped: dict[str, list[str]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of rows (trading days)."""
        return self.returns.shape[0]

    @property
    def d(self) -> int:
        """Number of assets (columns)."""
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


# Ticker symbol hygiene
def normalise_ticker(symbol: str) -> str:
    """Yahoo form: '-' instead of '.' for share classes (BRK.B -> BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


# S&P 500 constituents (fixed universe)
def _fetch_wikipedia_tables() -> list[pd.DataFrame]:
    """Wikipedia S&P 500 page tables: 0 = current constituents, 1 = changes."""
    import requests

    resp = requests.get(WIKI_SP500_URL, headers=_HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_current_constituents(use_cache: bool = True) -> pd.DataFrame:
    """Today's S&P 500 constituents, indexed by normalised ticker."""
    cache = CACHE_DIR / "sp500_constituents.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    tables = _fetch_wikipedia_tables()
    raw = tables[0]
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


# S&P 500 historical changes (intersection universe)
def get_constituent_changes(use_cache: bool = True) -> pd.DataFrame:
    """S&P 500 add/remove history from Wikipedia, oldest first. Rows with
    unparseable dates are dropped (the table has gaps)."""
    cache = CACHE_DIR / "sp500_changes.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    tables = _fetch_wikipedia_tables()
    raw = tables[1].copy()
    # Flatten the 2-level header, e.g. ("Added","Ticker") -> "Added_Ticker".
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
    # Cells can be "AAPL[1]" or "AAPL Apple Inc."; take the first token.
    token = text.split()[0].split("[")[0]
    return normalise_ticker(token)


def membership_at(date: str | pd.Timestamp, use_cache: bool = True) -> set[str]:
    """Membership as of date: start from today's set and undo every later
    change."""
    date = pd.Timestamp(date)
    members = set(get_current_constituents(use_cache).index)
    changes = get_constituent_changes(use_cache)
    future = changes[changes["date"] > date].sort_values("date", ascending=False)
    for _, row in future.iterrows():
        if row["added"]:
            members.discard(row["added"])
        if row["removed"]:
            members.add(row["removed"])
    return members


def intersection_universe(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    use_cache: bool = True,
) -> list[str]:
    """Tickers in the index for the entire [start, end] window. Approximate:
    ignores renames and remove-then-readd cases."""
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


# Price download
def download_prices(
    tickers: Sequence[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    use_cache: bool = True,
    cache_key: str | None = None,
) -> pd.DataFrame:
    """Download daily auto-adjusted close prices via yfinance. Tickers that
    fail (delisted, bad symbol) are logged and omitted."""
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

    # Multiple tickers give a (field, ticker) MultiIndex.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Drop all-NaN columns (unresolvable tickers).
    all_nan = [c for c in prices.columns if prices[c].isna().all()]
    if all_nan:
        logger.warning("No data for %d tickers: %s", len(all_nan), sorted(all_nan))
        prices = prices.drop(columns=all_nan)

    if cache_key:
        prices.to_parquet(CACHE_DIR / f"prices_{cache_key}.parquet")
    return prices


# Cleaning stages
def handle_missing(
    prices: pd.DataFrame,
    max_missing: float = 0.05,
    ffill_limit: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop sparse assets, forward-fill small gaps, align to common dates."""
    missing_frac = prices.isna().mean()
    too_sparse = missing_frac[missing_frac > max_missing].index.tolist()
    clean = prices.drop(columns=too_sparse)
    if too_sparse:
        logger.info(
            "Dropped %d assets with >%.0f%% missing days", len(too_sparse), 100 * max_missing
        )
    # Forward-fill short gaps; leaves leading NaNs.
    clean = clean.ffill(limit=ffill_limit)
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
    """Drop assets failing the ADF test (kept if p < alpha). Equity log-returns
    are almost always stationary, so this is a guard that rarely fires."""
    pvals = adf_pvalues(returns)
    non_stationary = pvals[(pvals >= alpha) | pvals.isna()].index.tolist()
    kept = returns.drop(columns=non_stationary)
    if non_stationary:
        logger.info("Dropped %d non-stationary assets (ADF p >= %.3f)", len(non_stationary), alpha)
    return kept, pvals, non_stationary


def standardise(returns: pd.DataFrame) -> pd.DataFrame:
    """Zero mean, unit variance per column."""
    return (returns - returns.mean()) / returns.std(ddof=0)


# Orchestrator
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
    """Build the model-ready log-return Dataset.

    An explicit tickers list bypasses Wikipedia and the approach setting.
    max_assets truncates the resolved universe (scaling tests).
    """
    # Universe
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

    # Prices
    cache_key = f"{approach_label}_{start}_{end}_{len(universe)}"
    prices = download_prices(universe, start, end, use_cache, cache_key=cache_key)

    # Missing data
    prices, dropped_sparse = handle_missing(prices, max_missing=max_missing)

    # Log-returns
    returns = compute_log_returns(prices)

    # Stationarity
    returns, pvals, dropped_nonstat = filter_stationary(returns, alpha=adf_alpha)
    prices = prices[returns.columns]

    # Standardise
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
