"""Portfolio performance metrics: Sharpe, Sortino, Calmar, CER, drawdown,
turnover, concentration. Inputs are pd.Series of period returns (daily,
252 periods/year by default). CER = mean - 0.5 * gamma_RA * var, annualised.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Risk-adjusted return measures
def annualised_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    sigma = r.std(ddof=0)
    if sigma < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / sigma)


def annualised_sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    downside = r[r < 0]
    if downside.empty:
        return float("inf")
    dd_sigma = np.sqrt((downside ** 2).mean())
    if dd_sigma < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / dd_sigma)


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    cagr = annualised_return(r, periods_per_year)
    mdd = abs(max_drawdown(r))
    if mdd < 1e-12:
        return float("inf")
    return float(cagr / mdd)


# Distribution- and multiplicity-aware Sharpe (Bailey & Lopez de Prado).
# These operate on per-period (non-annualised) Sharpes; PSR/DSR are unitless
# probabilities, so all Sharpes entering them must share per-period units.

_EULER_MASCHERONI = 0.5772156649015329


def _per_period_sharpe_moments(returns: pd.Series) -> tuple[float, float, float, int]:
    """(sr_hat, skew, kurt, T) for the per-period Sharpe. kurt is non-excess
    (3.0 for normal), the Bailey & Lopez de Prado convention."""
    from scipy import stats

    r = returns.dropna().to_numpy(dtype=float)
    T = len(r)
    if T < 3:
        return 0.0, 0.0, 3.0, T
    sigma = r.std(ddof=0)
    if sigma < 1e-12:
        return 0.0, 0.0, 3.0, T
    sr_hat = float(r.mean() / sigma)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # non-excess
    return sr_hat, skew, kurt, T


def probabilistic_sharpe_ratio(
    returns: pd.Series, sr_benchmark: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): probability
    the true per-period SR exceeds sr_benchmark, adjusting for skew and
    kurtosis. sr_benchmark must be per-period."""
    from scipy.stats import norm

    sr_hat, skew, kurt, T = _per_period_sharpe_moments(returns)
    if T < 3:
        return 0.0
    denom_sq = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    denom = np.sqrt(max(denom_sq, 1e-12))
    z = (sr_hat - sr_benchmark) * np.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected max of n_trials IID Sharpe estimates under zero true skill:
    the DSR benchmark SR* of Bailey & Lopez de Prado (2014). sr_variance is
    the cross-trial variance, in the same per-period units as the Sharpes."""
    from scipy.stats import norm

    if n_trials <= 1 or sr_variance <= 0.0:
        return 0.0
    sqrt_v = np.sqrt(sr_variance)
    g = _EULER_MASCHERONI
    term = (1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials) \
        + g * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sqrt_v * term)


def deflated_sharpe_ratio(
    returns: pd.Series, all_trial_sharpes
) -> float:
    """Deflated Sharpe Ratio: PSR against the multiplicity-adjusted SR*.
    all_trial_sharpes are the per-period Sharpes of every configuration tried
    (including this one); with a single trial the DSR collapses to PSR(0)."""
    sharpes = np.asarray([s for s in all_trial_sharpes if np.isfinite(s)], dtype=float)
    n = len(sharpes)
    sr_star = expected_max_sharpe(float(np.var(sharpes, ddof=1)) if n > 1 else 0.0, n)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr_star)


# Drawdown
def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (a negative number)."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    nav = (1.0 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def time_underwater(returns: pd.Series) -> int:
    """Longest run of consecutive periods below the prior peak."""
    r = returns.dropna()
    if r.empty:
        return 0
    nav = (1.0 + r).cumprod()
    peak = nav.cummax()
    under = (nav < peak).astype(int)
    if under.sum() == 0:
        return 0
    runs = (under != under.shift()).cumsum()[under == 1]
    return int(runs.value_counts().max())


# Return / volatility
def annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    cagr = (1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0
    return float(cagr)


def annualised_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    downside = r[r < 0]
    if downside.empty:
        return 0.0
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))


# Concentration
def herfindahl_index(weights: pd.Series) -> float:
    """Sum of squared weights; 1/N for an equal-weighted portfolio."""
    w = weights.fillna(0.0).to_numpy()
    return float((w ** 2).sum())


def effective_n(weights: pd.Series) -> float:
    """1 / HHI: number of effective positions."""
    hhi = herfindahl_index(weights)
    return float("inf") if hhi < 1e-12 else float(1.0 / hhi)


def max_weight(weights: pd.Series) -> float:
    return float(weights.fillna(0.0).max())


# Turnover
def one_way_annualised_turnover(
    rebalance_weights: list[pd.Series], rebalances_per_year: int = 12
) -> float:
    """mean(0.5 * sum|w[t] - w[t-1]|) * rebalances_per_year."""
    if len(rebalance_weights) < 2:
        return 0.0
    deltas = []
    for prev, cur in zip(rebalance_weights[:-1], rebalance_weights[1:]):
        union = sorted(set(prev.index) | set(cur.index))
        p = prev.reindex(union).fillna(0.0)
        c = cur.reindex(union).fillna(0.0)
        deltas.append(float(0.5 * (c - p).abs().sum()))
    return float(np.mean(deltas) * rebalances_per_year)


# Certainty-Equivalent Return
def certainty_equivalent_return(
    returns: pd.Series, gamma_ra: float = 3.0, periods_per_year: int = 252
) -> float:
    """CER = mean - 0.5 * gamma_RA * var, annualised."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    mu = r.mean()
    var = r.var(ddof=0)
    cer_per_period = mu - 0.5 * gamma_ra * var
    return float(cer_per_period * periods_per_year)


# One-shot summary
def performance_summary(
    returns: pd.Series,
    weights_history: list[pd.Series] | None = None,
    rebalances_per_year: int = 12,
    periods_per_year: int = 252,
    gamma_ras: tuple[float, ...] = (1.0, 3.0, 5.0),
) -> dict:
    """Every metric in one call, as a flat dict."""
    out: dict = {
        "annualised_return": annualised_return(returns, periods_per_year),
        "annualised_volatility": annualised_volatility(returns, periods_per_year),
        "downside_deviation": downside_deviation(returns, periods_per_year),
        "annualised_sharpe": annualised_sharpe(returns, periods_per_year),
        "annualised_sortino": annualised_sortino(returns, periods_per_year),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "time_underwater": time_underwater(returns),
    }
    for g in gamma_ras:
        out[f"cer_gamma{g}"] = certainty_equivalent_return(returns, g, periods_per_year)
    if weights_history is not None and weights_history:
        out["turnover_one_way_annualised"] = one_way_annualised_turnover(
            weights_history, rebalances_per_year
        )
        out["herfindahl_avg"] = float(np.mean([herfindahl_index(w) for w in weights_history]))
        out["effective_n_avg"] = float(np.mean([effective_n(w) for w in weights_history]))
        out["max_weight_avg"] = float(np.mean([max_weight(w) for w in weights_history]))
    return out


__all__ = [
    "annualised_sharpe",
    "annualised_sortino",
    "calmar_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "max_drawdown",
    "time_underwater",
    "annualised_return",
    "annualised_volatility",
    "downside_deviation",
    "herfindahl_index",
    "effective_n",
    "max_weight",
    "one_way_annualised_turnover",
    "certainty_equivalent_return",
    "performance_summary",
]
