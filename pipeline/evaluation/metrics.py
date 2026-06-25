"""Portfolio performance metrics (annualised Sharpe, Sortino, Calmar, CER,
drawdown, turnover, concentration).

Inputs are always ``pd.Series`` of period returns (daily by convention). The
``periods_per_year`` default of 252 matches the NYSE trading-day count.

The Certainty-Equivalent Return (CER) at risk aversion ``γ_RA`` follows the
Howard et al. convention used in the methodology chapter:

    CER = mean - 0.5 * γ_RA * var

(per-period; annualise by multiplying by ``periods_per_year``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Risk-adjusted return measures
# ============================================================================
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


# ============================================================================
# Distribution- and multiplicity-aware Sharpe (Bailey & Lopez de Prado)
# ============================================================================
# The plain Sharpe ratio assumes IID-normal returns and ignores how many
# strategy configurations were tried before the reported one was selected.
# Daily equity returns are heavily non-normal (excess kurtosis ~16-17 on this
# universe), and this study evaluates ~41 configurations, so both corrections
# matter. These helpers operate on the *raw per-period* return series (not the
# annualised Sharpe): the Probabilistic and Deflated Sharpe ratios are unitless
# probabilities and only require that every Sharpe entering them is expressed in
# the same per-period units. See ``scripts/robust_stats.py`` for the report
# battery built on top of these.

_EULER_MASCHERONI = 0.5772156649015329


def _per_period_sharpe_moments(returns: pd.Series) -> tuple[float, float, float, int]:
    """Return ``(sr_hat, skew, kurt, T)`` for the per-period (non-annualised)
    Sharpe and the higher moments used by the PSR/DSR formulae.

    ``kurt`` is the *non-excess* kurtosis (3.0 for a normal distribution), which
    is the convention in Bailey & Lopez de Prado.
    """
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
    """Probabilistic Sharpe Ratio: ``P(true per-period SR > sr_benchmark)``.

    Bailey & Lopez de Prado (2012): with the estimated per-period Sharpe
    ``SR_hat``, skewness ``g3``, non-excess kurtosis ``g4`` and ``T`` samples,

        PSR = Phi( (SR_hat - SR*) * sqrt(T-1)
                   / sqrt(1 - g3*SR_hat + ((g4-1)/4)*SR_hat^2) ).

    ``sr_benchmark`` is the threshold Sharpe (per-period), e.g. 0 for "better
    than random" or another strategy's per-period Sharpe for a head-to-head.
    Returns a probability in [0, 1].
    """
    from scipy.stats import norm

    sr_hat, skew, kurt, T = _per_period_sharpe_moments(returns)
    if T < 3:
        return 0.0
    denom_sq = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    denom = np.sqrt(max(denom_sq, 1e-12))
    z = (sr_hat - sr_benchmark) * np.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum of ``n_trials`` IID Sharpe estimates under the null of
    zero true skill — the Deflated Sharpe benchmark ``SR*`` of Bailey & Lopez de
    Prado (2014):

        SR* = sqrt(V) * [ (1 - gamma) * Z^-1(1 - 1/N)
                          +     gamma  * Z^-1(1 - 1/(N*e)) ],

    where ``V`` is the cross-trial variance of the Sharpe estimates, ``N`` the
    number of trials, ``gamma`` the Euler-Mascheroni constant and ``Z^-1`` the
    standard-normal quantile. ``sr_variance`` must be in the same (per-period)
    units as the Sharpes passed to :func:`deflated_sharpe_ratio`.
    """
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
    """Deflated Sharpe Ratio: PSR evaluated against the multiplicity-adjusted
    benchmark ``SR*`` from :func:`expected_max_sharpe`.

    ``all_trial_sharpes`` is the collection of *per-period* Sharpe estimates of
    every configuration tried (including ``returns``'s own). The DSR is the
    probability that the strategy's true Sharpe exceeds the expected best Sharpe
    obtainable from that many trials under the null of no skill — i.e. PSR after
    deflating for selection bias. As ``N -> 1`` (a single trial) ``SR* -> 0`` and
    the DSR collapses to ``PSR(SR* = 0)``.
    """
    sharpes = np.asarray([s for s in all_trial_sharpes if np.isfinite(s)], dtype=float)
    n = len(sharpes)
    sr_star = expected_max_sharpe(float(np.var(sharpes, ddof=1)) if n > 1 else 0.0, n)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr_star)


# ============================================================================
# Drawdown
# ============================================================================
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
    # Run-length encoding of the under-water indicator.
    runs = (under != under.shift()).cumsum()[under == 1]
    return int(runs.value_counts().max())


# ============================================================================
# Return / volatility
# ============================================================================
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


# ============================================================================
# Concentration
# ============================================================================
def herfindahl_index(weights: pd.Series) -> float:
    """``Σ w_i^2``; 1/N for an equal-weighted portfolio."""
    w = weights.fillna(0.0).to_numpy()
    return float((w ** 2).sum())


def effective_n(weights: pd.Series) -> float:
    """``1 / HHI`` — number of "effective" positions in the portfolio."""
    hhi = herfindahl_index(weights)
    return float("inf") if hhi < 1e-12 else float(1.0 / hhi)


def max_weight(weights: pd.Series) -> float:
    return float(weights.fillna(0.0).max())


# ============================================================================
# Turnover
# ============================================================================
def one_way_annualised_turnover(
    rebalance_weights: list[pd.Series], rebalances_per_year: int = 12
) -> float:
    """``mean(0.5 * Σ|w[t] - w[t-1]|) * rebalances_per_year``."""
    if len(rebalance_weights) < 2:
        return 0.0
    deltas = []
    for prev, cur in zip(rebalance_weights[:-1], rebalance_weights[1:]):
        union = sorted(set(prev.index) | set(cur.index))
        p = prev.reindex(union).fillna(0.0)
        c = cur.reindex(union).fillna(0.0)
        deltas.append(float(0.5 * (c - p).abs().sum()))
    return float(np.mean(deltas) * rebalances_per_year)


# ============================================================================
# Certainty-Equivalent Return
# ============================================================================
def certainty_equivalent_return(
    returns: pd.Series, gamma_ra: float = 3.0, periods_per_year: int = 252
) -> float:
    """``CER = mean - 0.5·γ_RA·var``; annualised."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    mu = r.mean()
    var = r.var(ddof=0)
    cer_per_period = mu - 0.5 * gamma_ra * var
    return float(cer_per_period * periods_per_year)


# ============================================================================
# One-shot summary
# ============================================================================
def performance_summary(
    returns: pd.Series,
    weights_history: list[pd.Series] | None = None,
    rebalances_per_year: int = 12,
    periods_per_year: int = 252,
    gamma_ras: tuple[float, ...] = (1.0, 3.0, 5.0),
) -> dict:
    """Compute every metric in one call. Returns a flat dict for easy DataFrame conversion."""
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
