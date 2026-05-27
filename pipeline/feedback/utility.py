"""Sensitivity-weighted credit attribution + EMA driver utility update.

Per ``Closed-Loop Causal-HSP Portfolio.md`` §Feedback loop mechanics:

* **Credit attribution**: drivers that more strongly shaped this period's
  portfolio get more credit (positive or negative).

      influence_d = Σ_i |w_i · s_{i,d}|
      credit_d    = R[t] · influence_d / Σ_{d'∈S} influence_{d'}

* **EMA update** (decay ``γ``):

      U_d[t] = γ · credit_d[t]  +  (1 - γ) · U_d[t-1]    for d ∈ selected
      U_d[t] =                       U_d[t-1]            for d ∉ selected

* **Lookahead discipline**: the storage layer (``feedback.storage``) keys the
  utility table by **holding-period-end** date, not rebalance date — see
  module docstring there. This module just computes the credit / update math.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Credit attribution
# ============================================================================
@dataclass
class CreditAttribution:
    """Per-driver credit for one holding period.

    ``Σ_d credit[d] == R`` by construction (the reward is partitioned over
    selected drivers).
    """

    rebalance_date: pd.Timestamp
    holding_end: pd.Timestamp
    reward: float
    influences: pd.Series  # raw |w·s| sums per selected driver
    credits: pd.Series      # normalised: credits sum to ``reward``


def sensitivity_weighted_credit(
    weights: pd.Series,
    sensitivities: pd.DataFrame,
    reward: float,
    rebalance_date: pd.Timestamp,
    holding_end: pd.Timestamp,
) -> CreditAttribution:
    """Distribute ``reward`` over selected drivers by sensitivity-weighted influence.

    Parameters
    ----------
    weights:
        ``pd.Series`` of portfolio weights (sum to 1), indexed by asset name.
    sensitivities:
        ``(N, K)`` DataFrame with assets as rows, *selected* drivers as
        columns. Same asset ordering as ``weights`` (subset OK; missing
        assets get weight 0).
    reward:
        Realised holding-period reward ``R[t]`` (e.g. excess Sharpe vs 1/N).
    rebalance_date, holding_end:
        Timestamps recorded on the attribution.

    Returns
    -------
    :class:`CreditAttribution` with ``credits`` summing to ``reward``.
    Driver-utility update consumes the ``.credits`` series.
    """
    assets = list(sensitivities.index)
    w = weights.reindex(assets).fillna(0.0).to_numpy()
    S = sensitivities.to_numpy()
    # influence_d = Σ_i |w_i · s_{i, d}|
    influences = np.abs(w[:, None] * S).sum(axis=0)
    total = influences.sum()
    if total < 1e-12:
        credits = np.zeros_like(influences)
    else:
        credits = reward * influences / total
    return CreditAttribution(
        rebalance_date=pd.Timestamp(rebalance_date),
        holding_end=pd.Timestamp(holding_end),
        reward=reward,
        influences=pd.Series(influences, index=sensitivities.columns, name="influence"),
        credits=pd.Series(credits, index=sensitivities.columns, name="credit"),
    )


# ============================================================================
# EMA update
# ============================================================================
def ema_update(
    prior: pd.Series,
    credit: pd.Series,
    gamma: float,
    selected: Sequence[str] | None = None,
) -> pd.Series:
    """Driver-level exponential-moving-average update.

    Parameters
    ----------
    prior:
        ``U_d[t-1]`` -- the utility carried from the previous rebalance,
        indexed by driver name. May contain drivers not in ``selected``;
        they carry through unchanged.
    credit:
        ``credit_d[t]`` from :func:`sensitivity_weighted_credit`. Only the
        drivers in ``selected`` contribute to the update.
    gamma:
        EMA decay. ``γ ∈ (0, 1]``. ``γ → 0`` ⇒ utility never moves;
        ``γ → 1`` ⇒ utility = latest credit.
    selected:
        Driver names that were *selected* this rebalance. Drivers not
        selected get their prior carried through verbatim. If ``None``,
        defaults to the index of ``credit``.

    Returns
    -------
    Updated ``U_d[t]`` series with the union of (prior ∪ selected) drivers.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1]; got {gamma}")
    selected = list(selected) if selected is not None else list(credit.index)
    union = sorted(set(prior.index) | set(credit.index))
    prior_full = prior.reindex(union).fillna(0.0)
    credit_full = credit.reindex(union).fillna(0.0)
    selected_mask = pd.Series(False, index=union)
    selected_mask.loc[selected_mask.index.intersection(selected)] = True
    updated = prior_full.where(
        ~selected_mask, gamma * credit_full + (1.0 - gamma) * prior_full
    )
    updated.name = "utility"
    return updated


__all__ = [
    "CreditAttribution",
    "sensitivity_weighted_credit",
    "ema_update",
]
