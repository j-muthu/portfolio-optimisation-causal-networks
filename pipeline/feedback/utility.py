"""Sensitivity-weighted credit attribution and EMA driver-utility update.

influence_d = sum_i |w_i * s_{i,d}|; credit_d partitions R[t] over selected
drivers in proportion to influence; the EMA update touches selected drivers
only. Lookahead discipline lives in feedback.storage, not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Credit attribution
@dataclass
class CreditAttribution:
    """Per-driver credit for one holding period; credits sum to the reward."""

    rebalance_date: pd.Timestamp
    holding_end: pd.Timestamp
    reward: float
    influences: pd.Series  # raw |w*s| sums per selected driver
    credits: pd.Series      # normalised to sum to reward


def sensitivity_weighted_credit(
    weights: pd.Series,
    sensitivities: pd.DataFrame,
    reward: float,
    rebalance_date: pd.Timestamp,
    holding_end: pd.Timestamp,
) -> CreditAttribution:
    """Distribute reward over selected drivers by sensitivity-weighted
    influence.

    sensitivities is (assets x selected drivers); assets missing from weights
    get weight 0. reward is the realised holding-period reward R[t].
    """
    assets = list(sensitivities.index)
    w = weights.reindex(assets).fillna(0.0).to_numpy()
    S = sensitivities.to_numpy()
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


# EMA update
def ema_update(
    prior: pd.Series,
    credit: pd.Series,
    gamma: float,
    selected: Sequence[str] | None = None,
) -> pd.Series:
    """EMA update of driver utility: selected drivers get
    gamma * credit + (1 - gamma) * prior; unselected carry the prior through.
    selected defaults to credit's index.
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
