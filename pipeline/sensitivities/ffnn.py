"""Per-window FFNN sensitivity estimation (HSP-style, multi-head batched).

Fits lagged drivers to contemporaneous asset returns with a shared-body,
per-asset-head MLP (architecture search by validation RMSE), then extracts
``S[t] ∈ ℝ^{N × K}`` as the window-averaged ``torch.func.jacrev`` Jacobian.
Per-window results are cached under ``cache/ffnn/``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Defer torch imports so the module can be inspected without torch.
def _torch():
    # source code available at: https://github.com/pytorch/pytorch
    import torch
    return torch


from pipeline._vendored import THESIS_ROOT

CACHE_DIR = THESIS_ROOT / "cache" / "ffnn"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Device selection
def best_device() -> str:
    """``"mps"`` on Apple Silicon, ``"cuda"`` if available, else ``"cpu"``."""
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Model
def _build_mlp(input_dim: int, n_assets: int, depth: int, width: int):
    """Multi-head MLP: shared hidden layers, one linear output head per asset."""
    torch = _torch()
    # source code available at: https://github.com/pytorch/pytorch
    import torch.nn as nn

    layers = []
    in_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.GELU())
        in_dim = width
    body = nn.Sequential(*layers) if layers else nn.Identity()
    head = nn.Linear(in_dim, n_assets)
    return nn.Sequential(body, head)


# Lagged-input builder
def make_lagged_inputs(
    drivers: pd.DataFrame, assets: pd.DataFrame, lags: int
) -> tuple["torch.Tensor", "torch.Tensor", pd.DatetimeIndex]:
    """Stack driver values at ``t-1 .. t-lags`` against asset returns at ``t``.

    Returns ``(X, Y, dates)`` for the rows where all lag values exist.
    """
    torch = _torch()
    selected = list(drivers.columns)
    cols = []
    for k in range(1, lags + 1):
        cols.append(drivers[selected].shift(k))
    X_df = pd.concat(cols, axis=1)
    common = X_df.dropna().index.intersection(assets.index)
    X_np = np.ascontiguousarray(X_df.loc[common].to_numpy(dtype=np.float32))
    Y_np = np.ascontiguousarray(assets.loc[common].to_numpy(dtype=np.float32))
    X = torch.from_numpy(X_np.copy())
    Y = torch.from_numpy(Y_np.copy())
    return X, Y, common


# Per-window fit + jacobian
@dataclass
class SensitivityWindow:
    """FFNN output for one rolling window.

    ``S`` is the ``(N, K)`` window-averaged Jacobian; for ``lags > 1`` it is
    summed across lag blocks before averaging.
    """

    rebalance_date: pd.Timestamp
    selected_drivers: list[str]
    asset_names: list[str]
    S: np.ndarray
    arch: dict
    val_rmse: float
    n_train: int
    n_val: int
    metadata: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.S.shape[0]

    @property
    def K(self) -> int:
        return self.S.shape[1]


# Training helpers
def _train_one_arch(
    X_tr, Y_tr, X_va, Y_va,
    depth: int, width: int,
    epochs: int, lr: float, weight_decay: float,
    device: str, seed: int,
    mask_tr=None, mask_va=None,
):
    """Train one architecture; return ``(model, val_rmse)``.

    Optional masks exclude per-(sample, asset) cells from the loss and val
    RMSE, so pre-inception zero-fills don't bias an asset's head.
    """
    torch = _torch()
    torch.manual_seed(seed)
    model = _build_mlp(X_tr.shape[1], Y_tr.shape[1], depth, width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_tr_d, Y_tr_d = X_tr.to(device), Y_tr.to(device)
    X_va_d, Y_va_d = X_va.to(device), Y_va.to(device)
    if mask_tr is not None:
        mask_tr = mask_tr.to(device)
    if mask_va is not None:
        mask_va = mask_va.to(device)
    best_val = float("inf")
    best_state = None
    patience, since_improvement = 20, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_d)
        sq = (pred - Y_tr_d) ** 2
        if mask_tr is not None:
            denom = mask_tr.sum().clamp(min=1.0)
            loss = (sq * mask_tr).sum() / denom
        else:
            loss = sq.mean()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            sq_va = (model(X_va_d) - Y_va_d) ** 2
            if mask_va is not None:
                denom_va = mask_va.sum().clamp(min=1.0)
                val_loss = float(((sq_va * mask_va).sum() / denom_va).sqrt())
            else:
                val_loss = float(sq_va.mean().sqrt())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def _compute_sensitivities(
    model, X_train, n_assets: int, lags: int, K_drivers: int, device: str,
) -> np.ndarray:
    """Average per-asset Jacobian via ``torch.func.jacrev``, summed across
    lag blocks back to the K-dimensional driver axis.
    """
    torch = _torch()
    # source code available at: https://github.com/pytorch/pytorch
    from torch.func import jacrev

    model = model.to(device).eval()

    def model_apply(x):
        return model(x.unsqueeze(0)).squeeze(0)

    jac_fn = jacrev(model_apply)
    # shape (n_samples, n_assets, K * lags)
    with torch.no_grad():
        jacs = torch.vmap(jac_fn)(X_train.to(device))
    mean_jac = jacs.mean(dim=0).detach().cpu().numpy()  # (n_assets, K * lags)
    S = mean_jac.reshape(n_assets, lags, K_drivers).sum(axis=1)
    return S


# Per-window orchestrator
def fit_sensitivities_window(
    drivers: pd.DataFrame,
    assets: pd.DataFrame,
    selected_drivers: Sequence[str],
    rebalance_date: pd.Timestamp,
    lags: int = 1,
    depths: tuple[int, ...] = (1, 2),
    widths: tuple[int, ...] = (16, 32, 64),
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    val_frac: float = 0.20,
    device: str | None = None,
    seed: int = 0,
    use_cache: bool = True,
    cache_key: str | None = None,
    asset_eligibility: pd.DataFrame | None = None,
) -> SensitivityWindow:
    """Fit one window's multi-head FFNN and extract per-asset sensitivities.

    Panels arrive z-scored from the discovery pipeline. ``asset_eligibility``
    (bool, same index/columns as ``assets``) restricts each asset's head to
    rows with real data; without it the FFNN learns "predict zero" on
    pre-inception days, biasing the Jacobian for late-inception assets.
    Returns a :class:`SensitivityWindow`.
    """
    torch = _torch()
    device = device or best_device()
    drivers = drivers[list(selected_drivers)]
    K = len(selected_drivers)
    N = assets.shape[1]
    asset_names = list(assets.columns)

    X, Y, dates = make_lagged_inputs(drivers, assets, lags)
    n = X.shape[0]
    if n < 50:
        raise ValueError(
            f"too few samples after lag construction ({n}); need ≥ 50 for "
            "the architecture search to be meaningful"
        )
    split = int(n * (1 - val_frac))
    X_tr, Y_tr = X[:split], Y[:split]
    X_va, Y_va = X[split:], Y[split:]

    # Per-(sample, asset) eligibility mask aligned to the lagged-input dates.
    mask_tr = mask_va = None
    elig_signature = b""
    if asset_eligibility is not None:
        aligned = asset_eligibility.reindex(dates).reindex(columns=asset_names)
        if aligned.isna().any().any():
            logger.warning(
                "asset_eligibility had NaN values after reindex (treating as ineligible)"
            )
        mask_np = aligned.fillna(False).astype(np.float32).to_numpy()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_np))
        mask_tr, mask_va = mask_t[:split], mask_t[split:]
        elig_signature = mask_np.tobytes()[:4096]

    if cache_key is None:
        h = hashlib.sha256()
        h.update(rebalance_date.isoformat().encode())
        h.update("|".join(selected_drivers).encode())
        h.update("|".join(asset_names).encode())
        h.update(np.ascontiguousarray(X.numpy()).tobytes()[:4096])
        h.update(np.ascontiguousarray(Y.numpy()).tobytes()[:4096])
        h.update(f"lags={lags} d={depths} w={widths}".encode())
        h.update(elig_signature)
        cache_key = h.hexdigest()[:16]

    cache_path = CACHE_DIR / f"{cache_key}.pt"
    if use_cache and cache_path.exists():
        # Tolerant read: a torn/corrupt cache file must not crash the
        # backtest; fall through and recompute.
        try:
            bundle = torch.load(cache_path, weights_only=False)
            logger.debug("FFNN cache hit: %s", cache_path.name)
            return SensitivityWindow(
                rebalance_date=rebalance_date,
                selected_drivers=list(selected_drivers),
                asset_names=asset_names,
                S=bundle["S"],
                arch=bundle["arch"],
                val_rmse=bundle["val_rmse"],
                n_train=int(X_tr.shape[0]),
                n_val=int(X_va.shape[0]),
                metadata={"cache_hit": True, "lags": lags, "device": device, "cache_key": cache_key},
            )
        except Exception as exc:
            logger.warning("FFNN cache read failed (%s: %s) — recomputing",
                           cache_path.name, exc)

    # Architecture search.
    best_model = None
    best_arch: dict = {}
    best_val = float("inf")
    for depth in depths:
        for width in widths:
            model, val_rmse = _train_one_arch(
                X_tr, Y_tr, X_va, Y_va,
                depth=depth, width=width,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                device=device, seed=seed,
                mask_tr=mask_tr, mask_va=mask_va,
            )
            if val_rmse < best_val:
                best_val = val_rmse
                best_model = model
                best_arch = {"depth": depth, "width": width}

    assert best_model is not None
    S = _compute_sensitivities(best_model, X_tr, N, lags, K, device)
    if use_cache:
        # Atomic write via temp file + os.replace, so a concurrent reader
        # never sees a torn file.
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(
            {"S": S, "arch": best_arch, "val_rmse": best_val,
             "state_dict": best_model.state_dict()},
            tmp_path,
        )
        os.replace(tmp_path, cache_path)

    logger.info(
        "FFNN window %s: K=%d, N=%d, n_train=%d, arch=%s, val_rmse=%.4f",
        rebalance_date.date(), K, N, X_tr.shape[0], best_arch, best_val,
    )
    return SensitivityWindow(
        rebalance_date=rebalance_date,
        selected_drivers=list(selected_drivers),
        asset_names=asset_names,
        S=S,
        arch=best_arch,
        val_rmse=best_val,
        n_train=int(X_tr.shape[0]),
        n_val=int(X_va.shape[0]),
        metadata={"cache_hit": False, "lags": lags, "device": device, "cache_key": cache_key},
    )


__all__ = [
    "best_device",
    "make_lagged_inputs",
    "SensitivityWindow",
    "fit_sensitivities_window",
]
