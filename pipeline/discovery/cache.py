"""Content-keyed disk cache for the per-window causal-graph fit.

The fit depends only on the window data and the discovery hyper-parameters.
It does not depend on K, alpha or gamma, so sweeps over those refit the
identical graph. I key the cache on the exact fit inputs so that every later
sweep configuration reuses the first fit. A corrupt file is recomputed, and
writes are atomic (a temporary file followed by os.replace), so concurrent
runs are safe. The cache is used only with ``use_cache=True``. Otherwise the
function is a passthrough that touches no disk.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT

logger = logging.getLogger(__name__)

CACHE_DIR = THESIS_ROOT / "cache" / "discovery"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def discovery_cache_key(
    joint_window: pd.DataFrame,
    driver_columns: Sequence[str],
    asset_columns: Sequence[str],
    method: str,
    discovery_kwargs: dict,
) -> str:
    """Return a stable 24-character hex key over the exact discovery fit
    inputs.

    I hash the full window content plus the column ordering, the method and
    the hyper-parameters, so any change to the data or the parameters gives
    a new key.
    """
    h = hashlib.sha256()
    h.update(method.encode())
    h.update(b"\x00drivers\x00")
    h.update("|".join(driver_columns).encode())
    h.update(b"\x00assets\x00")
    h.update("|".join(asset_columns).encode())
    # Hash the full content, i.e. the float64 values plus the date index.
    values = np.ascontiguousarray(joint_window.to_numpy(dtype=np.float64))
    h.update(values.tobytes())
    idx = joint_window.index
    if isinstance(idx, pd.DatetimeIndex):
        # Normalise to microsecond resolution before hashing. The datetime
        # unit of the index can change between us and ns when the environment
        # changes, which silently re-keys every window on identical data (this
        # happened on 2026-08-15). The committed cache was keyed with
        # microsecond bytes, so "us" keeps every existing key reachable.
        h.update(idx.as_unit("us").asi8.tobytes())
    else:
        h.update("|".join(map(str, idx)).encode())
    h.update(repr(sorted((discovery_kwargs or {}).items())).encode())
    return h.hexdigest()[:24]


def load_or_compute_discovery(
    compute_fn: Callable[[], object],
    *,
    joint_window: pd.DataFrame,
    driver_columns: Sequence[str],
    asset_columns: Sequence[str],
    method: str,
    discovery_kwargs: dict,
    use_cache: bool,
):
    """Return the cached discovery window if there is one. Otherwise compute
    it and cache it.

    ``compute_fn`` is a zero-argument function that runs the actual fit. Its
    result must be picklable. With ``use_cache=False`` this is a passthrough.
    """
    if not use_cache:
        return compute_fn()

    key = discovery_cache_key(
        joint_window, driver_columns, asset_columns, method, discovery_kwargs
    )
    cache_path = CACHE_DIR / f"{key}.pkl"

    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                obj = pickle.load(fh)
            logger.debug("discovery cache hit (%s): %s", method, cache_path.name)
            return obj
        except Exception as exc:  # the file is torn or corrupt, so recompute
            logger.warning(
                "discovery cache read failed (%s: %s) — recomputing",
                cache_path.name, exc,
            )

    obj = compute_fn()

    # Write atomically through a unique temporary file and os.replace.
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        logger.debug("discovery cache write (%s): %s", method, cache_path.name)
    except Exception as exc:  # a failed write must never fail the fit
        logger.warning("discovery cache write failed (%s): %s", cache_path.name, exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

    return obj


__all__ = ["discovery_cache_key", "load_or_compute_discovery", "CACHE_DIR"]
