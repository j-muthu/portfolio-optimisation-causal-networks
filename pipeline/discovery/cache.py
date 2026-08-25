"""Content-keyed disk cache for the per-window causal-graph fit.

The fit depends only on the window data and discovery hyper-parameters, not
on K, alpha or gamma, so sweeps over those refit the identical graph; caching
by the exact fit inputs lets every later sweep config reuse the first fit.
Reads are tolerant (corrupt file -> recompute) and writes are atomic
(tmp file + os.replace), so concurrent runs are safe. Opt-in via
``use_cache=True``; otherwise a pure passthrough that touches no disk.
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
    """Stable 24-char hex key over the exact discovery fit inputs.

    Hashes the full window content plus column ordering, method and
    hyper-parameters, so any change to the data or params yields a new key.
    """
    h = hashlib.sha256()
    h.update(method.encode())
    h.update(b"\x00drivers\x00")
    h.update("|".join(driver_columns).encode())
    h.update(b"\x00assets\x00")
    h.update("|".join(asset_columns).encode())
    # Full content fingerprint: float64 values plus the date index.
    values = np.ascontiguousarray(joint_window.to_numpy(dtype=np.float64))
    h.update(values.tobytes())
    idx = joint_window.index
    if isinstance(idx, pd.DatetimeIndex):
        # Normalise to microsecond resolution before hashing. The index's
        # datetime unit can flip (us <-> ns) with environment changes,
        # silently re-keying every window on byte-identical data (this
        # happened on 2026-08-15). The committed cache was keyed with
        # us-resolution bytes, so "us" keeps every existing key reachable.
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
    """Return a cached discovery window if present, else compute and cache it.

    ``compute_fn`` is a zero-arg thunk running the actual fit; its result must
    be picklable. With ``use_cache=False`` this is a pure passthrough.
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
        except Exception as exc:  # torn/corrupt: recompute
            logger.warning(
                "discovery cache read failed (%s: %s) — recomputing",
                cache_path.name, exc,
            )

    obj = compute_fn()

    # Atomic write: unique temp file then os.replace.
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        logger.debug("discovery cache write (%s): %s", method, cache_path.name)
    except Exception as exc:  # best-effort persistence, never fail the fit
        logger.warning("discovery cache write failed (%s): %s", cache_path.name, exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

    return obj


__all__ = ["discovery_cache_key", "load_or_compute_discovery", "CACHE_DIR"]
