"""Content-keyed disk cache for the per-window causal-graph fit.

The DYNOTEARS / VARLiNGAM fit is the dominant per-rebalance cost (~4 min/window
at d≈135, L-BFGS-B bound). Crucially it depends ONLY on the window data and the
discovery hyper-parameters — **not** on K, α, or γ, which act downstream in
selection / feedback. So a K-sensitivity or α/γ sweep refits the *identical*
graph dozens of times. This cache stores the fitted window object keyed by the
exact fit inputs, so the first run at a given (method, window, params) pays the
fit and every later sweep config reuses it.

Because the key is the full window content + params (not the rebalance date or
tag), V0′/V1/V2 under the same discovery method at the same window share one
cached graph automatically.

Safety mirrors the hardened FFNN cache (``pipeline/sensitivities/ffnn.py``):

* **tolerant read** — a torn/corrupt file (e.g. two concurrent runs writing the
  same key) never crashes the backtest; we fall through and recompute.
* **atomic write** — serialise to ``{key}.{pid}.tmp`` then ``os.replace`` into
  place (atomic on POSIX), so a concurrent reader sees old-or-complete, never a
  torn file. This makes cache-warm parallel sweep runs safe.

Opt-in only: callers pass ``use_cache=True``. With ``use_cache=False`` the
helper is a straight passthrough to ``compute_fn`` and touches no disk, so
existing reproductions and the test suite are bit-for-bit unchanged.
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
    """Stable 24-char hex key over the *exact* discovery fit inputs.

    Hashes the full window content (not a prefix) so two distinct windows can
    never collide, and includes the column ordering, the method, and the
    discovery hyper-parameters so a changed ``lambda_w`` / ``prune`` / etc.
    yields a different key (no false hit).
    """
    h = hashlib.sha256()
    h.update(method.encode())
    h.update(b"\x00drivers\x00")
    h.update("|".join(driver_columns).encode())
    h.update(b"\x00assets\x00")
    h.update("|".join(asset_columns).encode())
    # Full content fingerprint — float64 values + the date index — so any data
    # change invalidates the entry. ~540 KB/window → sub-ms to hash.
    values = np.ascontiguousarray(joint_window.to_numpy(dtype=np.float64))
    h.update(values.tobytes())
    idx = joint_window.index
    if isinstance(idx, pd.DatetimeIndex):
        h.update(idx.asi8.tobytes())
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

    ``compute_fn`` is a zero-arg thunk that runs the actual
    ``run_dynotears_joint_window`` / ``run_varlingam_joint_window`` fit. The
    returned object is a ``JointDynotearsWindow`` / ``JointVarLingamWindow``
    dataclass of numpy arrays — picklable as-is.

    With ``use_cache=False`` this is a pure passthrough (no disk touched).
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
        except Exception as exc:  # torn/corrupt → recompute
            logger.warning(
                "discovery cache read failed (%s: %s) — recomputing",
                cache_path.name, exc,
            )

    obj = compute_fn()

    # Atomic write: unique temp file then os.replace (atomic on POSIX).
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        logger.debug("discovery cache write (%s): %s", method, cache_path.name)
    except Exception as exc:  # best-effort persistence — never fail the fit
        logger.warning("discovery cache write failed (%s): %s", cache_path.name, exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

    return obj


__all__ = ["discovery_cache_key", "load_or_compute_discovery", "CACHE_DIR"]
