"""Byte-exact verification of the discovery cache (J4 enabler).

These tests gate the multi-day J4 sweeps: a cache that returned anything other
than the exact graph a fresh fit produces would silently corrupt sweep results.
We therefore assert hit==miss to the array level, that ``use_cache=False`` never
touches disk, and that a changed hyper-parameter yields a different key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.data.alignment import build_joint_matrix
from pipeline.discovery import cache as disc_cache
from pipeline.discovery.cache import (
    discovery_cache_key,
    load_or_compute_discovery,
)
from pipeline.discovery.dynotears import run_dynotears_joint_window


def _small_joint():
    rng = np.random.default_rng(seed=42)
    T, d_drivers, d_assets = 150, 6, 4
    cal = pd.bdate_range("2020-01-02", periods=T)
    drivers = pd.DataFrame(
        rng.standard_normal((T, d_drivers)),
        index=cal, columns=[f"D{i}" for i in range(d_drivers)],
    )
    assets = pd.DataFrame(
        rng.standard_normal((T, d_assets)),
        index=cal, columns=[f"A{i}" for i in range(d_assets)],
    )
    joint = build_joint_matrix(drivers, assets, calendar=cal, drop_na="any")
    return joint.frame, list(joint.driver_columns), list(joint.asset_columns)


def _fit(frame, dcols, acols, **kw):
    return run_dynotears_joint_window(
        frame, dcols, acols, p=1, w_threshold=0.01, **kw
    )


def test_cache_hit_is_byte_exact(tmp_path, monkeypatch):
    """A cached fit must equal a fresh fit to the array level."""
    monkeypatch.setattr(disc_cache, "CACHE_DIR", tmp_path)
    frame, dcols, acols = _small_joint()
    kwargs = {"p": 1, "w_threshold": 0.01}

    fresh = _fit(frame, dcols, acols)  # ground truth, no cache

    # First call: miss → computes + writes.
    miss = load_or_compute_discovery(
        lambda: _fit(frame, dcols, acols),
        joint_window=frame, driver_columns=dcols, asset_columns=acols,
        method="dynotears", discovery_kwargs=kwargs, use_cache=True,
    )
    # Second call: hit → loads from disk. Thunk would raise if ever called.
    hit = load_or_compute_discovery(
        lambda: (_ for _ in ()).throw(AssertionError("cache miss on warm key")),
        joint_window=frame, driver_columns=dcols, asset_columns=acols,
        method="dynotears", discovery_kwargs=kwargs, use_cache=True,
    )

    for got in (miss, hit):
        assert np.array_equal(got.W, fresh.W)
        assert len(got.A) == len(fresh.A)
        for a_got, a_fresh in zip(got.A, fresh.A):
            assert np.array_equal(a_got, a_fresh)
        assert got.fit_loss == fresh.fit_loss
        assert got.converged == fresh.converged
        assert list(got.columns) == list(fresh.columns)

    # Exactly one cache file written.
    assert len(list(tmp_path.glob("*.pkl"))) == 1


def test_use_cache_false_touches_no_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(disc_cache, "CACHE_DIR", tmp_path)
    frame, dcols, acols = _small_joint()
    out = load_or_compute_discovery(
        lambda: _fit(frame, dcols, acols),
        joint_window=frame, driver_columns=dcols, asset_columns=acols,
        method="dynotears", discovery_kwargs={}, use_cache=False,
    )
    assert out is not None
    assert list(tmp_path.glob("*")) == []


def test_key_isolation():
    """Distinct inputs → distinct keys (no false hits)."""
    frame, dcols, acols = _small_joint()
    base = discovery_cache_key(frame, dcols, acols, "dynotears", {"lambda_w": 0.05})

    # Changed hyper-parameter.
    assert base != discovery_cache_key(
        frame, dcols, acols, "dynotears", {"lambda_w": 0.10}
    )
    # Changed method.
    assert base != discovery_cache_key(frame, dcols, acols, "varlingam", {"lambda_w": 0.05})
    # Changed data content.
    perturbed = frame.copy()
    perturbed.iloc[0, 0] += 1e-6
    assert base != discovery_cache_key(
        perturbed, dcols, acols, "dynotears", {"lambda_w": 0.05}
    )
    # Same inputs → identical key (deterministic).
    assert base == discovery_cache_key(
        frame, dcols, acols, "dynotears", {"lambda_w": 0.05}
    )


def test_cache_key_invariant_to_datetime_index_unit():
    """The key must not depend on the index's datetime64 resolution: the
    joint index inherits its unit from whichever cache parquet was written
    most finely, and an environment change once flipped it (us -> ns),
    silently re-keying 1,684 byte-identical fits."""
    import numpy as np
    import pandas as pd

    from pipeline.discovery.cache import discovery_cache_key

    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    df = pd.DataFrame(rng.standard_normal((30, 4)),
                      index=idx, columns=list("abcd"))
    keys = set()
    for unit in ("s", "ms", "us", "ns"):
        d2 = df.copy()
        d2.index = df.index.as_unit(unit)
        keys.add(discovery_cache_key(d2, ["a"], ["b", "c", "d"],
                                     "dynotears", {}))
    assert len(keys) == 1, f"key varies with index unit: {keys}"
