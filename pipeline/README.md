# Causal Discovery Pipeline for S&P 500

Implementation of the two causal-discovery plans (DYNOTEARS and VARLiNGAM) over
a shared S&P 500 data pipeline. See `../Causal Discovery Pipelines for S&P 500.md`
for the design rationale.

## Setup

The pipeline runs in a Python 3.13 virtual environment (`thesis/.venv`). The
vendored `causalnex` and `lingam` libraries pin much older stacks, so they are
**not** pip-installed — `_vendored.py` imports their source trees directly and
applies the compatibility shims needed for modern pandas.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r pipeline/requirements.txt
```

Run everything from the thesis root so `pipeline` is importable as a package.

## Smoke test

End-to-end check on ~20 large-cap tickers — covers the verification plans from
both Plan A and Plan B:

```bash
.venv/bin/python -m pipeline.smoke_test
```

## Modules

| Module | Role |
|--------|------|
| `_vendored.py` | Import bridge to the vendored `causalnex`/`lingam` source trees + pandas compat shims. Not called directly. |
| `data.py` | Shared data pipeline — universe resolution, yfinance download, log-returns, ADF stationarity check, standardisation. Entry point: `build_dataset`. |
| `rolling_dynotears.py` | Plan A — rolling-window DYNOTEARS. Entry point: `run_rolling_dynotears`. |
| `rolling_varlingam.py` | Plan B — rolling-window VARLiNGAM. Entry point: `run_rolling_varlingam`. |
| `graph_analysis.py` | Step 2 — graph metrics, regime detection, sector flow, causal-order drift, head-to-head comparison. |
| `portfolio.py` | Step 3 (later phase) — Hierarchical Risk Parity from causal matrices (v1/v2/v3 variants). |

## Quick start

```python
from pipeline import build_dataset, run_rolling_dynotears, run_rolling_varlingam
from pipeline import analyse_rolling, detect_regime_changes

# Approach 1 (fixed universe) — the recommended starting point.
ds = build_dataset(start="2014-01-01", end="2024-12-31", approach="fixed")

# Approach 3 (intersection universe) — for the survivorship-bias robustness check.
ds_robust = build_dataset(start="2014-01-01", end="2024-12-31", approach="intersection")

dyn = run_rolling_dynotears(ds, window=504, step=21, p=1, n_jobs=-1)
var = run_rolling_varlingam(ds, window=504, step=21, lags=1, criterion="bic")

metrics = analyse_rolling(dyn)
regimes = detect_regime_changes(metrics, n_sigma=2.0)
```

## Conventions

- **Matrix direction**: every adjacency matrix this package exposes follows the
  `i -> j` convention — `M[i, j]` is the causal effect of asset `i` on asset
  `j`. VARLiNGAM's raw `j -> i` output is transposed on extraction so DYNOTEARS
  and VARLiNGAM results are directly comparable.
- **DYNOTEARS acyclicity**: the contemporaneous `W` is post-processed by
  `enforce_dag` so it is a genuine DAG (the continuous constraint only reaches
  `h(W) <= h_tol`). Lagged `A` matrices may contain cycles and are left as-is.
- **Caching**: downloaded prices and Wikipedia tables are cached under
  `thesis/cache/`. Delete that directory to force a refresh.

## Scaling to d=500

- DYNOTEARS: pass `n_jobs=-1` to `run_rolling_dynotears` for per-window
  parallelism; tune `lambda_w`/`lambda_a` (or pass `lambda_grid` for per-window
  cross-validation).
- VARLiNGAM: the OLS VAR is underdetermined at `d=500`. Pass
  `var_method="ridge"` to `run_rolling_varlingam` to use the ridge-regularised
  Stage-1 VAR (`estimate_var_coefs`).
