# Causal discovery for portfolio optimisation

MSc thesis code. It fits causal graphs (DYNOTEARS, VARLiNGAM, ridge-Granger)
over S&P 100 daily returns plus macro drivers, turns each graph into portfolio
weights, and backtests monthly rebalances from 2007 to 2024. The report is in
`final_report/main.tex`.

Everything in the report is reproduced by `reproduction.py`. It runs the
launchers in `scripts/` as subprocesses from the repository root, in order,
and stops at the first failure. Backtest cells whose
`results/<tag>/closed_loop.pkl` bundle already exists are skipped, so the run
is idempotent and can be restarted after a failure. Child output goes to
`results/reproduction_logs/<name>.log`.

## 1. Set up

Python 3.13 is required. The vendored `causalnex/`, `lingam/` and
`nts-notears/` trees are pruned copies (only the modules the pipeline
imports, plus their licences) and are imported directly, so do not
pip-install them.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r pipeline/requirements.txt
```

## 2. Get the data

Everything reads from `cache/`, which is not tracked in git. The first run
fills it. Asset prices come from CRSP through WRDS and need a WRDS line in
`~/.pgpass` (see `pipeline/data/wrds_backend.py`); without it the code falls
back to Yahoo Finance, which lacks delisted tickers and adjusts prices
differently, so it will not reproduce the reported numbers. Macro drivers
(FRED, Yahoo) and the S&P 500 membership history download automatically and
need no key. The asset universe is frozen in `scripts/phase_i_universe.txt`
(99 tickers); do not edit it, because every result depends on the same column
set.

Once `cache/` is populated nothing makes a network call, and graph fits are
served from `cache/discovery/` keyed on the exact window bytes. A backtest
cell then takes about a minute instead of hours.

## 3. Run it

```bash
.venv/bin/python reproduction.py
```

A cold run fits every discovery graph from scratch and takes hours per
(method, window) pair. Useful flags:

```bash
python reproduction.py --list                 # the steps
python reproduction.py --only phase2 --dry-run
python reproduction.py --from collate         # start part-way through
python reproduction.py --skip checks
python reproduction.py --force                # re-run cells that already have a bundle
python reproduction.py --k 17                 # override the calibrated K (the report uses 17)
```

The steps, in order:

| Step | What it does |
| --- | --- |
| `preflight` | Python version, vendored trees, WRDS credential |
| `tests` | the unit-test suite (88 tests, under a minute) |
| `phase1` | Phase I variants V0, V0', V1, V2 at 252 and 504 days; V1 calibrates K first |
| `phase1_sweeps` | J4a K sweep, J4b alpha/gamma grid, `collate_j4` |
| `phase2` | the graph-to-allocator ablation grid (DYNOTEARS, VARLiNGAM, Granger) |
| `phase2_sweeps` | sparsity-threshold and transaction-cost sweeps |
| `gates` | cache-hit gate, DAG diagnostics, and the D0/V0' replication gate |
| `checks` | directional prior, FFNN seed audit, NTS-NOTEARS probe |
| `collate` | Phase II matrix and contrasts, regimes, robust statistics |
| `figures` | the Phase II figures |
| `oos` | the 2025-26 out-of-sample slice (Yahoo prices, separate caches) |
| `compile` | `latexmk -pdf main.tex` (skipped if latexmk is absent) |

Outputs land in `results/` (CSVs and `results/figures/`) and
`final_report/_generated/` (LaTeX macros that `main.tex` inputs).

## Notes on reproducibility

- DYNOTEARS and VARLiNGAM fits are deterministic. The same window produces a
  bit-identical graph across processes and machines.
- The Phase I V1/V2 path trains a small PyTorch network for sensitivities. On
  Apple Silicon (MPS) this is not bit-reproducible even at a fixed seed. The
  seed audit in the `checks` step quantifies the spread.
- The `results/*/closed_loop.pkl` bundles are gitignored because of size. Only
  the CSVs, figures and generated macros are tracked.

### The distance-projection correction (2026-09-05)

Until 2026-09-05 every HRP and HERC allocator nearest-PSD projected its
clustering distance matrix before linkage (inherited from the Phase I code).
A Euclidean distance matrix has one positive eigenvalue, so the projection
collapsed the distance to rank one and single linkage degenerated to a chain.
The projection is now off by default and kept behind
`psd_project_distance=True` in `pipeline/portfolio/directed.py` and
`causal_hsp.py` so the original Phase I bundle can still be replayed. All
affected cells, batteries and figures were re-run. The report's Appendix
records what changed.
