# Causal discovery for portfolio optimisation

MSc thesis code. It fits causal graphs (DYNOTEARS, VARLiNGAM, ridge-Granger)
over S&P 100 daily returns plus macro drivers, turns each graph into portfolio
weights, and backtests monthly rebalances from 2007 to 2024. The report is in
`final_report/main.tex`.

This file tells you how to reproduce the numbers, tables and figures in the
report from scratch. Everything below is run from the repository root.

## 1. Set up

Python 3.13 is required. The vendored `causalnex/`, `lingam/` and
`nts-notears/` trees are imported directly, so do not pip-install them.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r pipeline/requirements.txt
```

Check the install:

```bash
.venv/bin/python -m pytest tests -q
```

All 88 tests should pass in under a minute.

## 2. Get the data

Every script reads from `cache/`, which is not tracked in git. The first run
of any backtest fills it. You need:

- **Asset prices.** CRSP daily prices via WRDS are the primary source. Put a
  WRDS line in `~/.pgpass` (see `pipeline/data/wrds_backend.py`). If WRDS is
  unavailable the code falls back to Yahoo Finance, which gives slightly
  different prices and therefore slightly different Sharpe ratios.
- **Macro drivers.** Downloaded automatically from FRED and Yahoo. No key
  needed.
- **S&P 500 membership history.** Downloaded automatically.

The asset universe is frozen in `scripts/phase_i_universe.txt` (99 tickers).
Do not edit it; every result depends on the same column set.

Once `cache/` is populated no script makes a network call, and graph fits are
served from `cache/discovery/` keyed on the exact window bytes. A full
backtest then takes about a minute instead of hours.

## 3. Phase I: the HSP driver-selection variants

Run these in order. The first V1 run calibrates K and prints `chosen K=N`;
pass that K to the others. The report uses K=17.

```bash
python -m scripts.run_phase_i --variant V1 --window 252
python -m scripts.run_phase_i --variant V0 --window 252 --k 17
python -m scripts.run_phase_i --variant V2 --window 252 --k 17
python -m scripts.run_phase_i --variant V0prime --window 252 --k 17
python -m scripts.run_phase_i --variant V1 --window 252 --discovery-method varlingam
python -m scripts.run_phase_i --variant V2 --window 252 --discovery-method varlingam
```

Repeat with `--window 504`. Each run writes
`results/phase_i_<variant>_w<window>/closed_loop.pkl`.

For the K sensitivity sweep (Appendix, J4a) add `--k K --tag-suffix _kK` for
K in 10, 14, 17, 20, 25. For the alpha/gamma sweep (J4b) run V2 with
`--alpha A --gamma G --tag-suffix _aA_gG` over the 3x3 grid
A in {0.4, 0.6, 0.8}, G in {0.1, 0.3, 0.5}. Then:

```bash
python -m scripts.collate_j4
```

## 4. Phase II: the graph-to-allocator ablation

One command per cell. Methods are `dynotears`, `varlingam`, `granger`.
Allocators are `D0 D0s D1 D2 D2s D3 D4` (the graph family), the controls
`D0lw D0df D0pc`, the graph-blind anchors `CORR EW IVP`, and the HERC cells
`HERCC HERC0 HERC1`. Windows are 189, 252, 378, 504.

```bash
python -m scripts.run_phase_ii --method dynotears --allocator D1 --window 252
```

The full grid used in the report is every DYNOTEARS and VARLiNGAM allocator
at all four windows, plus Granger at window 252 only. A bash loop covers it:

```bash
for m in dynotears varlingam; do
  for a in D0 D0s D1 D2 D2s D3 D4; do
    for w in 189 252 378 504; do
      python -m scripts.run_phase_ii --method $m --allocator $a --window $w
    done
  done
done
for a in D0lw D0df D0pc CORR EW IVP HERCC HERC0 HERC1; do
  for w in 189 252 378 504; do
    python -m scripts.run_phase_ii --method dynotears --allocator $a --window $w
  done
done
for a in D0 D1 D2 D3; do
  python -m scripts.run_phase_ii --method granger --allocator $a --window 252
done
```

Sanity check: the DYNOTEARS D0 w252 cell must reproduce
`phase_i_v0prime_w252` to within 1e-3. Compare the two `closed_loop.pkl` NAVs by hand.

Two small sweeps feed the appendix:

```bash
# sparsity threshold (D0, D2, D3 at w252)
python -m scripts.run_phase_ii --method dynotears --allocator D2 --window 252 --tau 0.05 --tag-suffix _tau0.05
# transaction cost (D0, D1, D2s at w252 and w504)
python -m scripts.run_phase_ii --method dynotears --allocator D1 --window 252 --transaction-cost-bps 10 --tag-suffix _cost10
```

Use tau in {0.01, 0.05, 0.1} and costs in {0, 10, 20} bps.

## 5. Build the tables, statistics and figures

Run in this order. Each script reads the `results/*/closed_loop.pkl` bundles
and writes CSVs to `results/`, LaTeX macros to `final_report/_generated/`,
and PNGs to `results/figures/`.

```bash
python -m scripts.collate_phase_ii        # results/phase_ii_matrix.csv, phase_ii_contrasts.csv
python -m scripts.regime_analysis         # results/regime_analysis/*.csv
python -m scripts.robust_stats            # Phase I PSR/DSR/SPA/MCS + _generated/robust_stats.tex
python -m scripts.robust_stats --phase-ii # same battery over the 146-trial universe
python -m scripts.plot_thesis_figures     # Phase I figures
python -m scripts.plot_phase_ii_figures   # Phase II figures
```

The bootstrap contrasts use 10,000 resamples and take a few minutes.

## 6. Optional checks reported in the appendix

```bash
python -m scripts.extract_asset_graphs       # cache-hit gate + DAG diagnostics
python -m scripts.verify_directional_prior   # J1 directional prior check
python -m scripts.run_seed_audit --n-seeds 20  # E7 FFNN seed audit (slow, needs torch)
python -m scripts.probe_nts_notears          # J5 NTS-NOTEARS probe
```

## 7. Out-of-sample 2025-26 slice

This uses Yahoo Finance prices only (CRSP ends 2024-12-31) and keeps its
own caches under `cache/prices_oos` and `cache/drivers_oos`.

```bash
for a in CORR D0 D0s D1 D2 D2s D0df; do
  for w in 189 252 378 504; do
    python -m scripts.run_oos_slice --allocator $a --window $w
  done
done
python -m scripts.collate_oos   # results/oos_slice.csv + _generated/oos_stats.tex
```

## 8. Compile the report

```bash
cd final_report && latexmk -pdf main.tex
```

`main.tex` inputs the generated macro files from step 5 and step 7 and the
figures under `results/figures/`.

## Notes on reproducibility

- DYNOTEARS and VARLiNGAM fits are deterministic. The same window produces
  a bit-identical graph across processes and machines.
- The Phase I V1/V2 path trains a small PyTorch network for sensitivities.
  On Apple Silicon (MPS) this is not bit-reproducible even at a fixed seed.
  The seed audit in step 6 quantifies the spread.
- Results depend on the price source. The report numbers come from CRSP.
  A Yahoo-only rebuild will differ in the third decimal place of Sharpe.
- The `results/*/closed_loop.pkl` bundles are gitignored because of size.
  Only the CSVs, figures and generated macros are tracked.
