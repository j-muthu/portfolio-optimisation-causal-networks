"""Full rolling-window causal-discovery experiment driver.

Runs DYNOTEARS and/or VARLiNGAM across the configured universe and date range,
then persists everything needed for downstream analysis so a multi-hour run is
never lost:

* ``<method>.pkl``           -- the full rolling result object (pickled).
* ``<method>_windows.csv``   -- per-window edge counts / parameters.
* ``<method>_metrics.csv``   -- density, avg weight, inter-window distance.
* ``<method>_regimes.csv``   -- windows flagged as regime changes.
* ``<method>_metrics.png``   -- the metrics plotted over time.
* ``head_to_head.csv``       -- DYNOTEARS-vs-VARLiNGAM comparison (if both run).
* ``causal_order_drift.csv`` -- VARLiNGAM causal-order stability (if VARLiNGAM).
* ``run.log``                -- full log of the run.
* ``dataset.pkl``            -- the exact returns matrix used (for reproducibility).
* ``checkpoints/``           -- per-window pickles written as each window finishes.

Results land in ``thesis/results/<tag>/``.

Checkpoint/resume: every completed window is pickled under ``checkpoints/`` as it
finishes, so an interrupted run can be resumed simply by re-running the same
command -- already-computed windows are loaded instead of recomputed. Checkpoints
are keyed by ``--tag`` and window index only: when changing windowing or
algorithm parameters, use a new ``--tag`` (or delete ``checkpoints/``) so stale
windows are not reused.

Examples
--------
Scaling test on ~100 assets (the plan's "find where the VAR step breaks")::

    .venv/bin/python -m pipeline.run_experiment --max-assets 100 --tag sp100

Full fixed-universe run (Approach 1, ~500 assets, 10 years -- a long job)::

    .venv/bin/python -m pipeline.run_experiment --approach fixed --tag full \\
        --n-jobs -1 --var-method ridge

Survivorship-bias robustness check (Approach 3)::

    .venv/bin/python -m pipeline.run_experiment --approach intersection --tag robust
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

from pipeline._vendored import THESIS_ROOT
from pipeline.data import build_dataset
from pipeline.discovery.diagnostics import (
    analyse_rolling,
    causal_order_drift,
    compare_rolling,
    detect_regime_changes,
    plot_metrics,
)
from pipeline.discovery.dynotears import run_rolling_dynotears
from pipeline.discovery.varlingam import run_rolling_varlingam

logger = logging.getLogger("pipeline.run_experiment")

RESULTS_ROOT = THESIS_ROOT / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rolling-window causal-discovery experiment driver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--start", default="2014-01-01", help="study period start")
    p.add_argument("--end", default="2024-12-31", help="study period end")
    p.add_argument(
        "--approach", choices=["fixed", "intersection"], default="fixed",
        help="universe: fixed (Approach 1) or intersection (Approach 3)",
    )
    p.add_argument(
        "--max-assets", type=int, default=None,
        help="cap the universe size (for scaling tests)",
    )
    # Windowing
    p.add_argument("--window", type=int, default=504, help="window length (days)")
    p.add_argument("--step", type=int, default=21, help="window stride (days)")
    # Methods
    p.add_argument(
        "--methods", choices=["dynotears", "varlingam", "both"], default="both",
        help="which method(s) to run",
    )
    # DYNOTEARS
    p.add_argument("--p", type=int, default=1, help="DYNOTEARS lag order")
    p.add_argument("--lambda-w", type=float, default=0.05, help="DYNOTEARS intra L1")
    p.add_argument("--lambda-a", type=float, default=0.05, help="DYNOTEARS inter L1")
    p.add_argument("--w-threshold", type=float, default=0.01, help="edge-weight floor")
    # VARLiNGAM
    p.add_argument("--lags", type=int, default=1, help="VARLiNGAM max VAR lag")
    p.add_argument(
        "--criterion", choices=["aic", "bic", "hqic", "fpe"], default="bic",
        help="VARLiNGAM lag-selection criterion",
    )
    p.add_argument(
        "--var-method", choices=["builtin", "ols", "ridge"], default="builtin",
        help="VARLiNGAM Stage-1 VAR estimator (ridge for large d)",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=0,
        help="VARLiNGAM bootstrap resamples per window (0 = skip)",
    )
    # Run control
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help="parallel windows (-1 = all cores)",
    )
    p.add_argument(
        "--tag", default="run",
        help="results subdirectory name; re-running the same tag resumes from checkpoints",
    )
    p.add_argument("--no-cache", action="store_true", help="ignore cached data")
    return p.parse_args(argv)


def _setup_logging(output_dir: Path) -> None:
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "run.log", mode="w"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _save_rolling(result, name: str, output_dir: Path) -> None:
    """Persist a rolling result: pickle + per-window CSV + metrics + plot."""
    with open(output_dir / f"{name}.pkl", "wb") as fh:
        pickle.dump(result, fh)
    result.to_frame().to_csv(output_dir / f"{name}_windows.csv", index=False)

    metrics = analyse_rolling(result)
    metrics.to_csv(output_dir / f"{name}_metrics.csv")
    regimes = detect_regime_changes(metrics, n_sigma=2.0)
    regimes.to_csv(output_dir / f"{name}_regimes.csv")
    plot_metrics(
        metrics,
        output_dir / f"{name}_metrics.png",
        title=f"{name} rolling causal-graph metrics",
        regime_changes=regimes,
    )
    logger.info(
        "%s: %d windows, %d regime-change candidate(s)", name, len(result), len(regimes)
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = RESULTS_ROOT / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    started = time.time()
    logger.info("Experiment '%s' -> %s", args.tag, output_dir)
    logger.info("Config: %s", vars(args))

    # --- Data ---------------------------------------------------------------
    dataset = build_dataset(
        start=args.start,
        end=args.end,
        approach=args.approach,
        max_assets=args.max_assets,
        use_cache=not args.no_cache,
    )
    logger.info("Dataset: %r", dataset)
    (output_dir / "dataset_meta.txt").write_text(
        f"{dataset!r}\n"
        f"n={dataset.n} d={dataset.d}\n"
        f"meta={dataset.meta}\n"
        f"dropped={dataset.dropped}\n"
    )
    # Snapshot the exact dataset so results stay reproducible even if yfinance
    # later revises historical prices.
    with open(output_dir / "dataset.pkl", "wb") as fh:
        pickle.dump(dataset, fh)

    checkpoint_dir = output_dir / "checkpoints"
    run_dyn = args.methods in ("dynotears", "both")
    run_var = args.methods in ("varlingam", "both")
    dyn = var = None

    # --- DYNOTEARS ----------------------------------------------------------
    if run_dyn:
        logger.info("=== Running rolling DYNOTEARS ===")
        t0 = time.time()
        dyn = run_rolling_dynotears(
            dataset,
            window=args.window,
            step=args.step,
            p=args.p,
            lambda_w=args.lambda_w,
            lambda_a=args.lambda_a,
            w_threshold=args.w_threshold,
            n_jobs=args.n_jobs,
            checkpoint_dir=checkpoint_dir,
        )
        _save_rolling(dyn, "dynotears", output_dir)
        logger.info("DYNOTEARS done in %.1f min", (time.time() - t0) / 60)

    # --- VARLiNGAM ----------------------------------------------------------
    if run_var:
        logger.info("=== Running rolling VARLiNGAM ===")
        t0 = time.time()
        var = run_rolling_varlingam(
            dataset,
            window=args.window,
            step=args.step,
            lags=args.lags,
            criterion=args.criterion,
            var_method=args.var_method,
            n_bootstrap=args.n_bootstrap,
            n_jobs=args.n_jobs,
            checkpoint_dir=checkpoint_dir,
        )
        _save_rolling(var, "varlingam", output_dir)
        drift = causal_order_drift(var)
        drift.to_csv(output_dir / "causal_order_drift.csv")
        logger.info("VARLiNGAM done in %.1f min", (time.time() - t0) / 60)

    # --- Head-to-head -------------------------------------------------------
    if dyn is not None and var is not None:
        logger.info("=== Head-to-head comparison ===")
        comparison = compare_rolling(dyn, var)
        comparison.to_csv(output_dir / "head_to_head.csv")
        logger.info(
            "Mean edge Jaccard (W vs B0): %.3f", comparison["edge_jaccard"].mean()
        )

    logger.info(
        "Experiment '%s' complete in %.1f min -> %s",
        args.tag, (time.time() - started) / 60, output_dir,
    )


if __name__ == "__main__":
    main()
