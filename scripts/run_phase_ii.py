"""Run one (method, allocator, window, tau) cell of the Phase-II ablation.

Everything upstream of the allocator is the identical Phase-I protocol, so
the discovery cache hits the Phase-I graph fits. All data pre-cached; no
WRDS calls.

Usage
-----
    # Replication gate (must reproduce phase_i_v0prime_w252 to <=1e-3):
    python -m scripts.run_phase_ii --method dynotears --allocator D0 --window 252

    # A direction-aware cell:
    python -m scripts.run_phase_ii --method dynotears --allocator D2 --window 252

    # E3 sparsity sweep:
    python -m scripts.run_phase_ii --method dynotears --allocator D2 \\
        --window 252 --tau 0.05
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from pipeline.data.drivers import DRIVER_CATALOGUE
from pipeline.portfolio.directed import ALLOCATORS
from pipeline.shakedown import run_shakedown

# Same exclusions as Phase I, mandatory for the discovery-cache keys to match.
DROP_DRIVERS = {"hyg_lqd_logret", "vvix"}

_TRACKED_UNIVERSE = pathlib.Path(__file__).resolve().parent / "phase_i_universe.txt"
_CACHE_UNIVERSE = pathlib.Path(__file__).resolve().parent.parent / "cache" / "phase_i_universe.txt"
UNIVERSE_FILE = _TRACKED_UNIVERSE if _TRACKED_UNIVERSE.exists() else _CACHE_UNIVERSE

DATA_START = "2005-01-03"
BACKTEST_START = "2007-01-03"
DATA_END = "2024-12-31"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Phase II D-variant backtest launcher.")
    p.add_argument("--method", required=True,
                   choices=["dynotears", "varlingam", "granger"])
    p.add_argument("--allocator", required=True, choices=list(ALLOCATORS))
    p.add_argument("--window", type=int, default=252, help="lookback (252 or 504)")
    p.add_argument("--tau", type=float, default=0.0,
                   help="magnitude threshold on the asset–asset block (E3 sweep)")
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument("--granger-lambda", type=float, default=1e-2,
                   help="ridge penalty for --method granger")
    p.add_argument(
        "--tag-suffix", default="",
        help="appended to the result-dir tag (e.g. '_tau0.05', '_cost10')",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("phase_ii")

    universe = UNIVERSE_FILE.read_text().strip().split(",")
    driver_specs = [s for s in DRIVER_CATALOGUE if s.name not in DROP_DRIVERS]

    # Discovery kwargs must match Phase I exactly for the cache keys to hit.
    # GRANGER windows are new fits, density-matched to DYNOTEARS.
    if args.method == "varlingam":
        discovery_kwargs: dict | None = {"prune": False}
    elif args.method == "granger":
        discovery_kwargs = {
            "lambda_ridge": args.granger_lambda,
            "density_match_dynotears": True,
        }
    else:
        discovery_kwargs = None

    # CORR, EW/IVP and HERCC are graph-blind, so their tags are method-independent.
    if args.allocator == "CORR":
        tag = f"phase_ii_corr_hrp_w{args.window}"
    elif args.allocator in ("EW", "IVP"):
        tag = f"phase_ii_{args.allocator.lower()}_w{args.window}"
    elif args.allocator == "HERCC":
        tag = f"phase_ii_herc_corr_w{args.window}"
    else:
        tag = f"phase_ii_{args.method}_{args.allocator}_w{args.window}"
    if args.tau > 0.0:
        tag += f"_tau{args.tau:g}"
    if args.tag_suffix:
        tag += args.tag_suffix

    log.info(
        "Phase II %s × %s | window=%d | tau=%g | %d assets | %d drivers | %s..%s",
        args.method, args.allocator, args.window, args.tau, len(universe),
        len(driver_specs), BACKTEST_START, DATA_END,
    )

    res = run_shakedown(
        start=DATA_START,
        end=DATA_END,
        backtest_start=BACKTEST_START,
        universe_override=universe,
        driver_specs=driver_specs,
        K_default=10,                 # unused on the asset_only path
        use_k_calibration=False,      # no drivers selected, nothing to calibrate
        window_size=args.window,
        lookback_days=args.window,
        holding_days=21,
        rebalance_step_days=21,
        transaction_cost_bps=args.transaction_cost_bps,
        selection_method="asset_only",
        discovery_method=args.method,
        allocator=args.allocator,
        graph_tau=args.tau,
        discovery_kwargs=discovery_kwargs,
        tag=tag,
        use_cache=True,
        discovery_cache=True,         # Phase II is cache-fed by design
    )

    print("\n" + "=" * 70)
    print(f"Phase II {args.method} × {args.allocator} (w{args.window}, τ={args.tau:g}) complete: {tag}")
    try:
        bt = res.closed_loop.backtest
        print(f"final NAV (gross/net): {bt.nav_gross.iloc[-1]:.4f} / {bt.nav_net.iloc[-1]:.4f}")
    except Exception as exc:  # cosmetic print must not abort a run
        log.warning("Could not print final NAV (%s); results are persisted regardless.", exc)
    print("=" * 70)


if __name__ == "__main__":
    main()
