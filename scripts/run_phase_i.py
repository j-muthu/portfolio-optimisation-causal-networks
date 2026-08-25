"""Phase I full 2007-2024 backtest launcher (V0/V1/V2).

All data is pre-cached, so this makes no WRDS calls. K calibration runs once
(on the V1 launch); V0 and V2 reuse the result via --k.

Usage
-----
    # 1. V1 first: calibrates K, prints "chosen K=N", then backtests.
    python -m scripts.run_phase_i --variant V1 --window 252

    # 2. Once K is known, launch V0 + V2 reusing it:
    python -m scripts.run_phase_i --variant V0 --window 252 --k N
    python -m scripts.run_phase_i --variant V2 --window 252 --k N
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from pipeline.data.drivers import DRIVER_CATALOGUE
from pipeline.shakedown import run_shakedown

# Excluded for the full-sample run: they don't exist before 2007-04, and
# baa10y_diff already covers the credit-spread role.
DROP_DRIVERS = {"hyg_lqd_logret", "vvix"}

# 99-ticker G.7 universe. Tracked copy lives next to this script; falls back
# to the cache/ copy if absent.
_TRACKED_UNIVERSE = pathlib.Path(__file__).resolve().parent / "phase_i_universe.txt"
_CACHE_UNIVERSE = pathlib.Path(__file__).resolve().parent.parent / "cache" / "phase_i_universe.txt"
UNIVERSE_FILE = _TRACKED_UNIVERSE if _TRACKED_UNIVERSE.exists() else _CACHE_UNIVERSE

# Fixed sample boundaries.
DATA_START = "2005-01-03"      # joint matrix naturally starts 2006-01
BACKTEST_START = "2007-01-03"  # full GFC captured at window 252
DATA_END = "2024-12-31"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Phase I full-sample backtest launcher.")
    p.add_argument("--variant", required=True, choices=["V0", "V0prime", "V1", "V2"])
    p.add_argument("--window", type=int, default=252, help="lookback window (252 or 504)")
    p.add_argument(
        "--discovery-method", choices=["dynotears", "varlingam"], default="dynotears",
        help="causal-discovery backend for V1/V2 (ignored for V0 cum-corr). "
             "VARLiNGAM runs calibrate their own K (bootstrap-prob Stage A scores "
             "differ from DYNOTEARS edge magnitudes, so K does not transfer).",
    )
    p.add_argument(
        "--k", type=int, default=None,
        help="reuse a pre-calibrated K (V0/V2). Omit for V1 to run K calibration.",
    )
    p.add_argument("--k-calibration-B", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.6, help="V2 causal/utility blend")
    p.add_argument("--gamma", type=float, default=0.3, help="V2 utility EMA decay")
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument(
        "--tag-suffix", default="",
        help="appended to the result-dir tag (e.g. '_k10', '_a0.4_g0.1') so "
             "J4 sweep runs don't clobber the committed Phase I bundles.",
    )
    p.add_argument(
        "--discovery-cache", action="store_true",
        help="reuse a content-keyed disk cache of the per-window causal-graph "
             "fit (K/alpha/gamma-independent). First run per (method, window) "
             "populates it; later sweep configs reuse it. Off by default so "
             "default reproductions are bit-for-bit unchanged.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("phase_i")

    universe = UNIVERSE_FILE.read_text().strip().split(",")
    driver_specs = [s for s in DRIVER_CATALOGUE if s.name not in DROP_DRIVERS]
    log.info(
        "Phase I %s | window=%d | %d assets | %d drivers (dropped %s) | %s..%s",
        args.variant, args.window, len(universe), len(driver_specs),
        sorted(DROP_DRIVERS), BACKTEST_START, DATA_END,
    )

    # Variant -> (selection_method, discovery_method, K-calibration on?)
    if args.variant == "V0":
        # V0 (cum-corr) never uses discovery or K calibration.
        selection_method, discovery_method = "correlation", "dynotears"
        use_kcal = False
    elif args.variant == "V0prime":
        # V0' asset-only Causal-HRP: no drivers, no FFNN, no K calibration.
        selection_method, discovery_method = "asset_only", "dynotears"
        use_kcal = False
    else:  # V1 / V2: causal greedy on the chosen discovery backend
        selection_method, discovery_method = "causal_greedy", args.discovery_method
        use_kcal = args.k is None

    if args.k is not None:
        log.info("Reusing pre-calibrated K=%d (skipping K calibration)", args.k)

    # VARLiNGAM at d=132 must disable lingam's adaptive-lasso pruning: it
    # fails when a late-order variable has more predecessors than window
    # samples. Not needed anyway; the mask is enforced by post-fit projection.
    discovery_kwargs = {"prune": False} if discovery_method == "varlingam" else None

    # Suffix non-default discovery so VARLiNGAM runs don't clobber the
    # committed DYNOTEARS bundles; DYNOTEARS keeps the unsuffixed tag.
    if args.variant != "V0" and discovery_method != "dynotears":
        tag = f"phase_i_{args.variant.lower()}_{discovery_method}_w{args.window}"
    else:
        tag = f"phase_i_{args.variant.lower()}_w{args.window}"
    # J4 sweep runs write to distinct result dirs via the suffix.
    if args.tag_suffix:
        tag = f"{tag}{args.tag_suffix}"

    res = run_shakedown(
        start=DATA_START,
        end=DATA_END,
        backtest_start=BACKTEST_START,
        universe_override=universe,
        driver_specs=driver_specs,
        K_default=args.k if args.k is not None else 10,
        use_k_calibration=use_kcal,
        k_calibration_B=args.k_calibration_B,
        k_calibration_n_jobs=-1,
        k_calibration_permuted_max_iter=20,
        window_size=args.window,
        lookback_days=args.window,
        holding_days=21,
        rebalance_step_days=21,
        transaction_cost_bps=args.transaction_cost_bps,
        alpha=args.alpha,
        gamma_ema=args.gamma,
        selection_method=selection_method,
        discovery_method=discovery_method,
        discovery_kwargs=discovery_kwargs,
        tag=tag,
        use_cache=True,
        discovery_cache=args.discovery_cache,
    )

    print("\n" + "=" * 70)
    print(f"Phase I {args.variant} (window {args.window}) complete: {tag}")
    try:
        bt = res.closed_loop.backtest
        print(f"final NAV (gross/net): {bt.nav_gross.iloc[-1]:.4f} / {bt.nav_net.iloc[-1]:.4f}")
    except Exception as exc:  # cosmetic print must not abort a multi-hour run
        log.warning("Could not print final NAV (%s); results are persisted regardless.", exc)
    print("=" * 70)


if __name__ == "__main__":
    main()
