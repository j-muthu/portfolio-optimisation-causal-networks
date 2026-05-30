"""Phase I — full 2007-2024 thesis backtest launcher.

The headline run: V0 / V1 / V2 over the full sample, fixed 99-ticker universe
(the G.7 universe, for direct G.7↔Phase-I comparability), 33-driver pool
(HYG/LQD + VVIX dropped — they don't exist before 2007-04 and baa10y_diff
already covers the credit-spread role, so dropping them recovers full GFC
capture; joint matrix then starts 2006-01).

All data is pre-cached (cache/prices/ + cache/drivers/), so this makes ZERO
WRDS calls — no Duo prompt. Verified by the offline-build guard in pre-flight.

K calibration is run ONCE (by the V1 launch) and the resulting K is reused by
V0 and V2 via ``--k`` so the 3h calibration isn't paid three times.

Usage
-----
    # 1. V1 first — calibrates K, prints "chosen K=N", then backtests.
    python -m scripts.run_phase_i --variant V1 --window 252

    # 2. Once V1's K-cal lands (~3h in), launch V0 + V2 reusing that K:
    python -m scripts.run_phase_i --variant V0 --window 252 --k N
    python -m scripts.run_phase_i --variant V2 --window 252 --k N

    # 3. Window-504 robustness appendix (same pattern, --window 504).
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from pipeline.data.drivers import DRIVER_CATALOGUE
from pipeline.shakedown import run_shakedown

# Drivers excluded for the full-sample run (see module docstring).
DROP_DRIVERS = {"hyg_lqd_logret", "vvix"}

# 99-ticker universe (the G.7 universe, for G.7↔Phase-I comparability). Tracked
# copy lives next to this script; falls back to the cache/ copy if absent.
_TRACKED_UNIVERSE = pathlib.Path(__file__).resolve().parent / "phase_i_universe.txt"
_CACHE_UNIVERSE = pathlib.Path(__file__).resolve().parent.parent / "cache" / "phase_i_universe.txt"
UNIVERSE_FILE = _TRACKED_UNIVERSE if _TRACKED_UNIVERSE.exists() else _CACHE_UNIVERSE

# Fixed sample boundaries.
DATA_START = "2005-01-03"      # cache coverage; joint naturally starts 2006-01
BACKTEST_START = "2007-01-03"  # first rebalance; full GFC captured at window 252
DATA_END = "2024-12-31"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Phase I full-sample backtest launcher.")
    p.add_argument("--variant", required=True, choices=["V0", "V1", "V2"])
    p.add_argument("--window", type=int, default=252, help="lookback window (252 or 504)")
    p.add_argument(
        "--k", type=int, default=None,
        help="reuse a pre-calibrated K (V0/V2). Omit for V1 to run K calibration.",
    )
    p.add_argument("--k-calibration-B", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.6, help="V2 causal/utility blend")
    p.add_argument("--gamma", type=float, default=0.3, help="V2 utility EMA decay")
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
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

    # Variant → (selection_method, discovery_method, K-calibration on?)
    if args.variant == "V0":
        selection_method, discovery_method = "correlation", "dynotears"
        use_kcal = False  # V0 (cum-corr) never uses K calibration
    elif args.variant == "V1":
        selection_method, discovery_method = "causal_greedy", "dynotears"
        use_kcal = args.k is None  # calibrate unless a K was supplied
    else:  # V2
        selection_method, discovery_method = "causal_greedy", "dynotears"
        use_kcal = args.k is None

    if args.k is not None:
        log.info("Reusing pre-calibrated K=%d (skipping K calibration)", args.k)

    tag = f"phase_i_{args.variant.lower()}_w{args.window}"

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
        tag=tag,
        use_cache=True,
    )

    print("\n" + "=" * 70)
    print(f"Phase I {args.variant} (window {args.window}) complete: {tag}")
    try:
        bt = res.closed_loop.backtest
        print(f"final NAV (gross/net): {bt.nav_gross.iloc[-1]:.4f} / {bt.nav_net.iloc[-1]:.4f}")
    except Exception as exc:  # never let a cosmetic print abort a multi-hour run
        log.warning("Could not print final NAV (%s); results are persisted regardless.", exc)
    print("=" * 70)


if __name__ == "__main__":
    main()
