"""2025-26 out-of-sample slice (PREDICTIONS_OOS.md).

Runs the Phase-II harness unchanged on 2025-01 -> 2026-08, DYNOTEARS only,
for the decomposition family and the de-factored control. The protocol and
the interpretations of every outcome were committed in
``PREDICTIONS_OOS.md`` before any out-of-sample price was fetched.

Cache isolation (IMPORTANT): the in-sample price/driver caches hold the
CRSP and driver series the whole report regenerates from, and the cache
writer REPLACES a ticker's parquet wholesale. This runner therefore
redirects both caches to ``cache/prices_oos`` / ``cache/drivers_oos``
before anything fetches, and disables the WRDS backend for the slice
(CRSP daily ends 2024-12-31, verified 2026-08-15, so asset prices for
the slice come from Yahoo Finance; the report discloses the source
switch). The discovery cache is content-keyed, so new fits simply join
it under new keys.

Usage:  python -m scripts.run_oos_slice --allocator D1 --window 252
"""
from __future__ import annotations

import argparse
import logging

from pipeline._vendored import THESIS_ROOT
from pipeline.data import assets, drivers

# --- cache isolation + backend pin, BEFORE any fetch ---------------------
assets.PRICES_DIR = THESIS_ROOT / "cache" / "prices_oos"
assets.PRICES_DIR.mkdir(parents=True, exist_ok=True)
assets.fetch_from_wrds = lambda *a, **k: None  # CRSP daily ends 2024-12-31
drivers.CACHE_DIR = THESIS_ROOT / "cache" / "drivers_oos"
drivers.CACHE_DIR.mkdir(parents=True, exist_ok=True)

from pipeline.data.drivers import DRIVER_CATALOGUE          # noqa: E402
from pipeline.portfolio.directed import ALLOCATORS          # noqa: E402
from pipeline.shakedown import run_shakedown                # noqa: E402
from scripts.run_phase_ii import DROP_DRIVERS, UNIVERSE_FILE  # noqa: E402

# Fixed in PREDICTIONS_OOS.md (data start padded so the 504-day lookback
# is fully burned in at the first 2025 rebalance).
DATA_START = "2022-07-01"
BACKTEST_START = "2025-01-02"
DATA_END = "2026-08-14"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="2025-26 OOS slice launcher.")
    p.add_argument("--allocator", required=True, choices=list(ALLOCATORS))
    p.add_argument("--window", type=int, default=252,
                   choices=[189, 252, 378, 504])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("oos_slice")

    universe = UNIVERSE_FILE.read_text().strip().split(",")
    driver_specs = [s for s in DRIVER_CATALOGUE if s.name not in DROP_DRIVERS]

    if args.allocator == "CORR":
        tag = f"oos_corr_hrp_w{args.window}"
    else:
        tag = f"oos_dynotears_{args.allocator}_w{args.window}"

    log.info("OOS slice dynotears x %s | window=%d | %s..%s (backtest from %s)",
             args.allocator, args.window, DATA_START, DATA_END, BACKTEST_START)

    res = run_shakedown(
        start=DATA_START,
        end=DATA_END,
        backtest_start=BACKTEST_START,
        universe_override=universe,
        driver_specs=driver_specs,
        K_default=10,
        use_k_calibration=False,
        window_size=args.window,
        lookback_days=args.window,
        holding_days=21,
        rebalance_step_days=21,
        transaction_cost_bps=5.0,
        selection_method="asset_only",
        discovery_method="dynotears",
        allocator=args.allocator,
        graph_tau=0.0,
        discovery_kwargs=None,
        tag=tag,
        use_cache=True,
        discovery_cache=True,
    )

    print("=" * 70)
    print(f"OOS dynotears x {args.allocator} (w{args.window}) complete: {tag}")
    try:
        bt = res.closed_loop.backtest
        print(f"final NAV (gross/net): {bt.nav_gross.iloc[-1]:.4f} / "
              f"{bt.nav_net.iloc[-1]:.4f} | rebalances: {len(bt.rebalances)}")
    except Exception as exc:  # cosmetic only
        log.warning("Could not print final NAV (%s); results persisted.", exc)
    print("=" * 70)


if __name__ == "__main__":
    main()
