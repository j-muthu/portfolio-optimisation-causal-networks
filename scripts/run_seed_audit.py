"""Phase-II E7 — seed audit of the Phase-I FFNN path.

Ce Guo's critique, quantified: the HSP-style pipeline routes allocation
through a per-window FFNN whose Jacobian is the sensitivity matrix, and the
FFNN fit depends on the torch seed. This script re-runs the committed
V1-DYNOTEARS w252 K=17 configuration (byte-identical everywhere else —
discovery graphs come from the content-keyed cache) across FFNN seeds with
the FFNN cache disabled, so the *only* varying ingredient is the network
initialisation. The output table reports the Sharpe distribution across
seeds and where the committed number sits inside it. Every Phase-II
D-variant is deterministic by construction (no NN anywhere), which is the
contrast the report draws.

Honesty note: on Apple-silicon MPS, torch does not guarantee bit-wise
reproducibility even at a fixed seed; any seed-0 divergence from the
committed bundle is itself framework nondeterminism and is reported as such.

Resumable: a seed whose bundle already exists is skipped, so the multi-hour
run can be interrupted and relaunched freely.

Usage
-----
    python -m scripts.run_seed_audit --n-seeds 10          # the minimum version
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time

import numpy as np
import pandas as pd

from pipeline._vendored import THESIS_ROOT
from pipeline.data.drivers import DRIVER_CATALOGUE
from pipeline.shakedown import run_shakedown
from scripts.run_phase_ii import (
    BACKTEST_START,
    DATA_END,
    DATA_START,
    DROP_DRIVERS,
    UNIVERSE_FILE,
)

RESULTS = THESIS_ROOT / "results"
OUT_CSV = RESULTS / "seed_audit.csv"

# The committed Phase-I V1 w252 configuration (results/phase_i_v1_w252 config).
K = 17
WINDOW = 252
ALPHA, GAMMA, BURN_IN = 0.6, 0.3, 3


def _sharpe(nav: pd.Series) -> float:
    r = nav.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252))


def _metrics(tag: str) -> dict:
    with (RESULTS / tag / "closed_loop.pkl").open("rb") as fh:
        bt = pickle.load(fh)["backtest"]
    nav = bt.nav_net
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = float(nav.iloc[-1] ** (1 / years) - 1)
    dd = float((nav / nav.cummax() - 1).min())
    return {
        "sharpe": _sharpe(nav),
        "cagr": cagr,
        "max_drawdown": dd,
        "nav_net": float(nav.iloc[-1]),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="E7 FFNN seed audit (V1 w252 K=17).")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--start-seed", type=int, default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("seed_audit")

    universe = UNIVERSE_FILE.read_text().strip().split(",")
    driver_specs = [s for s in DRIVER_CATALOGUE if s.name not in DROP_DRIVERS]

    rows = []
    for seed in range(args.start_seed, args.start_seed + args.n_seeds):
        tag = f"phase_ii_seed_audit_v1_w{WINDOW}_s{seed}"
        bundle = RESULTS / tag / "closed_loop.pkl"
        if bundle.exists():
            log.info("seed %d: bundle exists — skipping the run", seed)
        else:
            t0 = time.time()
            log.info("seed %d: launching V1 w%d K=%d (FFNN cache OFF)", seed, WINDOW, K)
            run_shakedown(
                start=DATA_START, end=DATA_END, backtest_start=BACKTEST_START,
                universe_override=universe, driver_specs=driver_specs,
                K_default=K, use_k_calibration=False,
                window_size=WINDOW, lookback_days=WINDOW,
                holding_days=21, rebalance_step_days=21,
                transaction_cost_bps=5.0,
                alpha=ALPHA, gamma_ema=GAMMA, burn_in_rebalances=BURN_IN,
                selection_method="causal_greedy", discovery_method="dynotears",
                sensitivities_kwargs={"seed": seed, "use_cache": False},
                tag=tag, use_cache=True, discovery_cache=True,
            )
            log.info("seed %d done in %.1f min", seed, (time.time() - t0) / 60)

        m = _metrics(tag)
        m["seed"] = seed
        rows.append(m)
        # Persist incrementally so a crash loses nothing.
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        log.info("seed %d: Sharpe %.4f (csv updated)", seed, m["sharpe"])

    df = pd.DataFrame(rows).set_index("seed")
    committed = _metrics("phase_i_v1_w252")["sharpe"]
    q = float((df["sharpe"] < committed).mean())
    print("\n" + "=" * 70)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("-" * 70)
    print(
        f"Sharpe across {len(df)} seeds: min {df.sharpe.min():.4f} | "
        f"p25 {df.sharpe.quantile(.25):.4f} | median {df.sharpe.median():.4f} | "
        f"p75 {df.sharpe.quantile(.75):.4f} | max {df.sharpe.max():.4f}"
    )
    print(f"Committed V1 w252 Sharpe {committed:.4f} sits at the {q:.0%} percentile")
    print(f"→ {OUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
