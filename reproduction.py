#!/usr/bin/env python3
"""Reproduce every number, table and figure in the report in one go.

    python reproduction.py            # everything, in order
    python reproduction.py --list     # show the steps
    python reproduction.py --only phase2 --dry-run
    python reproduction.py --from collate

The script is a thin driver over the launchers in ``scripts/``. It runs each
one as a subprocess from the repository root, in the order the README gives,
and stops at the first failure. Backtest cells whose
``results/<tag>/closed_loop.pkl`` bundle already exists are skipped, so the
script is idempotent and can be re-run after a failure; pass ``--force`` to
re-run them. Child output goes to ``results/reproduction_logs/<name>.log``.

Data access: asset prices come from CRSP through WRDS and need a WRDS line in
``~/.pgpass``; without it the code falls back to Yahoo Finance, which does not
reproduce the reported numbers. Drivers (FRED, Yahoo) need no key. The first
run of each (method, window) fits every discovery graph from scratch, which
takes hours per cell; once ``cache/discovery/`` is populated each backtest
takes about a minute.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
LOGS = RESULTS / "reproduction_logs"
REPORT = REPO / "final_report"

WINDOWS = (189, 252, 378, 504)
PHASE_I_WINDOWS = (252, 504)
GRAPH_ALLOCATORS = ("D0", "D0s", "D1", "D2", "D2s", "D3", "D4")
CONTROL_ALLOCATORS = ("D0lw", "D0df", "D0pc", "CORR", "EW", "IVP",
                      "HERCC", "HERC0", "HERC1")
GRANGER_ALLOCATORS = ("D0", "D1", "D2", "D3")
K_SWEEP = (10, 14, 17, 20, 25)
ALPHA_SWEEP = (0.4, 0.6, 0.8)
GAMMA_SWEEP = (0.1, 0.3, 0.5)
TAU_SWEEP = (0.01, 0.05, 0.1)
COST_SWEEP = (0, 10, 20)
OOS_ALLOCATORS = ("CORR", "D0", "D0s", "D1", "D2", "D2s", "D0df")
N_SEEDS = 20


# Small helpers
@dataclass
class Options:
    force: bool = False
    dry_run: bool = False
    k_override: int | None = None
    chosen_k: dict[int, int] = field(default_factory=dict)


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def say(msg: str) -> None:
    print(f"[{_stamp()}] {msg}", flush=True)


def bundle_exists(tag: str) -> bool:
    return (RESULTS / tag / "closed_loop.pkl").exists()


def run(name: str, argv: list[str], opts: Options, log_name: str | None = None) -> Path:
    """Run ``python -m scripts.<...>`` from the repo root, logging to a file."""
    cmd = [sys.executable, "-m", *argv]
    log_path = LOGS / f"{log_name or name}.log"
    say(f"{name}: {' '.join(argv)}")
    if opts.dry_run:
        return log_path
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
    minutes = (time.time() - t0) / 60
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-30:]
        print("\n".join(tail), file=sys.stderr)
        raise SystemExit(f"{name} failed (exit {proc.returncode}); "
                         f"full log at {log_path}")
    say(f"{name}: done in {minutes:.1f} min")
    return log_path


def run_cell(name: str, tag: str, argv: list[str], opts: Options) -> Path:
    """Run a backtest launcher unless its bundle already exists."""
    if bundle_exists(tag) and not opts.force:
        say(f"{name}: {tag} exists, skipping")
        return LOGS / f"{tag}.log"
    return run(name, argv, opts, log_name=tag)


def read_k(tag: str, log_path: Path) -> int | None:
    """K chosen by the V1 calibration, from the log or, failing that, the bundle."""
    if log_path.exists():
        m = re.findall(r"chosen K=(\d+)", log_path.read_text())
        if m:
            return int(m[-1])
    path = RESULTS / tag / "closed_loop.pkl"
    if path.exists():
        with path.open("rb") as fh:
            config = pickle.load(fh)["config"]
        k = config.get("K_used", config.get("K"))
        return int(k) if k is not None else None
    return None


# Steps
def step_preflight(opts: Options) -> None:
    if sys.version_info < (3, 13):
        raise SystemExit(f"Python 3.13 is required; this is {sys.version.split()[0]}")
    for tree in ("causalnex/causalnex", "lingam/lingam", "nts-notears"):
        if not (REPO / tree).is_dir():
            raise SystemExit(f"vendored tree missing: {tree}")
    if not (REPO / "scripts" / "phase_i_universe.txt").exists():
        raise SystemExit("scripts/phase_i_universe.txt is missing; the frozen "
                         "99-ticker universe is required")
    for name in ("Assets_SPX.xlsx", "Drivers_no_SB_Sectors.xlsx"):
        if not (REPO / "HSP" / name).exists():
            raise SystemExit(f"HSP/{name} is missing; tests/test_hsp_reference.py needs it")
    pgpass = Path.home() / ".pgpass"
    has_wrds = pgpass.exists() and "wrds" in pgpass.read_text().lower()
    if not has_wrds:
        say("WARNING: no WRDS line in ~/.pgpass. Prices will fall back to Yahoo "
            "Finance, which does not reproduce the reported numbers.")
    say(f"preflight ok: python {sys.version.split()[0]}, repo {REPO}")


def step_tests(opts: Options) -> None:
    run("tests", ["pytest", "tests", "-q"], opts)


def step_phase1(opts: Options) -> None:
    """The six Phase I variants at 252 and 504 days. V1 calibrates K first."""
    for w in PHASE_I_WINDOWS:
        v1_tag = f"phase_i_v1_w{w}"
        log = run_cell("phase1", v1_tag,
                       ["scripts.run_phase_i", "--variant", "V1", "--window", str(w)], opts)
        k = opts.k_override or read_k(v1_tag, log)
        if k is None:
            if opts.dry_run:
                k = 17
            else:
                raise SystemExit(f"could not read the calibrated K for window {w}")
        opts.chosen_k[w] = k
        say(f"phase1: window {w} uses K={k}")
        for variant in ("V0", "V2", "V0prime"):
            tag = f"phase_i_{variant.lower()}_w{w}"
            run_cell("phase1", tag,
                     ["scripts.run_phase_i", "--variant", variant, "--window", str(w),
                      "--k", str(k)], opts)
        for variant in ("V1", "V2"):
            tag = f"phase_i_{variant.lower()}_varlingam_w{w}"
            run_cell("phase1", tag,
                     ["scripts.run_phase_i", "--variant", variant, "--window", str(w),
                      "--discovery-method", "varlingam"], opts)


def step_phase1_sweeps(opts: Options) -> None:
    """J4a K sensitivity (V0 and V1) and J4b alpha/gamma grid (V2, 252 days)."""
    for w in PHASE_I_WINDOWS:
        for k in K_SWEEP:
            for variant in ("V0", "V1"):
                tag = f"phase_i_{variant.lower()}_w{w}_k{k}"
                run_cell("j4a", tag,
                         ["scripts.run_phase_i", "--variant", variant, "--window", str(w),
                          "--k", str(k), "--tag-suffix", f"_k{k}"], opts)
    k252 = opts.chosen_k.get(252) or opts.k_override or read_k("phase_i_v1_w252", LOGS / "phase_i_v1_w252.log") or 17
    for a in ALPHA_SWEEP:
        for g in GAMMA_SWEEP:
            tag = f"phase_i_v2_w252_a{a}_g{g}"
            run_cell("j4b", tag,
                     ["scripts.run_phase_i", "--variant", "V2", "--window", "252",
                      "--k", str(k252), "--alpha", str(a), "--gamma", str(g),
                      "--tag-suffix", f"_a{a}_g{g}"], opts)
    run("collate_j4", ["scripts.collate_j4"], opts)


def phase2_tag(method: str, allocator: str, window: int) -> str:
    if allocator == "CORR":
        return f"phase_ii_corr_hrp_w{window}"
    if allocator in ("EW", "IVP"):
        return f"phase_ii_{allocator.lower()}_w{window}"
    if allocator == "HERCC":
        return f"phase_ii_herc_corr_w{window}"
    return f"phase_ii_{method}_{allocator}_w{window}"


def step_phase2(opts: Options) -> None:
    """The graph-to-allocator ablation grid."""
    for method in ("dynotears", "varlingam"):
        for a in GRAPH_ALLOCATORS:
            for w in WINDOWS:
                run_cell("phase2", phase2_tag(method, a, w),
                         ["scripts.run_phase_ii", "--method", method,
                          "--allocator", a, "--window", str(w)], opts)
    for a in CONTROL_ALLOCATORS:
        for w in WINDOWS:
            run_cell("phase2", phase2_tag("dynotears", a, w),
                     ["scripts.run_phase_ii", "--method", "dynotears",
                      "--allocator", a, "--window", str(w)], opts)
    for a in GRANGER_ALLOCATORS:
        run_cell("phase2", phase2_tag("granger", a, 252),
                 ["scripts.run_phase_ii", "--method", "granger",
                  "--allocator", a, "--window", "252"], opts)


def step_phase2_sweeps(opts: Options) -> None:
    """Sparsity-threshold and transaction-cost sweeps for the appendix."""
    for a in ("D0", "D2", "D3"):
        for tau in TAU_SWEEP:
            tag = f"{phase2_tag('dynotears', a, 252)}_tau{tau:g}"
            run_cell("tau", tag,
                     ["scripts.run_phase_ii", "--method", "dynotears", "--allocator", a,
                      "--window", "252", "--tau", str(tau)], opts)
    for a in ("D0", "D1", "D2s"):
        for w in (252, 504):
            for c in COST_SWEEP:
                tag = f"{phase2_tag('dynotears', a, w)}_cost{c}"
                run_cell("cost", tag,
                         ["scripts.run_phase_ii", "--method", "dynotears", "--allocator", a,
                          "--window", str(w), "--transaction-cost-bps", str(c),
                          "--tag-suffix", f"_cost{c}"], opts)


def step_gates(opts: Options) -> None:
    """Cache-hit gate, DAG diagnostics, and the replication gate."""
    run("cache_gate", ["scripts.extract_asset_graphs"], opts)
    if opts.dry_run:
        say("replication gate: (dry run)")
        return
    a = RESULTS / "phase_ii_dynotears_D0_w252" / "closed_loop.pkl"
    b = RESULTS / "phase_i_v0prime_w252" / "closed_loop.pkl"
    if not (a.exists() and b.exists()):
        say("replication gate: bundles missing, skipped")
        return
    with a.open("rb") as fh:
        nav_a = pickle.load(fh)["backtest"].nav_net
    with b.open("rb") as fh:
        nav_b = pickle.load(fh)["backtest"].nav_net
    diff = float((nav_a - nav_b).abs().max())
    if diff > 1e-3:
        raise SystemExit(f"replication gate FAILED: max |NAV diff| = {diff:.3e} > 1e-3")
    say(f"replication gate passed: max |NAV diff| = {diff:.3e}")


def step_checks(opts: Options) -> None:
    """Appendix checks that the statistics and figures read."""
    run("directional_prior", ["scripts.verify_directional_prior"], opts)
    run("seed_audit", ["scripts.run_seed_audit", "--n-seeds", str(N_SEEDS)], opts)
    run("nts_probe", ["scripts.probe_nts_notears"], opts)


def step_collate(opts: Options) -> None:
    run("collate_phase_ii", ["scripts.collate_phase_ii"], opts)
    run("regime_analysis", ["scripts.regime_analysis"], opts)
    run("robust_stats", ["scripts.robust_stats"], opts)
    run("robust_stats_phase_ii", ["scripts.robust_stats", "--phase-ii"], opts)


def step_figures(opts: Options) -> None:
    run("plot_thesis_figures", ["scripts.plot_thesis_figures"], opts)
    run("plot_phase_ii_figures", ["scripts.plot_phase_ii_figures"], opts)


def step_oos(opts: Options) -> None:
    """The 2025-26 out-of-sample slice (Yahoo prices, separate caches)."""
    for a in OOS_ALLOCATORS:
        for w in WINDOWS:
            tag = f"oos_corr_hrp_w{w}" if a == "CORR" else f"oos_dynotears_{a}_w{w}"
            run_cell("oos", tag,
                     ["scripts.run_oos_slice", "--allocator", a, "--window", str(w)], opts)
    run("collate_oos", ["scripts.collate_oos"], opts)


def step_compile(opts: Options) -> None:
    if shutil.which("latexmk") is None:
        say("compile: latexmk not found, skipping (run `latexmk -pdf main.tex` in final_report/)")
        return
    say("compile: latexmk -pdf main.tex")
    if opts.dry_run:
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "compile.log").open("w") as fh:
        proc = subprocess.run(["latexmk", "-pdf", "main.tex"], cwd=REPORT,
                              stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"compile failed; log at {LOGS / 'compile.log'}")
    say(f"compile: {REPORT / 'main.pdf'}")


STEPS: list[tuple[str, str, Callable[[Options], None]]] = [
    ("preflight", "Python version, vendored trees, WRDS credential", step_preflight),
    ("tests", "unit-test suite", step_tests),
    ("phase1", "Phase I variants V0, V0', V1, V2 at 252 and 504 days", step_phase1),
    ("phase1_sweeps", "J4a K sweep, J4b alpha/gamma grid, collate_j4", step_phase1_sweeps),
    ("phase2", "Phase II ablation grid (DYNOTEARS, VARLiNGAM, Granger)", step_phase2),
    ("phase2_sweeps", "sparsity-threshold and transaction-cost sweeps", step_phase2_sweeps),
    ("gates", "cache-hit gate, DAG diagnostics, replication gate", step_gates),
    ("checks", "directional prior, FFNN seed audit, NTS-NOTEARS probe", step_checks),
    ("collate", "phase_ii matrix and contrasts, regimes, robust stats", step_collate),
    ("figures", "Phase I and Phase II figures", step_figures),
    ("oos", "2025-26 out-of-sample slice", step_oos),
    ("compile", "latexmk the report", step_compile),
]
STEP_NAMES = [s[0] for s in STEPS]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    p.add_argument("--only", nargs="+", choices=STEP_NAMES, metavar="STEP",
                   help="run only these steps")
    p.add_argument("--from", dest="from_step", choices=STEP_NAMES, metavar="STEP",
                   help="start from this step")
    p.add_argument("--skip", nargs="+", choices=STEP_NAMES, default=[], metavar="STEP",
                   help="skip these steps")
    p.add_argument("--force", action="store_true",
                   help="re-run backtest cells whose bundle already exists")
    p.add_argument("--dry-run", action="store_true", help="print commands without running")
    p.add_argument("--k", type=int, default=None,
                   help="override the calibrated K for V0/V2 (the report uses 17)")
    args = p.parse_args(argv)

    if args.list:
        for name, desc, _ in STEPS:
            print(f"{name:15s} {desc}")
        return

    selected = STEP_NAMES
    if args.only:
        selected = [s for s in STEP_NAMES if s in args.only]
    elif args.from_step:
        selected = STEP_NAMES[STEP_NAMES.index(args.from_step):]
    selected = [s for s in selected if s not in args.skip]

    opts = Options(force=args.force, dry_run=args.dry_run, k_override=args.k)
    t0 = time.time()
    for name, desc, fn in STEPS:
        if name not in selected:
            continue
        say(f"=== {name}: {desc}")
        fn(opts)
    say(f"all done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
