"""Robust-stats battery for the final report.

Computes PSR/DSR, White's Reality Check, Hansen's SPA, the Model Confidence
Set, and the closed-loop reward SNR over every evaluated configuration.
Reads the gitignored results/<tag>/closed_loop.pkl bundles, so it must run
locally. Writes results/robust_stats.csv and the _generated/robust_stats.tex
macros for both report variants.

Run:  python -m scripts.robust_stats
"""
from __future__ import annotations

import logging
import pathlib
import pickle

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.evaluation.bootstrap import stationary_block_indices
from pipeline.evaluation.metrics import (
    annualised_sharpe,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

log = logging.getLogger("robust_stats")

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
GEN = REPO / "final_report" / "_generated"
GEN_NO_HSP = REPO / "final_report_no_hsp" / "_generated"

# The trial universe the DSR deflates against. Missing bundles are skipped
# and logged, so this runs on whatever subset is present.
WINDOWS = (252, 504)
KS = (10, 14, 17, 20, 25)
AG_GRID = ((0.4, 0.1), (0.4, 0.3), (0.4, 0.5),
           (0.6, 0.1), (0.6, 0.3), (0.6, 0.5),
           (0.8, 0.1), (0.8, 0.3), (0.8, 0.5))

# Headline variant tags per window (the universe SPA/RC/MCS choose between).
HEADLINE = {
    "V0":            "phase_i_v0_w{w}",
    "V0prime":       "phase_i_v0prime_w{w}",
    "V1-DYNOTEARS":  "phase_i_v1_w{w}",
    "V2-DYNOTEARS":  "phase_i_v2_w{w}",
    "V1-VARLiNGAM":  "phase_i_v1_varlingam_w{w}",
    "V2-VARLiNGAM":  "phase_i_v2_varlingam_w{w}",
}


def _all_trial_tags() -> dict[str, str]:
    """name -> bundle tag, for every distinct configuration evaluated."""
    tags: dict[str, str] = {}
    for w in WINDOWS:
        for name, stem in HEADLINE.items():
            tags[f"{name}_w{w}"] = stem.format(w=w)
        for k in KS:                       # K-sensitivity sweep (V0, V1)
            tags[f"V0_w{w}_k{k}"] = f"phase_i_v0_w{w}_k{k}"
            tags[f"V1_w{w}_k{k}"] = f"phase_i_v1_w{w}_k{k}"
    for a, g in AG_GRID:                    # alpha/gamma feedback grid (w252)
        tags[f"V2_w252_a{a}_g{g}"] = f"phase_i_v2_w252_a{a}_g{g}"
    return tags


# Phase-II direction-aware allocators. D0 duplicates V0prime byte-for-byte
# under DYNOTEARS; drop_duplicate_configs removes it before SPA/MCS.
PHASE_II_ALLOCS = ("D0", "D0s", "D1", "D2", "D2s", "D3", "D4")
# The reported family, a restriction chosen AFTER the four-window results were
# seen, so every family-wise test runs twice: pre-specified (primary) and
# reported (labelled as narrowed).
REPORTED_ALLOCS = ("D0", "D0s", "D1", "D2", "D2s")
# Direction-aware members of each family (candidates vs the D0/V0prime anchor).
DIRECTION_FAMILY = ("1", "2", "2s")
DIRECTION_FAMILY_PRE = ("1", "2", "2s", "3", "4")
# Post-hoc mechanism controls, DYNOTEARS only. In the deflation universe but
# in NEITHER family-wise comparison set (introduced after both were fixed).
CONTROL_ALLOCS = ("D0lw", "D0df", "D0pc")
# Graph-blind anchors: deflation universe only.
ANCHOR_TAGS = {"EW": "phase_ii_ew_w{w}", "IVP": "phase_ii_ivp_w{w}"}
# HERC cells. No "DYNO-" prefix on purpose: the family filters key on that
# prefix and the HERC cells belong to neither SPA family.
HERC_TAGS = {"HERCC": "phase_ii_herc_corr_w{w}",
             "HERC0": "phase_ii_dynotears_HERC0_w{w}",
             "HERC1": "phase_ii_dynotears_HERC1_w{w}"}
PHASE_II_METHODS = (("dynotears", "DYNO"), ("varlingam", "VARL"))
PHASE_II_WINDOWS = (189, 252, 378, 504)


def _phase_ii_tags() -> dict[str, str]:
    """name -> bundle tag for every Phase-II configuration evaluated."""
    tags: dict[str, str] = {}
    for w in PHASE_II_WINDOWS:
        tags[f"CORR-HRP_w{w}"] = f"phase_ii_corr_hrp_w{w}"
        for method, short in PHASE_II_METHODS:
            for a in PHASE_II_ALLOCS:
                tags[f"{short}-{a}_w{w}"] = f"phase_ii_{method}_{a}_w{w}"
        for a in CONTROL_ALLOCS:
            tags[f"DYNO-{a}_w{w}"] = f"phase_ii_dynotears_{a}_w{w}"
        for a, stem in ANCHOR_TAGS.items():
            tags[f"{a}_w{w}"] = stem.format(w=w)
        for a, stem in HERC_TAGS.items():
            tags[f"{a}_w{w}"] = stem.format(w=w)
    for a in ("D0", "D1", "D2", "D3"):      # E2 GRANGER arm (w252 only)
        tags[f"GRAN-{a}_w252"] = f"phase_ii_granger_{a}_w252"
    for tau in ("0.01", "0.05", "0.1"):     # E3 τ sweep (DYNO w252)
        for a in ("D0", "D2", "D3"):
            tags[f"DYNO-{a}_w252_tau{tau}"] = f"phase_ii_dynotears_{a}_w252_tau{tau}"
    return tags


# Bundle loading
def _load_backtest(tag: str):
    path = RESULTS / tag / "closed_loop.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)["backtest"]


def load_return_matrix(tags: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Load each tag's daily net returns into an inner-joined matrix.

    Returns (returns_df, loaded_names).
    """
    series, loaded = {}, []
    for name, tag in tags.items():
        bt = _load_backtest(tag)
        if bt is None:
            log.warning("missing bundle: %s (%s) — skipped", name, tag)
            continue
        s = pd.Series(bt.nav_net).astype(float)
        s.index = pd.to_datetime(s.index)
        series[name] = s.sort_index().pct_change().dropna()
        loaded.append(name)
    if not series:
        return pd.DataFrame(), []
    df = pd.concat(series, axis=1).dropna()
    return df, loaded


def load_reward_series(tag: str) -> pd.Series | None:
    """Per-rebalance realised reward R[t] (excess Sharpe vs 1/N)."""
    bt = _load_backtest(tag)
    if bt is None:
        return None
    return pd.Series(
        [r.holding_reward for r in bt.rebalances],
        index=pd.to_datetime([r.rebalance_date for r in bt.rebalances]),
    )


# Pure analysis functions
def per_period_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    sigma = r.std(ddof=0)
    return float(r.mean() / sigma) if sigma > 1e-12 else 0.0


def drop_duplicate_configs(returns: pd.DataFrame, names: list[str]) -> list[str]:
    """Drop names whose daily-return series exactly equals an earlier-kept one.

    V2 equals V1 byte-for-byte (inert loop); duplicates double-count a strategy
    and give the studentised MCS a zero differential variance.
    """
    kept: list[str] = []
    for c in names:
        if not any(np.array_equal(returns[c].to_numpy(), returns[k].to_numpy())
                   for k in kept):
            kept.append(c)
    return kept


def psr_dsr_table(returns: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """PSR (vs 0 and vs baseline) and DSR for every column of returns.

    The DSR deflates against all per-period trial Sharpes in the matrix.
    """
    pp = {c: per_period_sharpe(returns[c]) for c in returns.columns}
    all_pp = list(pp.values())
    base_pp = pp[baseline]
    rows = []
    for c in returns.columns:
        rows.append({
            "config": c,
            "sharpe_ann": round(annualised_sharpe(returns[c]), 4),
            "psr_vs_zero": round(probabilistic_sharpe_ratio(returns[c], 0.0), 4),
            "psr_vs_baseline": round(probabilistic_sharpe_ratio(returns[c], base_pp), 4),
            "dsr": round(deflated_sharpe_ratio(returns[c], all_pp), 4),
            "n_trials": len(all_pp),
        })
    return pd.DataFrame(rows).sort_values("sharpe_ann", ascending=False).reset_index(drop=True)


def run_spa(returns: pd.DataFrame, benchmark: str, candidates: list[str],
            block_size: int = 21, reps: int = 10000, seed: int = 42) -> dict:
    """White's Reality Check + Hansen's SPA: do any candidates beat the
    benchmark? Loss = negative return, so a low p-value means a candidate
    significantly out-returns it. Falls back to a hand-rolled Reality Check
    if arch is absent.
    """
    bench = -returns[benchmark].to_numpy(dtype=float)
    models = -returns[candidates].to_numpy(dtype=float)
    try:
        from arch.bootstrap import SPA
        spa = SPA(bench, models, block_size=block_size, reps=reps, seed=seed)
        spa.compute()
        pv = spa.pvalues
        return {"rc_lower": float(pv["lower"]),
                "spa_consistent": float(pv["consistent"]),
                "spa_upper": float(pv["upper"]),
                "engine": "arch"}
    except Exception as exc:  # pragma: no cover - exercised only without arch
        log.warning("arch SPA unavailable (%s); using hand-rolled Reality Check", exc)
        return {"rc_lower": _handrolled_reality_check(returns, benchmark, candidates,
                                                      block_size, reps, seed),
                "spa_consistent": float("nan"), "spa_upper": float("nan"),
                "engine": "fallback"}


def _handrolled_reality_check(returns: pd.DataFrame, benchmark: str,
                              candidates: list[str], block_size: int,
                              reps: int, seed: int) -> float:
    """White (2000) Reality Check p-value via the stationary block bootstrap."""
    diffs = returns[candidates].sub(returns[benchmark], axis=0).to_numpy(dtype=float)
    n = diffs.shape[0]
    obs = float(np.max(diffs.mean(axis=0)))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(reps):
        idx = stationary_block_indices(n, float(block_size), rng)
        boot = diffs[idx]
        centred = boot.mean(axis=0) - diffs.mean(axis=0)
        if float(np.max(centred)) >= obs:
            count += 1
    return (count + 1) / (reps + 1)


def run_mcs(returns: pd.DataFrame, configs: list[str], size: float = 0.10,
            block_size: int = 21, reps: int = 10000, seed: int = 42) -> tuple[list[str], pd.DataFrame]:
    """Model Confidence Set over configs at confidence 1-size (loss =
    negative return). Returns (included_names, pvalue_frame).
    """
    from arch.bootstrap import MCS
    losses = -returns[configs]
    mcs = MCS(losses, size=size, block_size=block_size, reps=reps, seed=seed, method="R")
    mcs.compute()
    included = [c for c in configs if c in set(mcs.included)]
    return included, mcs.pvalues


def measurement_problem(reward: pd.Series, holding_days: int = 21) -> dict:
    """Quantify why a monthly reward is too noisy to learn from.

    Reports the reward's mean, std, SNR, and the theoretical sampling SE of
    a Sharpe estimated from holding_days observations.
    """
    r = reward.dropna()
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    snr = abs(mean) / std if std > 1e-12 else float("inf")
    se_sharpe = float(np.sqrt(1.0 / holding_days))
    return {"reward_mean": mean, "reward_std": std, "reward_snr": snr,
            "sharpe_se_window": se_sharpe, "n_rebalances": int(len(r))}


def pooled_excess_kurtosis(returns: pd.DataFrame) -> float:
    """Mean across configurations of the daily-return excess kurtosis."""
    return float(np.mean([stats.kurtosis(returns[c].dropna(), fisher=True, bias=False)
                          for c in returns.columns]))


def full_sample_sharpe_se(returns: pd.Series, periods_per_year: int = 252) -> float:
    """SE of the full-sample annualised Sharpe (Lo 2002, iid approximation).
    Distinct from measurement_problem's per-holding-window SE."""
    r = returns.dropna()
    sr_d = per_period_sharpe(r)
    return float(np.sqrt((1.0 + 0.5 * sr_d ** 2) / len(r)) * np.sqrt(periods_per_year))


def contrast_se_mde(returns: pd.DataFrame, a: str, b: str,
                    reps: int = 10000, block: float = 21.0,
                    seed: int = 42) -> tuple[float, float]:
    """Block-bootstrap SE of the annualised ΔSharpe(a - b), plus the one-sided
    MDE at 5% size / 80% power. The joint panel is resampled so the
    correlation between the strategies is preserved."""
    from pipeline.evaluation.metrics import annualised_sharpe as _sh
    arr = returns[[a, b]].dropna().to_numpy(dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        idx = stationary_block_indices(n, block, rng)
        panel = arr[idx]
        diffs[i] = _sh(pd.Series(panel[:, 0])) - _sh(pd.Series(panel[:, 1]))
    se = float(diffs.std(ddof=1))
    mde = float((stats.norm.ppf(0.95) + stats.norm.ppf(0.80)) * se)
    return se, mde


# Macro emission for the report
def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else "NaN"


def write_macros(path: pathlib.Path, macros: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% AUTO-GENERATED by scripts/robust_stats.py — do not edit by hand.",
             "% Regenerate locally (bundles required): python -m scripts.robust_stats",
             ""]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    path.write_text("\n".join(lines) + "\n")


# Orchestration
def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Robust-stats battery (PSR/DSR/SPA/RC/MCS).")
    ap.add_argument(
        "--phase-ii", action="store_true",
        help="extend the trial universe with the Phase-II D-variant bundles; "
             "writes robust_stats_phase_ii.csv and does NOT touch the report "
             "macros (the committed Phase-I battery stays frozen until the "
             "report-reorientation stage).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tags = _all_trial_tags()
    if args.phase_ii:
        tags = {**tags, **_phase_ii_tags()}
    returns, loaded = load_return_matrix(tags)
    if returns.empty:
        log.error("No bundles found under %s — this script must run where the "
                  "gitignored closed_loop.pkl bundles live.", RESULTS)
        return
    n_trials = returns.shape[1]
    log.info("loaded %d configurations over %d common days", n_trials, len(returns))
    if not args.phase_ii and n_trials != 41:
        log.warning("expected 41 trials, loaded %d (partial result set)", n_trials)

    # PSR / DSR table over every trial (baseline = V0 at w252).
    baseline = "V0_w252" if "V0_w252" in returns.columns else loaded[0]
    table = psr_dsr_table(returns, baseline=baseline)

    # SPA / Reality Check + MCS over the w252 headline universe vs V0.
    # Identical columns break the studentised MCS, so drop duplicates first.
    headline252 = drop_duplicate_configs(
        returns, [f"{n}_w252" for n in HEADLINE if f"{n}_w252" in returns.columns])
    causal252 = [c for c in headline252 if not c.startswith("V0_")]
    spa = run_spa(returns, benchmark="V0_w252", candidates=causal252)

    # MCS universe: the w252 headline, plus (in --phase-ii mode) every
    # distinct w252 D-variant.
    mcs_universe = list(headline252)
    spa_phase_ii = None
    spa_phase_ii_pre = None
    spa_direction: dict[int, dict] = {}
    spa_direction_pre: dict[int, dict] = {}
    spa_skeleton: dict[int, dict] = {}
    spa_skeleton_pre: dict[int, dict] = {}
    spa_pooled: dict[str, dict | None] = {"skel": None, "skel_pre": None,
                                          "dir": None, "dir_pre": None}

    def _spa_family(bench: str, cands: list[str]) -> dict | None:
        cands = [c for c in drop_duplicate_configs(returns, [bench] + cands)
                 if c != bench]
        if bench in returns.columns and cands:
            return run_spa(returns, benchmark=bench, candidates=cands)
        return None

    if args.phase_ii:
        def _d252(excluded: tuple[str, ...]) -> list[str]:
            # Mechanism controls are candidates of neither family.
            drop = excluded + CONTROL_ALLOCS
            return [c for c in returns.columns
                    if c.startswith(("DYNO-", "VARL-", "GRAN-", "CORR-"))
                    and c.endswith("_w252") and "_tau" not in c
                    and not any(f"-{a}_" in c for a in drop)]
        # Pre-registered E1 family first, then the reported (narrowed) family.
        d252_pre = drop_duplicate_configs(returns, mcs_universe + _d252(()))
        d_only_pre = [c for c in d252_pre if c not in mcs_universe]
        if d_only_pre:
            spa_phase_ii_pre = run_spa(returns, benchmark="V0_w252",
                                       candidates=d_only_pre)
        d252 = drop_duplicate_configs(returns, mcs_universe + _d252(("D3", "D4")))
        d_only = [c for c in d252 if c not in mcs_universe]
        if d_only:
            spa_phase_ii = run_spa(returns, benchmark="V0_w252", candidates=d_only)
        mcs_universe = d252
        # Does ANY direction-aware allocator beat its symmetrised control D0,
        # per window? Where no V0prime bundle exists the benchmark falls back
        # to DYNO-D0 (byte-identical where both exist).
        for w in PHASE_II_WINDOWS:
            bench = f"V0prime_w{w}"
            if bench not in returns.columns:
                bench = f"DYNO-D0_w{w}"
            for fam, out in ((DIRECTION_FAMILY_PRE, spa_direction_pre),
                             (DIRECTION_FAMILY, spa_direction)):
                cands = [c for c in returns.columns
                         if c.split("_")[0] in {f"{s}-D{v}" for s in ("DYNO", "VARL")
                                                for v in fam}
                         and c.endswith(f"_w{w}") and "_tau" not in c]
                res = _spa_family(bench, cands)
                if res is not None:
                    out[w] = res
        # Does ANY causal-structure strategy beat CORR-HRP, per window?
        for w in PHASE_II_WINDOWS:
            bench = f"CORR-HRP_w{w}"
            for fam, out in ((PHASE_II_ALLOCS, spa_skeleton_pre),
                             (REPORTED_ALLOCS, spa_skeleton)):
                cands = [c for c in returns.columns
                         if (c.startswith(tuple(f"{s}-{a}_" for s in ("DYNO", "VARL")
                                                for a in fam))
                             or c == f"V0prime_w{w}")
                         and c.endswith(f"_w{w}") and "_tau" not in c]
                res = _spa_family(bench, cands)
                if res is not None:
                    out[w] = res
        # Pooled SPA over the union of the four windows, pricing the search
        # across strategies AND windows jointly. Benchmark = the family's
        # control at the one-year window.
        def _pool_skel(fam: tuple[str, ...]) -> list[str]:
            return [c for c in returns.columns
                    if (c.startswith(tuple(f"{s}-{a}_" for s in ("DYNO", "VARL")
                                           for a in fam))
                        or c.startswith("V0prime_w"))
                    and "_tau" not in c]

        def _pool_dir(fam: tuple[str, ...]) -> list[str]:
            return [c for c in returns.columns
                    if c.split("_")[0] in {f"{s}-D{v}" for s in ("DYNO", "VARL")
                                           for v in fam}
                    and "_tau" not in c]

        skel_bench = "CORR-HRP_w252"
        dir_bench = ("V0prime_w252" if "V0prime_w252" in returns.columns
                     else "DYNO-D0_w252")
        if skel_bench in returns.columns:
            spa_pooled["skel_pre"] = _spa_family(skel_bench, _pool_skel(PHASE_II_ALLOCS))
            spa_pooled["skel"] = _spa_family(skel_bench, _pool_skel(REPORTED_ALLOCS))
        spa_pooled["dir_pre"] = _spa_family(dir_bench, _pool_dir(DIRECTION_FAMILY_PRE))
        spa_pooled["dir"] = _spa_family(dir_bench, _pool_dir(DIRECTION_FAMILY))
    mcs_in, _ = run_mcs(returns, mcs_universe)
    # MCS membership is only meaningful for the MCS universe.
    table["in_mcs90"] = [bool(c in mcs_in) if c in mcs_universe else ""
                         for c in table.config]
    RESULTS.mkdir(exist_ok=True)
    out_csv = RESULTS / ("robust_stats_phase_ii.csv" if args.phase_ii else "robust_stats.csv")
    table.to_csv(out_csv, index=False)
    print("\n=== PSR / DSR by configuration ===")
    print(table.to_string(index=False))
    print(f"\n=== SPA/RC (causal vs V0, w252) ===\n{spa}")
    if spa_phase_ii_pre is not None:
        print(f"\n=== SPA/RC (Phase-II D-variants vs V0, w252, PRE-REGISTERED family) ===\n{spa_phase_ii_pre}")
    if spa_phase_ii is not None:
        print(f"\n=== SPA/RC (Phase-II D-variants vs V0, w252, reported family) ===\n{spa_phase_ii}")
    for w, res in spa_direction_pre.items():
        print(f"\n=== SPA/RC (direction-aware vs D0/V0prime, w{w}, PRE-REGISTERED family) ===\n{res}")
    for w, res in spa_direction.items():
        print(f"\n=== SPA/RC (direction-aware vs D0/V0prime, w{w}, reported family) ===\n{res}")
    for w, res in spa_skeleton_pre.items():
        print(f"\n=== SPA/RC (causal structure vs CORR-HRP, w{w}, PRE-REGISTERED family) ===\n{res}")
    for w, res in spa_skeleton.items():
        print(f"\n=== SPA/RC (causal structure vs CORR-HRP, w{w}, reported family) ===\n{res}")
    for key, label in (("skel_pre", "causal structure vs CORR-HRP w252, POOLED windows, PRE-REGISTERED"),
                       ("skel", "causal structure vs CORR-HRP w252, POOLED windows, reported"),
                       ("dir_pre", "direction-aware vs D0/V0prime w252, POOLED windows, PRE-REGISTERED"),
                       ("dir", "direction-aware vs D0/V0prime w252, POOLED windows, reported")):
        if spa_pooled.get(key) is not None:
            print(f"\n=== SPA/RC ({label} family) ===\n{spa_pooled[key]}")
    print(f"\n=== MCS 90% set (w252 universe, n={len(mcs_universe)}) ===\n{mcs_in}")

    # Measurement problem on the closed loop (V2/V1 w252).
    rtag = "phase_i_v2_w252" if (RESULTS / "phase_i_v2_w252").exists() else "phase_i_v1_w252"
    reward = load_reward_series(rtag)
    mp = measurement_problem(reward) if reward is not None else {}
    print(f"\n=== measurement problem ({rtag}) ===\n{mp}")

    # Full-sample SE and contrast-level precision (power context).
    se_full = (full_sample_sharpe_se(returns["DYNO-D1_w252"])
               if "DYNO-D1_w252" in returns.columns else float("nan"))
    se_total = mde_total = se_orient = mde_orient = float("nan")
    if args.phase_ii:
        if {"DYNO-D1_w252", "CORR-HRP_w252"} <= set(returns.columns):
            se_total, mde_total = contrast_se_mde(returns, "DYNO-D1_w252",
                                                  "CORR-HRP_w252")
        if {"DYNO-D2s_w252", "DYNO-D0_w252"} <= set(returns.columns):
            se_orient, mde_orient = contrast_se_mde(returns, "DYNO-D2s_w252",
                                                    "DYNO-D0_w252")
        print(f"\n=== precision === full-sample Sharpe SE {se_full:.3f} | "
              f"ΔSharpe SE (D1−CORR) {se_total:.3f}, MDE80 {mde_total:.3f} | "
              f"ΔSharpe SE (D2s−D0) {se_orient:.3f}, MDE80 {mde_orient:.3f}")

    # Self-check: reproduce the headline Sharpes.
    print("\n=== self-check: headline annualised Sharpe ===")
    for n in ("V0_w252", "V1-DYNOTEARS_w252", "V1-VARLiNGAM_w252", "V0prime_w252"):
        if n in returns.columns:
            print(f"  {n:24s} {annualised_sharpe(returns[n]):.4f}")

    # Report macros are owned by the unified --phase-ii battery; the 41-trial
    # default mode must not overwrite them with a smaller deflation universe.
    if not args.phase_ii:
        print(f"\nsaved → {out_csv} (report macros unchanged — regenerate with --phase-ii)")
        return

    def cell(config: str, col: str) -> float:
        hit = table.loc[table.config == config, col]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    # The informative quantity is who is EXCLUDED from the MCS. Excluded names
    # are rendered with the report's display macros (no codenames in prose).
    display = {
        "V0": r"\hspbase{}",
        "V0prime": r"\skelsamp{}",
        "V1-DYNOTEARS": r"\hspcausal{}",
        "V1-VARLiNGAM": r"\hspcausal{} (VARLiNGAM)",
        "V2-DYNOTEARS": r"\hsploop{}",
        "V2-VARLiNGAM": r"\hsploop{} (VARLiNGAM)",
        "CORR-HRP": r"\HRP{}",
        "DYNO-D0": r"\skelsamp{}", "DYNO-D0s": r"\skelalt{}",
        "DYNO-D1": r"\skelsem{}", "DYNO-D2": r"\orientsamp{}",
        "DYNO-D2s": r"\orientsem{}",
        "VARL-D0": r"\skelsamp{} (VARLiNGAM)", "VARL-D0s": r"\skelalt{} (VARLiNGAM)",
        "VARL-D1": r"\skelsem{} (VARLiNGAM)", "VARL-D2": r"\orientsamp{} (VARLiNGAM)",
        "VARL-D2s": r"\orientsem{} (VARLiNGAM)",
        "DYNO-D0lw": "the Ledoit-Wolf control", "DYNO-D0df": "the de-factored control",
    }
    excluded = [c for c in mcs_universe if c not in mcs_in]
    pretty_excluded = ", ".join(
        display.get(e.replace("_w252", ""), e.replace("_w252", ""))
        for e in excluded) or "none"

    # E7 seed audit (written by scripts/run_seed_audit.py).
    seed_macros: dict[str, str] = {}
    seed_csv = RESULTS / "seed_audit.csv"
    if seed_csv.exists():
        sa = pd.read_csv(seed_csv)
        committed = float(sa.loc[sa.seed == 0, "sharpe"].iloc[0])
        seed_macros = {
            "rsSeedN":      str(len(sa)),
            "rsSeedMin":    _fmt(sa.sharpe.min()),
            "rsSeedMedian": _fmt(sa.sharpe.median()),
            "rsSeedMax":    _fmt(sa.sharpe.max()),
            "rsSeedRange":  _fmt(sa.sharpe.max() - sa.sharpe.min()),
            "rsSeedPct":    str(int(round(100 * float((sa.sharpe < committed).mean())))),
        }

    macros = {
        "rsNtrials": str(n_trials),
        "rsKurtosis": _fmt(pooled_excess_kurtosis(returns), 1),
        # per-variant PSR (vs zero) and DSR, w252
        "rsVzeroPSR":    _fmt(cell("V0_w252", "psr_vs_zero")),
        "rsVzeroDSR":    _fmt(cell("V0_w252", "dsr")),
        "rsVprimePSR":   _fmt(cell("V0prime_w252", "psr_vs_zero")),
        "rsVprimeDSR":   _fmt(cell("V0prime_w252", "dsr")),
        "rsVoneDynoPSR": _fmt(cell("V1-DYNOTEARS_w252", "psr_vs_zero")),
        "rsVoneDynoDSR": _fmt(cell("V1-DYNOTEARS_w252", "dsr")),
        "rsVoneVarPSR":  _fmt(cell("V1-VARLiNGAM_w252", "psr_vs_zero")),
        "rsVoneVarDSR":  _fmt(cell("V1-VARLiNGAM_w252", "dsr")),
        # window suffixes: Wone=189, Wthree=378, Wfive=504; unsuffixed = w252
        "rsCorrSharpe":       _fmt(cell("CORR-HRP_w252", "sharpe_ann")),
        "rsCorrSharpeWone":   _fmt(cell("CORR-HRP_w189", "sharpe_ann")),
        "rsCorrSharpeWthree": _fmt(cell("CORR-HRP_w378", "sharpe_ann")),
        "rsCorrSharpeWfive":  _fmt(cell("CORR-HRP_w504", "sharpe_ann")),
        "rsCorrPSR":          _fmt(cell("CORR-HRP_w252", "psr_vs_zero")),
        "rsCorrDSR":          _fmt(cell("CORR-HRP_w252", "dsr")),
        "rsDoneSharpe":       _fmt(cell("DYNO-D1_w252", "sharpe_ann")),
        "rsDoneSharpeWone":   _fmt(cell("DYNO-D1_w189", "sharpe_ann")),
        "rsDoneSharpeWthree": _fmt(cell("DYNO-D1_w378", "sharpe_ann")),
        "rsDoneSharpeWfive":  _fmt(cell("DYNO-D1_w504", "sharpe_ann")),
        "rsDonePSR":     _fmt(cell("DYNO-D1_w252", "psr_vs_zero")),
        "rsDoneDSR":     _fmt(cell("DYNO-D1_w252", "dsr")),
        "rsDtwosSharpe":       _fmt(cell("DYNO-D2s_w252", "sharpe_ann")),
        "rsDtwosSharpeWone":   _fmt(cell("DYNO-D2s_w189", "sharpe_ann")),
        "rsDtwosSharpeWthree": _fmt(cell("DYNO-D2s_w378", "sharpe_ann")),
        "rsDtwosSharpeWfive":  _fmt(cell("DYNO-D2s_w504", "sharpe_ann")),
        "rsDtwosDSR":    _fmt(cell("DYNO-D2s_w252", "dsr")),
        "rsDtwosDSRWthree": _fmt(cell("DYNO-D2s_w378", "dsr")),
        "rsDoneDSRWthree":  _fmt(cell("DYNO-D1_w378", "dsr")),
        "rsDzeroSharpe":       _fmt(cell("DYNO-D0_w252", "sharpe_ann")),
        "rsDzeroSharpeWone":   _fmt(cell("DYNO-D0_w189", "sharpe_ann")),
        "rsDzeroSharpeWthree": _fmt(cell("DYNO-D0_w378", "sharpe_ann")),
        "rsDzeroSharpeWfive":  _fmt(cell("DYNO-D0_w504", "sharpe_ann")),
        # mechanism controls
        "rsDlwSharpe":       _fmt(cell("DYNO-D0lw_w252", "sharpe_ann")),
        "rsDlwSharpeWone":   _fmt(cell("DYNO-D0lw_w189", "sharpe_ann")),
        "rsDlwSharpeWthree": _fmt(cell("DYNO-D0lw_w378", "sharpe_ann")),
        "rsDlwSharpeWfive":  _fmt(cell("DYNO-D0lw_w504", "sharpe_ann")),
        "rsDdfSharpe":       _fmt(cell("DYNO-D0df_w252", "sharpe_ann")),
        "rsDdfSharpeWone":   _fmt(cell("DYNO-D0df_w189", "sharpe_ann")),
        "rsDdfSharpeWthree": _fmt(cell("DYNO-D0df_w378", "sharpe_ann")),
        "rsDdfSharpeWfive":  _fmt(cell("DYNO-D0df_w504", "sharpe_ann")),
        "rsDdfDSRWthree":    _fmt(cell("DYNO-D0df_w378", "dsr")),
        # skeleton-channel control
        "rsDpcSharpe":       _fmt(cell("DYNO-D0pc_w252", "sharpe_ann")),
        "rsDpcSharpeWone":   _fmt(cell("DYNO-D0pc_w189", "sharpe_ann")),
        "rsDpcSharpeWthree": _fmt(cell("DYNO-D0pc_w378", "sharpe_ann")),
        "rsDpcSharpeWfive":  _fmt(cell("DYNO-D0pc_w504", "sharpe_ann")),
        # naive anchors (one-year cells)
        "rsEwSharpe":        _fmt(cell("EW_w252", "sharpe_ann")),
        "rsIvpSharpe":       _fmt(cell("IVP_w252", "sharpe_ann")),
        # HERC cells (one-year)
        "rsHercCorrSharpe":  _fmt(cell("HERCC_w252", "sharpe_ann")),
        "rsHercSkelSharpe":  _fmt(cell("HERC0_w252", "sharpe_ann")),
        "rsHercSemSharpe":   _fmt(cell("HERC1_w252", "sharpe_ann")),
        # data-snooping battery: unsuffixed = reported (narrowed) family,
        # *Pre = the pre-specified E1 family (quoted first in the report)
        "rsSpaRC":         _fmt(spa["rc_lower"]),
        "rsSpaConsistent": _fmt(spa["spa_consistent"]),
        "rsSpaDvar":       _fmt(spa_phase_ii["spa_consistent"]) if spa_phase_ii else "--",
        "rsSpaDvarPre":    _fmt(spa_phase_ii_pre["spa_consistent"]) if spa_phase_ii_pre else "--",
        "rsSpaDirWone":    _fmt(spa_direction[189]["spa_consistent"]) if 189 in spa_direction else "--",
        "rsSpaDirWtwo":    _fmt(spa_direction[252]["spa_consistent"]) if 252 in spa_direction else "--",
        "rsSpaDirWthree":  _fmt(spa_direction[378]["spa_consistent"]) if 378 in spa_direction else "--",
        "rsSpaDirWfive":   _fmt(spa_direction[504]["spa_consistent"]) if 504 in spa_direction else "--",
        "rsSpaDirWonePre":    _fmt(spa_direction_pre[189]["spa_consistent"]) if 189 in spa_direction_pre else "--",
        "rsSpaDirWtwoPre":    _fmt(spa_direction_pre[252]["spa_consistent"]) if 252 in spa_direction_pre else "--",
        "rsSpaDirWthreePre":  _fmt(spa_direction_pre[378]["spa_consistent"]) if 378 in spa_direction_pre else "--",
        "rsSpaDirWfivePre":   _fmt(spa_direction_pre[504]["spa_consistent"]) if 504 in spa_direction_pre else "--",
        "rsSpaSkelWone":   _fmt(spa_skeleton[189]["spa_consistent"]) if 189 in spa_skeleton else "--",
        "rsSpaSkelWtwo":   _fmt(spa_skeleton[252]["spa_consistent"]) if 252 in spa_skeleton else "--",
        "rsSpaSkelWthree": _fmt(spa_skeleton[378]["spa_consistent"]) if 378 in spa_skeleton else "--",
        "rsSpaSkelWfive":  _fmt(spa_skeleton[504]["spa_consistent"]) if 504 in spa_skeleton else "--",
        "rsSpaSkelWonePre":   _fmt(spa_skeleton_pre[189]["spa_consistent"]) if 189 in spa_skeleton_pre else "--",
        "rsSpaSkelWtwoPre":   _fmt(spa_skeleton_pre[252]["spa_consistent"]) if 252 in spa_skeleton_pre else "--",
        "rsSpaSkelWthreePre": _fmt(spa_skeleton_pre[378]["spa_consistent"]) if 378 in spa_skeleton_pre else "--",
        "rsSpaSkelWfivePre":  _fmt(spa_skeleton_pre[504]["spa_consistent"]) if 504 in spa_skeleton_pre else "--",
        "rsSpaSkelPooled":    _fmt(spa_pooled["skel"]["spa_consistent"]) if spa_pooled["skel"] else "--",
        "rsSpaSkelPooledPre": _fmt(spa_pooled["skel_pre"]["spa_consistent"]) if spa_pooled["skel_pre"] else "--",
        "rsSpaDirPooled":     _fmt(spa_pooled["dir"]["spa_consistent"]) if spa_pooled["dir"] else "--",
        "rsSpaDirPooledPre":  _fmt(spa_pooled["dir_pre"]["spa_consistent"]) if spa_pooled["dir_pre"] else "--",
        "rsMcsSize":       str(len(mcs_in)),
        "rsMcsUniverse":   str(len(mcs_universe)),
        "rsMcsExcluded":   pretty_excluded,
        "rsMcsVzeroIn":    "is" if "V0_w252" in mcs_in else "is not",
        # closed-loop measurement problem
        "rsRewardMean": _fmt(mp.get("reward_mean", float("nan"))),
        "rsRewardStd":  _fmt(mp.get("reward_std", float("nan"))),
        "rsRewardSNR":  _fmt(mp.get("reward_snr", float("nan")), 2),
        "rsSharpeSE":   _fmt(mp.get("sharpe_se_window", float("nan")), 2),
        # full-sample and contrast-level precision
        "rsSharpeSEFull":      _fmt(se_full, 2),
        "rsContrastSETotal":   _fmt(se_total, 3),
        "rsContrastMDETotal":  _fmt(mde_total, 3),
        "rsContrastSEOrient":  _fmt(se_orient, 3),
        "rsContrastMDEOrient": _fmt(mde_orient, 3),
        **seed_macros,
    }
    write_macros(GEN / "robust_stats.tex", macros)
    print(f"\nsaved → {out_csv}")
    print(f"saved → {GEN}/robust_stats.tex (report macros, unified battery)")
    if GEN_NO_HSP.parent.is_dir():
        # rsMcsExcluded names an HSP allocator, unused in the no-HSP report,
        # so it is emitted empty there.
        write_macros(GEN_NO_HSP / "robust_stats.tex",
                     {**macros, "rsMcsExcluded": ""})
        print(f"saved → {GEN_NO_HSP}/robust_stats.tex (no-HSP variant)")


if __name__ == "__main__":
    main()
