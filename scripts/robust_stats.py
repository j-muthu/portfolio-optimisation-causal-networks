"""Robust-stats battery for the final report — the distribution- and
data-snooping-aware adjudication the second marker asked for.

For every evaluated configuration it computes, on the daily net-return series:

* the **Probabilistic Sharpe Ratio** (PSR) vs zero and vs the V0 baseline;
* the **Deflated Sharpe Ratio** (DSR), correcting for the multiplicity of the
  ~41 configurations tried (Bailey & Lopez de Prado);
* **White's Reality Check** and **Hansen's SPA** — does the best causal variant
  beat the correlation baseline once data-snooping is accounted for;
* the **Model Confidence Set** (Hansen, Lunde & Nason) over the variant universe,
  reporting which variants survive at the 90% level (and whether V0 does);
* the **closed-loop measurement problem**: the per-rebalance reward R[t] is a
  Sharpe estimated from a ~21-day window, so its sampling noise swamps its mean —
  quantified as the reward signal-to-noise ratio, the update-frequency face of the
  loop's inertness.

Inputs are the persisted Phase-I bundles (``results/<tag>/closed_loop.pkl``),
which are gitignored — so this script runs **locally** where the bundles live.
It reuses the bundle-reading convention of ``scripts/regime_analysis.py`` and the
metric helpers of ``pipeline.evaluation.metrics``.

Outputs:
  * ``results/robust_stats.csv``                     — the per-variant table;
  * ``final_report_lean/_generated/robust_stats.tex`` — \\newcommand macros that
    the report \\input's, so no number is transcribed by hand.

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
GEN = REPO / "final_report_lean" / "_generated"

# ----------------------------------------------------------------------------
# The configuration universe (the "trials" whose multiplicity the DSR corrects).
# Mirrors the tag convention used across scripts/{regime_analysis,collate_j4,
# plot_thesis_figures}.py. Missing bundles are skipped and logged, so this runs
# on whatever subset of the full grid is present.
# ----------------------------------------------------------------------------
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


# Phase-II direction-aware allocators (fixed-graph ablation). D0 duplicates
# V0prime byte-for-byte under DYNOTEARS — that is the designed replication
# check; drop_duplicate_configs removes it before SPA/MCS.
PHASE_II_ALLOCS = ("D0", "D0s", "D1", "D2", "D2s", "D3", "D4")
PHASE_II_METHODS = (("dynotears", "DYNO"), ("varlingam", "VARL"))


def _phase_ii_tags() -> dict[str, str]:
    """name -> bundle tag for every Phase-II configuration evaluated."""
    tags: dict[str, str] = {}
    for w in WINDOWS:
        for method, short in PHASE_II_METHODS:
            for a in PHASE_II_ALLOCS:
                tags[f"{short}-{a}_w{w}"] = f"phase_ii_{method}_{a}_w{w}"
    for a in ("D0", "D1", "D2", "D3"):      # E2 GRANGER arm (w252 only)
        tags[f"GRAN-{a}_w252"] = f"phase_ii_granger_{a}_w252"
    for tau in ("0.01", "0.05", "0.1"):     # E3 τ sweep (DYNO w252)
        for a in ("D0", "D2", "D3"):
            tags[f"DYNO-{a}_w252_tau{tau}"] = f"phase_ii_dynotears_{a}_w252_tau{tau}"
    return tags


# ============================================================================
# Bundle loading (impure)
# ============================================================================
def _load_backtest(tag: str):
    path = RESULTS / tag / "closed_loop.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)["backtest"]


def load_return_matrix(tags: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Load each tag's daily net returns into an aligned matrix.

    Returns ``(returns_df, loaded_names)`` where columns are configuration names
    and rows are the dates common to all loaded series (inner join).
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
    """Per-rebalance realised reward R[t] (excess Sharpe vs 1/N) for one config."""
    bt = _load_backtest(tag)
    if bt is None:
        return None
    return pd.Series(
        [r.holding_reward for r in bt.rebalances],
        index=pd.to_datetime([r.rebalance_date for r in bt.rebalances]),
    )


# ============================================================================
# Pure analysis functions (unit-tested without bundles)
# ============================================================================
def per_period_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    sigma = r.std(ddof=0)
    return float(r.mean() / sigma) if sigma > 1e-12 else 0.0


def drop_duplicate_configs(returns: pd.DataFrame, names: list[str]) -> list[str]:
    """Drop names whose daily-return series exactly equals an earlier-kept one.

    The closed loop is inert, so V2 ≡ V1 byte-for-byte; feeding both into SPA/MCS
    double-counts a single strategy and — worse — gives a zero differential
    variance that makes the studentised MCS divide by zero and abort. Keeping the
    first occurrence (V1 precedes V2 in the universe order) yields the set of
    *distinct* strategies, which is what the data-snooping tests should compare.
    """
    kept: list[str] = []
    for c in names:
        if not any(np.array_equal(returns[c].to_numpy(), returns[k].to_numpy())
                   for k in kept):
            kept.append(c)
    return kept


def psr_dsr_table(returns: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """PSR (vs 0 and vs ``baseline``) and DSR for every column of ``returns``.

    The DSR deflates against the full set of per-period trial Sharpes in
    ``returns`` — i.e. corrects for having tried this many configurations.
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
            block_size: int = 21, reps: int = 2000, seed: int = 42) -> dict:
    """White's Reality Check + Hansen's SPA: do any ``candidates`` beat the
    ``benchmark``? Loss = negative return (lower loss = higher return), so a low
    p-value means a candidate significantly out-returns the benchmark.

    Returns the SPA lower (~Reality Check), consistent and upper p-values. Falls
    back to a hand-rolled stationary-block Reality Check if ``arch`` is absent.
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
    """White (2000) Reality Check p-value via the Politis-Romano stationary block
    bootstrap on the max-over-candidates mean return differential vs benchmark.
    """
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
            block_size: int = 21, reps: int = 2000, seed: int = 42) -> tuple[list[str], pd.DataFrame]:
    """Model Confidence Set over ``configs`` at confidence ``1-size`` (loss =
    negative return). Returns ``(included_names, pvalue_frame)``.
    """
    from arch.bootstrap import MCS
    losses = -returns[configs]
    mcs = MCS(losses, size=size, block_size=block_size, reps=reps, seed=seed, method="R")
    mcs.compute()
    included = [c for c in configs if c in set(mcs.included)]
    return included, mcs.pvalues


def measurement_problem(reward: pd.Series, holding_days: int = 21) -> dict:
    """Quantify why a monthly reward is too noisy to learn from.

    R[t] is an (excess) Sharpe estimated over the holding window, so a single
    month carries one noisy draw. We report its empirical mean and std, the
    per-observation signal-to-noise ratio |mean|/std, and the theoretical
    sampling SE of a Sharpe estimated from ``holding_days`` observations
    (≈ sqrt(1/n) for small SR), the floor that noise cannot fall below.
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


# ============================================================================
# Macro emission for the report
# ============================================================================
def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else "NaN"


def write_macros(path: pathlib.Path, macros: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% AUTO-GENERATED by scripts/robust_stats.py — do not edit by hand.",
             "% Regenerate locally (bundles required): python -m scripts.robust_stats",
             ""]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    path.write_text("\n".join(lines) + "\n")


# ============================================================================
# Orchestration
# ============================================================================
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

    # --- PSR / DSR table over every trial (baseline = correlation V0 at w252) ---
    baseline = "V0_w252" if "V0_w252" in returns.columns else loaded[0]
    table = psr_dsr_table(returns, baseline=baseline)

    # --- SPA / Reality Check + MCS over the w252 headline universe vs V0 ---
    # Compare only DISTINCT strategies: V2 ≡ V1 (inert loop), and identical
    # columns break the studentised MCS, so drop exact duplicates first.
    headline252 = drop_duplicate_configs(
        returns, [f"{n}_w252" for n in HEADLINE if f"{n}_w252" in returns.columns])
    causal252 = [c for c in headline252 if not c.startswith("V0_")]
    spa = run_spa(returns, benchmark="V0_w252", candidates=causal252)

    # Universe for the MCS: the w252 headline, plus (in --phase-ii mode) every
    # distinct w252 D-variant, so the set answers "which strategies survive
    # jointly once the Phase-II arm is on the table".
    mcs_universe = list(headline252)
    spa_phase_ii = None
    spa_direction: dict[int, dict] = {}
    if args.phase_ii:
        d252 = [c for c in returns.columns
                if c.startswith(("DYNO-", "VARL-", "GRAN-")) and c.endswith("_w252")
                and "_tau" not in c]
        d252 = drop_duplicate_configs(returns, mcs_universe + d252)
        d_only = [c for c in d252 if c not in mcs_universe]
        if d_only:
            spa_phase_ii = run_spa(returns, benchmark="V0_w252", candidates=d_only)
        mcs_universe = d252
        # The family-wise adjudication of the Phase-II question itself: does
        # ANY direction-aware allocator beat its own symmetrised control D0
        # (≡ V0prime), per window? This is the fixed-graph direction effect
        # with the multiplicity of the whole direction-aware grid priced in.
        for w in WINDOWS:
            bench = f"V0prime_w{w}"
            cands = [c for c in returns.columns
                     if c.split("_")[0] in {f"{s}-D{v}" for s in ("DYNO", "VARL")
                                            for v in ("1", "2", "2s", "3", "4")}
                     and c.endswith(f"_w{w}") and "_tau" not in c]
            cands = [c for c in drop_duplicate_configs(returns, [bench] + cands)
                     if c != bench]
            if bench in returns.columns and cands:
                spa_direction[w] = run_spa(returns, benchmark=bench, candidates=cands)
    mcs_in, _ = run_mcs(returns, mcs_universe)
    # mark MCS membership on the table (only meaningful for the MCS universe)
    table["in_mcs90"] = [bool(c in mcs_in) if c in mcs_universe else ""
                         for c in table.config]
    RESULTS.mkdir(exist_ok=True)
    out_csv = RESULTS / ("robust_stats_phase_ii.csv" if args.phase_ii else "robust_stats.csv")
    table.to_csv(out_csv, index=False)
    print("\n=== PSR / DSR by configuration ===")
    print(table.to_string(index=False))
    print(f"\n=== SPA/RC (causal vs V0, w252) ===\n{spa}")
    if spa_phase_ii is not None:
        print(f"\n=== SPA/RC (Phase-II D-variants vs V0, w252) ===\n{spa_phase_ii}")
    for w, res in spa_direction.items():
        print(f"\n=== SPA/RC (direction-aware vs D0/V0prime, w{w}) ===\n{res}")
    print(f"\n=== MCS 90% set (w252 universe, n={len(mcs_universe)}) ===\n{mcs_in}")

    # --- measurement problem on the closed loop (V2/V1 w252) ---
    rtag = "phase_i_v2_w252" if (RESULTS / "phase_i_v2_w252").exists() else "phase_i_v1_w252"
    reward = load_reward_series(rtag)
    mp = measurement_problem(reward) if reward is not None else {}
    print(f"\n=== measurement problem ({rtag}) ===\n{mp}")

    # --- self-check: reproduce the headline Sharpes ---
    print("\n=== self-check: headline annualised Sharpe ===")
    for n in ("V0_w252", "V1-DYNOTEARS_w252", "V1-VARLiNGAM_w252", "V0prime_w252"):
        if n in returns.columns:
            print(f"  {n:24s} {annualised_sharpe(returns[n]):.4f}")

    # --- emit report macros (per named w252 variant + battery verdicts) ---
    if args.phase_ii:
        # Report macros stay frozen on the committed 41-trial Phase-I battery
        # until the report-reorientation stage; the Phase-II battery lives in
        # robust_stats_phase_ii.csv + FINDINGS.md only.
        print(f"\nsaved → {out_csv} (report macros NOT regenerated in --phase-ii mode)")
        return
    def cell(config: str, col: str) -> float:
        hit = table.loc[table.config == config, col]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    pretty_mcs = ", ".join(
        m.replace("_w252", "").replace("V0prime", "V0$'$") for m in mcs_in) or "none"
    macros = {
        "rsNtrials": str(n_trials),
        "rsKurtosis": _fmt(pooled_excess_kurtosis(returns), 1),
        # per-variant PSR (vs zero) and DSR (multiplicity-deflated), w252 headline
        "rsVzeroPSR":    _fmt(cell("V0_w252", "psr_vs_zero")),
        "rsVzeroDSR":    _fmt(cell("V0_w252", "dsr")),
        "rsVprimePSR":   _fmt(cell("V0prime_w252", "psr_vs_zero")),
        "rsVprimeDSR":   _fmt(cell("V0prime_w252", "dsr")),
        "rsVoneDynoPSR": _fmt(cell("V1-DYNOTEARS_w252", "psr_vs_zero")),
        "rsVoneDynoDSR": _fmt(cell("V1-DYNOTEARS_w252", "dsr")),
        "rsVoneVarPSR":  _fmt(cell("V1-VARLiNGAM_w252", "psr_vs_zero")),
        "rsVoneVarDSR":  _fmt(cell("V1-VARLiNGAM_w252", "dsr")),
        # data-snooping battery (best causal variant vs V0, w252)
        "rsSpaRC":         _fmt(spa["rc_lower"]),
        "rsSpaConsistent": _fmt(spa["spa_consistent"]),
        "rsMcsSize":       str(len(mcs_in)),
        "rsMcsMembers":    pretty_mcs,
        "rsMcsVzeroIn":    "is" if "V0_w252" in mcs_in else "is not",
        # closed-loop measurement problem
        "rsRewardMean": _fmt(mp.get("reward_mean", float("nan"))),
        "rsRewardStd":  _fmt(mp.get("reward_std", float("nan"))),
        "rsRewardSNR":  _fmt(mp.get("reward_snr", float("nan")), 2),
        "rsSharpeSE":   _fmt(mp.get("sharpe_se_window", float("nan")), 2),
    }
    write_macros(GEN / "robust_stats.tex", macros)
    print(f"\nsaved → {RESULTS}/robust_stats.csv")
    print(f"saved → {GEN}/robust_stats.tex (report macros)")


if __name__ == "__main__":
    main()
