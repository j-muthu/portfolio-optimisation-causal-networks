"""Graph analysis and regime detection, method-agnostic.

Consumes the per-window adjacency matrices from either rolling method (same
``i -> j`` convention) and computes density, inter-window distance, sector
flow, causal-order drift, and head-to-head comparisons.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Single-matrix metrics
def graph_density(matrix: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of possible directed edges that are present.

    Self-loops are excluded, so the denominator is ``d * (d - 1)``.
    """
    d = matrix.shape[0]
    if d < 2:
        return 0.0
    n_edges = int(np.count_nonzero(np.abs(matrix) > threshold))
    diag = int(np.count_nonzero(np.abs(np.diag(matrix)) > threshold))
    return (n_edges - diag) / (d * (d - 1))


def average_edge_weight(matrix: np.ndarray, threshold: float = 0.0) -> float:
    """Mean absolute weight over the *present* edges (0.0 if the graph is empty)."""
    weights = np.abs(matrix[np.abs(matrix) > threshold])
    return float(weights.mean()) if weights.size else 0.0


def graph_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius norm of ``a - b`` -- the structural distance between two graphs."""
    return float(np.linalg.norm(a - b, "fro"))


def edge_jaccard(a: np.ndarray, b: np.ndarray, threshold: float = 0.0) -> float:
    """Jaccard overlap of the edge *sets* of two graphs (ignores weights/signs)."""
    ea = np.abs(a) > threshold
    eb = np.abs(b) > threshold
    union = np.count_nonzero(ea | eb)
    return float(np.count_nonzero(ea & eb) / union) if union else 1.0


# Result adapters
def _extract_sequence(result) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """Pull ``(matrices, dates, columns)`` from either rolling result type."""
    if hasattr(result, "w_stack"):
        return result.w_stack(), result.dates, result.columns
    if hasattr(result, "b0_stack"):
        return result.b0_stack(), result.dates, result.columns
    raise TypeError(f"unrecognised rolling result: {type(result)!r}")


# Time-series of metrics + regime detection
def analyse_rolling(result, threshold: float = 0.0) -> pd.DataFrame:
    """Per-window graph metrics for a rolling result.

    Returns a DataFrame indexed by window end-date with columns:
    ``n_edges``, ``density``, ``avg_weight`` and ``distance_prev`` (Frobenius
    distance to the previous window's graph; ``NaN`` for the first window).
    """
    matrices, dates, _ = _extract_sequence(result)
    rows = []
    for i, m in enumerate(matrices):
        rows.append(
            {
                "end_date": dates[i],
                "n_edges": int(np.count_nonzero(np.abs(m) > threshold)),
                "density": graph_density(m, threshold),
                "avg_weight": average_edge_weight(m, threshold),
                "distance_prev": (
                    graph_distance(m, matrices[i - 1]) if i > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("end_date")


def detect_regime_changes(
    metrics: pd.DataFrame, n_sigma: float = 2.0
) -> pd.DataFrame:
    """Flag windows whose graph jumped abnormally far from the previous one.

    A window is a candidate regime change when ``distance_prev`` exceeds
    ``mean + n_sigma * std`` of all consecutive distances. Returns the
    flagged subset of ``metrics`` with an added ``z_score`` column.
    """
    dist = metrics["distance_prev"].dropna()
    if dist.empty:
        return metrics.iloc[0:0].assign(z_score=pd.Series(dtype=float))
    mu, sigma = dist.mean(), dist.std(ddof=0)
    z = (metrics["distance_prev"] - mu) / (sigma if sigma > 0 else 1.0)
    flagged = metrics.assign(z_score=z)
    flagged = flagged[flagged["z_score"] > n_sigma]
    logger.info(
        "Detected %d candidate regime changes (>%.1f sigma): %s",
        len(flagged), n_sigma, [d.date() for d in flagged.index],
    )
    return flagged


# Sector-level causal flow
def sector_flow(
    matrix: np.ndarray,
    columns: list[str],
    sectors: dict[str, str],
    normalise: bool = True,
) -> pd.DataFrame:
    """Aggregate edge weights into a sector-by-sector causal-flow matrix.

    Entry ``[s_from, s_to]`` is the total absolute weight from one sector to
    another; ``normalise=True`` divides by the number of ordered asset pairs.
    """
    labels = [sectors.get(c, "Unknown") for c in columns]
    unique = sorted(set(labels))
    idx = {s: k for k, s in enumerate(unique)}
    label_idx = np.array([idx[s] for s in labels])

    flow = np.zeros((len(unique), len(unique)))
    counts = np.zeros((len(unique), len(unique)))
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i == j:
                continue
            si, sj = label_idx[i], label_idx[j]
            flow[si, sj] += abs(matrix[i, j])
            counts[si, sj] += 1

    if normalise:
        flow = np.divide(flow, counts, out=np.zeros_like(flow), where=counts > 0)
    return pd.DataFrame(flow, index=unique, columns=unique)


# VARLiNGAM-specific: causal-order drift
def causal_order_drift(result) -> pd.DataFrame:
    """Track how VARLiNGAM's discovered causal order shifts across windows.

    Kendall's tau between consecutive rank vectors; a sharp drop signals a
    re-shuffling of which assets are causally upstream. Returns a DataFrame
    with ``kendall_tau`` and ``n_position_changes`` per window.
    """
    from scipy.stats import kendalltau

    if not hasattr(result, "windows") or not result.windows:
        raise TypeError("causal_order_drift expects a RollingVarLingamResult")
    if not hasattr(result.windows[0], "causal_order"):
        raise TypeError("causal_order_drift only applies to VARLiNGAM results")

    d = len(result.columns)

    def _ranks(order: list[int]) -> np.ndarray:
        rank = np.empty(d, dtype=int)
        for pos, var in enumerate(order):
            rank[var] = pos
        return rank

    rows = []
    prev = None
    for w in result.windows:
        ranks = _ranks(w.causal_order)
        if prev is None:
            tau, changes = np.nan, np.nan
        else:
            tau = kendalltau(prev, ranks).statistic
            changes = int(np.count_nonzero(prev != ranks))
        rows.append(
            {"end_date": w.end_date, "kendall_tau": tau, "n_position_changes": changes}
        )
        prev = ranks
    return pd.DataFrame(rows).set_index("end_date")


# Head-to-head graph comparison
def compare_graphs(
    a: np.ndarray, b: np.ndarray, threshold: float = 0.0
) -> dict[str, float]:
    """Compare two adjacency matrices (same convention, same asset ordering).

    Returns Frobenius distance, edge-set Jaccard, sign agreement on common
    edges, and weight correlation over edges present in either graph.
    """
    ea = np.abs(a) > threshold
    eb = np.abs(b) > threshold
    common = ea & eb
    sign_agree = (
        float(np.mean(np.sign(a[common]) == np.sign(b[common])))
        if np.any(common)
        else np.nan
    )
    either = ea | eb
    if np.count_nonzero(either) > 1:
        weight_corr = float(np.corrcoef(a[either], b[either])[0, 1])
    else:
        weight_corr = np.nan
    return {
        "frobenius_distance": graph_distance(a, b),
        "edge_jaccard": edge_jaccard(a, b, threshold),
        "sign_agreement": sign_agree,
        "weight_correlation": weight_corr,
        "n_edges_a": int(np.count_nonzero(ea)),
        "n_edges_b": int(np.count_nonzero(eb)),
    }


def compare_rolling(dyn_result, var_result, threshold: float = 0.0) -> pd.DataFrame:
    """Window-by-window comparison of a DYNOTEARS run against a VARLiNGAM run.

    Both runs must share windows and asset ordering. One row per window.
    """
    w_stack, dates, _ = _extract_sequence(dyn_result)
    b_stack, _, _ = _extract_sequence(var_result)
    if len(w_stack) != len(b_stack):
        raise ValueError(
            f"window count mismatch: DYNOTEARS={len(w_stack)} VARLiNGAM={len(b_stack)}"
        )
    rows = []
    for i in range(len(w_stack)):
        rec = {"end_date": dates[i]}
        rec.update(compare_graphs(w_stack[i], b_stack[i], threshold))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("end_date")


# Plotting
def plot_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path,
    title: str = "Rolling causal-graph metrics",
    regime_changes: pd.DataFrame | None = None,
) -> Path:
    """Plot density, average weight and inter-window distance over time.

    Flagged ``regime_changes`` windows get vertical lines. Returns the path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    panels = [
        ("density", "Graph density"),
        ("avg_weight", "Avg. edge weight"),
        ("distance_prev", "Distance to prev. window"),
    ]
    for ax, (col, label) in zip(axes, panels):
        ax.plot(metrics.index, metrics[col], marker="o", ms=3, lw=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if regime_changes is not None:
            for date in regime_changes.index:
                ax.axvline(date, color="crimson", ls="--", alpha=0.6, lw=1)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Window end date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    logger.info("Saved metrics plot to %s", output_path)
    return output_path


# VARLiNGAM HSIC residual-independence diagnostics
def summarise_error_independence(window) -> dict:
    """Summarise a window's HSIC p-value matrix as scalars.

    ``rejection_rate`` is the fraction of off-diagonal p-values < 0.05;
    much above 0.05 means LiNGAM is misspecified for this window. Empty dict
    if the window has no ``error_indep_pvalues``.
    """
    pvalues = getattr(window, "error_indep_pvalues", None)
    if pvalues is None:
        return {}
    pvalues = np.asarray(pvalues)
    triu = np.triu_indices_from(pvalues, k=1)
    off_diag = pvalues[triu]
    return {
        "rejection_rate": float((off_diag < 0.05).mean()),
        "min_pvalue": float(off_diag.min()),
        "median_pvalue": float(np.median(off_diag)),
        "n_pairs": int(len(off_diag)),
    }


def summarise_error_independence_panel(result) -> pd.DataFrame:
    """Aggregate :func:`summarise_error_independence` across a rolling result.

    One row per window with HSIC p-values populated; skipped windows omitted.
    """
    rows = []
    for w in result.windows:
        s = summarise_error_independence(w)
        if not s:
            continue
        rows.append({"end_date": w.end_date, **s})
    return pd.DataFrame(rows).set_index("end_date") if rows else pd.DataFrame()
