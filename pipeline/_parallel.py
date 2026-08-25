"""Shared rolling-window execution: parallelism, progress logging, checkpointing.

Progress is logged from the parent process because joblib workers don't
inherit logging handlers. Completed windows are pickled so runs can resume.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Callable, Sequence

Job = tuple[int, int, int]  # (window index, start row, end row)


def execute_windows(
    jobs: Sequence[Job],
    call_fn: Callable[[Job], object],
    n_jobs: int,
    label: str,
    *,
    checkpoint_dir: str | Path | None = None,
    log_every: int = 1,
) -> list:
    """Run ``call_fn`` over every job; returns results sorted by ``.index``.

    Each result must carry an integer ``.index``. Checkpoints are keyed only
    by ``label`` + index, so use a fresh ``checkpoint_dir`` when windowing or
    algorithm parameters change.
    """
    log = logging.getLogger(f"pipeline.{label}")
    total = len(jobs)
    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)

    def _ckpt_path(index: int) -> Path:
        return ckpt / f"{label}_window_{index:04d}.pkl"  # type: ignore[union-attr]

    # Partition into already-checkpointed and to-do
    done: list = []
    todo: list[Job] = []
    for job in jobs:
        index = job[0]
        if ckpt is not None and _ckpt_path(index).exists():
            with open(_ckpt_path(index), "rb") as fh:
                done.append(pickle.load(fh))
        else:
            todo.append(job)

    if done:
        log.info(
            "%s: resuming -- %d/%d windows already checkpointed",
            label, len(done), total,
        )
    if not todo:
        return sorted(done, key=lambda r: r.index)

    # Run the remaining jobs, recording each as it completes
    results: list = list(done)
    started = time.time()
    n_complete = len(done)

    def _record(result) -> None:
        nonlocal n_complete
        n_complete += 1
        if ckpt is not None:
            with open(_ckpt_path(result.index), "wb") as fh:
                pickle.dump(result, fh)
        results.append(result)
        if n_complete % log_every == 0 or n_complete == total:
            elapsed = (time.time() - started) / 60
            computed = n_complete - len(done)
            eta = (elapsed / computed) * (total - n_complete) if computed else 0.0
            log.info(
                "%s: %d/%d windows (%.1f min elapsed, ~%.1f min remaining)",
                label, n_complete, total, elapsed, eta,
            )

    if n_jobs == 1:
        for job in todo:
            _record(call_fn(job))
    else:
        from joblib import Parallel, delayed

        stream = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(call_fn)(job) for job in todo
        )
        for result in stream:
            _record(result)

    return sorted(results, key=lambda r: r.index)
