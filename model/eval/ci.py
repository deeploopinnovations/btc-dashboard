"""
eval/ci.py
=====================================================================
`mean_ci`, moved here VERBATIM from eval/direction.py so code that needs only
the interval -- the forward-holdout scorers run by a scheduled workflow on
serving's dependencies -- does not import the benchmark (torch, sklearn).
eval/direction.py re-exports it, so every existing import is unchanged.
"""
from __future__ import annotations

import numpy as np


def mean_ci(d, n_rep: int = 20_000, seed: int = 0, alpha: float = 0.05,
            block_len: int | None = None) -> dict:
    """A CI for the mean that is DEFINED at every usable n, plus the diagnostics
    that matter when n is small.

    `block_bootstrap_ci` returns (nan, nan) below n = 20. That guard is correct
    for its intended argument -- a per-episode loss difference with fewer than
    20 points means the caller made a mistake -- but it is a trap for any
    caller whose unit is a FOLD. `eval/anchor_freshness.py` fed it six yearly
    deltas, got (nan, nan), and its pre-registered condition

        spike QLIKE CI excludes zero favourably

    evaluated to False -- because `nan < 0` is False, not because the data said
    so. The verdict happened to be right; the mechanism that produced it was
    not looking at the data at all. A rule decided by a NaN is not a rule.

    Three numbers are returned rather than one, because at n = 6 the choice of
    estimator is a real degree of freedom and hiding it would be a way to pick
    the answer:

      ci95        the moving-block interval, blocks of round(n^(1/3)) unless
                  `block_len` overrides it. This is the PRE-REGISTERED
                  estimator and the one a verdict uses.

    `block_len` exists because n^(1/3) is a rule of thumb about SAMPLE SIZE and
    knows nothing about the dependence it is supposed to absorb. When the unit
    is an episode drawn from an hourly-anchored table with an H-hour forward
    window, consecutive observations share H-1 of their H hours, and the
    dependence range is H regardless of n. At H = 168 and n = 49,000 the rule of
    thumb gives 37 -- roughly a fifth of the overlap -- and the resulting
    interval is too narrow for the reason that matters: it is treating four
    fifths of a shared window as independent evidence. Callers that know their
    dependence range should say so.
      ci95_iid    the ordinary bootstrap. For yearly folds separated by an
                  embargo the dependence the blocks model is largely absent,
                  so the block interval is if anything conservative; when the
                  two disagree in sign of coverage, say so rather than picking.
      n_favourable / n  the sign count, which no bootstrap can launder.
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        raise ValueError(f"mean_ci needs at least 2 finite observations, got {n}")
    L = max(1, int(round(n ** (1 / 3))) if block_len is None else int(block_len))
    L = min(L, n)
    nb = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)

    # CHUNKED. The obvious vectorisation builds an (n_rep, n) index array, which
    # is 20,000 x 49,118 = 982M int64 = 7.9 GB and killed the direction
    # benchmark with SIGKILL. It was fine at n = 1,681 (0.3 GB) and became a
    # problem the moment STATS_PROTOCOL made the paired per-episode estimator
    # primary -- i.e. the fix that improved the statistics moved this onto the
    # hot path. Chunking bounds peak memory at CHUNK x n regardless of n_rep.
    chunk = max(1, min(n_rep, int(2e7 // max(n, 1)) or 1))
    blk = np.empty(n_rep, dtype=np.float64)
    iid = np.empty(n_rep, dtype=np.float64)
    off = np.arange(L)[None, None, :]
    done = 0
    while done < n_rep:
        m = min(chunk, n_rep - done)
        starts = rng.integers(0, n - L + 1, size=(m, nb))
        idx = (starts[:, :, None] + off).reshape(m, -1)[:, :n]
        blk[done:done + m] = d[idx].mean(axis=1)
        iid[done:done + m] = d[rng.integers(0, n, size=(m, n))].mean(axis=1)
        done += m
    q = lambda a: (float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2)))
    # NOTE the key name: `ci95` carries whatever `alpha` was requested, so at
    # alpha = 0.025 it is a 97.5% interval despite the name. `level` is
    # returned alongside precisely so a caller reporting a Bonferroni-adjusted
    # interval cannot label it 95% by copying the key.
    return {"mean": float(d.mean()), "n": n, "block_len": L,
            "level": float(1.0 - alpha),
            "ci95": list(q(blk)), "ci95_iid": list(q(iid)),
            "n_negative": int((d < 0).sum()), "n_positive": int((d > 0).sum()),
            "sd": float(d.std(ddof=1)) if n > 1 else float("nan")}
