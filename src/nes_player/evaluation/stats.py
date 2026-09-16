"""The three tests every number in this project is quoted with.

They were written out by hand in eight scripts — `confirm.py`, `soft_choice.py`,
`curve.py`, `battery.py`, `gate.py`, `plan_probe.py`, `analyse_draws.py`,
`variance.py` — each slightly differently. Of the error classes this repository
has retracted for, a wrong interval is the only one that neither a frame nor a
scripted check can catch: it looks exactly like a right one. So they live here
once, with a test that has answers known in advance.

Everything is paired: the arms are run on the same evaluation seeds and the
difference is taken seed by seed, which is what makes intervals this narrow
possible at all.
"""

from __future__ import annotations

from math import comb, sqrt

import numpy as np

BOOTSTRAP = 20000
PERMUTATIONS = 20000

# Two-sided t at 95%, by degrees of freedom. Small-sample work here rarely goes
# past a dozen training runs, and pulling in scipy for one lookup is not worth
# the dependency.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 15: 2.131,
       20: 2.086, 30: 2.042}


def _t95(df: int) -> float:
    if df in T95:
        return T95[df]
    known = sorted(k for k in T95 if k <= df)
    return T95[known[-1]] if known else 12.706


def paired_bootstrap(d: np.ndarray, n: int = BOOTSTRAP,
                     seed: int = 0) -> tuple[float, float, float, int]:
    """Mean difference, its 95% percentile interval, and pairs won.

    `d` is one difference per pair, already subtracted. Resampling pairs rather
    than the two arms separately is the whole point: it keeps the pairing.
    """
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), int((d > 0).sum())


def permutation_p(d: np.ndarray, n: int = PERMUTATIONS, seed: int = 0) -> float:
    """Two-sided paired permutation test: flip the sign of each difference.

    The +1 in both places is deliberate — it counts the observed arrangement,
    so the p-value can never be zero, which it is not.
    """
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float(((np.abs((signs * d).mean(1)) >= obs).sum() + 1) / (n + 1))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar on paired yes/no outcomes, two-sided.

    Only the seeds where the two arms disagree carry information; the ones
    where both cleared or both died say nothing about which is better.
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def t_interval(x: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% t interval, for the handful of training runs per arm.

    A bootstrap over six numbers resamples the same six values and reports an
    interval narrower than the data support; with this few observations the t
    interval is the honest one.
    """
    x = np.asarray(x, float)
    se = x.std(ddof=1) / sqrt(len(x))
    half = _t95(len(x) - 1) * se
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


def detectable(sd: float, runs: int) -> float:
    """Smallest effect a paired comparison of this many runs can resolve.

    Worth computing before a campaign rather than after it: an effect below
    this line was unmeasurable from the start, which is a different statement
    from "did not replicate" and saves the machine time.
    """
    return float(_t95(runs - 1) * sd / sqrt(runs))


def variance_split(per_run: list[np.ndarray]) -> dict[str, float]:
    """Split run-to-run spread into evaluation noise and training noise.

    `per_run` is one array of per-seed differences per training run. The spread
    between runs mixes two independent things: the training lottery, and the
    finite evaluation sample. They call for opposite remedies — more training
    runs against the first, more evaluation seeds against the second — so
    buying either without splitting them first is guesswork.
    """
    means = np.array([np.mean(d) for d in per_run], float)
    ev = float(np.mean([np.var(d, ddof=1) / len(d) for d in per_run]))
    total = float(means.var(ddof=1))
    # With a handful of runs the difference of two variance estimates can come
    # out negative, which means the training lottery is too small to separate
    # from the evaluation sample rather than that it is less than nothing. The
    # share is capped at 1 so it stays a share; train_noise then reads zero.
    return {
        "mean": float(means.mean()),
        "sd": sqrt(total),
        "eval_noise": sqrt(ev),
        "train_noise": sqrt(max(total - ev, 0.0)),
        "eval_share": min(1.0, ev / total) if total > 0 else 1.0,
    }
