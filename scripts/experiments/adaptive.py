"""Adaptive draw allocation, calibrated offline on the stored draw matrices.

The scheme under test (the reviewer's): every candidate always gets two
paired rollouts; from those the expected regret of stopping is estimated;
only where it is high are the third and fourth rollouts paid for — for all
six candidates, so nothing is ever pruned and bc in particular cannot be cut.

What the diagnostics on the stored matrices established before this file
took its final form (kept because each step changed the design):

  * a Gaussian expected-regret trigger with a GLOBAL calibrated sigma is
    anti-informative — its top-25% "riskiest" points carry 19-36% of the
    real 2->4 saving, worse than random escalation.  The noise is strongly
    heteroscedastic and one global sigma washes out exactly the signal.
  * the winner-death trigger is empty: deaths are already priced into the
    penalised returns (floor = worst live - DEATH_MARGIN).
  * the two components that do work are the reviewer's other two: winner
    instability between the paired draws (escalating the ~43% of flip
    points captures ~70-75% of the saving), and the point's OWN observed
    draw disagreement.

The trigger used here is therefore the hybrid: escalate every point whose
two draws disagree on the winner; rank the stable points by expected
stopping regret with a PER-POINT sigma — the fifteen pairwise d0-d1
disagreements at that point, shrunk toward the train-levels global
(k_local=15, k0=6).  Expected regret per rival with deficit delta:

    E[(q_c - q_w)^+] = s * phi(delta/s) + delta * Phi(delta/s),  summed.

On held-out levels this puts 47-58% of the saving into the top 25% of
escalations (ratio ~2), where the global-sigma version managed 0.2-0.4.

Two evaluations:

1. Battery matrices, six levels, leave-one-level-out: sigma and the
   threshold frozen on five levels, applied to the sixth.  Reference is the
   4-draw mean, so full escalation has zero regret by construction; the
   honest comparison is the straight line between uniform-2 and full
   (= random escalation at the same average cost).

2. draw_matrix_big (level 1-1, 16 draws): the procedure only sees draws
   0-3; the reference is the mean of draws 4-15, independent of every
   choice it makes.  Here uniform-2, full-4 and the adaptive rule all have
   real regret and the reviewer's win criterion applies directly: the
   adaptive point must fall below the straight line between the fixed
   budgets.  Sigma comes from the five non-1-1 levels.

    uv run python scripts/experiments/adaptive.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_draws import DEATH_MARGIN  # noqa: E402
from battery import LEVELS  # noqa: E402

ERF = np.vectorize(math.erf)
FRACS = (0.1, 0.25, 0.5, 0.75)
K0 = 6.0                    # shrinkage weight of the global sigma
FLIP = 1e6                  # rank offset that puts flips ahead of everything


def penalised(path):
    """(P, C, N) returns with deaths floored below the worst live draw."""
    z = np.load(path)
    r = z["returns"].astype(np.float64)
    d = z["died"]
    if r.ndim == 4:
        r, d = r[:, 0], d[:, 0]
    d = d.astype(bool)
    for p in range(len(r)):
        live = r[p][~d[p]]
        floor = (live.min() if live.size else 0.0) - DEATH_MARGIN
        r[p][d[p]] = floor
    return r


def pair_sigma2(r):
    """Pooled variance of a single-draw pairwise difference. r: (P, C, N)."""
    P, C, _ = r.shape
    tot, cnt = 0.0, 0
    for a in range(C):
        for b in range(a + 1, C):
            tot += (r[:, a] - r[:, b]).var(1, ddof=1).sum()
            cnt += P
    return tot / cnt


def local_sigma2(d2):
    """Per-point single-draw pair variance from the two paid draws.

    d2: (P, C, 2).  Each of the C(C-1)/2 pairs contributes one df via
    0.5*(D_ab(draw0) - D_ab(draw1))^2; averaged over pairs.
    """
    P, C, _ = d2.shape
    loc = np.zeros(P)
    for a in range(C):
        for b in range(a + 1, C):
            Dab = (d2[:, a, 0] - d2[:, b, 0]) - (d2[:, a, 1] - d2[:, b, 1])
            loc += 0.5 * Dab ** 2
    return loc / (C * (C - 1) / 2)


def trigger(d2, global_s2):
    """The escalation score from the two paid draws. d2: (P, C, 2)."""
    m2 = d2.mean(2)
    n, C = m2.shape
    k = C * (C - 1) / 2
    s = np.sqrt((k * local_sigma2(d2) + K0 * global_s2) / (k + K0) / 2)
    w = m2.argmax(1)
    delta = m2 - m2[np.arange(n), w][:, None]
    z = delta / s[:, None]
    es = s[:, None] * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) \
        + delta * 0.5 * (1.0 + ERF(z / math.sqrt(2)))
    es[np.arange(n), w] = 0.0
    flip = d2[:, :, 0].argmax(1) != d2[:, :, 1].argmax(1)
    return es.sum(1) + FLIP * flip


def points(r_proc, q_ref):
    """Per-point regrets of the 2-draw and 4-draw picks vs the reference."""
    n = len(q_ref)
    best = q_ref.max(1)
    m2 = r_proc[:, :, :2].mean(2)
    reg2 = best - q_ref[np.arange(n), m2.argmax(1)]
    reg4 = best - q_ref[np.arange(n), r_proc.mean(2).argmax(1)]
    return reg2, reg4


def row(mask, reg2, reg4, extra=None):
    f = float(mask.mean())
    save = reg2 - reg4
    out = {
        "frac": round(f, 3), "cost": round(12 + 12 * f, 1),
        "regret": round(float(np.where(mask, reg4, reg2).mean()), 2),
        "line": round(float((1 - f) * reg2.mean() + f * reg4.mean()), 2),
        "lift": round(float(save[mask].sum() / save.sum())
                      if save.sum() > 0 else 0.0, 3),
    }
    if extra:
        out.update(extra)
    return out


def sweep(stat, reg2, reg4):
    rows = []
    n = len(stat)
    for f in FRACS:
        mask = np.zeros(n, bool)
        mask[np.argsort(-stat)[:int(round(f * n))]] = True
        rows.append(row(mask, reg2, reg4))
    return rows


def main() -> int:
    r = {L: penalised(f"runs/knowledge/battery_{L}.npz") for L in LEVELS}

    # --- 1. leave-one-level-out on the battery matrices -------------------
    for hold in LEVELS:
        train = [L for L in LEVELS if L != hold]
        g2 = float(np.mean([pair_sigma2(r[L]) for L in train]))
        tr_stat = np.concatenate(
            [trigger(r[L][:, :, :2], g2) for L in train])
        taus = {f: float(np.quantile(tr_stat, 1 - f)) for f in FRACS}

        reg2, reg4 = points(r[hold], r[hold].mean(2))
        stat = trigger(r[hold][:, :, :2], g2)
        frozen = [row(stat > tau, reg2, reg4, {"target_frac": f})
                  for f, tau in taus.items()]
        print(json.dumps({
            "holdout": hold, "reg2": round(float(reg2.mean()), 2),
            "frozen_tau": frozen,
            "sweep": sweep(stat, reg2, reg4),
        }), flush=True)

    # --- 2. independent-reference validation on draw_matrix_big ----------
    big = penalised("runs/knowledge/draw_matrix_big.npz")
    g2 = float(np.mean(
        [pair_sigma2(r[L]) for L in LEVELS if L != "default"]))
    proc, ref = big[:, :, :4], big[:, :, 4:].mean(2)
    reg2, reg4 = points(proc, ref)
    stat = trigger(proc[:, :, :2], g2)
    tr_stat = np.concatenate(
        [trigger(r[L][:, :, :2], g2) for L in LEVELS if L != "default"])
    frozen = [row(stat > float(np.quantile(tr_stat, 1 - f)), reg2, reg4,
                  {"target_frac": f}) for f in FRACS]
    print(json.dumps({
        "validation": "draw_matrix_big level 1-1, reference = draws 4..15",
        "uniform2": {"cost": 12, "regret": round(float(reg2.mean()), 2)},
        "full4": {"cost": 24, "regret": round(float(reg4.mean()), 2)},
        "frozen_tau": frozen,
        "sweep": sweep(stat, reg2, reg4),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
