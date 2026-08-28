"""Did the cheap label trade quantity for quality?

Eighty thousand decisions were collected at four draws because that costs 848
emulator frames against 9360. Before any conclusion rests on them, the label
itself needs measuring: on the six-candidate task, four draws agreed with
sixteen on only 69.1% of top-1 picks. A binary target should be steadier, but
that is a prediction, not a measurement.

Everything here is offline, from `draw_matrix_rep.npz` — 300 states, each
labelled twice by an independent sixteen-draw matrix. Panels of four are cut
from those sixteen, so a four-draw label can be compared against another
four-draw label on the same state, and against an independent sixteen.

    uv run python scripts/experiments/label_audit.py \
        runs/knowledge/draw_matrix_rep.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def delta(z, rep: int, idx) -> np.ndarray:
    """Jump minus defer, from the given draws of one repeat."""
    from analyse_draws import DEATH_MARGIN

    names = [str(n) for n in z["names"]]
    bc_i, jump_i = names.index("bc"), names.index("jump now")
    r = z["returns"][:, rep].astype(np.float64).copy()
    d = z["died"][:, rep]
    for p in range(len(r)):
        live = r[p][~d[p]]
        floor = (live.min() if live.size else 0.0) - DEATH_MARGIN
        r[p][d[p]] = floor
    return r[:, jump_i, idx].mean(1) - r[:, bc_i, idx].mean(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    z = np.load(args.matrix)
    n_draw = z["returns"].shape[3]
    k = args.draws
    if 2 * k > n_draw:
        raise SystemExit(f"need {2 * k} draws, matrix has {n_draw}")

    ref = delta(z, 1, np.arange(n_draw))          # independent 16-draw truth
    perm = rng.permutation(n_draw)
    a = delta(z, 0, perm[:k])
    b = delta(z, 0, perm[k:2 * k])
    full0 = delta(z, 0, np.arange(n_draw))

    # A decision where the two candidates are identical is not a decision; it
    # would otherwise count as perfect agreement and flatter every number here.
    tie = (a == 0) & (b == 0) & (ref == 0)
    live = ~tie
    print(f"{len(ref)} states, {int(tie.sum())} exact ties dropped, "
          f"{int(live.sum())} left")

    def agree(u, v):
        return float((np.sign(u[live]) == np.sign(v[live])).mean())

    print()
    print(f"sign of delta agrees, {k} draws vs another {k} on the same state: "
          f"{agree(a, b):.3f}")
    print(f"sign of delta agrees, {k} draws vs an independent 16:            "
          f"{agree(a, ref):.3f}")
    print(f"sign of delta agrees, 16 draws vs an independent 16:            "
          f"{agree(full0, ref):.3f}")

    def auc(score, pos):
        x, y = score[pos], score[~pos]
        if not len(x) or not len(y):
            return float("nan")
        gt = (x[:, None] > y[None, :]).astype(float)
        gt += 0.5 * (x[:, None] == y[None, :])
        return float(gt.mean())

    pos = ref[live] > 0
    print()
    print(f"as a ranking of the independent 16-draw value:")
    print(f"  {k}-draw label   AUC {auc(a[live], pos):.3f}")
    print(f"  16-draw label  AUC {auc(full0[live], pos):.3f}")

    print()
    sd_mc = float(((a - b) ** 2)[live].mean() / 2) ** 0.5
    print(f"sd of a {k}-draw delta, from two panels on one state: {sd_mc:6.1f} px")
    print(f"sd of delta across states (16-draw):                  "
          f"{ref[live].std():6.1f} px")
    for lim in (5, 10, 20):
        print(f"  |delta| below {lim:2d} px in {float((np.abs(ref[live]) < lim).mean()):.0%}"
              f" of live decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
