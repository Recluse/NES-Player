"""How much of the label's variation does the representation hide?

Two states that look identical to the network can still carry different
answers — because the label is a Monte Carlo estimate and noisy, or because
the representation is missing something. Those are different diagnoses with
different fixes, and argmax disagreement rates cannot separate them: the
difference of two disagreement probabilities is not a decomposition of
anything. Variances are.

Write `delta_i,r` for the jump-minus-defer value at state `i`, estimated by
repeat `r` of sixteen draws. With `delta_i,r = delta_i + e_i,r`:

    same state, two repeats     E[(d_i1 - d_i2)^2] / 2  =  var_MC
    two states that share z     E[(d_i1 - d_j1)^2] / 2  =  var_MC
                                                         + (delta_i-delta_j)^2/2

so the excess of the second over the first estimates the variation the
representation is hiding. Reported as a root-mean-square in pixels, next to
the spread between candidates, which is what a chooser has to resolve.

The closeness threshold is a percentile of the pairwise distance distribution,
fixed by the geometry of the inputs and computed before any label is read.

    uv run python scripts/experiments/variance.py \
        runs/knowledge/draw_matrix_rep.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MAX_PAIRS = 200_000


def _delta(z, rep: int) -> np.ndarray:
    """Jump-minus-defer, per point, from one repeat, death made terminal."""
    from analyse_draws import DEATH_MARGIN

    names = [str(n) for n in z["names"]]
    bc_i, jump_i = names.index("bc"), names.index("jump now")
    r = z["returns"][:, rep].astype(np.float64).copy()      # (P, C, N)
    d = z["died"][:, rep]
    for p in range(len(r)):
        live = r[p][~d[p]]
        floor = (live.min() if live.size else 0.0) - DEATH_MARGIN
        r[p][d[p]] = floor
    return r[:, jump_i, :].mean(1) - r[:, bc_i, :].mean(1), r


def main() -> int:
    from plan_probe import features

    ap = argparse.ArgumentParser()
    ap.add_argument("matrix")
    ap.add_argument("--variants", nargs="+",
                    default=["strip", "oam", "privileged"])
    ap.add_argument("--close-pct", type=float, default=0.5,
                    help="percentile of the pairwise distance distribution "
                         "below which two states count as sharing z; fixed "
                         "from the inputs alone, before any label is read")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    z = np.load(args.matrix)
    if z["returns"].ndim != 4 or z["returns"].shape[1] < 2:
        raise SystemExit("this matrix has one repeat; collect with --repeats 2")

    d0, r0 = _delta(z, 0)
    d1, r1 = _delta(z, 1)
    # A decision where jumping and deferring produce byte-identical futures is
    # not a decision. Under common random numbers those come out exactly equal,
    # with zero Monte Carlo noise, and they would otherwise dominate any set of
    # "states that look alike" and drive the estimate to nonsense.
    tie = ((r0[:, 2] == r0[:, 0]).all(1) & (r1[:, 2] == r1[:, 0]).all(1))
    print(f"{int(tie.sum())} of {len(tie)} decisions are exact ties "
          f"({tie.mean():.0%}) and are dropped")
    keep_pt = ~tie
    d0, d1, r0 = d0[keep_pt], d1[keep_pt], r0[keep_pt]
    n = len(d0)
    var_mc = float(((d0 - d1) ** 2).mean() / 2.0)
    spread = float(r0.mean(2).std(1).mean())
    print(f"{n} states, two independent sixteen-draw labels each")
    print(f"  Monte Carlo sd of the label      {np.sqrt(var_mc):7.1f} px")
    print(f"  spread between candidates        {spread:7.1f} px")
    print(f"  sd of delta across all states    {d0.std():7.1f} px")
    print()

    for variant in args.variants:
        imgs, vecs = features(z, variant)
        imgs, vecs = imgs[keep_pt], vecs[keep_pt]
        parts = [vecs.astype(np.float64)]
        if variant != "privileged":
            parts.append(imgs.reshape(n, -1).astype(np.float64) / 255.0)
        f = np.concatenate(parts, axis=1)
        f = (f - f.mean(0)) / (f.std(0) + 1e-9)
        sq = (f ** 2).sum(1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * f @ f.T, 0.0))
        iu = np.triu_indices(n, 1)
        dv = dist[iu]
        if len(dv) > MAX_PAIRS:
            keep = rng.choice(len(dv), MAX_PAIRS, replace=False)
        else:
            keep = np.arange(len(dv))
        thr = float(np.percentile(dv[keep], args.close_pct))
        a, b = iu[0][keep], iu[1][keep]
        near = dv[keep] <= thr
        far = dv[keep] >= np.percentile(dv[keep], 50.0)

        def half_sq(mask):
            if not mask.any():
                return float("nan")
            return float((((d0[a[mask]] - d0[b[mask]]) ** 2) / 2.0).mean())

        v_near, v_far = half_sq(near), half_sq(far)
        hidden = max(v_near - var_mc, 0.0)
        print(f"{variant}: threshold at the {args.close_pct}th percentile "
              f"= {thr:.2f}, {int(near.sum())} close pairs")
        print(f"  close pairs, half squared difference   "
              f"{np.sqrt(v_near):7.1f} px")
        print(f"  hidden by the representation (excess)  "
              f"{np.sqrt(hidden):7.1f} px")
        print(f"  distant pairs, for reference           "
              f"{np.sqrt(v_far):7.1f} px")
        print(f"  hidden / candidate spread              "
              f"{np.sqrt(hidden) / max(spread, 1e-9):7.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
