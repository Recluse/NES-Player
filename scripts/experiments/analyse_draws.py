"""What the saved futures say, before anything is trained on them.

Reads a matrix from `draw_matrix.py` — sixteen returns per candidate per
decision — and answers offline the questions that used to cost an emulator
hour each:

* how repeatable a teacher of N draws is, for nested N built from the same
  rollouts, so N is not confounded with a fresh sample;
* whether common random numbers actually shrink the paired differences, which
  is the quantity a ranking uses;
* whether the greedy T=0 continuation's choice is a useful surrogate or just a
  different answer;
* what a risk-sensitive selection rule would pick, and what it costs.

Agreement on top-1 alone is misleading where candidates tie, so the top-*set*
— everything within a standard error of the best — is reported beside it.

    uv run python scripts/experiments/analyse_draws.py \
        runs/knowledge/draw_matrix_crn.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEATH_MARGIN = 200.0     # how far below the worst live return a death sits


def penalised(z) -> np.ndarray:
    """Returns with death made terminal, per point.

    A run that dies after 80 px is not worth 80 px. The floor is the worst
    surviving return in the point minus a margin, which says "worse than
    anything that lived" on the scale the numbers already use.
    """
    r = z["returns"].astype(np.float64).copy()      # (P, C, N)
    d = z["died"]
    for p in range(len(r)):
        live = r[p][~d[p]]
        floor = (live.min() if live.size else 0.0) - DEATH_MARGIN
        r[p][d[p]] = floor
    return r


def penalised_det(z, ref: np.ndarray) -> np.ndarray:
    """The greedy continuation's returns, on the same scale as `ref`."""
    r = z["det_returns"].astype(np.float64).copy()  # (P, C)
    d = z["det_died"]
    for p in range(len(r)):
        live = ref[p][~z["died"][p]]
        floor = (live.min() if live.size else 0.0) - DEATH_MARGIN
        r[p][d[p]] = floor
    return r


def top_set(vals: np.ndarray, err: np.ndarray) -> list[set]:
    """Candidates not separated from the best by one standard error."""
    out = []
    for v, e in zip(vals, err, strict=True):
        b = int(v.argmax())
        tol = e[b] + e
        out.append({i for i in range(len(v)) if v[i] >= v[b] - tol[i]})
    return out


def panels(r: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint panels of n draws each, from the same sixteen."""
    p, c, total = r.shape
    a = np.empty((p, c, n))
    b = np.empty((p, c, n))
    for i in range(p):
        idx = rng.permutation(total)
        a[i] = r[i][:, idx[:n]]
        b[i] = r[i][:, idx[n:2 * n]]
    return a, b


def repeatability(r: np.ndarray, truth: np.ndarray, rng) -> None:
    print(f"{'N':>3} {'top-1 agree':>12} {'top-set agree':>14} "
          f"{'sd of paired diff':>18} {'regret vs 16':>13} {'entropy':>8}")
    for n in (1, 2, 4, 8):
        if 2 * n > r.shape[2]:
            continue          # two disjoint panels of n do not fit
        a, b = panels(r, n, rng)
        ma, mb = a.mean(2), b.mean(2)
        pick_a, pick_b = ma.argmax(1), mb.argmax(1)
        agree = float((pick_a == pick_b).mean())
        ea = a.std(2, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(ma)
        eb = b.std(2, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mb)
        sets_a, sets_b = top_set(ma, ea), top_set(mb, eb)
        set_agree = float(np.mean([len(x & y) / len(x | y)
                                   for x, y in zip(sets_a, sets_b, strict=True)]))
        # What a ranking uses: the difference between two candidates, and how
        # much that difference moves between two independent panels.
        da = ma[:, :, None] - ma[:, None, :]
        db = mb[:, :, None] - mb[:, None, :]
        iu = np.triu_indices(r.shape[1], 1)
        spread = float((da - db)[:, iu[0], iu[1]].std())
        regret = float((truth.max(1)
                        - truth[np.arange(len(truth)), pick_a]).mean())
        # Soft target: how often this panel's own bootstrap names each
        # candidate best. Its entropy says how much of the label is a coin.
        boot = np.stack([a[:, :, rng.integers(0, n, n)].mean(2)
                         for _ in range(64)])
        p_i = np.stack([(boot.argmax(2) == k).mean(0)
                        for k in range(r.shape[1])], axis=1)
        ent = float((-(p_i * np.log(p_i + 1e-9)).sum(1)).mean())
        print(f"{n:>3} {agree:>11.1%} {set_agree:>13.1%} "
              f"{spread:>15.1f} px {regret:>10.1f} px {ent:>8.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix")
    ap.add_argument("--compare", default=None,
                    help="a second matrix over the same points, to price "
                         "common random numbers against independent draws")
    ap.add_argument("--subsets", action="store_true",
                    help="value of every subset of the five "
                         "templates, bc always available")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    z = np.load(args.matrix)
    names = [str(n) for n in z["names"]]
    r = penalised(z)
    truth = r.mean(2)                      # the best estimate available
    print(f"{args.matrix}: {r.shape[0]} points, {r.shape[2]} draws, "
          f"crn={bool(z['crn'])}")
    print(f"deaths in {(z['died'].any((1, 2))).mean():.1%} of points; "
          f"spread across candidates {truth.std(1).mean():.1f} px")
    print()
    repeatability(r, truth, rng)

    print()
    # Scoring a rule by the statistic it optimises is circular — the mean wins
    # by construction. Each rule is computed on one panel of eight and scored
    # on the other, so every rule is judged out of sample by the same measure.
    print("selection rules: computed on eight draws, scored on the other eight")
    a, b = panels(r, 8, np.random.default_rng(args.seed + 1))
    held = b.mean(2)
    rules = {
        "mean": a.mean(2),
        "median": np.median(a, axis=2),
        "lower quartile": np.percentile(a, 25, axis=2),
        "CVaR worst 25%": np.sort(a, axis=2)[:, :, :2].mean(2),
        "mean - 1 sd": a.mean(2) - a.std(2, ddof=1),
    }
    for label, v in rules.items():
        pick = v.argmax(1)
        reg = (held.max(1) - held[np.arange(len(held)), pick]).mean()
        print(f"  {label:16} regret {reg:6.1f} px   "
              f"picks bc {float((pick == 0).mean()):.0%}")
    print(f"  {'perfect':16} regret {0.0:6.1f} px   "
          f"(the held-out panel's own best)")

    det = penalised_det(z, z["returns"].astype(np.float64))
    pick = det.argmax(1)
    reg = (truth.max(1) - truth[np.arange(len(truth)), pick]).mean()
    same = float((pick == truth.argmax(1)).mean())
    print()
    print("the greedy T=0 continuation, the target that produced a probe worth "
          "+449:")
    print(f"  regret against the honest sixteen {reg:6.1f} px")
    print(f"  same best candidate as the honest sixteen {same:.1%}")
    print(f"  picks bc {float((pick == 0).mean()):.0%}, "
          f"honest sixteen picks bc {float((truth.argmax(1) == 0).mean()):.0%}")
    print(f"  candidate order: {names}")

    if args.subsets:
        # What each template is worth, offline: restrict the oracle to a
        # subset and see how much of the full set's value survives. BC is
        # always available, so the subsets are over the five templates. This
        # is not the game metric — it is the value the *scorer* could reach —
        # but it costs nothing and it says whether a six-way output is needed.
        import itertools
        best = truth.max(1)
        rows = []
        for k in range(6):
            for combo in itertools.combinations(range(1, 6), k):
                keep = [0, *combo]
                got = truth[:, keep].max(1)
                rows.append((float((got - truth[:, 0]).mean()), combo))
        rows.sort(key=lambda t: -t[0])
        full = float((best - truth[:, 0]).mean())
        print()
        print(f"value over bc alone, by subset (full set {full:.1f} px)")
        for k in range(6):
            same = [r for r in rows if len(r[1]) == k]
            top = max(same, key=lambda t: t[0])
            print(f"  {k} templates: best {top[0]:6.1f} px "
                  f"({100 * top[0] / full:3.0f}% of full)  "
                  f"{[names[i] for i in top[1]]}")
        print("  each template alone:")
        for i in range(1, 6):
            v = next(r[0] for r in rows if r[1] == (i,))
            print(f"    {names[i]:12} {v:6.1f} px")

    if args.compare:
        zc = np.load(args.compare)
        rc = penalised(zc)
        k = min(len(r), len(rc))
        iu = np.triu_indices(r.shape[1], 1)
        print()
        print("paired differences, this matrix against " + args.compare)
        for n in (2, 4, 8):
            if 2 * n > min(r.shape[2], rc.shape[2]):
                continue
            out = []
            for m in (r[:k], rc[:k]):
                a, b = panels(m, n, np.random.default_rng(args.seed))
                da = a.mean(2)[:, :, None] - a.mean(2)[:, None, :]
                db = b.mean(2)[:, :, None] - b.mean(2)[:, None, :]
                out.append((da - db)[:, iu[0], iu[1]].std())
            print(f"  N={n}: crn={bool(z['crn'])} {out[0]:6.1f} px   "
                  f"crn={bool(zc['crn'])} {out[1]:6.1f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
