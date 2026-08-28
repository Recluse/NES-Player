"""The confirmatory test, exactly as pre-registered.

Reads the per-seed rows written by `oracle_mpc.py` for the three arms of
`docs/preregistration.md` and applies the tests named there, so the analysis
cannot drift after the numbers are in.

    uv run python scripts/experiments/confirm.py \
        bc.log habit.log student.log --student "probe h=48"
"""

import argparse
import json
from pathlib import Path

import numpy as np

PERMUTATIONS = 10000


def rows(paths: list[str]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


def permutation(d: np.ndarray, rng, n: int = PERMUTATIONS) -> float:
    """Two-sided paired permutation test: flip the sign of each difference."""
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float(((np.abs((signs * d).mean(1)) >= obs).sum() + 1) / (n + 1))


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar on paired completions, two-sided."""
    from math import comb

    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--student", required=True, help="arm name of the student")
    ap.add_argument("--habit", default="always jump now")
    ap.add_argument("--policy", default="bc")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    arms = rows(args.logs)
    missing = [a for a in (args.student, args.habit, args.policy)
               if a not in arms]
    if missing:
        raise SystemExit(f"no rows for {missing}; found {sorted(arms)}")
    seeds = sorted(set.intersection(*(set(arms[a]) for a in
                                      (args.student, args.habit, args.policy))))
    print(f"paired seeds {len(seeds)}: {seeds[0]}-{seeds[-1]}")

    def col(arm, key="best_x"):
        return np.array([arms[arm][s][key] for s in seeds], float)

    for arm in (args.policy, args.habit, args.student):
        v = col(arm)
        print(f"  {arm:22} median {np.median(v):7.1f}  mean {v.mean():7.1f}  "
              f"clears {int((v > 4000).sum()):2d}/{len(v)}  "
              f"deaths {int(col(arm, 'deaths').sum()):3d}")

    print()
    print("pre-registered: the student must beat both, p < 0.05")
    passed = []
    for other in (args.policy, args.habit):
        d = col(args.student) - col(other)
        p = permutation(d, rng)
        boot = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        passed.append(p < 0.05 and d.mean() > 0)
        print(f"  student minus {other:18} {d.mean():+8.1f} px  "
              f"[{lo:+.0f},{hi:+.0f}]  p={p:.4f}  "
              f"{'PASS' if passed[-1] else 'fail'}")

    print()
    print("reported alongside, not criteria")
    for other in (args.policy, args.habit):
        p = mcnemar(col(args.student) > 4000, col(other) > 4000)
        print(f"  McNemar on 1-1 completions vs {other:18} p={p:.4f}")

    print()
    if all(passed):
        print("CONFIRMED: better than the policy and better than the habit")
    elif passed[0]:
        print("PARTIAL: better than the policy; not shown better than the habit")
    else:
        print("FAILED: not shown better than the policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
