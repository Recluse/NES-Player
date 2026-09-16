"""Recompute every published number from the logs and say which stopped agreeing.

`docs/experiments.md` is five and a half thousand lines, and checking one number
in it means rereading the run that produced it. Four claims on that page have
been retracted, all four caught by someone noticing. This is so the fifth does
not have to be noticed: `docs/claims.jsonl` carries one machine-readable line
per published number, and this recomputes them all from the logs on disk.

    uv run python scripts/experiments/recheck_claims.py
    uv run python scripts/experiments/recheck_claims.py --id attention-loss

Nothing is run and nothing is trained; if a log is missing the claim is skipped
and said to be skipped, which is different from passing.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from nes_player.evaluation.stats import (
    detectable, paired_bootstrap, t_interval, variance_split,
)

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "runs/oracle_adaptive"
CLAIMS = ROOT / "docs/claims.jsonl"

# An interval recomputed from the same logs should land where it landed. The
# bootstrap resamples, so allow a pixel or two of drift, not more.
TOLERANCE = 3


def per_seed(path: Path) -> dict[int, float]:
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            out[r["seed"]] = float(r["best_x"])
    return out


def differences(a: list[str], b: list[str]) -> list[np.ndarray]:
    """One array of per-seed differences per paired run."""
    out = []
    for pa, pb in zip(a, b, strict=True):
        x, y = per_seed(LOGS / pa), per_seed(LOGS / pb)
        seeds = sorted(set(x) & set(y))
        if not seeds:
            raise ValueError(f"{pa} and {pb} share no evaluation seeds")
        out.append(np.array([x[s] - y[s] for s in seeds], float))
    return out


def check(c: dict) -> tuple[str, str]:
    missing = [p for p in c["a"] + c["b"] if not (LOGS / p).exists()]
    if missing:
        return "skip", f"{len(missing)} log(s) missing, first {missing[0]}"

    per_run = differences(c["a"], c["b"])
    if c["unit"] == "run":                      # a training run is one observation
        means = np.array([d.mean() for d in per_run])
        got, lo, hi = t_interval(means)
        wins = int((means > 0).sum())
        extra = variance_split(per_run)
        note = (f"sd {extra['sd']:.0f}, eval {extra['eval_share'] * 100:.0f}%, "
                f"resolves {detectable(extra['sd'], len(means)):.0f}")
    else:                                       # an evaluation seed is one observation
        d = np.concatenate(per_run)
        got, lo, hi, wins = paired_bootstrap(d)
        note = f"{len(d)} seeds"

    off = [f"effect {got:+.0f} vs {c['effect']:+d}" if abs(got - c["effect"]) > TOLERANCE else "",
           f"low {lo:+.0f} vs {c['lo']:+d}" if abs(lo - c["lo"]) > TOLERANCE else "",
           f"high {hi:+.0f} vs {c['hi']:+d}" if abs(hi - c["hi"]) > TOLERANCE else "",
           f"wins {wins} vs {c['wins']}" if wins != c["wins"] else ""]
    off = [o for o in off if o]
    if off:
        return "MOVED", "; ".join(off)
    return "ok", f"{got:+.0f} [{lo:+.0f}, {hi:+.0f}]  {wins}/{c['of']}  {note}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None, help="check one claim instead of all")
    args = ap.parse_args()

    claims = [json.loads(line) for line in CLAIMS.read_text().splitlines()
              if line.strip() and "_comment" not in line[:20]]
    if args.id:
        claims = [c for c in claims if c["id"] == args.id]
        if not claims:
            raise SystemExit(f"no claim with id {args.id!r}")

    worst = 0
    for c in claims:
        verdict, detail = check(c)
        print(f"{verdict:>5}  {c['id']:<26} {detail}")
        if verdict == "MOVED":
            print(f"       {c['claim']}  ({c['doc']})")
            worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
