#!/usr/bin/env python3
"""Hierarchical subgoals experiment: does subgoal chaining at a short horizon
match flat planning at a long horizon?

Metroid-class problem: action-result windows longer than the horizon. The
hypothesis is that chaining subgoals lets a short-horizon planner cover a
long window by breaking it into reachable steps.

Arms (32 paired seeds 5000-5031):
  1. flat h=48         — baseline: short horizon, no subgoals
  2. flat h=128        — long horizon (the expensive fix)
  3. hierarchical h=48 — subgoal chain + short horizon (the cheap fix)

Metric: best_x (progress). If (3) ≈ (2), subgoal chaining replaces
horizon extension.

    uv run python scripts/experiments/hierarchical_exp.py --game DoubleDragonII-Nes-v0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

SEEDS = list(range(5000, 5032))


def run_arm(game: str, checkpoint: str, prior: str, seed0: int, n_runs: int,
            frames: int, horizons: list[int], pos_from_scan: bool,
            use_subgoals: bool, n_subgoals: int, label: str) -> list[dict]:
    """Run one arm of the experiment via oracle_mpc."""
    from oracle_mpc import main as oracle_main
    import argparse as ap_mod

    out_rows = []

    # We monkey-patch sys.argv to drive oracle_mpc
    argv = ["oracle_mpc.py", checkpoint, "--game", game,
            "--seed0", str(seed0), "--runs", str(n_runs),
            "--frames", str(frames)]
    for h in horizons:
        argv += ["--horizons", str(h)]
    if prior:
        argv += ["--prior", prior]
    if pos_from_scan:
        argv += ["--pos-from-scan"]
    if use_subgoals:
        argv += ["--subgoals", str(n_subgoals)]

    old_argv = sys.argv
    sys.argv = argv
    try:
        # oracle_mpc prints rows as JSON; we capture from its output
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            oracle_main()
        output = buf.getvalue()
    finally:
        sys.argv = old_argv

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{") and '"arm"' in line:
            try:
                row = json.loads(line)
                row["label"] = label
                out_rows.append(row)
            except json.JSONDecodeError:
                pass

    return out_rows


def summarise(rows: list[dict], label: str) -> dict:
    by_seed: dict[int, float] = {}
    for r in rows:
        by_seed[r["seed"]] = max(by_seed.get(r["seed"], 0), r.get("best_x", 0))
    vals = [by_seed[s] for s in sorted(by_seed)]
    return {
        "label": label,
        "n": len(vals),
        "mean": float(np.mean(vals)) if vals else 0,
        "median": float(np.median(vals)) if vals else 0,
        "min": float(np.min(vals)) if vals else 0,
        "max": float(np.max(vals)) if vals else 0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="DoubleDragonII-Nes-v0")
    p.add_argument("--checkpoint", default="runs/bc_doubledragonii")
    p.add_argument("--prior", default="assets/priors/prior_DoubleDragonII-Nes-v0.json")
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--runs", type=int, default=32)
    p.add_argument("--seed0", type=int, default=5000)
    args = p.parse_args()

    print(f"=== Hierarchical subgoals: {args.game} ===")
    print(f"Seeds: {args.seed0}-{args.seed0 + args.runs - 1}")

    results = {}

    # Arm 1: flat h=48
    print("\n--- Arm 1: flat h=48 ---")
    t0 = time.time()
    rows = run_arm(args.game, args.checkpoint, args.prior,
                   args.seed0, args.runs, args.frames, [48],
                   pos_from_scan=True, use_subgoals=False,
                   n_subgoals=0, label="flat_h48")
    results["flat_h48"] = summarise(rows, "flat h=48")
    print(f"  mean={results['flat_h48']['mean']:.0f}  ({time.time()-t0:.0f}s)")

    # Arm 2: flat h=128
    print("\n--- Arm 2: flat h=128 ---")
    t0 = time.time()
    rows = run_arm(args.game, args.checkpoint, args.prior,
                   args.seed0, args.runs, args.frames, [128],
                   pos_from_scan=True, use_subgoals=False,
                   n_subgoals=0, label="flat_h128")
    results["flat_h128"] = summarise(rows, "flat h=128")
    print(f"  mean={results['flat_h128']['mean']:.0f}  ({time.time()-t0:.0f}s)")

    # Arm 3: hierarchical h=48 (subgoal chain)
    print("\n--- Arm 3: hierarchical h=48 (subgoals) ---")
    t0 = time.time()
    rows = run_arm(args.game, args.checkpoint, args.prior,
                   args.seed0, args.runs, args.frames, [48],
                   pos_from_scan=True, use_subgoals=True,
                   n_subgoals=8, label="hier_h48")
    results["hier_h48"] = summarise(rows, "hierarchical h=48")
    print(f"  mean={results['hier_h48']['mean']:.0f}  ({time.time()-t0:.0f}s)")

    # Comparison
    print("\n=== Comparison ===")
    flat48 = results["flat_h48"]["mean"]
    flat128 = results["flat_h128"]["mean"]
    hier48 = results["hier_h48"]["mean"]
    print(f"flat h=48:          {flat48:.0f}")
    print(f"flat h=128:         {flat128:.0f}")
    print(f"hierarchical h=48:  {hier48:.0f}")
    print(f"\nhier h=48 vs flat h=48:  {hier48 - flat48:+.0f}")
    print(f"hier h=48 vs flat h=128: {hier48 - flat128:+.0f}")
    if flat128 > flat48:
        recovery = (hier48 - flat48) / (flat128 - flat48) * 100
        print(f"Subgoal chaining recovers {recovery:.0f}% of the horizon gap")

    out = {"game": args.game, "seeds": [args.seed0, args.seed0 + args.runs - 1],
           "results": results}
    out_path = ROOT / "runs" / f"hierarchical_{args.game}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
