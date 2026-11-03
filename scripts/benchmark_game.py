#!/usr/bin/env python3
"""Benchmark script: run the full pipeline on a new game.

Automates the per-game pipeline:
  1. fetch_manual.py  — download + OCR manual
  2. llm_lexicon.py   — extract facts (requires manual LLM call)
  3. auto_prior.py    — generate gold + prior
  4. scans            — hero_tiles, controllability
  5. explore + train-bc — collect data + train policy
  6. oracle_mpc.py    — confirmatory

Usage (after writing the lexicon JSON):
  uv run python scripts/benchmark_game.py --game DoubleDragonII-Nes-v0 \
      --lexicon manuals/parsed/doubledragonii_llm.json \
      --manual-url "https://archive.org/download/..."

Measures per-game pipeline time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str], timeout: int = 600) -> float:
    """Run a pipeline step and return elapsed time."""
    print(f"\n--- {name} ---")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, timeout=timeout,
                           capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED ({result.returncode}): {result.stderr[-500:]}")
    else:
        print(f"OK ({elapsed:.1f}s)")
    return elapsed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--lexicon", required=True, help="LLM lexicon JSON (manual step)")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--prior", default="")
    p.add_argument("--pos-from-scan", action="store_true")
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--horizons", type=int, nargs="+", default=[48, 96, 128])
    p.add_argument("--dry-run", action="store_true", help="print commands only")
    args = p.parse_args()

    game = args.game
    lexicon = Path(args.lexicon)
    prior_path = args.prior or f"assets/priors/prior_{game}.json"
    gold_path = f"assets/priors/gold/{game.lower().replace('-nes-v0', '')}_us.json"

    print(f"=== Benchmark: {game} ===")
    times = {}

    # Step 1: auto-prior (gold + prior from lexicon)
    cmd = [sys.executable, "scripts/auto_prior.py",
           "--lexicon", str(lexicon), "--game", game,
           "--out-gold", gold_path, "--out-prior", prior_path]
    if args.dry_run:
        print(f"[1] auto_prior: {' '.join(cmd)}")
    else:
        times["auto_prior"] = run_step("auto_prior", cmd)

    # Step 2: hero tiles
    cmd = [sys.executable, "scripts/experiments/hero_tiles.py", game]
    if args.dry_run:
        print(f"[2] hero_tiles: {' '.join(cmd)}")
    else:
        times["hero_tiles"] = run_step("hero_tiles", cmd, 300)

    # Step 3: controllability (position bytes)
    cmd = [sys.executable, "scripts/experiments/controllability.py", game]
    if args.dry_run:
        print(f"[3] controllability: {' '.join(cmd)}")
    else:
        times["controllability"] = run_step("controllability", cmd, 300)

    # Step 4: explore data collection
    cmd = [sys.executable, "-m", "nes_player.cli", "explore",
           "--game", game, "--integrations", "integrations/",
           "--max-frames", "30000", "--loop",
           "--record", f"datasets/explore_{game}"]
    if args.dry_run:
        print(f"[4] explore: {' '.join(cmd)}  (run in background, 10 min)")
    else:
        print("  NOTE: explore needs background run (~10 min), skipping")

    # Step 5: BC training
    cmd = [sys.executable, "-m", "nes_player.cli", "train-bc",
           "--episode", f"datasets/explore_{game}",
           "--out", f"runs/bc_{game.lower().replace('-nes-v0', '')}",
           "--epochs", "1", "--max-episodes", "10"]
    if args.dry_run:
        print(f"[5] train-bc: {' '.join(cmd)}")
    else:
        print("  NOTE: needs explore data first, skipping")

    # Step 6: confirmatory
    checkpoint = args.checkpoint or f"runs/bc_{game.lower().replace('-nes-v0', '')}"
    cmd = [sys.executable, "scripts/experiments/oracle_mpc.py", checkpoint,
           "--game", game, "--seed0", "5000", "--runs", str(args.seeds),
           "--frames", str(args.frames)]
    for h in args.horizons:
        cmd += ["--horizons", str(h)]
    if args.pos_from_scan:
        cmd += ["--pos-from-scan"]
    if Path(prior_path).exists():
        cmd += ["--prior", prior_path]
    if args.dry_run:
        print(f"[6] confirmatory: {' '.join(cmd)}")
    else:
        times["confirmatory"] = run_step("confirmatory", cmd, 3600)

    # Summary
    total = sum(times.values())
    print(f"\n=== Pipeline time: {total:.0f}s ({total/60:.1f} min) ===")
    for k, v in times.items():
        print(f"  {k}: {v:.0f}s")

    out = {"game": game, "steps": {k: round(v, 1) for k, v in times.items()},
           "total_s": round(total, 1)}
    out_path = ROOT / "runs" / f"benchmark_{game}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
