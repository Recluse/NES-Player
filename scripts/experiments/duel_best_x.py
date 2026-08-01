"""Duel checkpoints or flags by best_x, the progress metric from debug RAM.

Each variant is N headless runs of 3600 frames. Results are JSON lines.

Usage:
  uv run python scripts/experiments/duel_best_x.py --game SuperMarioBros-Nes-v0 \
      "bc:runs/bc_smb_av" "planner:runs/bc_smb_av --planner --ghost runs/ego_smb4"
A variant is "name:checkpoint [extra play flags]".
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
ap.add_argument("--integrations", default=None)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--frames", type=int, default=3600)
ap.add_argument("--temperature", default="0.9")
ap.add_argument("variants", nargs="+", help='"name:checkpoint [flags]"')
args = ap.parse_args()

for v in args.variants:
    name, spec = v.split(":", 1)
    ckpt, *extra = spec.split()
    for run in range(args.runs):
        cmd = ["uv", "run", "nes-player", "play", "--game", args.game,
               "--checkpoint", ckpt, "--auto-start",
               "--temperature", args.temperature,
               "--max-frames", str(args.frames), *extra]
        if args.integrations:
            cmd += ["--integrations", args.integrations]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                             check=False).stdout
        xs = [int(m) for m in re.findall(r"best_x=(\d+)", out)]
        print(json.dumps({"variant": name, "run": run,
                          "best_x": max(xs) if xs else -1}), flush=True)
