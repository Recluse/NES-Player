"""Validate progress-from-pixels against RAM.

On a game with a known memory map, run N rollouts and compare the accumulated
camera scroll measured from pixels with the true xscroll from debug RAM. A high
correlation means a visual reward can stand in for telemetry on the games where
no memory map exists. Measured: Pearson r = 0.872.

Usage: uv run python scripts/experiments/visual_progress_check.py [--rollouts 8]
"""

import argparse
import json

import numpy as np

from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.policy.bc import BCPolicy
from nes_player.policy.improve import VisualProgress

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
ap.add_argument("--checkpoint", default="runs/bc_smb_attn3")
ap.add_argument("--rollouts", type=int, default=8)
ap.add_argument("--frames", type=int, default=1800)
ap.add_argument("--temperature", type=float, default=1.1)
a = ap.parse_args()

policy = BCPolicy(a.checkpoint)
em = StableRetroAdapter(a.game, include_debug=True)
pairs = []
for r in range(a.rollouts):
    obs = em.reset()
    policy.reset()
    vis, pressed, ram_best = VisualProgress(), frozenset(), 0
    for i in range(a.frames):
        policy.push_audio(obs.audio_pcm)
        if i % 4 == 0:
            pressed = policy.act(obs.frame_rgb, temperature=a.temperature)[0]
        if 60 <= i < 64:
            pressed = frozenset({"START"})
        obs = em.step_buttons([pressed])
        vis.update(obs.frame_rgb)
        d = obs.debug or {}
        hi = d.get("xscrollHi", 0)
        if i > 200 and hi != 255:   # before the game starts these addresses hold garbage
            ram_best = max(ram_best, hi * 256 + d.get("xscrollLo", 0))
    pairs.append((vis.total, ram_best))
    print(json.dumps({"rollout": r, "visual": round(vis.total, 1), "ram_x": ram_best}),
          flush=True)
em.close()

v = np.array([p[0] for p in pairs])
x = np.array([p[1] for p in pairs])
print(f"pearson r = {np.corrcoef(v, x)[0, 1]:.3f} (n={len(pairs)})")
