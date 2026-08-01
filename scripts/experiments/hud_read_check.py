"""Validate the HUD reader against RAM.

Train HudReader on the frames of an episode with no labels at all, then compare
what it reads against the true score and timer from debug RAM on held-out
frames. On Super Mario Bros. the timer comes out at r = 1.000 with 95.8% exact
matches.

Usage: uv run python scripts/experiments/hud_read_check.py [--game ...] [--frames 2400]
"""

import argparse

import numpy as np

from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.perception.text import HudReader
from nes_player.policy.bc import BCPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
ap.add_argument("--checkpoint", default="runs/bc_smb_attn3")
ap.add_argument("--integrations", default=None)
ap.add_argument("--frames", type=int, default=2400)
a = ap.parse_args()

policy = BCPolicy(a.checkpoint)
em = StableRetroAdapter(a.game, integration_dir=a.integrations, include_debug=True)
obs = em.reset()
frames, truth, pressed = [], [], frozenset()
for i in range(a.frames):
    policy.push_audio(obs.audio_pcm)
    if i % 4 == 0:
        pressed = policy.act(obs.frame_rgb, temperature=1.0)[0]
    if 60 <= i < 64:
        pressed = frozenset({"START"})
    obs = em.step_buttons([pressed])
    frames.append(obs.frame_rgb.copy())
    truth.append(dict(obs.debug or {}))
em.close()

split = int(len(frames) * 0.6)
reader = HudReader().fit(frames[:split])
print(f"digits learned: {len(reader.digits)}, digit cells: {len(reader.cells)}, "
      f"groups: {[len(g) for g in reader.groups]}")
if not reader.groups:
    raise SystemExit("the digits were not learned")

reads = [reader.read(f) for f in frames[split:]]
keys = [k for k in ("score", "time", "coins", "lives") if k in truth[0]]
cols = np.array([r + [-1] * (len(reader.groups) - len(r)) for r in reads], dtype=float)
print("sample readings:", reads[:3], "| RAM:",
      [{k: truth[split + i].get(k) for k in keys} for i in range(3)])
for gi in range(cols.shape[1]):
    v = cols[:, gi]
    ok = v >= 0
    g = reader.groups[gi]
    if ok.sum() < 20 or v[ok].std() == 0:
        print(f"group {gi} {g[0]}..{g[-1]}: rarely read, or constant "
              f"({ok.sum()} frames)")
        continue
    best = None
    for k in keys:
        t = np.array([row.get(k, 0) for row in truth[split:]], dtype=float)
        if t.std() == 0:
            continue
        r = float(np.corrcoef(v[ok], t[ok])[0, 1])
        exact = float((v[ok] == t[ok]).mean())
        if best is None or abs(r) > abs(best[1]):
            best = (k, r, exact)
    if best:
        print(f"group {gi} {g[0]}..{g[-1]} ({len(g)} digits) vs RAM {best[0]}: "
              f"r={best[1]:.3f}, exact {best[2]:.1%}, read in {ok.mean():.0%} of frames")
