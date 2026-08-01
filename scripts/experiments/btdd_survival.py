"""Behavioural cloning against a random agent on Battletoads & Double Dragon.

All three metrics avoid RAM entirely:

- progress — accumulated forward camera scroll: how far the agent actually got;
- survival — frames until the continue or title screen. Level 1 is blue-cyan
  (G-B is about -1) while those screens are green-yellow (G-B above 25);
- health — mean length of the cyan health bars in the HUD, verified live on the
  TAS episode: 432 px full, 222 after damage, then recovery.

A control on the TAS itself showed the metric works — the expert scores 164.6
against 42.2 for random. Note also that fewer than six runs on this game is
noise; the first three gave the opposite result to the next three.

Usage: uv run python scripts/experiments/btdd_survival.py [--runs 3] [--frames 6000]
"""

import argparse
import json

import cv2
import numpy as np

from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.policy.bc import BCPolicy, mask_to_pressed

GAME = "BattletoadsDoubleDragon-Nes-v0"
END_GB = 15.0    # above this, it is no longer level 1 gameplay
END_HOLD = 30    # consecutive frames required, so a flash does not count
START_PULSES = frozenset(range(60, 900, 60))   # title -> menu -> game


HUD_H = 32


def survival(policy: BCPolicy | None, seed: int, frames: int, repeat: int = 4,
             temperature: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    em = StableRetroAdapter(GAME, integration_dir="integrations", include_debug=False)
    if policy is not None:
        policy.reset()
    obs = em.reset()
    pressed, bad = frozenset(), 0
    prev, progress = None, 0.0
    health: list[int] = []

    def result(n: int) -> dict:
        return {"frames_survived": n, "progress": round(progress, 1),
                "health": round(float(np.mean(health)), 1) if health else 0.0}

    try:
        for i in range(frames):
            if i <= 900:
                # The menu phase is identical for every agent: START pulses only
                # and no input, otherwise the random agent drags the cursor
                # around and never enters the level
                pressed = frozenset({"START"}) if i in START_PULSES else frozenset()
            elif i % repeat == 0:
                if policy is None:
                    pressed = mask_to_pressed(int(rng.integers(0, 256))) - {"START", "SELECT"}
                else:
                    policy.push_audio(obs.audio_pcm)
                    pressed = policy.act(obs.frame_rgb, temperature)[0] - {"START", "SELECT"}
            obs = em.step_buttons([pressed])
            f = obs.frame_rgb.astype(np.float32)
            gb = float((f[..., 1] - f[..., 2]).mean())
            if i > 900:   # past the menus
                play = cv2.cvtColor(obs.frame_rgb, cv2.COLOR_RGB2GRAY)[HUD_H:].astype(np.float32)
                if prev is not None and gb <= END_GB:
                    (dx, _), _ = cv2.phaseCorrelate(prev, play)
                    if abs(dx) < 16:   # reject the jumps of a scene change
                        progress += -dx   # camera moving right gives a negative dx
                    hud = obs.frame_rgb[:24]
                    health.append(int(((hud[..., 1] > 120) & (hud[..., 2] > 120)
                                       & (hud[..., 0] < 100)).sum()))
                prev = play
                bad = bad + 1 if gb > END_GB else 0
                if bad >= END_HOLD:
                    return result(i - END_HOLD)
        return result(frames)
    finally:
        em.close()


ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="runs/bc_btdd_attn")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--frames", type=int, default=6000)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--repeat", type=int, default=4)
ap.add_argument("--skip-random", action="store_true")
a = ap.parse_args()

pol = BCPolicy(a.checkpoint)
agents = [("bc", pol)] if a.skip_random else [("random", None), ("bc", pol)]
for name, p in agents:
    for r in range(a.runs):
        res = survival(p, seed=r, frames=a.frames, repeat=a.repeat,
                       temperature=a.temperature)
        # Progress per 1000 frames of gameplay, comparable with the TAS at 164.6
        res["per1000"] = round(
            res["progress"] / max(1, res["frames_survived"] - 900) * 1000, 1)
        print(json.dumps({"agent": name, "run": r, **res}), flush=True)
