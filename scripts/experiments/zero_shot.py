"""Zero-shot: the agent meets a game for the first time, with no demonstrations.

Three agents on games that appeared in no training run:

- random   — random buttons, the lower bound;
- instinct — instincts, which calibrate the controls in place, untrained;
- base     — the multi-game BC base with no fine-tuning at all.

Metrics, both without RAM:

- score — THE SCORE READ OFF THE SCREEN, with digits learned from counter
  dynamics. This generalises: it works on single-screen games and on vertical
  scrollers alike;
- progress — accumulated horizontal camera scroll, which correlates 0.872 with
  the true value but is only meaningful for horizontally scrolling games.

The reader is trained once per game, on a preliminary random run, and the same
one serves every agent — otherwise the comparison is not fair.

Usage: uv run python scripts/experiments/zero_shot.py [--runs 3] [--frames 3600]
"""

import argparse
import json

import numpy as np

from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.perception.text import HudReader
from nes_player.policy.bc import BCPolicy, mask_to_pressed
from nes_player.policy.improve import VisualProgress
from nes_player.policy.instinct import InstinctPolicy

# Games absent from the base and from any tuning: a genuine first encounter
GAMES = [
    ("Castlevania-Nes-v0", None),
    ("BubbleBobble-Nes-v0", None),
    ("1943-Nes-v0", None),
]
START_PULSES = frozenset(range(60, 900, 60))


def fit_reader(game: str, integrations, frames: int = 2400) -> tuple:
    """Train the HUD reader on a random run and find the score counter: it
    never decreases, unlike a timer, and it does change at least once."""
    rng = np.random.default_rng(0)
    em = StableRetroAdapter(game, integration_dir=integrations)
    obs = em.reset()
    shots, pressed = [], frozenset()
    for i in range(frames):
        if i in START_PULSES:
            pressed = frozenset({"START"})
        elif i % 4 == 0:
            pressed = mask_to_pressed(int(rng.integers(0, 256))) - {"START"}
        obs = em.step_buttons([pressed])
        if i > 900 and i % 4 == 0:
            shots.append(obs.frame_rgb.copy())
    em.close()
    reader = HudReader().fit(shots[:240])
    if not reader.groups:
        return reader, None
    # The score is the LONGEST non-decreasing counter, usually six digits.
    # Requiring it to grow during training would be wrong: a random agent may
    # score nothing at all, leaving the counter at zero for the whole run.
    series = np.array([reader.read(f) for f in shots[::4]], dtype=float)
    best, best_len = None, 0
    for gi in range(series.shape[1]):
        v = series[:, gi]
        v = v[v >= 0]
        if len(v) < 10:
            continue
        if float((np.diff(v) < 0).mean()) < 0.02 and len(reader.groups[gi]) > best_len:
            best, best_len = gi, len(reader.groups[gi])
    return reader, best


def run(game: str, integrations, agent: str, seed: int, frames: int,
        checkpoint: str, reader=None, score_group=None) -> dict:
    rng = np.random.default_rng(seed)
    em = StableRetroAdapter(game, integration_dir=integrations)
    policy = None
    if agent == "instinct":
        policy = InstinctPolicy()
    elif agent == "base":
        policy = BCPolicy(checkpoint)
        policy.reset()
    obs = em.reset()
    vis, pressed = VisualProgress(), frozenset()
    best_score = -1
    try:
        for i in range(frames):
            if i in START_PULSES:   # the menus are handled identically for all agents
                pressed = frozenset({"START"})
            elif agent == "random":
                if i % 4 == 0:
                    pressed = mask_to_pressed(int(rng.integers(0, 256))) - {"START"}
            elif agent == "instinct":
                pressed = policy.step(obs.frame_rgb)[0]
            else:
                policy.push_audio(obs.audio_pcm)
                if i % 4 == 0:
                    pressed = policy.act(obs.frame_rgb, temperature=0.9)[0] - {"START"}
            obs = em.step_buttons([pressed])
            if i > 900:
                vis.update(obs.frame_rgb)
                if score_group is not None and i % 12 == 0:
                    vals = reader.read(obs.frame_rgb)
                    if score_group < len(vals) and vals[score_group] >= 0:
                        best_score = max(best_score, vals[score_group])
        return {"progress_per1000": round(vis.total / (frames - 900) * 1000, 1),
                "score_read": best_score}
    finally:
        em.close()


ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--frames", type=int, default=3600)
ap.add_argument("--checkpoint", default="runs/bc_base41_attn1")
a = ap.parse_args()

for game, integ in GAMES:
    reader, score_group = fit_reader(game, integ)
    print(json.dumps({"game": game, "digits_learned": len(reader.digits),
                      "score_group": score_group}), flush=True)
    for agent in ("random", "instinct", "base"):
        for seed in range(a.runs):
            try:
                res = run(game, integ, agent, seed, a.frames, a.checkpoint,
                          reader=reader, score_group=score_group)
            except Exception as e:   # no ROM or integration: skip it openly
                print(json.dumps({"game": game, "agent": agent, "error": str(e)[:80]}),
                      flush=True)
                break
            print(json.dumps({"game": game, "agent": agent, "seed": seed, **res}),
                  flush=True)
