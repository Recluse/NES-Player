"""Behavioural cloning against random: xscroll progress over N frames.

Usage: uv run python scripts/eval_bc_vs_random.py runs/bc_smb [episodes] [frames]
"""

import json
import sys
from pathlib import Path

import numpy as np

from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.policy.bc import BCPolicy, mask_to_pressed


def progress(env, act_fn, frames: int) -> int:
    obs = env.reset(seed=0)
    best = 0
    for _ in range(frames):
        obs = env.step_buttons([act_fn(obs)])
        d = obs.debug
        # xscroll resets between levels, so track the maximum world progress
        best = max(best, d["levelHi"] * 100000 + d["levelLo"] * 10000
                   + d["xscrollHi"] * 256 + d["xscrollLo"])
    return best


def main() -> None:
    run_dir = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 3600

    meta = json.loads(Path(f"{run_dir}/meta.json").read_text())
    masks = meta["vocab_masks"]
    rng = np.random.default_rng(0)

    env = StableRetroAdapter("SuperMarioBros-Nes-v0", include_debug=True)
    policy = BCPolicy(run_dir)

    for name, fn in [
        ("random", lambda obs: mask_to_pressed(int(rng.choice(masks)))),
        ("bc", lambda obs: policy.act(obs.frame_rgb)[0]),
    ]:
        scores = []
        for _ in range(episodes):
            policy.reset()
            scores.append(progress(env, fn, frames))
        print(f"{name}: progress={scores} mean={np.mean(scores):.0f}")
    env.close()


if __name__ == "__main__":
    main()
