"""How far into a Super Mario Bros. level does a checkpoint actually get?

The duel scores by camera scroll, which is a stand-in for progress on games that
do not report their own. Mario does report it: `xscrollHi/Lo` is the hero's
position in the level, it is not a clock, and it differs between policies by
hundreds. Where the game will tell us, ask the game.

    uv run python scripts/experiments/mario_reach.py runs/bc_smb_old runs/bc_smb_new
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

GAME = "SuperMarioBros-Nes-v0"
LEVEL_END = 3266     # 1-1 is about this long, for reading the numbers as a share


def run(checkpoint: str, seed: int, frames: int, temperature: float,
        jump_hold: int = 32, repeat: int = 4) -> dict:
    from nes_player.cli.play import JumpShaper
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.bc import BCPolicy

    # The policy samples from its own softmax, so a run is only repeatable if
    # that draw is seeded too. Without this a run that finishes the level cannot
    # be played again to be recorded.
    np.random.seed(seed)
    env = StableRetroAdapter(GAME, include_debug=True, state="default")
    policy = BCPolicy(checkpoint)
    # Jump height on the NES is how long A is held, and the policy cannot say
    # "hold": it emits one button set per frame. `play` turns a momentary A into
    # a real jump with this, and a measurement without it is measuring a
    # different agent — one that can only hop a single frame's worth and cannot
    # clear the first pipe of 1-1.
    jump = JumpShaper(jump_hold) if jump_hold else None
    obs = env.reset(seed=seed)
    for _ in range(seed * 37):
        obs = env.step_buttons([frozenset()])
    best_x, deaths, lives = 0, 0, None
    level, levels = None, 0
    held: frozenset = frozenset()
    for _i in range(frames):
        d = obs.debug or {}
        x = int(d.get("xscrollHi", 0)) * 256 + int(d.get("xscrollLo", 0))
        best_x = max(best_x, x)
        lv = d.get("lives")
        if lv is not None and 0 <= lv <= 9:
            if lives is not None and lv < lives:
                deaths += 1
            lives = lv
        now = int(d.get("levelHi", 0)) * 4 + int(d.get("levelLo", 0))
        if level is not None and now > level:
            levels += 1               # finished one; keep going and say so
        level = now
        policy.push_audio(obs.audio_pcm)
        policy.observe(obs.frame_rgb, obs.audio_pcm)
        if _i % repeat == 0:
            held, _ = policy.decide(temperature=temperature)
        pressed = held
        if jump is not None:
            pressed = jump.apply(pressed)
        obs = env.step_buttons([pressed - {"START", "SELECT"}])
    env.close()
    return {"seed": seed, "best_x": best_x, "deaths": deaths, "levels": levels,
            "share": round(best_x / LEVEL_END, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--repeat", type=int, default=4,
                    help="emulator frames between decisions, as in play")
    ap.add_argument("--jump-hold", type=int, default=32,
                    help="frames to hold A; 0 disables the shaper")
    args = ap.parse_args()

    out: dict[str, list[dict]] = {c: [] for c in args.checkpoints}
    for seed in range(args.runs):
        for ckpt in args.checkpoints:
            r = run(ckpt, seed, args.frames, args.temperature, args.jump_hold,
                    args.repeat)
            out[ckpt].append(r)
            print(json.dumps({"checkpoint": ckpt, **r}), flush=True)

    print()
    base = out[args.checkpoints[0]]
    for ckpt, rows in out.items():
        xs = [r["best_x"] for r in rows]
        diffs = [r["best_x"] - b["best_x"] for r, b in zip(rows, base, strict=True)]
        print(f"{Path(ckpt).name:16} x {statistics.mean(xs):7.0f} "
              f"(best {max(xs)}, {statistics.mean(xs) / LEVEL_END:.0%} of 1-1)  "
              f"deaths {statistics.mean(r['deaths'] for r in rows):4.1f}  "
              f"levels {sum(r['levels'] for r in rows)}  "
              f"vs base {statistics.mean(diffs):+7.0f} "
              f"({sum(d > 0 for d in diffs)}W/{sum(d < 0 for d in diffs)}L)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
