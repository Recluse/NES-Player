"""Compare Double Dragon checkpoints by playing, not by validation accuracy.

Validation accuracy answers "how predictable is the policy this model cloned",
which is a different question from "how well does it play". A policy that stands
in a corner repeating one manoeuvre is trivially predictable and scores well; a
policy that reacts to where the enemies are does not.

So the checkpoints are made to play. Both arms run in one process on the same
seeds, each seed starting the model at a different point in the level, since
these models are near-deterministic at low temperature.

    uv run python scripts/experiments/dd_checkpoint_duel.py \
        runs/bc_dd_attn2 runs/bc_dd_attn3 --runs 8
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

GAME = "DoubleDragon-Nes-v0"
STATE = "default"


def run(checkpoint: str, frames: int, seed: int, temperature: float) -> dict:
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.improve import VisualProgress

    env = StableRetroAdapter(GAME, include_debug=True, state=STATE)
    policy = BCPolicy(checkpoint)
    progress = VisualProgress()
    obs = env.reset(seed=seed)
    for _ in range(seed * 37):
        obs = env.step_buttons([frozenset()])
    score0, attacks = None, 0
    for i in range(frames):
        d = obs.debug or {}
        if score0 is None and i > 100:
            score0 = d.get("score", 0)
        score = max(0, d.get("score", 0) - score0) if score0 is not None else 0
        policy.push_audio(obs.audio_pcm)
        pressed, _ = policy.act(obs.frame_rgb, temperature=temperature)
        pressed = pressed - {"START", "SELECT"}   # START pauses; the menus are behind us
        attacks += "B" in pressed
        obs = env.step_buttons([pressed])
        progress.update(obs.frame_rgb)
    env.close()
    return {"score": score, "progress": round(progress.total, 1), "attacks": attacks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    out: dict[str, list[dict]] = {c: [] for c in args.checkpoints}
    for seed in range(args.runs):
        for ckpt in args.checkpoints:
            r = run(ckpt, args.frames, seed, args.temperature)
            r["seed"] = seed
            out[ckpt].append(r)
            print(json.dumps({"checkpoint": ckpt, **r}), flush=True)

    print()
    base = out[args.checkpoints[0]]
    for ckpt, rows in out.items():
        scores = [r["score"] for r in rows]
        diffs = [r["score"] - b["score"] for r, b in zip(rows, base, strict=True)]
        spread = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{Path(ckpt).name:16} score {statistics.mean(scores):6.1f} ±{spread:5.1f}  "
              f"progress {statistics.mean(r['progress'] for r in rows):7.1f}  "
              f"attack frames {statistics.mean(r['attacks'] for r in rows):5.0f}  "
              f"vs base {statistics.mean(diffs):+6.1f} "
              f"({sum(d > 0 for d in diffs)}W/{sum(d < 0 for d in diffs)}L)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
