"""Does splitting a merged blob help the agent fight?

In a beat-em-up two sprites at contact range become one connected component.
The tracker used to hand that blob to whichever track was nearer, so the other
fighter vanished and the sign of "which way is the enemy" became noise — the
agent struck away from the enemy. `_split_detection` cuts the blob between the
two predicted positions instead.

This measures whether that changes anything a player would notice. Both arms
run in the same process on identical seeds, with splitting disabled in one by
neutralising the function, so nothing else can differ.

    uv run python scripts/experiments/clinch_ab.py --runs 3 --frames 3000
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nes_player.perception import motion  # noqa: E402
from nes_player.policy.instinct import InstinctPolicy  # noqa: E402

GAME = "DoubleDragon-Nes-v0"
STATE = "default"     # its title screen cannot be passed from power-on


def run(frames: int, seed: int, split: bool, engage_dy: int | None = None,
        left_edge: int | None = None) -> dict:
    """One episode. `seed` idles a different number of frames before play starts.

    The instinct policy is deterministic, so passing a seed to the emulator
    changes nothing at all — every run comes back byte-identical and three runs
    look like agreement when they are one measurement repeated. Starting the
    policy at a different point in the level is what actually varies the run.
    """
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy import instinct

    real = motion._split_detection
    real_dy = instinct.ENGAGE_DY
    real_edge = instinct.LEFT_EDGE
    if not split:
        motion._split_detection = lambda *a, **k: None
    if engage_dy is not None:
        instinct.ENGAGE_DY = engage_dy
    if left_edge is not None:
        instinct.LEFT_EDGE = left_edge
    try:
        from nes_player.policy.improve import VisualProgress

        env = StableRetroAdapter(GAME, include_debug=True, state=STATE)
        policy = InstinctPolicy(knowledge_path=f"runs/knowledge/{GAME}.json")
        progress = VisualProgress()
        obs = env.reset(seed=seed)
        for _ in range(seed * 37):          # idle: a different slice of the level
            obs = env.step_buttons([frozenset()])
        score0, hits, aligning, closing, pinned = None, 0, 0, 0, 0
        for i in range(frames):
            d = obs.debug or {}
            if score0 is None and i > 100:
                score0 = d.get("score", 0)
            score = max(0, d.get("score", 0) - score0) if score0 is not None else 0
            pressed, slots, _ = policy.step(obs.frame_rgb, score, False)
            hero = motion.pick_hero(slots)
            # The pathology this measures directly: the agent walks into the
            # left edge, gets no scroll, reads that as being stuck, and retreats
            # into the same wall again. Score barely notices; this does.
            pinned += hero is not None and hero.ctrl_prob > 0.7 and hero.cx < 28
            reason = policy.last_reason
            hits += "finishing" in reason
            aligning += "aligning" in reason
            closing += "closing in" in reason
            obs = env.step_buttons([pressed])
            progress.update(obs.frame_rgb)
        env.close()
        # Progress is the second opinion: in a beat-em-up the level only moves
        # on once the enemies are down, so camera scroll says whether the agent
        # is winning fights even when the score counter is noisy.
        return {"score": score, "progress": round(progress.total, 1),
                "hits": hits, "aligning": aligning, "closing": closing,
                "pinned": int(pinned)}
    finally:
        motion._split_detection = real
        instinct.ENGAGE_DY = real_dy
        instinct.LEFT_EDGE = real_edge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--engage-dy", type=int, nargs="*", default=None,
                    help="sweep the same-lane threshold with splitting on")
    ap.add_argument("--left-edge", action="store_true",
                    help="isolate the left-wall fix instead: on against off")
    args = ap.parse_args()

    if args.left_edge:
        arms = [("wall retreat", True, None, -1), ("wall fix", True, None, 28)]
    else:
        arms = [("merged", False, None, None), ("split", True, None, None)]
        for dy in args.engage_dy or []:
            arms.append((f"split dy={dy}", True, dy, None))

    out: dict[str, list[dict]] = {a[0]: [] for a in arms}
    for seed in range(args.runs):
        for arm, split, dy, edge in arms:
            r = run(args.frames, seed, split, dy, edge)
            r["seed"] = seed
            out[arm].append(r)
            print(json.dumps({"arm": arm, **r}), flush=True)

    print()
    base = out[arms[0][0]]
    for arm, rows in out.items():
        scores = [r["score"] for r in rows]
        prog = [r["progress"] for r in rows]
        spread = statistics.stdev(scores) if len(scores) > 1 else 0.0
        # Paired against the first arm: the seeds are the same runs, so the
        # win/loss split says more than two means with overlapping spreads.
        diffs = [r["score"] - b["score"] for r, b in zip(rows, base, strict=True)]
        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        print(f"{arm:14} score {statistics.mean(scores):6.1f} ±{spread:5.1f}  "
              f"progress {statistics.mean(prog):7.1f}  "
              f"hit {statistics.mean(r['hits'] for r in rows):5.0f}  "
              f"pinned {statistics.mean(r['pinned'] for r in rows):5.0f}  "
              f"vs base {statistics.mean(diffs):+6.1f} ({wins}W/{losses}L)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
