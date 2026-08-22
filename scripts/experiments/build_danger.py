"""Build the object memory over enough deaths for it to have an opinion.

A verdict of "danger" needs two deaths blamed on the same visual cluster, and
one 3000-frame duel supplies about three deaths in total — so every verdict
stays "unknown" and the planner's collision term never fires. Measured: zero
threats offered across 432 replans.

Nothing is wrong with the memory; it is being asked in the wrong lifetime. This
plays for as long as it takes, restarting when the lives run out, and saves what
it learned so a duel can load it instead of starting from nothing.

    uv run python scripts/experiments/build_danger.py --frames 40000

Deaths come from the lives counter in emulator memory, which is teacher-side
information — the same thing `--feedback privileged` reads. What the memory
stores is appearance and tallies; the policy still plays from pixels.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/bc_smb_new")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--frames", type=int, default=40000)
    ap.add_argument("--out", default=None,
                    help="default runs/knowledge/danger_{game}.npz")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()

    import numpy as np

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.perception.feedback import game_over
    from nes_player.perception.memory import ObjectMemory
    from nes_player.perception.sprites import SpriteTracker, sprite_boxes
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin

    np.random.seed(0)
    env = StableRetroAdapter(args.game, include_debug=True, state=args.state)
    policy = BCPolicy(args.checkpoint)
    tracker, memory = SpriteTracker(), ObjectMemory()
    obs = _begin(env)
    lives, deaths, restarts = None, 0, 0
    pressed: frozenset = frozenset()
    for i in range(args.frames):
        d = obs.debug or {}
        slots = tracker.update(obs.frame_rgb, pressed,
                               boxes=sprite_boxes(env._env.get_ram()))
        now = d.get("lives")
        died = lives is not None and now is not None and now < lives
        lives = now
        deaths += int(died)
        memory.update(obs.frame_rgb, slots, i, int(d.get("score", 0) or 0), died)
        if i % args.repeat == 0:
            pressed, _ = policy.act(obs.frame_rgb, args.temperature)
        obs = env.step_buttons([pressed - {"START", "SELECT"}])
        if game_over(d):
            obs = _begin(env)
            tracker = SpriteTracker()
            lives = None
            restarts += 1
    env.close()

    out = Path(args.out or f"runs/knowledge/danger_{args.game}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    memory.save(out)
    tally = {}
    for c in memory.clusters:
        tally.setdefault(c.verdict, 0)
        tally[c.verdict] += 1
    print(json.dumps({
        "path": str(out), "frames": args.frames, "deaths": deaths,
        "restarts": restarts, "clusters": len(memory.clusters),
        "verdicts": tally,
        "top": sorted(((c.deaths, c.contacts, c.verdict)
                       for c in memory.clusters), reverse=True)[:5],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
