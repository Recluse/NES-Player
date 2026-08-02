"""The audit's acceptance criteria for A-02/A-03, on the real emulator.

1. The same seed run fast and run in realtime gives the same action trace and
   the same decision frames. Under the old wall-clock loop it could not: the
   number of emulator frames between decisions depended on how fast the machine
   happened to be going.
2. Two arms — with and without an override standing in for the planner — get
   identical decision indices, so a difference between them is the override and
   not the schedule.
3. A checkpoint's frame stack advances once per emulator frame, so `long` really
   does reach 128 frames back.

    uv run python scripts/experiments/evaluator_check.py runs/bc_dd_long
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

GAME = "DoubleDragon-Nes-v0"
STATE = "default"


def main() -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.evaluation.evaluator import evaluate
    from nes_player.policy.bc import BCPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()

    def run(**kw):
        import numpy as np

        np.random.seed(0)          # the policy samples from its own softmax
        env = StableRetroAdapter(GAME, include_debug=True, state=STATE)
        pol = BCPolicy(args.checkpoint)
        r = evaluate(env, pol, frames=args.frames, action_repeat=args.repeat,
                     temperature=0.2, seed=0, **kw)
        env.close()
        return r, pol

    fast, pol = run()
    slow, _ = run(realtime=True)
    ok_pace = fast.trace() == slow.trace()
    print(f"1. fast vs realtime: {'IDENTICAL' if ok_pace else 'DIFFERENT'} "
          f"({len(fast.decision_frames)} decisions)")

    plain, _ = run()
    withalt, _ = run(override=lambda i, o, p: (p, "planner") if i % 8 == 0 else None)
    ok_ticks = plain.decision_frames == withalt.decision_frames
    print(f"2. both arms, same decision frames: {'YES' if ok_ticks else 'NO'}")

    span = max(pol.offsets)
    ok_stack = len(pol._stack) == span
    print(f"3. frame stack holds {len(pol._stack)} frames, offsets need {span}: "
          f"{'OK' if ok_stack else 'WRONG'}  offsets={pol.offsets}")
    print(f"   {args.frames} frames produced {len(fast.decision_frames)} decisions, "
          f"expected {args.frames // args.repeat}")

    return 0 if (ok_pace and ok_ticks and ok_stack) else 1


if __name__ == "__main__":
    raise SystemExit(main())
