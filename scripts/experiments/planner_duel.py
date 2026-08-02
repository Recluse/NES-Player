"""Does the planner help? Measured with both arms on the same decision ticks.

The earlier answer — "about 4x" — came from spawning `nes-player play` twice.
There the policy decided on a wall clock at 15 Hz while the planner replanned
every `--repeat` emulator frames, so the two arms were not only different
agents but different schedules, and a busier machine changed the comparison.
That number should not be repeated until this script replaces it.

Here one process runs both arms through the synchronous evaluator: observation
every frame, decisions on a fixed grid of frame indices, and the planner
offered exactly those same ticks and no others. The decision indices are
compared afterwards rather than assumed equal.

    uv run python scripts/experiments/planner_duel.py runs/bc_smb_av \
        --ghost runs/ego_smb4 --runs 8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

IDLE_STEP, IDLE_MAX = 37, 1800
CTRL_PROB = 0.7


def run(checkpoint: str, ghost_path: str | None, game: str, state: str | None,
        frames: int, seed: int, temperature: float, repeat: int) -> dict:
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.evaluation.evaluator import evaluate
    from nes_player.perception.memory import ObjectMemory
    from nes_player.perception.motion import MotionTracker
    from nes_player.policy.bc import BCPolicy

    env = StableRetroAdapter(game, include_debug=True, state=state)
    policy = BCPolicy(checkpoint)
    tracker, memory = MotionTracker(), ObjectMemory()
    planner = None
    if ghost_path:
        from nes_player.policy.planner import EgoPlanner
        from nes_player.world_model.ego import GhostPredictor

        planner = EgoPlanner(GhostPredictor(ghost_path))

    state_bag = {"best_x": 0, "slots": [], "verdicts": {}, "plans": 0}

    def on_frame(i, obs, action):
        d = obs.debug or {}
        x = d.get("xscroll", d.get("xscrollHi", 0) * 256 + d.get("xscrollLo", 0))
        if x < 6000:
            state_bag["best_x"] = max(state_bag["best_x"], x)
        # Perception runs every frame for both arms, so the only difference
        # between them is whether the plan is allowed to replace the action.
        slots = tracker.update(obs.frame_rgb, action)
        state_bag["slots"] = slots
        state_bag["verdicts"] = memory.update(obs.frame_rgb, slots, i, 0, False)

    def override(i, obs, pressed):
        if planner is None:
            return None
        top = [s for s in state_bag["slots"] if s.ctrl_prob > CTRL_PROB]
        if not top:
            return None       # hero not visible: the policy keeps the wheel
        dangers = [(s.cx, s.cy, s.vx, s.vy) for s in state_bag["slots"]
                   if s is not top[0] and state_bag["verdicts"].get(s.slot_id) == "danger"]
        plan = planner.plan(obs.frame_rgb, top[0], dangers)
        state_bag["plans"] += 1
        return plan.pressed, "planner"

    r = evaluate(env, policy, frames=frames, action_repeat=repeat,
                 temperature=temperature, seed=seed,
                 idle_start=IDLE_STEP * seed % IDLE_MAX,
                 override=override if planner else None, on_frame=on_frame)
    env.close()
    return {"seed": seed, "best_x": state_bag["best_x"], "plans": state_bag["plans"],
            "decisions": len(r.decision_frames),
            "decision_frames": r.decision_frames}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--ghost", default="runs/ego_smb4")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    arms = {"bc": None, "bc+planner": args.ghost}
    results: dict[str, list[dict]] = {}
    for name, ghost in arms.items():
        rows = []
        for seed in range(args.runs):
            np.random.seed(seed)      # both arms sample the same way
            row = run(args.checkpoint, ghost, args.game, args.state,
                      args.frames, seed, args.temperature, args.repeat)
            row["arm"] = name
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items()
                              if k != "decision_frames"}), flush=True)
        results[name] = rows

    a = np.array([r["best_x"] for r in results["bc+planner"]], float)
    b = np.array([r["best_x"] for r in results["bc"]], float)
    same_ticks = all(x["decision_frames"] == y["decision_frames"]
                     for x, y in zip(results["bc"], results["bc+planner"], strict=True))
    d = a - b
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
    print()
    print(f"same decision frames in both arms: {'YES' if same_ticks else 'NO'}")
    print(f"best_x   bc {b.mean():8.1f}   bc+planner {a.mean():8.1f}   "
          f"diff {d.mean():+8.1f}  t={t:+.2f}  "
          f"{int((d > 0).sum())}W/{int((d < 0).sum())}L")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
