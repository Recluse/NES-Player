"""What does standing still cost, as a function of how far ahead you look?

The failed student sends a third of every teacher row to `wait`, and the
reason nothing in the loss stopped it is that a spurious wait costs 2.1 px
inside the 144-frame horizon the label is measured over. In a run it costs
432. The suspicion is that the horizon, not the student, is what cannot see
inaction.

That is one measurement: at a decision, take `wait` and take the teacher's
choice, and compare where each ends up after H frames of the policy, for
several H. If the gap grows with H, the target is fixable by looking further.
If it stays at two pixels, the target is not the problem and the direction is
dead in twenty minutes rather than after another training run.

    uv run python scripts/experiments/wait_cost.py runs/bc_smb_new --points 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

HORIZON, COMMIT = 48, 16
PROBE_EVERY = 20
IDLE_STEP, IDLE_MAX = 37, 1800
DEATH = -400.0


def _play(env, policy, obs, plan, l0, repeat, temperature, horizons, p0):
    """Commit the plan, then hand over, sampling progress at each horizon."""
    from nes_player.policy.robustify import progress_of

    out, ob = {}, obs
    for k in range(COMMIT):
        ob = env.step_buttons([plan[k]])
        now = (ob.debug or {}).get("lives")
        if l0 is not None and now is not None and now < l0:
            return dict.fromkeys(horizons, DEATH)
    pressed: frozenset = frozenset()
    for k in range(max(horizons)):
        if k % repeat == 0:
            pr, _ = policy.act(ob.frame_rgb, temperature)
            pressed = pr - {"START", "SELECT"}
        ob = env.step_buttons([pressed])
        now = (ob.debug or {}).get("lives")
        if l0 is not None and now is not None and now < l0:
            for h in horizons:
                out.setdefault(h, DEATH)
            return {h: out.get(h, DEATH) for h in horizons}
        if k + 1 in horizons:
            out[k + 1] = float(progress_of(ob.debug or {}) - p0)
    return {h: out.get(h, DEATH) for h in horizons}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--points", type=int, default=200)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[96, 192, 288, 480])
    args = ap.parse_args()

    from oracle_mpc import templates

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.perception.feedback import game_over
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin
    from nes_player.policy.robustify import progress_of

    env = StableRetroAdapter(args.game, include_debug=True, state=args.state)
    policy = BCPolicy(args.checkpoint)
    cands = templates(HORIZON)
    names = ["bc", *[n for n, _ in cands]]
    wait_i = names.index("wait")
    hs = sorted(args.horizons)

    got: list = []
    seed = args.seed - 1
    played = 0
    while len(got) < args.points and played < 200000:
        seed += 1
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        policy.reset()
        obs = _begin(env)
        pressed: frozenset = frozenset()
        for _ in range(IDLE_STEP * seed % IDLE_MAX):
            obs = env.step_buttons([frozenset()])
        for _ in range(3000):
            played += 1
            if played % PROBE_EVERY == 0:
                here = env.save_state()
                stack = list(policy._stack)
                state = np.random.get_state()
                l0 = (obs.debug or {}).get("lives")
                p0 = progress_of(obs.debug or {})
                seq, o = [], obs
                for k in range(HORIZON):
                    if k % args.repeat == 0:
                        p, _ = policy.act(o.frame_rgb, args.temperature)
                        p = p - {"START", "SELECT"}
                    seq.append(p)
                    o = env.step_buttons([p])
                policy._stack = list(stack)
                seeds = rng.integers(0, 2**31 - 1, size=args.draws)
                vals = np.zeros((len(names), len(hs)))
                for c, (_, plan) in enumerate([("bc", seq), *cands]):
                    acc = np.zeros(len(hs))
                    for d in range(args.draws):
                        env.load_state(here)
                        policy._stack = list(stack)
                        np.random.seed(int(seeds[d]))
                        r = _play(env, policy, obs, plan, l0, args.repeat,
                                  args.temperature, hs, p0)
                        acc += np.array([r[h] for h in hs])
                    vals[c] = acc / args.draws
                got.append(vals)
                env.load_state(here)
                policy._stack = list(stack)
                np.random.set_state(state)
            if played % args.repeat == 0:
                pressed, _ = policy.act(obs.frame_rgb, args.temperature)
                pressed = pressed - {"START", "SELECT"}
            obs = env.step_buttons([pressed])
            if game_over(obs.debug or {}) or len(got) >= args.points:
                break
    env.close()

    v = np.stack(got)                       # (P, C, H)
    print(json.dumps({"points": len(v), "draws": args.draws,
                      "horizons": hs, "seeds": f"{args.seed}+"}, indent=2))
    print()
    # Everything grows with the horizon, so the absolute penalty says little.
    # What matters for a loss is whether waiting is expensive *relative to the
    # spread the loss is fitting* — that ratio is the gradient's opinion of it.
    print(f"{'horizon':>8} {'best - wait':>12} {'best - bc':>11} "
          f"{'spread':>8} {'wait/spread':>12} {'wait is best':>13}")
    for j, h in enumerate(hs):
        col = v[:, :, j]
        best = col.max(1)
        pen = (best - col[:, wait_i]).mean()
        spread = col.std(1).mean()
        print(f"{h:>8} {pen:11.1f}p "
              f"{(best - col[:, 0]).mean():10.1f}p "
              f"{spread:7.1f}p {pen / max(spread, 1e-9):11.2f} "
              f"{float((col.argmax(1) == wait_i).mean()):12.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
