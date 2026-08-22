"""Is the ceiling set by the value, or by the five things the planner can say?

Doom has no boundary at any horizon tried: a position is wholly recoverable or
wholly lost, and which of the five templates is taken never decides it. One
reading of that is that the templates are too coarse to express the difference
between living and dying — they are five fixed 48-frame sequences, and the
game is played at a finer grain than that.

This tests the reading directly, with no learning in it at all. The oracle
scores candidates through the console either way; the only difference is where
the candidates come from. One arm has the five templates. The other also
searches a space of macro-actions — the horizon cut into segments, each
segment one of a handful of primitives — by categorical cross-entropy method:
sample, score, keep the elites, refit, sample again.

If the searched arm goes further, the vocabulary is a real limit and worth
enlarging. If it does not, the templates are enough and everything left is in
the value.

    uv run python scripts/experiments/cem_macro.py runs/bc_smb_new --runs 8
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

HORIZON, COMMIT, TAIL = 48, 16, 96
SEG = 16                  # frames per segment of a macro-action
IDLE_STEP, IDLE_MAX = 37, 1800
DEATH = -400.0


def primitives():
    """The vocabulary a segment is written in, one entry per SEG frames."""
    from nes_player.policy.planner import JUMP, LEFT, NOOP, RUN

    WALK = frozenset({"RIGHT"})
    return [
        ("run", [RUN] * SEG),
        ("walk", [WALK] * SEG),
        ("jump", [JUMP] * 10 + [RUN] * (SEG - 10)),
        ("hop", [JUMP] * 5 + [RUN] * (SEG - 5)),
        ("wait", [NOOP] * SEG),
        ("left", [LEFT] * SEG),
    ]


def _score(env, policy, obs, plan, here, stack, l0, p0, repeat, tail_temp,
           draws=1):
    """The tail value of one plan: hold it, then hand over, then look.

    Over `draws` futures rather than one. Taking the best of many candidates
    scored on a single draw is what turned this search from +2168 into -2041.
    """
    from oracle_mpc import _continue, mario_x

    env.load_state(here)
    policy._stack = list(stack)
    # The baseline is read here, from the same register the score is read
    # from. Mixing a folded cross-level progress with a level-local x is how
    # a metric once capped every good run at the end of 1-1.
    x0 = mario_x(env)
    ob, died = obs, False
    for press in plan:
        ob = env.step_buttons([press])
        now = (ob.debug or {}).get("lives")
        if l0 is not None and now is not None and now < l0:
            died = True
            break
    if died:
        policy._stack = list(stack)
        return DEATH
    mid = env.save_state()
    reached, dead_n = [], 0
    for _ in range(draws):
        env.load_state(mid)
        policy._stack = list(stack)
        gone, _ = _continue(env, policy, ob, l0, TAIL, repeat, tail_temp)
        if gone:
            dead_n += 1
        else:
            reached.append(float(mario_x(env) - x0))
    policy._stack = list(stack)
    if dead_n * 2 > draws:
        return DEATH
    return float(np.mean(reached))


def _cem(env, policy, obs, here, stack, l0, p0, args, rng):
    """Search the macro-action space, and return the best plan found."""
    prims = primitives()
    n_seg = HORIZON // SEG
    logp = np.zeros((n_seg, len(prims)))     # uniform to start
    best_plan, best_val = None, -1e18
    for _ in range(args.iters):
        p = np.exp(logp) / np.exp(logp).sum(1, keepdims=True)
        picks = np.stack([rng.choice(len(prims), size=args.samples, p=p[s])
                          for s in range(n_seg)], axis=1)
        vals = []
        for row in picks:
            plan = [b for s in row for b in prims[s][1]]
            v = _score(env, policy, obs, plan, here, stack, l0, p0,
                       args.repeat, args.tail_temp, args.draws)
            vals.append(v)
            if v > best_val:
                best_val, best_plan = v, plan
        elite = picks[np.argsort(vals)[-args.elites:]]
        # Refit to the elites, with a floor so a primitive that lost one round
        # can still be drawn in the next.
        counts = np.stack([np.bincount(elite[:, s], minlength=len(prims))
                           for s in range(n_seg)]).astype(float)
        logp = np.log((counts + args.floor) / (counts.sum(1, keepdims=True)
                                               + args.floor * len(prims)))
    return best_plan, best_val


def run(args, seed: int, search: bool) -> dict:
    from oracle_mpc import mario_x, templates

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin
    from nes_player.policy.robustify import progress_of

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = StableRetroAdapter(args.game, include_debug=True, state=args.state)
    policy = BCPolicy(args.checkpoint)
    obs = _begin(env)
    for _ in range(IDLE_STEP * seed % IDLE_MAX):
        obs = env.step_buttons([frozenset()])

    cands = templates(HORIZON)
    best_x, deaths, lives = 0, 0, (obs.debug or {}).get("lives")
    chosen: Counter = Counter()
    held: list = []
    pressed: frozenset = frozenset()
    branch_frames = 0

    for i in range(args.frames):
        if not held:
            here = env.save_state()
            stack = list(policy._stack)
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
            env.load_state(here)

            scored = []
            for name, plan in [("bc", seq), *cands]:
                scored.append((_score(env, policy, obs, plan, here, stack, l0,
                                      p0, args.repeat, args.tail_temp,
                                      args.draws),
                               name, plan))
                branch_frames += HORIZON + TAIL
            if search:
                plan, val = _cem(env, policy, obs, here, stack, l0, p0, args,
                                 rng)
                branch_frames += args.iters * args.samples * (HORIZON + TAIL)
                scored.append((val, "cem", plan))
            env.load_state(here)
            policy._stack = list(stack)

            val, name, plan = max(scored, key=lambda t: t[0])
            chosen[name] += 1
            held = list(plan[:args.commit])

        pressed = held.pop(0)
        obs = env.step_buttons([pressed])
        d = obs.debug or {}
        now = d.get("lives")
        if lives is not None and now is not None and now < lives:
            deaths += 1
        lives = now
        best_x = max(best_x, progress_of(d))
    env.close()
    return {"seed": seed, "best_x": best_x, "deaths": deaths,
            "branch_frames": branch_frames, "chosen": dict(chosen)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--commit", type=int, default=COMMIT)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--tail-temp", type=float, default=0.9,
                    help="temperature of the continuation the value is\n                         measured under. At 0.9 the same plan scored\n                         twice differs by as much as different plans do,\n                         so taking the best of many selects for lucky\n                         draws; at 0 the score reproduces exactly")
    ap.add_argument("--draws", type=int, default=1,
                    help="futures to average each candidate over")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--elites", type=int, default=4)
    ap.add_argument("--floor", type=float, default=0.5,
                    help="pseudocount when refitting, so a primitive that lost "
                         "one round can still be drawn in the next")
    args = ap.parse_args()

    arms = {}
    for name, search in (("oracle templates", False), ("oracle +cem", True)):
        rows = []
        for seed in range(args.seed0, args.seed0 + args.runs):
            rows.append(run(args, seed, search))
            print(json.dumps({"arm": name, **rows[-1]}), flush=True)
        arms[name] = rows

    base = np.array([r["best_x"] for r in arms["oracle templates"]], float)
    print()
    for name, rows in arms.items():
        a = np.array([r["best_x"] for r in rows], float)
        d = a - base
        sd = d.std(ddof=1) if len(d) > 1 else 0.0
        t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
        print(f"{name:18} median {np.median(a):7.1f}  mean {a.mean():7.1f}  "
              f"clears {int((a > 4000).sum()):2d}/{len(a)}  "
              f"deaths {sum(r['deaths'] for r in rows):3d}  "
              f"vs templates {d.mean():+8.1f}  t={t:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
