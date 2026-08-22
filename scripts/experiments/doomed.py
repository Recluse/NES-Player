"""Where the game is lost, as opposed to where it ends.

Every fatal decision the controller makes turns out to have had no safe
alternative: of 115, none. A perfect veto over the six plans moved deaths from
48 to 49. So the choice that kills is not the choice that looks fatal — it is
some earlier one, taken while every option still looked fine.

This looks one level further. At a decision, for each candidate: play the
commitment, then ask the console whether *every* plan from the state it lands
in dies. That state is `doomed` — no longer a place where a plan is fatal, but
a place from which nothing survives. Three classes follow:

    safe        no candidate leads to a doomed state
    boundary    some candidates lead to doomed, some do not
    lost        every candidate leads to doomed

The boundary is the interesting one and the whole point of the script. It is
the last moment where the decision still matters, and it is invisible to
everything measured so far: the plans all survive their own horizon, the
regret between them is nothing, and the death is a second and a half away.

    uv run python scripts/experiments/doomed.py runs/bc_smb_new --runs 4
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

IDLE_STEP, IDLE_MAX = 37, 1800
HORIZON = 48
EVERY = 8      # analysed decisions are expensive; take one in this many


def _all_die(env, policy, obs, l0, temperature, repeat, tail=0) -> bool:
    """From here, does every one of the six plans die inside the horizon?

    With `tail`, a plan counts as surviving only if the policy, taking over
    where it ends, is still alive that many frames later. Doom rarely arrives
    inside one plan; this is how far ahead we are willing to look for it,
    at linear cost rather than the branching factor to the k.
    """
    from oracle_mpc import templates

    here = env.save_state()
    stack = list(policy._stack)
    seq, o = [], obs
    for k in range(HORIZON):
        if k % repeat == 0:
            p, _ = policy.act(o.frame_rgb, temperature)
            p = p - {"START", "SELECT"}
        seq.append(p)
        o = env.step_buttons([p])
    policy._stack = list(stack)

    for _, plan in [("bc", seq), *templates(HORIZON)]:
        env.load_state(here)
        policy._stack = list(stack)
        survived = True
        ob = obs
        for k in range(HORIZON + tail):
            if k < HORIZON:
                press = plan[k]
            elif (k - HORIZON) % repeat == 0:
                press, _ = policy.act(ob.frame_rgb, temperature)
                press = press - {"START", "SELECT"}
            ob = env.step_buttons([press])
            now = (ob.debug or {}).get("lives")
            if l0 is not None and now is not None and now < l0:
                survived = False
                break
        if survived:
            env.load_state(here)
            policy._stack = list(stack)
            return False
    env.load_state(here)
    policy._stack = list(stack)
    return True


def classify(env, policy, obs, temperature, repeat, commit, tail=0):
    """Which candidates from here lead somewhere nothing survives."""
    from oracle_mpc import mario_x, templates

    here = env.save_state()
    stack = list(policy._stack)
    l0 = (obs.debug or {}).get("lives")
    x0 = mario_x(env)

    seq, o = [], obs
    for k in range(HORIZON):
        if k % repeat == 0:
            p, _ = policy.act(o.frame_rgb, temperature)
            p = p - {"START", "SELECT"}
        seq.append(p)
        o = env.step_buttons([p])
    policy._stack = list(stack)
    env.load_state(here)

    names, doomed, died_now = [], [], []
    for name, plan in [("bc", seq), *templates(HORIZON)]:
        env.load_state(here)
        policy._stack = list(stack)
        dead = False
        ob = obs
        for k in range(commit):
            ob = env.step_buttons([plan[k]])
            now = (ob.debug or {}).get("lives")
            if l0 is not None and now is not None and now < l0:
                dead = True
                break
        names.append(name)
        died_now.append(dead)
        doomed.append(True if dead
                      else _all_die(env, policy, ob, l0, temperature, repeat,
                                    tail))
    env.load_state(here)
    policy._stack = list(stack)
    return {"names": names, "doomed": doomed, "died_now": died_now, "x": x0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--commit", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--tail", type=int, default=0,
                    help="frames of policy play after each plan before a state "
                         "counts as survived; how far ahead doom is looked for")
    ap.add_argument("--out", default="runs/knowledge/doomed.json")
    args = ap.parse_args()

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin

    kinds: Counter = Counter()
    rows = []
    for seed in range(args.seed0, args.seed0 + args.runs):
        np.random.seed(seed)
        env = StableRetroAdapter(args.game, include_debug=True, state=args.state)
        policy = BCPolicy(args.checkpoint)
        obs = _begin(env)
        for _ in range(IDLE_STEP * seed % IDLE_MAX):
            obs = env.step_buttons([frozenset()])
        pressed: frozenset = frozenset()
        n = 0
        for i in range(args.frames):
            if i % (args.commit * EVERY) == 0 and i > 0:
                r = classify(env, policy, obs, args.temperature, args.repeat,
                             args.commit, args.tail)
                d = r["doomed"]
                kind = ("lost" if all(d) else
                        "boundary" if any(d) else "safe")
                kinds[kind] += 1
                r.update(kind=kind, seed=seed, frame=i)
                rows.append(r)
                n += 1
            if i % args.repeat == 0:
                pressed, _ = policy.act(obs.frame_rgb, args.temperature)
                pressed = pressed - {"START", "SELECT"}
            obs = env.step_buttons([pressed])
        env.close()
        print(json.dumps({"seed": seed, "analysed": n, **kinds}), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    total = sum(kinds.values())
    print()
    print(f"analysed decisions: {total}")
    for k in ("safe", "boundary", "lost"):
        print(f"  {k:9} {kinds[k]:5d}  {kinds[k] / max(total, 1):6.1%}")
    b = [r for r in rows if r["kind"] == "boundary"]
    if b:
        # Which plans are the trap, and which the way out. At a boundary the
        # difference between them is the whole game, and nothing the
        # controller currently measures can see it.
        trap = Counter(n for r in b for n, d in zip(r["names"], r["doomed"],
                                                    strict=True) if d)
        out_of = Counter(n for r in b for n, d in zip(r["names"], r["doomed"],
                                                      strict=True) if not d)
        print(f"  at a boundary, doomed plans: {dict(trap)}")
        print(f"  at a boundary, safe plans:   {dict(out_of)}")
        print(f"  level x of boundaries: "
              f"{np.percentile([r['x'] for r in b], [10, 50, 90]).round().tolist()}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
