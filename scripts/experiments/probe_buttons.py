"""What does each button actually do in this game? Measure, don't assume.

The five planner behaviours were written by hand for Mario and happened to
fit Contra — until a held B turned out to fire one bullet and three days of
hypotheses fell to a tap. This probe replaces the assumption with a scan:
from a few states along a scripted advance, hold or tap every button and
every direction+button chord for sixty frames and record three things —

    dpos      forward position gained (the game's own counter)
    died      whether it cost a life
    sprites   new OAM sprites versus doing nothing (bullets are sprites)

The table is the game's control manual, read off the console. Templates
can be assembled from it instead of from Mario folklore.

    uv run python scripts/experiments/probe_buttons.py SuperMarioBros-Nes-v0
    uv run python scripts/experiments/probe_buttons.py ContraJ-Nes-v0 --state none
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_mpc import begin_any, game_pos  # noqa: E402

BUTTONS = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT"]
CHORDS = [frozenset({b}) for b in BUTTONS] + [
    frozenset({"RIGHT", "A"}), frozenset({"RIGHT", "B"}),
    frozenset({"RIGHT", "A", "B"}), frozenset({"UP", "B"}),
    frozenset({"UP", "RIGHT", "B"}), frozenset({"DOWN", "B"}),
    frozenset({"LEFT", "B"}), frozenset({"LEFT", "A"}),
    frozenset({"UP", "A"}), frozenset({"DOWN", "A"}),
]
PROBE_FRAMES = 60


def oam_count(env) -> int:
    ram = env._env.get_ram()
    ys = ram[0x200:0x200 + 256:4]
    return int((ys < 0xEF).sum())


def measure(env, game, state, chord, mode, l0):
    env.load_state(state)
    p0, s0 = game_pos(env, game), oam_count(env)
    died, smax = False, s0
    for k in range(PROBE_FRAMES):
        on = mode == "hold" or k % 4 < 2
        obs = env.step_buttons([chord if on else frozenset()])
        smax = max(smax, oam_count(env))
        now = (obs.debug or {}).get("lives")
        if l0 is not None and now is not None and now < l0:
            died = True
            break
    return game_pos(env, game) - p0, died, smax - s0


def main() -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--state", default="default")
    ap.add_argument("--advance", default="RIGHT",
                    help="button(s) held to walk to the probe states, "
                         "comma-separated")
    ap.add_argument("--at", type=int, nargs="+", default=[60, 300, 600],
                    help="frames of advance before each probe state")
    args = ap.parse_args()
    state = None if args.state in ("none", "") else args.state
    root = Path(__file__).resolve().parents[2] / "integrations"
    integ = str(root) if (root / args.game).exists() else None
    env = StableRetroAdapter(args.game, include_debug=True, state=state,
                             integration_dir=integ)
    obs = begin_any(env, args.game)
    adv = frozenset(args.advance.split(","))
    states, t = [], 0
    for target in sorted(args.at):
        while t < target:
            obs = env.step_buttons([adv])
            t += 1
        states.append(env.save_state())
    l0 = (obs.debug or {}).get("lives")

    # baseline: doing nothing from each state
    base = [measure(env, args.game, s, frozenset(), "hold", l0)
            for s in states]
    rows = []
    for chord in CHORDS:
        for mode in ("hold", "tap"):
            r = [measure(env, args.game, s, chord, mode, l0) for s in states]
            rows.append({
                "chord": "+".join(sorted(chord)), "mode": mode,
                "dpos": round(float(np.mean([x[0] for x in r])), 1),
                "died": round(float(np.mean([x[1] for x in r])), 2),
                "sprites": round(float(np.mean(
                    [x[2] - b[2] for x, b in zip(r, base, strict=True)])), 1),
            })
    env.close()
    rows.sort(key=lambda r: -r["dpos"])
    print(f"# {args.game}: {len(states)} probe states, {PROBE_FRAMES} frames each")
    print(f"{'chord':16} {'mode':5} {'dpos':>7} {'died':>5} {'sprites':>8}")
    for r in rows:
        print(f"{r['chord']:16} {r['mode']:5} {r['dpos']:7.1f} "
              f"{r['died']:5.2f} {r['sprites']:8.1f}")
    out = Path("runs/knowledge") / f"buttons_{args.game}.json"
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
