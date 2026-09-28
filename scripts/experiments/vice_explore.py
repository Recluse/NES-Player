"""Go-Explore on a stretch the planner cannot see through.

Vice's bridge collapses behind the hero, and a death costs a life only
140-260 frames after the fall, so a 48+160-frame lookahead reads standing
still as safe. Go-Explore does not need to see the end: it remembers every
place it has reached alive, goes back to the furthest and least tried, and
pushes on from there with random button runs.

Alive has to be judged the way the lag allows. The game timer stops the
moment the hero is hit, so a state is filed only once the timer has ticked
after it — then he was still playing at that point. States saved after the
last tick before a lost life are thrown away.

    uv run python scripts/experiments/vice_explore.py \
        --load-state runs/vice/room2.state \
        --pos runs/vice/pos_by_level.json --iterations 3000 --out runs/vice/ge
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GAME = "ViceProjectDoom-Nes-v0"
LIVES, TIMER, STAGE, ROOM, HERO_Y, HP = 866, 862, 51, 168, 528, 640
DOOM = 260  # frames a life takes to go after the hit (measured 140-260)  # STAGE kept only for the cell
ACTIONS = (  # button set, weight
    ({"RIGHT"}, 3), ({"RIGHT", "A"}, 4), ({"RIGHT", "B"}, 3),
    ({"RIGHT", "A", "B"}, 3), ({"A"}, 1), ({"B"}, 1), ({"LEFT"}, 1),
    ({"LEFT", "A"}, 1), ({"DOWN"}, 1), ({"DOWN", "A"}, 0.5), ({"UP"}, 1),
    ({"RIGHT", "DOWN"}, 1), (set(), 0.5),
    # "TAP" = B pressed every other frame: a held B swings once, and the
    # owner's way with Vice's bosses is to spam the katana, then back off
    ({"TAP"}, 3), ({"RIGHT", "TAP"}, 2), ({"LEFT", "TAP"}, 1),
)


@dataclass
class Entry:
    cell: tuple
    state: bytes
    actions: list
    pos: int
    hp: int = 0
    chosen: int = 0


def main() -> int:
    import oracle_mpc as m

    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("--load-state", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--explore", type=int, default=120)
    ap.add_argument("--cell-px", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flat", action="store_true",
                    help="choose where to explore from by novelty alone, no "
                         "pull towards x: for stretches that climb or double "
                         "back (3-1 goes up a ladder and left along the top)")
    ap.add_argument("--boss", type=int, default=0,
                    help="RAM address of the boss's health (Vice: 644, the "
                         "E bar); its drop becomes part of the cell and pulls "
                         "selection towards the damage done")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m.SCAN_POS[GAME] = json.loads(Path(args.pos).read_text())[
        "position_bytes_consistent"]
    rng = np.random.default_rng(args.seed)
    sets = [frozenset(a) for a, _ in ACTIONS]
    w = np.array([p for _, p in ACTIONS], float)
    w /= w.sum()
    env = StableRetroAdapter(GAME, include_debug=True, state="default",
                             integration_dir=None)
    env.reset(seed=0)
    env.load_state(Path(args.load_state).read_bytes())

    def here():
        ram = env._env.get_ram()
        pos = m.game_pos(env, GAME)
        # hero y in the cell: an upper girder and the one below it are not
        # the same place, and only one of them leads on
        boss = int(ram[args.boss]) if args.boss else 0
        return (int(ram[STAGE]), int(ram[ROOM]), pos // args.cell_px,
                int(ram[HERO_Y]) // 32, boss), pos

    archive: dict = {}
    c, p = here()
    archive[c] = Entry(c, env.save_state(), [], p)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for it in range(args.iterations):
        es = list(archive.values())
        # Newest room first, then whatever has been tried least. Within a
        # room x is no guide: under the bridge the rooms are single screens
        # joined by ladders, and weighting by x parked the search against the
        # right wall of room 109 for a thousand iterations.
        room = max(e.cell[1] for e in es)
        top = max(e.pos for e in es)
        # a mild pull towards the front as well: without it the search sat
        # in room 84 for 4000 iterations that the x-weighted one crossed in 2000
        low = min(e.cell[4] for e in es)
        wt = np.array([(1 / (1 + e.chosen)) * (10.0 if e.cell[1] == room else 1.0)
                       * (1.0 if args.flat else (0.3 + e.pos / top) ** 2)
                       * (10.0 if args.boss and e.cell[4] == low else 1.0)
                       for e in es])
        start = es[int(rng.choice(len(es), p=wt / wt.sum()))]
        start.chosen += 1
        env.load_state(start.state)
        ram = env._env.get_ram()
        lives0 = int(ram[LIVES])
        acts = list(start.actions)
        pending: list[tuple[int, Entry]] = []
        held, left, f = frozenset(), 0, 0

        def commit(e):
            old = archive.get(e.cell)
            # more health first, then the earlier arrival: the bridge
            # collapses on a clock, and Hart reached 2-2 with 4 of 20
            if old is None or (e.hp, -len(e.actions)) > (old.hp, -len(old.actions)):
                if old is not None:
                    e.chosen = old.chosen
                archive[e.cell] = e

        # A life goes 140-260 frames after the fatal moment, so a state is
        # filed only once the run has gone on DOOM frames past it alive.
        # The timer test it replaces let falls through: the timer ticks
        # while he drops, and 3-1's "furthest" state was a hero in a pit.
        while f < args.explore + DOOM:
            if left == 0:
                held = sets[int(rng.choice(len(sets), p=w))]
                left = int(rng.integers(4, 24))
            left -= 1
            b = held
            if "TAP" in b:
                b = (b - {"TAP"}) | ({"B"} if f % 2 == 0 else set())
            env.step_buttons([b])
            acts.append(sorted(b))
            f += 1
            ram = env._env.get_ram()
            if int(ram[LIVES]) < lives0:
                break
            while pending and pending[0][0] <= f - DOOM:
                commit(pending.pop(0)[1])
            if f <= args.explore:
                c, p = here()
                pending.append((f, Entry(c, env.save_state(), list(acts), p,
                                         int(ram[HP]))))

        if it % 100 == 0 or it == args.iterations - 1:
            best = max(archive.values(), key=lambda e: e.pos)
            (out / "best.state").write_bytes(best.state)
            if args.boss:
                # the furthest place is not the fight's best: log and keep
                # the state with the boss lowest, too
                weak = min(archive.values(), key=lambda e: (e.cell[4], -e.hp))
                (out / "boss.state").write_bytes(weak.state)
                (out / "boss.json").write_text(json.dumps({
                    "load_state": args.load_state, "boss": weak.cell[4],
                    "hp": weak.hp, "actions": weak.actions}))
                print(json.dumps({"it": it, "boss_low": weak.cell[4],
                                  "hp_there": weak.hp}), flush=True)
            print(json.dumps({"it": it, "cells": len(archive),
                              "best_pos": best.pos, "best_cell": best.cell}),
                  flush=True)

    best = max(archive.values(), key=lambda e: e.pos)
    (out / "best.state").write_bytes(best.state)
    (out / "best.json").write_text(json.dumps({
        "load_state": args.load_state, "pos": best.pos, "cell": best.cell,
        "actions": best.actions}))
    print("best", best.pos, best.cell, len(best.actions), "frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
