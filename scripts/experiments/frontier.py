"""A2: a frontier archive keyed by what the screen looks like, not by RAM.

The Go-Explore in `policy/go_explore.py` works, and its cell is a triple
from the Super Mario Bros. RAM map: level, camera, clock. That is the per-game
variable a universal agent is supposed to discover, and it fails exactly
where the planner already fails — Contra's base does not scroll, so every
state in it is one cell, and there is no gradient to explore along.

This archive keys cells on perception:

    scene     a coarse hash of the picture: 12x10 luma cells, three bits
              each. Two rooms of a base differ; two frames of the same
              room do not.
    body      where the thing the player controls is, in a 6x5 grid, from
              A0's own record of which pixels answer the buttons.
    alive     whether a life was just lost, so a corpse is not a place.

Exploration is the usual loop: pick a remembered place, restore the
console there, play a short burst of the scan's own templates, and keep
every new cell with the state that reached it. What comes out is a map of
where the game can be got to, and the states to resume from.

    uv run python scripts/experiments/frontier.py ContraJ-Nes-v0 \
        --load-state runs/oracle_adaptive/level2_start.state --iterations 200

The metric stays out of the key by construction: cells are perceptual, and
progress is reported from the game's own counter afterwards, so a run
cannot be scored by the thing it was steered with.
"""

import argparse
import json
import random
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_mpc import begin_any, game_pos, scan_templates  # noqa: E402

GRID_X, GRID_Y = 6, 5
QUANT = 64        # calibrated below; finer keys on sprite noise
BODY_X, BODY_Y = 6, 5


def scene_hash(frame: np.ndarray) -> int:
    """A coarse fingerprint of the picture: 6x5 cells, two bits each.

    Calibrated on Contra's base, 2200 frames through two rooms and the
    door between them. At 12x10 cells and three bits the first room alone
    gives 260 scenes — the hash is keying on where the sprites are. At
    6x5 and two bits it gives 22, the second room 4, and only 2 hashes are
    shared between them. Coarser still (4x4) and the rooms start to
    collapse into each other.
    """
    g = frame[..., :3].mean(-1)
    h, w = g.shape
    cells = g[: h // GRID_Y * GRID_Y, : w // GRID_X * GRID_X]
    cells = cells.reshape(GRID_Y, h // GRID_Y, GRID_X, w // GRID_X).mean((1, 3))
    return int(zlib.crc32((cells / QUANT).astype(np.uint8).tobytes()))


def body_cell(frame: np.ndarray, prev: np.ndarray | None) -> tuple:
    """Where the moving thing is, coarsely; the frame difference finds it.

    A0 locates the controlled body properly, with branches. Inside a search
    loop that is too expensive per step, and the cheap version is enough
    for a cell key: what moved between two frames is mostly the player.
    """
    if prev is None:
        return (0, 0)
    d = np.abs(frame[..., :3].mean(-1) - prev[..., :3].mean(-1))
    if d.max() < 8:
        return (0, 0)
    ys, xs = np.nonzero(d > d.max() * 0.5)
    if not len(xs):
        return (0, 0)
    return (int(np.median(xs)) * BODY_X // frame.shape[1],
            int(np.median(ys)) * BODY_Y // frame.shape[0])


@dataclass
class Entry:
    cell: tuple
    state: bytes
    pos: int
    lives: int
    chosen: int = 0


@dataclass
class Archive:
    entries: dict = field(default_factory=dict)
    found: int = 0

    def consider(self, e: Entry) -> bool:
        old = self.entries.get(e.cell)
        if old is None:
            self.entries[e.cell] = e
            self.found += 1
            return True
        # a place reached with more lives in hand is a better place to
        # resume from than the same place reached dying
        if (e.lives, e.pos) > (old.lives, old.pos):
            e.chosen = old.chosen
            self.entries[e.cell] = e
            return True
        return False

    def pick(self, rng: random.Random) -> Entry:
        """Prefer the least-explored places, ties broken at random."""
        best = min(e.chosen for e in self.entries.values())
        pool = [e for e in self.entries.values() if e.chosen == best]
        e = rng.choice(pool)
        e.chosen += 1
        return e


def main() -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--load-state", default="")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--burst", type=int, default=120,
                    help="frames played from a resumed place")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-best", default="",
                    help="write the state of the furthest place reached")
    args = ap.parse_args()

    integ_root = Path(__file__).resolve().parents[2] / "integrations"
    integ = str(integ_root) if (integ_root / args.game).exists() else None
    env = StableRetroAdapter(args.game, include_debug=True, state=None,
                             integration_dir=integ)
    if args.load_state:
        obs = env.reset(seed=0)
        env.load_state(Path(args.load_state).read_bytes())
        env._env.data.set_value("lives", 2)
        obs = env.step_buttons([frozenset()])
    else:
        obs = begin_any(env, args.game)

    templates = [p for _, p in scan_templates(args.burst, args.game)] \
        if (Path("runs/knowledge") / f"control_{args.game}.json").exists() \
        else [[frozenset({"B", "RIGHT"})] * args.burst,
              [frozenset({"A", "RIGHT"})] * args.burst,
              [frozenset({"RIGHT"})] * args.burst,
              [frozenset({"UP"})] * args.burst,
              [frozenset({"B"})] * args.burst]
    rng = random.Random(args.seed)

    arc = Archive()
    lives0 = (obs.debug or {}).get("lives")
    arc.consider(Entry((scene_hash(obs.frame_rgb), (0, 0)), env.save_state(),
                       game_pos(env, args.game), int(lives0 or 0)))
    best_pos, best_state = game_pos(env, args.game), env.save_state()
    deaths = 0

    for it in range(args.iterations):
        e = arc.pick(rng)
        env.load_state(e.state)
        prev = None
        plan = rng.choice(templates)
        lives = e.lives
        for k in range(args.burst):
            press = plan[k % len(plan)] if rng.random() < 0.8 \
                else rng.choice(templates)[0]
            obs = env.step_buttons([press])
            now = (obs.debug or {}).get("lives")
            if now is not None and now < lives:
                deaths += 1
                break
            lives = now if now is not None else lives
            cell = (scene_hash(obs.frame_rgb), body_cell(obs.frame_rgb, prev))
            prev = obs.frame_rgb
            pos = game_pos(env, args.game)
            arc.consider(Entry(cell, env.save_state(), pos, int(lives or 0)))
            if pos > best_pos:
                best_pos, best_state = pos, env.save_state()
        if (it + 1) % 25 == 0:
            print(f"{it + 1:4d} iterations: {len(arc.entries):5d} cells, "
                  f"{arc.found} found, best position {best_pos}, "
                  f"{deaths} deaths", flush=True)

    scenes = len({c[0] for c in arc.entries})
    print(f"done: {len(arc.entries)} cells over {scenes} distinct scenes, "
          f"best position {best_pos}")
    if args.save_best:
        Path(args.save_best).write_bytes(best_state)
        print("wrote", args.save_best)
    out = Path("runs/knowledge") / f"frontier_{args.game}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"game": args.game, "cells": len(arc.entries),
                               "scenes": scenes, "best_pos": best_pos,
                               "iterations": args.iterations,
                               "deaths": deaths}, indent=2))
    print("wrote", out)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
