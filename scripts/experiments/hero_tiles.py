"""A6: which sprites are the player? Ask the buttons, not the eye.

A term that rewards "get higher up the screen" needs to know where the
player is. The first attempt took the median of everything that moved
between two frames; in Contra's base that is enemy fire, and holding UP,
DOWN or nothing all produced the same number. The objective was steering
on noise, and it took a scripted check to notice.

The console knows. Every sprite in the table carries a tile index — the
game's own name for what is being drawn — and the player's tiles are the
ones whose sprites move with the buttons. So: run two synchronous
branches from one state, one holding a direction and one doing nothing,
and keep the tiles whose sprites end up displaced between them. Enemies,
bullets and scenery march identically in both, so they cancel; only what
answers the controller survives.

    uv run python scripts/experiments/hero_tiles.py ContraJ-Nes-v0 \
        --load-state runs/oracle_adaptive/base_deep.state

Writes runs/knowledge/hero_ContraJ-Nes-v0.json.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_mpc import begin_any  # noqa: E402

HIDDEN_Y = 0xEF


def oam(ram) -> np.ndarray:
    return np.asarray(ram[0x200:0x200 + 256]).reshape(64, 4)


def run(env, state, chord, frames: int) -> list:
    """Play `frames` under one chord, returning the sprite table per frame."""
    env.load_state(state)
    out = []
    for _ in range(frames):
        env.step_buttons([chord])
        out.append(oam(env._env.get_ram()).copy())
    return out


def tile_centres(table: np.ndarray) -> dict:
    """Mean position of each visible tile index in one frame."""
    vis = table[table[:, 0] < HIDDEN_Y]
    out: dict = {}
    for tile in np.unique(vis[:, 1]):
        rows = vis[vis[:, 1] == tile]
        out[int(tile)] = (float(rows[:, 3].mean()), float(rows[:, 0].mean()))
    return out


def displaced(a: np.ndarray, b: np.ndarray, axis: int, sign: int,
              min_px: float) -> set:
    """Tiles whose sprites sit further along `axis` in b than in a."""
    ca, cb = tile_centres(a), tile_centres(b)
    out = set()
    for tile, pos in cb.items():
        if tile in ca and sign * (pos[axis] - ca[tile][axis]) >= min_px:
            out.add(tile)
    return out


def main() -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--load-state", default="")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--min-px", type=float, default=4.0)
    ap.add_argument("--repeats", type=int, default=6,
                    help="probes from successive states; a tile must answer "
                         "in most of them, so one lucky frame is not a hero")
    args = ap.parse_args()

    integ_root = Path(__file__).resolve().parents[2] / "integrations"
    integ = str(integ_root) if (integ_root / args.game).exists() else None
    env = StableRetroAdapter(args.game, include_debug=True, state=None,
                             integration_dir=integ)
    if args.load_state:
        env.reset(seed=0)
        env.load_state(Path(args.load_state).read_bytes())
        env._env.data.set_value("lives", 2)
    else:
        begin_any(env, args.game)

    votes: Counter = Counter()
    for rep in range(args.repeats):
        here = env.save_state()
        idle = run(env, here, frozenset(), args.frames)
        for chord, axis, sign in ((frozenset({"RIGHT"}), 0, +1),
                                  (frozenset({"LEFT"}), 0, -1),
                                  (frozenset({"UP"}), 1, -1),
                                  (frozenset({"DOWN"}), 1, +1)):
            moved = run(env, here, chord, args.frames)
            votes.update(displaced(idle[-1], moved[-1], axis, sign,
                                   args.min_px))
        # Let time pass between probes rather than walking: walking pins
        # the player against a wall, where no direction moves him, and in a
        # lethal room it also gets him killed, after which every later
        # probe is blank. Both happened.
        env.load_state(here)
        for _ in range(20):
            env.step_buttons([frozenset()])

    # One answer is enough, because the test is already causal and
    # directional: the tile has to move the way the button points, in a
    # branch that differs from the idle one by nothing else. Most
    # directions are blocked in any given spot, so demanding several
    # answers demands the impossible.
    need = 1
    tiles = sorted(t for t, c in votes.items() if c >= need)
    print("tile votes (top):", votes.most_common(10))
    print(f"hero tiles (>= {need} votes):", tiles)

    out = Path("runs/knowledge") / f"hero_{args.game}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Merge, never replace: the player is drawn from different tiles in
    # different poses, so each pose adds a few. Overwriting a good scan
    # with a blank one from a spot where every direction is blocked is a
    # mistake this made once already.
    before = json.loads(out.read_text())["tiles"] if out.exists() else []
    tiles = sorted(set(before) | set(tiles))
    out.write_text(json.dumps({"game": args.game, "tiles": tiles,
                               "votes": dict(votes.most_common(24)),
                               "probes": args.repeats}, indent=2))
    print("tiles after merge:", tiles)
    print("wrote", out)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
