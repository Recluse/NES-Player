"""Find the console's tile map by what Mario's body proves, not by what pixels look like.

A first attempt matched every RAM window against a pixel "solid" mask and
failed: the mask calls clouds, bushes, enemies and text solid, and the map does
not, so plain agreement never beat the 0.822 of predicting empty everywhere.
The reference was wrong, not the search.

Physics gives a clean one. Mario's body cannot overlap a solid tile, so every
tile his box covers is **empty** — thousands of labels a run, free and certain.
And when he is standing, the tile under his feet is **solid**. Tiles with no
evidence stay unknown and are not scored.

With a reference that means what the map means, every window in RAM is read as
two screens of 13x16 and scored by balanced accuracy on labelled tiles only.

    uv run python scripts/experiments/find_tilemap.py runs/bc_smb_new
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MARIO_X, MARIO_Y, AIRBORNE = 0x3AD, 0xCE, 0x1D
ROWS, COLS = 13, 16          # the map's shape, one screen
BOX_W, BOX_H = 16, 16        # small Mario; big Mario only adds empty evidence


def collect(checkpoint, game, state, seeds, frames, repeat, temperature):
    """Occupancy evidence in world tiles, and the RAM beside it."""
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin

    solid: dict = {}
    empty: dict = {}
    snaps = []
    env = StableRetroAdapter(game, include_debug=True, state=state)
    policy = BCPolicy(checkpoint)
    for seed in seeds:
        np.random.seed(seed)
        policy.reset()
        obs = _begin(env)
        pressed: frozenset = frozenset()
        for i in range(frames):
            if i % repeat == 0:
                pressed, _ = policy.act(obs.frame_rgb, temperature)
                pressed = pressed - {"START", "SELECT"}
            obs = env.step_buttons([pressed])
            ram = env._env.get_ram()
            d = obs.debug or {}
            cam = int(d.get("xscrollHi", 0)) * 256 + int(d.get("xscrollLo", 0))
            wx = int(ram[0x6D]) * 256 + int(ram[0x86])
            sy = int(ram[MARIO_Y])
            if not (0 < sy < 208):
                continue
            # Every tile the body covers is empty; he cannot be inside a wall.
            for tx in range(wx // 16, (wx + BOX_W - 1) // 16 + 1):
                for ty in range(sy // 16, (sy + BOX_H - 1) // 16 + 1):
                    empty[(tx, ty)] = empty.get((tx, ty), 0) + 1
            # Standing: the tile under the feet is solid.
            if int(ram[AIRBORNE]) == 0:
                ty = (sy + BOX_H) // 16
                for tx in range(wx // 16, (wx + BOX_W - 1) // 16 + 1):
                    solid[(tx, ty)] = solid.get((tx, ty), 0) + 1
            if cam % 16 == 0 and i > 120:
                snaps.append((ram.copy(), cam))
    env.close()
    return solid, empty, snaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    solid, empty, snaps = collect(args.checkpoint, args.game, args.state,
                                  range(args.seeds), args.frames, args.repeat,
                                  args.temperature)
    # A tile the body ever occupied is empty, whatever else was seen there —
    # standing evidence sits one row below and should not collide, but a moving
    # platform or a mistake would show up here rather than silently.
    both = set(solid) & set(empty)
    truth = {k: True for k in solid if k not in empty}
    truth.update({k: False for k in empty})
    print(f"{len(snaps)} aligned frames; tiles: {sum(truth.values())} solid, "
          f"{len(truth) - sum(truth.values())} empty, "
          f"{len(both)} contradictory (dropped to empty)")

    rams = np.stack([s[0] for s in snaps])
    cams = np.array([s[1] for s in snaps])
    col0 = cams // 16

    # The labelled cells of each frame, in map coordinates.
    idx, lab = [], []
    for f, cam in enumerate(cams):
        for c in range(COLS):
            tx = col0[f] + c
            for r in range(ROWS):
                v = truth.get((tx, r))
                if v is None:
                    continue
                idx.append((f, r, tx))
                lab.append(v)
    if not idx:
        raise SystemExit("no labelled cells; collect more frames")
    idx = np.array(idx)
    lab = np.array(lab)
    print(f"{len(lab)} labelled cells, {lab.mean():.1%} solid")

    def balanced(pred):
        p, n = lab, ~lab
        if not p.any() or not n.any():
            return 0.0
        return 0.5 * (pred[p].mean() + (~pred[n]).mean())

    # Three things are guesses and all three are swept: where the map's first
    # row sits relative to the screen, whether it is stored by column or by
    # row, and whether a world column lands in the buffer by tile parity.
    best = []
    for off in range(0x400, 0x520):
        blk = rams[:, off:off + 2 * ROWS * COLS]
        if blk.shape[1] < 2 * ROWS * COLS:
            break
        for order in ("col", "row"):
            t = (blk.reshape(len(rams), 2 * COLS, ROWS).transpose(0, 2, 1)
                 if order == "col"
                 else blk.reshape(len(rams), ROWS, 2 * COLS))
            for dr in range(-3, 4):
                r = idx[:, 1] + dr
                ok = (r >= 0) & (r < ROWS)
                if ok.sum() < 500:
                    continue
                # The buffer is a ring of 32 columns and its phase is not
                # necessarily zero; fitting one global phase removes the
                # confound that otherwise hides in the base address.
                for k in range(32):
                    c = (idx[ok, 2] + k) % (2 * COLS)
                    pred = t[idx[ok, 0], r[ok], c] != 0
                    p, n = lab[ok], ~lab[ok]
                    if not p.any() or not n.any():
                        continue
                    sc = 0.5 * (pred[p].mean() + (~pred[n]).mean())
                    best.append((sc, hex(off), order, dr, k))
    best.sort(reverse=True)
    print("  baseline: predict empty everywhere -> balanced 0.500")
    for a, off, order, dr, k in best[:8]:
        print(f"  {off} {order} row{dr:+d} phase{k:+d}: balanced {a:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
