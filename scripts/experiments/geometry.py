"""The surface profile, read from the contact map instead of from pixels.

Sixteen rays ahead of the hero, in tile units. For ray k at dx = 12 + 8k
pixels in the facing direction: scan the contact map downward from the feet
for the first proven-solid tile, then up along the contiguous solid run to
its top. Three channels per ray, exactly as the design review specified:

    s[k]       surface height relative to the feet, in tiles, clip +/-6;
               0.0 where unknown or hole — magnitude carries only measured
               heights
    hole[k]    1 if the scan found provably-empty all the way down
    known[k]   1 if the ray's answer rests on evidence at all

The contact map only knows where Mario has been, so `known` is honest low
coverage rather than a defect: a consumer sees exactly which rays speak.

    from geometry import load_map, profile
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

RAYS = 16
RAY0_PX, RAY_STEP_PX = 12, 8
CLIP_TILES = 6.0
DOWN_TILES = 4          # how far below the feet a surface is looked for
MIN_EVIDENCE = 2        # occurrences before a tile counts as proven


def load_map(path: str | Path):
    """(solid, empty) boolean dicts keyed by (tx, ty), thresholded."""
    z = np.load(path)
    ev = z["evidence"]                    # (n, 4): tx, ty, solid, empty
    solid, empty = {}, {}
    for tx, ty, s_, e_ in ev:
        # Occupancy wins: a tile the body ever crossed is passable, however
        # often something also stood on its seam.
        if e_ >= MIN_EVIDENCE:
            empty[(int(tx), int(ty))] = True
        elif s_ >= MIN_EVIDENCE:
            solid[(int(tx), int(ty))] = True
    return solid, empty


def profile(solid: dict, empty: dict, wx: float, sy: float,
            facing: int = 1) -> np.ndarray:
    """(3, RAYS) float32: heights, holes, known — see module docstring."""
    out = np.zeros((3, RAYS), np.float32)
    feet_ty = int(sy + 16) // 16
    for k in range(RAYS):
        tx = int(wx + facing * (RAY0_PX + RAY_STEP_PX * k)) // 16
        top = None
        below_empty = 0
        for ty in range(feet_ty - int(CLIP_TILES), feet_ty + DOWN_TILES + 1):
            key = (tx, ty)
            if key in solid:
                if top is None or ty < top:
                    top = ty
            elif key in empty and ty >= feet_ty:
                below_empty += 1
        if top is not None:
            out[0, k] = float(np.clip(feet_ty - top, -CLIP_TILES, CLIP_TILES))
            out[2, k] = 1.0
        elif below_empty >= 2:
            # No solid found and at least two rows at or below the feet are
            # proven passable. Below a real pit Mario's fall leaves evidence
            # only part-way down, so demanding the whole scan be proven would
            # call every hole unknown.
            out[1, k] = 1.0
            out[2, k] = 1.0
    return out


def demo() -> None:
    """A synthetic map: flat floor, a two-tile hole, a step up. Fails loudly."""
    solid, empty = {}, {}
    for tx in range(0, 40):
        if tx in (12, 13):
            for ty in range(2, 13):     # provably empty all the way down
                empty[(tx, ty)] = True
            continue
        solid[(tx, 10)] = True          # floor at row 10
        for ty in range(2, 10):
            empty[(tx, ty)] = True
    for ty in (8, 9):
        solid[(20, ty)] = True          # a two-tile step

    p = profile(solid, empty, wx=100, sy=144)   # feet on the floor row
    assert p[2].all(), "every ray should be known on the synthetic map"
    flat = p[0][p[1] == 0]
    assert (np.abs(flat[:2]) <= 1).all(), f"near rays should be ~flat: {p[0]}"
    hole_rays = np.where(p[1] == 1)[0]
    assert len(hole_rays), "the hole should be visible"
    hx = [int(100 + 12 + 8 * k) // 16 for k in hole_rays]
    assert set(hx) <= {12, 13}, f"holes at wrong tiles: {hx}"
    step = [k for k in range(RAYS)
            if int(100 + 12 + 8 * k) // 16 == 20]
    assert all(p[0][k] == 2.0 for k in step), f"step should read +2: {p[0]}"
    print("demo ok:", np.array2string(p[0], precision=0))


if __name__ == "__main__":
    demo()
