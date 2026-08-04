"""Where the floor stops. The one thing on screen that kills and is not an object.

Everything else the agent reacts to is a sprite: it moves, so the motion tracker
finds it, and the console's own sprite table lists it. A pit is neither. It is
an absence of background tiles, it never moves, and to every perception module
in this project it simply does not exist — which is why the agent runs into one
at full speed. The owner put it exactly: it does not stand at the pit and it is
not stuck, it just never tries to jump.

That also rules out the obvious fix. A rule that reacts to being stuck cannot
help, because nothing gets stuck; the signal has to arrive *before* the edge.

Finding the floor needs no memory and no model. A NES level is drawn on a grid
of tiles over a flat background, and the background colour is whatever fills the
top of the screen. So a column of screen whose lower part is that colour has
nothing to stand on. Overworld or cave, blue sky or black, the same test works —
it asks "is there anything here" rather than "is this a floor".
"""

from __future__ import annotations

import numpy as np

SKY_ROWS = slice(40, 70)     # below the HUD, above anything standing on the floor
FLOOR_ROWS = slice(196, 222)  # the band a walkable floor occupies
BG_TOLERANCE = 40            # summed RGB distance still counted as the background
GAP_SHARE = 0.8              # share of the band that must be background to be a hole
MIN_GAP_PX = 12              # narrower than this is a seam between tiles, not a pit
# Past this share of the screen there is no floor anywhere, which does not mean
# a very wide pit — it means we are not looking at a level. A death fades the
# picture to black, and black matches the background by construction, so without
# this the detector reports a hole across the whole screen at every death.
NOT_A_LEVEL = 0.9


def background_colour(frame_rgb: np.ndarray) -> np.ndarray:
    """What the empty part of the screen looks like in this level."""
    return np.median(frame_rgb[SKY_ROWS].reshape(-1, 3), axis=0)


def floor_gaps(frame_rgb: np.ndarray) -> np.ndarray:
    """Per screen column: is there nothing to stand on?

    Narrow runs are filled in. Tile seams and the dark line under a brick row
    produce single background-coloured columns that are not holes, and a rule
    that jumps at those would jump constantly.
    """
    bg = background_colour(frame_rgb)
    band = frame_rgb[FLOOR_ROWS].astype(np.int16)
    empty = (np.abs(band - bg).sum(axis=2) < BG_TOLERANCE).mean(axis=0) > GAP_SHARE
    if empty.mean() > NOT_A_LEVEL:
        return np.zeros_like(empty)
    return _drop_narrow(empty, MIN_GAP_PX)


def _drop_narrow(mask: np.ndarray, min_run: int) -> np.ndarray:
    out = mask.copy()
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start < min_run:
                out[start:i] = False
            start = None
    if start is not None and len(mask) - start < min_run:
        out[start:] = False
    return out


def gap_ahead(frame_rgb: np.ndarray, hero_cx: float, facing: int = 1,
              reach: int = 64, gaps: np.ndarray | None = None) -> int | None:
    """Distance in pixels to the near edge of the next hole, or None.

    `facing` is +1 for right, -1 for left. Only the ground the hero is heading
    over is interesting; a hole behind is somewhere already survived.

    `gaps` is the column mask when the caller already has it; asking two
    questions of one frame should not scan it twice.
    """
    if gaps is None:
        gaps = floor_gaps(frame_rgb)
    x0 = int(round(hero_cx))
    width = len(gaps)
    for d in range(4, reach):
        x = x0 + facing * d
        if 0 <= x < width and gaps[x]:
            return d
    return None
