"""Finding objects by motion (spec §10.9) — the version with no neural network.

Per frame:

1. Background scroll, by phase correlation between consecutive frames.
2. Compensate for the scroll and difference the frames: what survives is what
   moved relative to the background.
3. Blobs become boxes; greedy matching by centre distance gives stable ids and
   an EMA velocity.
4. Which slot is "me" — correlate the sign of its world velocity with LEFT and
   RIGHT being pressed.

This is deliberately classical. Slot attention was tried on the same problem
and failed; see docs/experiments.md.
"""

from dataclasses import dataclass

import cv2
import numpy as np

HUD_H = 32   # the score band most games put on top; excluded from scroll and diff
MIN_AREA, MAX_AREA = 24, 3000
MATCH_DIST = 28.0
VEL_EMA = 0.4
STALE_AFTER = 8

# Splitting a merged blob back into its fighters. In a beat-em-up two sprites
# at contact range become one connected component: the loser of the greedy
# match ghosts at its last position while the winner's centre jumps into the
# gap between them. The sign of "which way is the enemy" then becomes noise,
# and the agent turns around and punches empty air. Measured on Double Dragon,
# holding the last confident direction did not help — the fix has to keep the
# two centres apart in the first place.
SPLIT_MIN_EXTENT = 14   # a blob thinner than this cannot hold two fighters
SPLIT_MIN_GAP = 7       # predicted centres closer than this are one thing


@dataclass
class Slot:
    slot_id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0
    missed: int = 0
    ctrl_score: float = 0.0   # EMA agreement between vx and LEFT/RIGHT presses
    small: bool = False       # blob under MIN_AREA: a bullet, or scenery

    @property
    def ctrl_prob(self) -> float:
        return 1 / (1 + np.exp(-self.ctrl_score / 2))


def _slice_stats(labels: np.ndarray, comp: int, lo: int, hi: int, axis: int,
                 hud_h: int) -> tuple[int, int, int, int, float, float] | None:
    """Bounding box and centroid of one component inside a slice of the frame.

    `lo`/`hi` bound the slice along `axis` in screen coordinates; `labels` is in
    play coordinates, hence the HUD offset. Returns None when the slice holds no
    pixels of that component, which happens when a sprite is fully occluded.
    """
    if axis == 0:
        sub = labels[:, max(0, lo):max(0, hi)]
        off_x, off_y = max(0, lo), hud_h
    else:
        sub = labels[max(0, lo - hud_h):max(0, hi - hud_h), :]
        off_x, off_y = 0, max(hud_h, lo)
    ys, xs = np.nonzero(sub == comp)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()) + off_x, int(xs.max()) + off_x
    y0, y1 = int(ys.min()) + off_y, int(ys.max()) + off_y
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1,
            float(xs.mean()) + off_x, float(ys.mean()) + off_y)


def _split_detection(labels: np.ndarray, comp: int, det: tuple, tracks: list) -> list | None:
    """Cut one merged blob between the tracks that are all claiming it.

    The cut goes along whichever axis separates their predicted positions more,
    at the midpoints between them, so each fighter keeps a centre on its own
    side of the clinch. Returns one detection per track, in the same order, or
    None when the blob is too small to be two things.
    """
    x, y, w, h, _, _, small = det
    if small or len(tracks) < 2:
        return None
    pred = [(s.cx + s.vx, s.cy + s.vy) for s in tracks]
    spread_x = max(p[0] for p in pred) - min(p[0] for p in pred)
    spread_y = max(p[1] for p in pred) - min(p[1] for p in pred)
    axis = 0 if spread_x >= spread_y else 1
    if max(spread_x, spread_y) < SPLIT_MIN_GAP:
        return None
    if (w if axis == 0 else h) < SPLIT_MIN_EXTENT:
        return None

    order = sorted(range(len(tracks)), key=lambda i: pred[i][axis])
    lo_edge, hi_edge = (x, x + w) if axis == 0 else (y, y + h)
    out: list = [None] * len(tracks)
    for rank, i in enumerate(order):
        lo = lo_edge if rank == 0 else int(round((pred[order[rank - 1]][axis]
                                                  + pred[i][axis]) / 2))
        hi = hi_edge if rank == len(order) - 1 else int(round(
            (pred[i][axis] + pred[order[rank + 1]][axis]) / 2))
        st = _slice_stats(labels, comp, lo, hi, axis, HUD_H)
        if st is None:
            return None          # one side is fully occluded: not a clean split
        out[i] = (*st, small)
    return out


class MotionTracker:
    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._slots: list[Slot] = []
        self._next_id = 0
        self.scroll_dx = 0.0

    def update(self, frame_rgb: np.ndarray, pressed: frozenset[str]) -> list[Slot]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        play = gray[HUD_H:]
        if self._prev_gray is None:
            self._prev_gray = play
            return []

        # 1. Background scroll
        (dx, dy), _ = cv2.phaseCorrelate(self._prev_gray, play)
        self.scroll_dx = dx

        # 2. Difference, compensated for the scroll
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        prev_shifted = cv2.warpAffine(self._prev_gray, m, (play.shape[1], play.shape[0]))
        diff = cv2.absdiff(play, prev_shifted).astype(np.uint8)
        diff[:2] = diff[-2:] = 0
        _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        dets, det_comp = [], []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # Small compact blobs — Contra's bullets are 2x2 to 8x8 — go through
            # a separate branch, so that MIN_AREA can stay high for everything
            # else instead of letting noise in everywhere.
            small = 4 <= area < MIN_AREA and w <= 8 and h <= 8
            if small or (MIN_AREA <= area <= MAX_AREA and w < 100 and h < 100):
                dets.append((x, y + HUD_H, w, h, centroids[i][0], centroids[i][1] + HUD_H,
                             small))
                det_comp.append(i)

        # 3. Tracking. Each slot picks its nearest detection; unlike a plain
        # greedy match, two slots are allowed to pick the same one, because in
        # a clinch they genuinely are one blob. Those get split below.
        claims: dict[int, list[Slot]] = {}
        for slot in self._slots:
            best, best_d = None, MATCH_DIST
            for k, d in enumerate(dets):
                dist = float(np.hypot(d[4] - slot.cx, d[5] - slot.cy))
                if dist < best_d:
                    best, best_d = k, dist
            if best is None:
                slot.missed += 1
                slot.vx *= 0.8   # stalled or lost: let the velocity decay
                slot.vy *= 0.8
                continue
            claims.setdefault(best, []).append(slot)

        used = set(claims)
        assigned: list[tuple[Slot, tuple]] = []
        for k, tracks in claims.items():
            if len(tracks) == 1:
                assigned.append((tracks[0], dets[k]))
                continue
            parts = _split_detection(labels, det_comp[k], dets[k], tracks)
            if parts is None:
                # Not separable: the nearest track keeps the blob, the rest go
                # stale exactly as they did before splitting existed.
                nearest = min(tracks, key=lambda s: np.hypot(dets[k][4] - s.cx,
                                                             dets[k][5] - s.cy))
                for s in tracks:
                    if s is nearest:
                        assigned.append((s, dets[k]))
                    else:
                        s.missed += 1
                        s.vx *= 0.8
                        s.vy *= 0.8
                continue
            assigned.extend(zip(tracks, parts, strict=True))

        for slot, det in assigned:
            x, y, w, h, cx, cy, small = det
            slot.small = small
            slot.vx = (1 - VEL_EMA) * slot.vx + VEL_EMA * (cx - slot.cx)
            slot.vy = (1 - VEL_EMA) * slot.vy + VEL_EMA * (cy - slot.cy)
            slot.bbox, slot.cx, slot.cy = (x, y, w, h), cx, cy
            slot.age += 1
            slot.missed = 0
            # 4. Controllability: the sign of the WORLD velocity against the
            # buttons. When the camera follows the hero his screen velocity is
            # about zero, so the camera has to be added back in — the camera
            # moving right gives a negative scroll_dx, hence the subtraction.
            # Without this correction the agent cannot recognise itself at all.
            direction = ("RIGHT" in pressed) - ("LEFT" in pressed)
            if direction:
                world_vx = slot.vx - self.scroll_dx
                agree = np.clip(world_vx * direction, -1.5, 1.5)
                slot.ctrl_score = 0.98 * slot.ctrl_score + 0.35 * float(agree)
        for k, d in enumerate(dets):
            if k not in used:
                x, y, w, h, cx, cy, small = d
                self._slots.append(Slot(self._next_id, (x, y, w, h), cx, cy, small=small))
                self._next_id += 1
        # The controlled object often stands still, so its slot is kept far
        # longer than the rest before being forgotten.
        self._slots = [
            s for s in self._slots
            if s.missed <= (300 if s.ctrl_prob > 0.7 else STALE_AFTER)
        ]
        self._prev_gray = play
        # Only 'live' small blobs are reported: something with a real world
        # velocity, like a bullet. Twinkling stars and scenery either move with
        # the background or not at all, so their world velocity is near zero.
        return [
            s for s in self._slots
            if s.age >= 3 and (
                not s.small or np.hypot(s.vx - self.scroll_dx, s.vy) > 1.5)
        ]
