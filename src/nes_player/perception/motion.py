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
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # Small compact blobs — Contra's bullets are 2x2 to 8x8 — go through
            # a separate branch, so that MIN_AREA can stay high for everything
            # else instead of letting noise in everywhere.
            small = 4 <= area < MIN_AREA and w <= 8 and h <= 8
            if small or (MIN_AREA <= area <= MAX_AREA and w < 100 and h < 100):
                dets.append((x, y + HUD_H, w, h, centroids[i][0], centroids[i][1] + HUD_H,
                             small))

        # 3. Tracking: greedy matching by centre distance
        used = set()
        for slot in self._slots:
            best, best_d = None, MATCH_DIST
            for k, d in enumerate(dets):
                if k in used:
                    continue
                dist = float(np.hypot(d[4] - slot.cx, d[5] - slot.cy))
                if dist < best_d:
                    best, best_d = k, dist
            if best is None:
                slot.missed += 1
                slot.vx *= 0.8   # stalled or lost: let the velocity decay
                slot.vy *= 0.8
                continue
            used.add(best)
            x, y, w, h, cx, cy, small = dets[best]
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
