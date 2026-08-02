"""Object positions straight from the console's sprite table.

The NES draws moving things as sprites, and the PPU is told where they are by
a 256-byte table it receives by DMA from CPU page $0200 every frame: 64
sprites, four bytes each — y, tile, attributes, x. Practically every game on
the machine uses that DMA, so reading page 2 gives exact positions of every
on-screen object in any game, with no per-game memory map to write.

This is deliberately NOT for the policy to use. The agent plays from pixels and
sound (spec §3), and nothing here is wired into it. It is for supervision and
measurement: the attention targets the network is trained against, and any
future teacher that is allowed to see the machine state while the student is
not. The cheat stays on the training side of the fence.

Existing datasets do not store RAM, but they do not need to: every episode
starts from a fixed state and the emulator is deterministic, so replaying the
recorded actions reproduces the run frame for frame — checked here rather than
assumed — and the sprite table can be read along the way.
"""

from pathlib import Path

import numpy as np

from nes_player.data.reader import Episode
from nes_player.emulator.controller import BUTTONS

SPRITES_VERSION = 1
OAM_PAGE = 0x200          # shadow OAM: the source of the per-frame DMA
OAM_SPRITES = 64
HIDDEN_Y = 0xEF           # the usual way to park an unused sprite off-screen
SPRITE_W = 8
# ponytail: 8x8 assumed. A game in 8x16 mode loses the bottom half of each
# sprite; at the (10, 11) attention grid that is a third of a cell, so it has
# not been worth reading PPUCTRL to find out. Revisit if masks look clipped.
SPRITE_H = 8
CHECK_EVERY = 600         # frames between replay-fidelity checks


def sprite_boxes(ram: np.ndarray) -> np.ndarray:
    """Visible sprites as (x, y) pairs, uint8, shape (n, 2)."""
    oam = ram[OAM_PAGE:OAM_PAGE + 4 * OAM_SPRITES].reshape(OAM_SPRITES, 4)
    vis = oam[oam[:, 0] < HIDDEN_Y]
    # The stored y is one less than the drawn y — the PPU renders a sprite on
    # the scanline after the one in the table.
    return np.stack([vis[:, 3], (vis[:, 0].astype(np.uint16) + 1)
                     .clip(0, 255).astype(np.uint8)], axis=1)


class ReplayMismatch(Exception):
    """The episode did not reproduce, so any RAM read from it would be fiction."""

    def __init__(self, frame: int, diff: float):
        super().__init__(f"replay diverged at frame {frame} (mean abs diff "
                         f"{diff:.2f}) — the dataset and the emulator disagree")
        self.frame, self.diff = frame, diff


def _open_matching_env(ep: Episode, first_action: int):
    """The emulator started where this episode started, or an error.

    Episodes do not record which integration state they began from, and getting
    it wrong produces a plausible-looking replay of a completely different run.
    So try the candidates and keep the one whose first frame is identical.
    """
    from nes_player.emulator.stable_retro import StableRetroAdapter

    want = np.asarray(ep.frames[0], np.int16)
    pressed = buttons_from_mask(first_action)
    closest = np.inf
    for state in (ep.metadata.get("state", "default"), None):
        env = StableRetroAdapter(ep.metadata["game"], include_debug=True, state=state)
        env.reset(seed=0)
        obs = env.step_buttons([pressed])
        diff = float(np.abs(obs.frame_rgb.astype(np.int16) - want).mean())
        if diff == 0.0:
            return env, state
        env.close()
        closest = min(closest, diff)
    raise ReplayMismatch(0, closest)


def buttons_from_mask(mask: int) -> frozenset[str]:
    return frozenset(b for i, b in enumerate(BUTTONS) if mask & (1 << i))


def episode_sprites(ep: Episode) -> np.ndarray:
    """Per-frame sprite positions, (N, 64, 2) uint8, zero-padded; cached.

    Padding rather than a ragged list keeps this a plain array that memory-maps
    and slices; a sprite count of zero at (0, 0) is indistinguishable from a
    real sprite parked there, which does not matter for a coverage mask.
    """
    cache = ep.path / f"sprites.v{SPRITES_VERSION}.npy"
    if cache.exists():
        return np.load(cache)

    actions = ep.actions
    n = int(actions.shape[0])
    env, _ = _open_matching_env(ep, int(actions[0, 0]))
    out = np.zeros((n, OAM_SPRITES, 2), np.uint8)
    try:
        out[0] = _pad(sprite_boxes(env._env.get_ram()))
        for i in range(1, n):
            obs = env.step_buttons([buttons_from_mask(int(actions[i, 0]))])
            out[i] = _pad(sprite_boxes(env._env.get_ram()))
            if i % CHECK_EVERY == 0 or i == n - 1:
                diff = float(np.abs(obs.frame_rgb.astype(np.int16)
                                    - np.asarray(ep.frames[i], np.int16)).mean())
                if diff > 0.01:
                    raise ReplayMismatch(i, diff)
    finally:
        env.close()
    np.save(cache, out)
    return out


def _pad(boxes: np.ndarray) -> np.ndarray:
    out = np.zeros((OAM_SPRITES, 2), np.uint8)
    out[:len(boxes)] = boxes[:OAM_SPRITES]
    return out


def sprite_mask(boxes: np.ndarray, frame_hw: tuple[int, int],
                out_hw: tuple[int, int], lead_dx: np.ndarray | None = None,
                ) -> np.ndarray:
    """Sprite positions as a coverage mask on the attention grid."""
    import cv2

    h, w = frame_hw
    full = np.zeros((h, w), np.uint8)
    for k, (x, y) in enumerate(boxes):
        if x == 0 and y == 0:
            continue          # padding
        x = int(x) + (0 if lead_dx is None else int(lead_dx[k]))
        x = max(0, min(x, w - SPRITE_W))
        y = min(int(y), h - SPRITE_H)
        if y < 0:
            continue
        full[y:y + SPRITE_H, x:x + SPRITE_W] = 255
    small = cv2.resize(full, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
    return (small > 16).astype(np.uint8)


class SpriteTracker:
    """Drop-in replacement for MotionTracker that reads the sprite table.

    Same surface — `update` returns Slots and sets `scroll_dx` — so the instinct
    policy can be run on exact object positions instead of inferred ones without
    touching its logic. That is the point: it isolates *perception* as a
    variable. If the policy plays no better with perfect object positions, the
    thing holding it back is what it decides, not what it sees.

    Camera scroll still comes from the pixels. It is not in the sprite table:
    sprite coordinates are relative to the screen, and where the screen is
    looking is a PPU register, not RAM.
    """

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._slots: list = []
        self._next_id = 0
        self.scroll_dx = 0.0

    def update(self, frame_rgb: np.ndarray, pressed: frozenset[str],
               ram: np.ndarray | None = None,
               boxes: np.ndarray | None = None) -> list:
        import cv2

        from nes_player.perception.motion import (
            HUD_H,
            MATCH_DIST,
            STALE_AFTER,
            VEL_EMA,
            Slot,
        )

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        play = gray[HUD_H:]
        if self._prev_gray is None:
            self._prev_gray = play
            return []
        (dx, _), _ = cv2.phaseCorrelate(self._prev_gray, play)
        self.scroll_dx = dx
        self._prev_gray = play
        # Live play hands over RAM; offline feature extraction hands over the
        # positions straight from the episode cache, so the run is not emulated
        # a second time just to read the same bytes again.
        if boxes is None:
            if ram is None:
                return []
            boxes = sprite_boxes(ram)

        # One character is drawn from several 8x8 sprites side by side, so the
        # boxes are painted and merged rather than reported one by one. The
        # dilation closes the seam between neighbouring tiles of one object
        # without joining two characters standing apart.
        h, w = frame_rgb.shape[:2]
        canvas = np.zeros((h, w), np.uint8)
        for x, y in boxes:
            if x == 0 and y == 0:
                continue          # padding in a cached table
            canvas[int(y):int(y) + SPRITE_H, int(x):int(x) + SPRITE_W] = 255
        canvas = cv2.dilate(canvas, np.ones((3, 3), np.uint8))
        canvas[:HUD_H] = 0          # the HUD is not part of the world
        n, _, stats, cent = cv2.connectedComponentsWithStats(canvas)
        # ponytail: two characters in a clinch merge into one component, as they
        # do for the motion tracker. Splitting them needs the per-sprite tile
        # ids; worth it only if the clinch turns out to matter.
        dets = [(stats[i][0], stats[i][1], stats[i][2], stats[i][3],
                 float(cent[i][0]), float(cent[i][1])) for i in range(1, n)]

        taken = set()
        for slot in self._slots:
            best, best_d = None, MATCH_DIST
            for k, d in enumerate(dets):
                if k in taken:
                    continue
                dist = float(np.hypot(d[4] - slot.cx, d[5] - slot.cy))
                if dist < best_d:
                    best, best_d = k, dist
            if best is None:
                slot.missed += 1
                slot.vx *= 0.8
                slot.vy *= 0.8
                continue
            taken.add(best)
            x, y, bw, bh, cx, cy = dets[best]
            slot.vx = (1 - VEL_EMA) * slot.vx + VEL_EMA * (cx - slot.cx)
            slot.vy = (1 - VEL_EMA) * slot.vy + VEL_EMA * (cy - slot.cy)
            slot.bbox, slot.cx, slot.cy = (x, y, bw, bh), cx, cy
            slot.age += 1
            slot.missed = 0
            slot.small = bw * bh < 64
            direction = ("RIGHT" in pressed) - ("LEFT" in pressed)
            if direction:
                world_vx = slot.vx - self.scroll_dx
                slot.ctrl_score = (0.98 * slot.ctrl_score
                                   + 0.35 * float(np.clip(world_vx * direction, -1.5, 1.5)))
        for k, d in enumerate(dets):
            if k not in taken:
                x, y, bw, bh, cx, cy = d
                self._slots.append(Slot(self._next_id, (x, y, bw, bh), cx, cy,
                                        small=bw * bh < 64))
                self._next_id += 1
        self._slots = [s for s in self._slots
                       if s.missed <= (300 if s.ctrl_prob > 0.7 else STALE_AFTER)]
        return [s for s in self._slots if s.age >= 3]


def main() -> int:
    """Draw the sprite table over an episode's frames, so it can be checked."""
    import argparse

    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episode")
    ap.add_argument("--out", default="sprites_check.jpg")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 300, 900, 1800])
    args = ap.parse_args()

    ep = Episode(Path(args.episode))
    boxes = episode_sprites(ep)
    panels = []
    for i in args.frames:
        img = np.asarray(ep.frames[i]).copy()
        n = 0
        for x, y in boxes[i]:
            if x == 0 and y == 0:
                continue
            n += 1
            cv2.rectangle(img, (int(x), int(y)),
                          (int(x) + SPRITE_W, int(y) + SPRITE_H), (0, 255, 0), 1)
        cv2.putText(img, f"f{i} sprites={n}", (3, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 0), 1, cv2.LINE_AA)
        panels.append(img)
    sheet = np.concatenate(panels, axis=0)
    cv2.imwrite(args.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
