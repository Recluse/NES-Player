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

SPRITES_VERSION = 2
OAM_PAGE = 0x200          # shadow OAM: the source of the per-frame DMA
OAM_SPRITES = 64
HIDDEN_Y = 0xEF           # the usual way to park an unused sprite off-screen
SPRITE_W = 8
# ponytail: 8x8 assumed. A game in 8x16 mode loses the bottom half of each
# sprite; at the (10, 11) attention grid that is a third of a cell, so it has
# not been worth reading PPUCTRL to find out. Revisit if masks look clipped.
SPRITE_H = 8
CHECK_EVERY = 600         # frames between replay-fidelity checks
# This rule is wrong about who the player is in roughly a quarter of frames —
# measured with the trained policy driving, an enemy was chosen as the hero in
# 26.9% of them (18.3 / 31.9 / 30.4 over three seeds). The mechanism is partly
# understood: a stationary player earns no score while the decay keeps running,
# so a stalled hero fades and a drifting enemy overtakes him, at ctrl_prob 0.42
# against 0.77. Two fixes were tried and neither survived measurement.
# Skipping the update when nothing moved: 26.9% -> 25.5%, inside the seed
# spread. Additionally penalising movement while no direction is pressed:
# 12.3% -> 22.0% on the seed it was built against, because Mario keeps sliding
# after the button is released and the penalty lands on him too.
# One fix survived measurement: vertical response to the jump button, below.
JUMP_EPS = 0.3


def sprite_boxes(ram: np.ndarray) -> np.ndarray:
    """Visible sprites as (x, y, tile), uint8, shape (n, 3).

    The tile index is the game's own name for a visual component, and it used
    to be dropped here. Position alone says where something is and nothing
    about what it is, which left identity to be guessed from a pixel crop —
    and that crop is mostly background, so it could tell two sightings apart
    but never say which known thing either one was.
    """
    oam = ram[OAM_PAGE:OAM_PAGE + 4 * OAM_SPRITES].reshape(OAM_SPRITES, 4)
    vis = oam[oam[:, 0] < HIDDEN_Y]
    # The stored y is one less than the drawn y — the PPU renders a sprite on
    # the scanline after the one in the table.
    return np.stack([vis[:, 3], (vis[:, 0].astype(np.uint16) + 1)
                     .clip(0, 255).astype(np.uint8), vis[:, 1]], axis=1)


#: Mario, as the console itself knows him. Screen x, screen y, and the two
#: signed velocity bytes, found by matching every byte in the page against the
#: quantity it should equal: $3AD is within 0.94 px of the camera-relative
#: world x, $CE reproduces the screen y exactly, and $57/$9F correlate +0.94
#: with the per-frame change in each. The additions put him in the same frame
#: as a tracked box, whose centre sits +8/+16 from the sprite's corner —
#: measured over 12000 frames, not derived, so it is a calibration.
MARIO = (0x3AD, 0xCE, 0x57, 0x9F)
MARIO_CENTRE = (8.0, 16.0)


class RamHero:
    """The hero as the console reports him, shaped like a tracked one."""

    slot_id, missed, age, ctrl_prob = -1, 0, 999, 1.0

    def __init__(self, ram: np.ndarray):
        x, y, vx, vy = (int(ram[a]) for a in MARIO)
        self.cx = x + MARIO_CENTRE[0]
        self.cy = y + MARIO_CENTRE[1]
        self.vx = float(vx - 256 if vx > 127 else vx)
        self.vy = float(vy - 256 if vy > 127 else vy)


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
    out = np.zeros((n, OAM_SPRITES, 3), np.uint8)
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
    out = np.zeros((OAM_SPRITES, 3), np.uint8)
    out[:len(boxes)] = boxes[:OAM_SPRITES]
    return out


def sprite_mask(boxes: np.ndarray, frame_hw: tuple[int, int],
                out_hw: tuple[int, int], lead_dx: np.ndarray | None = None,
                ) -> np.ndarray:
    """Sprite positions as a coverage mask on the attention grid."""
    import cv2

    h, w = frame_hw
    full = np.zeros((h, w), np.uint8)
    for k, (x, y, _tile) in enumerate(boxes):
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
        for x, y, _tile in boxes:
            if x == 0 and y == 0:
                continue          # padding in a cached table
            canvas[int(y):int(y) + SPRITE_H, int(x):int(x) + SPRITE_W] = 255
        canvas = cv2.dilate(canvas, np.ones((3, 3), np.uint8))
        canvas[:HUD_H] = 0          # the HUD is not part of the world
        n, _, stats, cent = cv2.connectedComponentsWithStats(canvas)
        # ponytail: two characters in a clinch still merge into one component.
        # The tile ids are here now, so splitting them is possible; it stays
        # undone until a clinch is shown to cost something.
        dets = [(stats[i][0], stats[i][1], stats[i][2], stats[i][3],
                 float(cent[i][0]), float(cent[i][1])) for i in range(1, n)]
        # Which tiles make up each object. An object is several 8x8 sprites and
        # the set of their tile indices is what the game calls the thing —
        # exact, and with no background in it, unlike a crop of the screen.
        det_tiles: list[set] = []
        for x, y, bw, bh, _cx, _cy in dets:
            det_tiles.append({
                int(t) for sx, sy, t in boxes
                if not (sx == 0 and sy == 0)
                and x - 1 <= int(sx) <= x + bw and y - 1 <= int(sy) <= y + bh})

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
            slot.tiles = frozenset(det_tiles[best])
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
            # Sideways evidence dries up when the player is not going anywhere,
            # and this agent stalls constantly: with RIGHT held, its world
            # velocity is +0.155 and the camera is barely moving. Jumping does
            # not need him to be going anywhere. Nothing else on screen rises
            # because A was pressed, so an object going up while A is held is
            # about as clean a signal of control as exists here.
            if "A" in pressed and slot.vy < -JUMP_EPS:
                slot.ctrl_score += 0.35 * min(-slot.vy, 1.5)
        for k, d in enumerate(dets):
            if k not in taken:
                x, y, bw, bh, cx, cy = d
                self._slots.append(Slot(self._next_id, (x, y, bw, bh), cx, cy,
                                        small=bw * bh < 64,
                                        tiles=frozenset(det_tiles[k])))
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
