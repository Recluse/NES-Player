"""Two sprites in a clinch must stay two objects.

The failure this pins is the open one from the experiment log: at contact range
a beat-em-up's hero and enemy become one connected component, the greedy match
gives it to whichever track is nearer, and the loser ghosts at its last
position. The winner's centre then jumps into the gap between the two, so
`enemy.cx - hero.cx` flips sign and the agent strikes away from the enemy.
"""

import numpy as np

from nes_player.perception.motion import HUD_H, MotionTracker, Slot, _split_detection

H, W = 224, 240

# A textured, static background. Without it phase correlation locks onto the
# only detailed thing in the frame — the sprite — reports its motion as camera
# scroll, and the compensated difference cancels the sprite out entirely. Real
# games have detailed backgrounds; a flat one is a property of the test.
# Tile-sized blocks, like a real NES background. Finer patterns alias under the
# sub-pixel warp that scroll compensation applies and light up the whole
# difference; a smooth gradient gives phase correlation nothing to hold on to,
# so it locks onto the sprite instead and cancels it out.
_rr, _cc = np.mgrid[0:H, 0:W]
_BACKGROUND = np.repeat(
    (60 + 60 * (((_rr // 16) + (_cc // 16)) % 2)).astype(np.uint8)[..., None], 3, axis=2)


def _frame(boxes) -> np.ndarray:
    """A dark frame with textured sprites: (x, y, w, h) in screen coordinates.

    The texture matters. A solid rectangle only differs from the previous frame
    along its leading and trailing edges, so a difference tracker sees two thin
    slivers rather than one object — an artefact of the test, not of the game.
    A real sprite carries internal detail that moves with it.
    """
    f = _BACKGROUND.copy()
    for x, y, w, h in boxes:
        rr, cc = np.mgrid[0:h, 0:w]
        f[y:y + h, x:x + w] = np.where(((rr // 3 + cc // 3) % 2)[..., None], 230, 120)
    return f


def _run(tracker, positions, pressed=frozenset()):
    """Feed a sequence of box-lists; return the slots after the last frame."""
    slots = []
    for boxes in positions:
        slots = tracker.update(_frame(boxes), pressed)
    return slots


def _approach(hero_x0, enemy_x0, steps, gap_end):
    """Two 16x24 sprites walking towards each other until they overlap."""
    y = HUD_H + 60
    out = []
    for i in range(steps):
        t = i / (steps - 1)
        hx = round(hero_x0 + t * ((enemy_x0 - gap_end) - hero_x0))
        out.append([(hx, y, 16, 24), (enemy_x0, y, 16, 24)])
    return out


def test_clinch_keeps_two_objects_on_the_correct_sides():
    tracker = MotionTracker()
    # Walk together until the boxes overlap by four pixels.
    frames = _approach(hero_x0=60, enemy_x0=140, steps=26, gap_end=12)
    slots = _run(tracker, frames)
    big = sorted((s for s in slots if not s.small), key=lambda s: s.cx)
    assert len(big) >= 2, "the clinch merged the fighters into a single object"
    left, right = big[0], big[-1]
    assert right.cx - left.cx > 6, "the two centres collapsed onto each other"
    assert left.slot_id != right.slot_id


def test_direction_sign_survives_the_clinch():
    """The number the beat-em-up logic actually uses is the sign of dx."""
    tracker = MotionTracker()
    frames = _approach(hero_x0=60, enemy_x0=140, steps=26, gap_end=12)
    signs = []
    for boxes in frames[10:]:
        slots = [s for s in tracker.update(_frame(boxes), frozenset()) if not s.small]
        if len(slots) < 2:
            continue
        ordered = sorted(slots, key=lambda s: s.cx)
        signs.append(np.sign(ordered[-1].cx - ordered[0].cx))
    assert signs, "no frames produced two objects at all"
    assert all(s > 0 for s in signs), "the direction between the fighters flipped"


def test_a_single_sprite_is_never_split():
    tracker = MotionTracker()
    y = HUD_H + 60
    frames = [[(60 + i * 3, y, 16, 24)] for i in range(20)]
    slots = _run(tracker, frames)
    assert len([s for s in slots if not s.small]) == 1


def _labels_with_blob(x, y, w, h):
    """A labels image holding one component, in play coordinates."""
    lab = np.zeros((H - HUD_H, W), dtype=np.int32)
    lab[y - HUD_H:y - HUD_H + h, x:x + w] = 1
    return lab


def _track(cx, cy, vx=0.0, vy=0.0):
    return Slot(slot_id=0, bbox=(0, 0, 1, 1), cx=cx, cy=cy, vx=vx, vy=vy)


def test_split_cuts_between_the_predicted_positions():
    x, y, w, h = 100, HUD_H + 60, 40, 24
    det = (x, y, w, h, x + w / 2, y + h / 2, False)
    left, right = _track(108, y + 12), _track(132, y + 12)
    parts = _split_detection(_labels_with_blob(x, y, w, h), 1, det, [left, right])
    assert parts is not None
    assert parts[0][4] < parts[1][4], "the left track must get the left half"
    assert parts[0][0] + parts[0][2] <= parts[1][0] + 1, "the halves must not overlap"


def test_split_refuses_a_blob_too_thin_to_be_two():
    x, y, w, h = 100, HUD_H + 60, 8, 24
    det = (x, y, w, h, x + w / 2, y + h / 2, False)
    parts = _split_detection(_labels_with_blob(x, y, w, h), 1,
                             det, [_track(102, y + 12), _track(106, y + 12)])
    assert parts is None


def test_split_refuses_when_the_tracks_coincide():
    """Two tracks on the same spot are one thing seen twice, not a clinch."""
    x, y, w, h = 100, HUD_H + 60, 40, 24
    det = (x, y, w, h, x + w / 2, y + h / 2, False)
    parts = _split_detection(_labels_with_blob(x, y, w, h), 1,
                             det, [_track(119, y + 12), _track(121, y + 12)])
    assert parts is None


def test_split_never_touches_small_blobs():
    """Bullets are small and numerous; splitting them would invent objects."""
    x, y, w, h = 100, HUD_H + 60, 40, 24
    det = (x, y, w, h, x + w / 2, y + h / 2, True)
    parts = _split_detection(_labels_with_blob(x, y, w, h), 1,
                             det, [_track(108, y + 12), _track(132, y + 12)])
    assert parts is None
