"""Attention aimed at where an object will be, not where it is.

A player shoots at where the enemy is going. The tracker already computes a
velocity for every object, so pointing the attention target ahead of them costs
one multiplication — and is a much cheaper thing to try than a model that
predicts frames.

These tests work on the extrapolation itself rather than on a trained network:
what has to hold is that a lead moves the mask in the direction of travel, by
the right amount, and never off the edge of the frame.
"""

import numpy as np

from nes_player.policy.bc import ATTN_HW


class _Slot:
    """The two fields the mask builder reads off a tracker slot."""

    def __init__(self, bbox, vx=0.0, vy=0.0):
        self.bbox, self.vx, self.vy = bbox, vx, vy


def _mask(slots, lead: int, shape=(224, 240)) -> np.ndarray:
    """The mask-building step of episode_attn_masks, in isolation."""
    import cv2

    h_full, w_full = shape
    full = np.zeros(shape, np.uint8)
    for s in slots:
        x, y, w, h = s.bbox
        if lead:
            x = int(round(x + s.vx * lead))
            y = int(round(y + s.vy * lead))
            x = max(0, min(x, w_full - w))
            y = max(0, min(y, h_full - h))
        full[y:y + h, x:x + w] = 255
    small = cv2.resize(full, (ATTN_HW[1], ATTN_HW[0]), interpolation=cv2.INTER_AREA)
    return (small > 16).astype(np.uint8)


def _centre_x(mask: np.ndarray) -> float:
    xs = np.nonzero(mask.any(axis=0))[0]
    return float(xs.mean()) if len(xs) else -1.0


def test_lead_moves_the_target_the_way_the_object_is_going():
    slot = _Slot((100, 100, 16, 24), vx=2.0)      # two pixels a frame, rightwards
    now = _centre_x(_mask([slot], lead=0))
    ahead = _centre_x(_mask([slot], lead=30))     # half a second
    assert ahead > now, "the mask must move in the direction of travel"


def test_lead_distance_follows_the_velocity():
    slow = _Slot((60, 100, 16, 24), vx=0.5)
    fast = _Slot((60, 100, 16, 24), vx=3.0)
    base = _centre_x(_mask([slow], lead=0))
    assert (_centre_x(_mask([fast], lead=30)) - base
            > _centre_x(_mask([slow], lead=30)) - base), "faster means further ahead"


def test_a_still_object_is_not_moved():
    slot = _Slot((100, 100, 16, 24))
    assert np.array_equal(_mask([slot], lead=0), _mask([slot], lead=30))


def test_extrapolation_stays_on_screen():
    """An object leaving the screen is still worth watching at the edge it
    leaves by; extrapolating it into nowhere would produce an empty target."""
    for vx in (-9.0, 9.0):
        m = _mask([_Slot((120, 100, 16, 24), vx=vx)], lead=60)
        assert m.any(), f"vx={vx}: the target vanished off the frame"


def test_zero_lead_is_exactly_the_old_behaviour():
    slots = [_Slot((40, 60, 16, 24), vx=1.5, vy=-0.5),
             _Slot((150, 120, 20, 20), vx=-2.0)]
    plain = np.zeros((224, 240), np.uint8)
    for s in slots:
        x, y, w, h = s.bbox
        plain[y:y + h, x:x + w] = 255
    import cv2
    want = (cv2.resize(plain, (ATTN_HW[1], ATTN_HW[0]),
                       interpolation=cv2.INTER_AREA) > 16).astype(np.uint8)
    assert np.array_equal(_mask(slots, lead=0), want)


def _mask_multi(slots, leads, shape=(224, 240)) -> np.ndarray:
    """The union form: every lead marks its own box, all in one target."""
    import cv2

    h_full, w_full = shape
    full = np.zeros(shape, np.uint8)
    for s in slots:
        bx, by, w, h = s.bbox
        for lv in leads:
            x = max(0, min(int(round(bx + s.vx * lv)), w_full - w))
            y = max(0, min(int(round(by + s.vy * lv)), h_full - h))
            full[y:y + h, x:x + w] = 255
    small = cv2.resize(full, (ATTN_HW[1], ATTN_HW[0]), interpolation=cv2.INTER_AREA)
    return (small > 16).astype(np.uint8)


def test_union_marks_both_now_and_later():
    """Offering both places, rather than replacing one with the other.

    A single lead moved the target off the enemy in front of the agent, and it
    stopped attacking. A union should keep that cell lit and light the future
    one as well.
    """
    slot = _Slot((100, 100, 16, 24), vx=2.5)
    now = _mask([slot], lead=0)
    later = _mask([slot], lead=30)
    both = _mask_multi([slot], (0, 30))
    assert (both & now).sum() == now.sum(), "the current position must stay marked"
    assert (both & later).sum() == later.sum(), "the future position must be marked too"
    assert both.sum() > now.sum(), "the union has to be larger than either alone"


def test_union_with_only_zero_is_the_plain_mask():
    slots = [_Slot((40, 60, 16, 24), vx=1.5), _Slot((150, 120, 20, 20), vx=-2.0)]
    assert np.array_equal(_mask_multi(slots, (0,)), _mask(slots, lead=0))


def test_union_of_a_still_object_does_not_grow():
    """Nothing moving means now and later are the same place."""
    slot = _Slot((100, 100, 16, 24))
    assert np.array_equal(_mask_multi([slot], (0, 15, 30)), _mask([slot], lead=0))
