"""The perceptual cell must separate places without keying on sprite noise.

Calibrated on Contra's base: at 12x10 cells and three bits, one room gave
260 distinct scenes; at 6x5 and two bits it gives about twenty, and the
next room shares almost none of them.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "scripts" / "experiments"))


def _room(seed: int, sprite_at: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.repeat(np.repeat(rng.integers(0, 255, (5, 6)), 48, 0), 43, 1)
    f = np.stack([f[:224, :256]] * 3, -1).astype(np.uint8)
    f[100:116, sprite_at:sprite_at + 16] = 255      # a sprite moved
    return f


def test_a_moved_sprite_is_the_same_place():
    import frontier

    a, b = _room(1, 40), _room(1, 90)
    assert frontier.scene_hash(a) == frontier.scene_hash(b)


def test_a_different_room_is_a_different_place():
    import frontier

    assert frontier.scene_hash(_room(1, 40)) != frontier.scene_hash(_room(2, 40))
