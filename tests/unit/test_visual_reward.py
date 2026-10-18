"""Proxy reward from bright flashes — the signal for long action-result windows."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nes_player.perception.visual_reward import VisualReward, make_reward  # noqa: E402


def _dark():
    return np.zeros((224, 240, 3), np.uint8)


def _flash():
    f = np.zeros((224, 240, 3), np.uint8)
    f[80:120, 100:160] = 255
    return f


def test_flash_is_reward():
    r = VisualReward()
    assert r.update(_dark(), 0) == 0
    assert r.update(_flash(), 1) == 1
    assert r.total == 1


def test_cooldown_blocks_double_count():
    r = VisualReward()
    r.update(_dark(), 0)
    r.update(_flash(), 1)
    # same flash next frame — cooldown holds
    assert r.update(_flash(), 2) == 0
    # cooldown is 8 update-calls; step past it with dark frames
    for i in range(3, 11):
        r.update(_dark(), i)
    assert r.update(_flash(), 11) == 1


def test_no_flash_no_reward():
    r = VisualReward()
    for i in range(30):
        assert r.update(_dark(), i) == 0
    assert r.total == 0


def test_gradual_change_is_not_a_flash():
    """A slow fade is not an explosion."""
    r = VisualReward()
    r.update(_dark(), 0)
    for i in range(20):
        f = np.full((224, 240, 3), min(255, i * 5), np.uint8)
        assert r.update(f, i) == 0
    assert r.total == 0


def test_make_reward_factory():
    assert isinstance(make_reward("visual"), VisualReward)
    assert isinstance(make_reward("proxy"), VisualReward)
    assert make_reward("off").update(_flash(), 0) == 0
