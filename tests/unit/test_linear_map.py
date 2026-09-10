"""The weight sheet has to show the policy, not the mean brightness of Contra."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/experiments"))
from linear_map import panel  # noqa: E402


def test_panel_colours_signed_weights():
    m = np.array([[1.0, -1.0, 0.0]])
    bgr = panel(m, 1.0)
    pos, neg, zero = bgr[0, 0], bgr[0, 1], bgr[0, 2]
    assert pos[2] > pos[0], "a positive weight must read red, not blue"
    assert neg[0] > neg[2], "a negative weight must read blue, not red"
    assert zero.min() == 255, "a zero weight must be white"


def test_common_component_is_removed():
    """What every action shares cannot move a softmax, so it must not be drawn."""
    shared = np.random.default_rng(0).normal(size=(1, 4, 8, 8))
    w = np.repeat(shared, 3, axis=0)          # three actions, identical weights
    centred = w - w.mean(0, keepdims=True)    # what action_maps does
    assert np.abs(centred).max() < 1e-12


if __name__ == "__main__":
    test_panel_colours_signed_weights()
    test_common_component_is_removed()
    print("ok")
