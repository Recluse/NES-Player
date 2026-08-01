import pytest

from nes_player.emulator.controller import BUTTONS, ControllerState


def test_impossible_combos_rejected():
    with pytest.raises(ValueError):
        ControllerState(left=True, right=True)
    with pytest.raises(ValueError):
        ControllerState(up=True, down=True)


def test_valid_masks_roundtrip():
    left = 1 << BUTTONS.index("LEFT")
    right = 1 << BUTTONS.index("RIGHT")
    up = 1 << BUTTONS.index("UP")
    down = 1 << BUTTONS.index("DOWN")
    for mask in range(256):
        if (mask & left and mask & right) or (mask & up and mask & down):
            with pytest.raises(ValueError):
                ControllerState.from_mask(mask)
        else:
            assert ControllerState.from_mask(mask).to_mask() == mask


def test_retro_array_mapping():
    retro_buttons = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
    state = ControllerState(a=True, right=True)
    assert state.to_retro_array(retro_buttons) == [0, 0, 0, 0, 0, 0, 0, 1, 1]
