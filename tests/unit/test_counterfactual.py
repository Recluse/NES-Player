"""Counterfactual branches: the file format, and the sprite table the ego
trajectory is now built from."""

import numpy as np

from nes_player.emulator.controller import BUTTONS
from nes_player.perception.sprites import SPRITES_VERSION
from nes_player.policy.bc import ActionVocab
from nes_player.policy.planner import RUN
from nes_player.world_model.counterfactual import ACTIONS, _mask, load_packs
from nes_player.world_model.ego import CROP, SEQ, _sprite_boxes


def test_mask_matches_the_controller():
    m = _mask(RUN)
    assert {b for i, b in enumerate(BUTTONS) if m >> i & 1} == RUN


def test_branch_long_enough_for_a_window():
    from nes_player.world_model.counterfactual import BRANCH

    assert BRANCH >= SEQ + 1, "a branch too short yields no training window"


def _write(path, crops):
    n, steps = crops.shape[:2]
    np.savez_compressed(
        path, crops=crops,
        pos=np.zeros((n, steps, 2), np.float32),
        valid=np.ones((n, steps), bool),
        masks=np.stack([np.full(steps, _mask(p), np.int64) for p in ACTIONS]),
    )
    return ActionVocab.from_actions(np.array([_mask(p) for p in ACTIONS]))


def test_load_packs_round_trip(tmp_path):
    n, steps = len(ACTIONS), 6
    # Each branch has to look different, or it is a moment where the buttons
    # did nothing and load_packs is right to throw it away.
    crops = np.stack([np.full((steps, CROP, CROP, 3), i, np.uint8)
                      for i in range(n)])
    path = tmp_path / "b.npz"
    vocab = _write(path, crops)
    packs = load_packs(path, vocab)
    assert len(packs) == n
    assert packs[0]["crops"].shape == (steps, CROP, CROP, 3)
    # One branch holds one action, and the four branches hold different ones.
    assert all(len(set(p["labels"].tolist())) == 1 for p in packs)
    assert len({p["labels"][0] for p in packs}) == n


def test_load_packs_drops_a_dead_quad(tmp_path):
    """Four identical branches are the attract-mode demo, not a choice."""
    n, steps = len(ACTIONS), 6
    path = tmp_path / "b.npz"
    vocab = _write(path, np.zeros((n, steps, CROP, CROP, 3), np.uint8))
    assert load_packs(path, vocab) == []


class _Ep:
    def __init__(self, path):
        self.path = path


def test_sprite_boxes_accepts_the_older_table(tmp_path):
    np.save(tmp_path / "sprites.v1.npy", np.ones((5, 64, 2), np.uint8))
    boxes = _sprite_boxes(_Ep(tmp_path))
    assert boxes.shape == (5, 64, 3)
    assert (boxes[..., 2] == 0).all(), "a v1 table has no tile ids to offer"


def test_sprite_boxes_prefers_the_current_table(tmp_path):
    np.save(tmp_path / "sprites.v1.npy", np.ones((5, 64, 2), np.uint8))
    np.save(tmp_path / f"sprites.v{SPRITES_VERSION}.npy",
            np.full((5, 64, 3), 7, np.uint8))
    assert (_sprite_boxes(_Ep(tmp_path))[..., 2] == 7).all()
