"""Tensor shapes of the models, and action-vocabulary serialisation."""

import numpy as np
import torch

from nes_player.perception.av_align import DIM, AVAlign
from nes_player.perception.slots import IN_HW, K_SLOTS, SlotAE
from nes_player.policy.bc import (
    FRAME_STACK,
    INPUT_HW,
    MEL_FRAMES,
    MEL_N,
    ActionVocab,
    BCNet,
    BCNetAV,
)


def test_bcnet_shapes():
    m = BCNet(n_actions=13)
    y = m(torch.zeros(2, FRAME_STACK * 3, *INPUT_HW))
    assert y.shape == (2, 13)


def test_bcnet_av_shapes():
    m = BCNetAV(n_actions=7)
    y = m(torch.zeros(2, FRAME_STACK * 3, *INPUT_HW), torch.zeros(2, 1, MEL_N, MEL_FRAMES))
    assert y.shape == (2, 7)


def test_av_align_shapes():
    m = AVAlign()
    g, a = m(torch.zeros(3, 3, *INPUT_HW), torch.zeros(3, 1, MEL_N, MEL_FRAMES))
    assert g.shape[:2] == (3, DIM) and g.dim() == 4
    assert a.shape == (3, DIM)
    # embeddings are normalised
    assert torch.allclose(a.norm(dim=1), torch.ones(3), atol=1e-4)


def test_slot_ae_shapes():
    m = SlotAE()
    recon, alpha = m(torch.rand(2, 3, *IN_HW))
    assert recon.shape == (2, 3, *IN_HW)
    assert alpha.shape == (2, K_SLOTS, *IN_HW)
    # alpha is a distribution over slots at every pixel
    assert torch.allclose(alpha.sum(1), torch.ones(2, *IN_HW), atol=1e-4)


def test_action_vocab_roundtrip():
    actions = np.array([0, 1, 3, 3, 129, 1])
    v = ActionVocab.from_actions(actions)
    enc = v.encode(actions)
    assert [v.masks[i] for i in enc] == [0, 1, 3, 3, 129, 1]
    assert "NOOP" in v.names
    # round-tripping through meta.json gives back the same masks
    v2 = ActionVocab(masks=list(v.masks))
    assert np.array_equal(v2.encode(actions), enc)


def test_a_wider_teacher_reloads_at_its_own_width(tmp_path):
    """A checkpoint that does not record its width cannot be reloaded: the
    loader would build the default 256 and fail on the state dict."""
    import json

    import torch

    from nes_player.policy.state_teacher import StateNet, StatePolicy

    net = StateNet(7, width=64)
    (tmp_path / "meta.json").write_text(json.dumps({
        "vocab_masks": [0, 1, 8, 64, 128, 130, 131], "width": 64}))
    torch.save(net.state_dict(), tmp_path / "model.pt")
    loaded = StatePolicy(tmp_path)
    assert loaded.net.net[0].out_features == 64


def test_a_checkpoint_from_before_the_width_was_recorded_still_loads(tmp_path):
    import json

    import torch

    from nes_player.policy.state_teacher import StateNet, StatePolicy

    net = StateNet(7)
    (tmp_path / "meta.json").write_text(json.dumps({
        "vocab_masks": [0, 1, 8, 64, 128, 130, 131]}))
    torch.save(net.state_dict(), tmp_path / "model.pt")
    assert StatePolicy(tmp_path).net.net[0].out_features == 256
