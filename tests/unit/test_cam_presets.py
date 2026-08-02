"""Grad-CAM must work for every memory preset, not just the narrowest one.

`compute_cam` used to assemble four consecutive frames whatever the checkpoint
expected, so wide/long/epic raised a channel-count error — swallowed by the
GUI's blanket except, which is why nobody noticed the overlay was dead.
"""

import json

import numpy as np
import pytest
import torch

from nes_player.policy.bc import FRAME_OFFSETS, BCNet, BCPolicy


@pytest.fixture(params=sorted(FRAME_OFFSETS))
def checkpoint(request, tmp_path):
    """A tiny untrained checkpoint with this preset's geometry."""
    offsets = FRAME_OFFSETS[request.param]
    masks = [0, 128]
    net = BCNet(len(masks), in_ch=len(offsets) * 3)
    out = tmp_path / request.param
    out.mkdir()
    torch.save(net.state_dict(), out / "model.pt")
    (out / "meta.json").write_text(json.dumps({
        "vocab_masks": masks, "vocab_names": ["NOOP", "RIGHT"],
        "frame_offsets": list(offsets), "memory": request.param,
        "modality": "video", "input_hw": [112, 120]}))
    return out


def test_cam_works_for_every_preset(checkpoint):
    policy = BCPolicy(checkpoint)
    frame = np.zeros((224, 240, 3), np.uint8)
    for _ in range(4):
        policy.act(frame)
    cam = policy.compute_cam(frame)
    assert cam.ndim == 2
    assert np.isfinite(cam).all()
    assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0


def test_cam_does_not_consume_the_frame(checkpoint):
    """Looking is not deciding: the GUI calls this between decisions."""
    policy = BCPolicy(checkpoint)
    frame = np.zeros((224, 240, 3), np.uint8)
    policy.act(frame)
    before = len(policy._stack)
    policy.compute_cam(frame)
    assert len(policy._stack) == before
