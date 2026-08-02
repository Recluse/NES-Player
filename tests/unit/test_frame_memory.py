"""The dilated frame stack: a longer window at the same cost.

Four consecutive frames span 67 ms, which shows a velocity and nothing else.
Spacing the samples geometrically reaches seconds back without adding channels.
What these tests hold in place is that training and inference sample the *same*
frames — a mismatch there would not crash, it would just quietly feed the
network a different input than it was trained on.
"""

import json

import numpy as np
import torch

from nes_player.policy.bc import (
    DEFAULT_OFFSETS,
    FRAME_OFFSETS,
    INPUT_HW,
    ActionVocab,
    BCNet,
    BCPolicy,
    EpisodeBCDataset,
)


def _frames(n: int) -> np.ndarray:
    """Frames whose top-left pixel encodes the frame number, so a sampled
    stack can be read back and checked against the offsets."""
    f = np.zeros((n, *INPUT_HW, 3), np.uint8)
    for i in range(n):
        f[i, 0, 0, 0] = i
    return f


def test_offsets_are_newest_last_and_cover_their_span():
    for name, offs in FRAME_OFFSETS.items():
        assert offs[-1] == 1, f"{name}: the newest frame must be the last one"
        assert list(offs) == sorted(offs, reverse=True), f"{name}: must run oldest to newest"
        assert len(set(offs)) == len(offs), f"{name}: duplicate offsets waste a channel"


def test_dataset_samples_exactly_the_requested_frames():
    offs = (8, 4, 2, 1)
    ds = EpisodeBCDataset(_frames(40), np.zeros(40, np.int64), offsets=offs)
    x, _ = ds[0]                       # first usable sample sits at i = max(offs)
    assert x.shape[0] == len(offs) * 3
    got = [int(round(float(x[c * 3, 0, 0]) * 255)) for c in range(len(offs))]
    assert got == [8 - o for o in offs], f"sampled {got}, expected {[8 - o for o in offs]}"


def test_wider_memory_costs_samples_not_channels_per_frame():
    short = EpisodeBCDataset(_frames(200), np.zeros(200, np.int64),
                             offsets=FRAME_OFFSETS["short"])
    long_ = EpisodeBCDataset(_frames(200), np.zeros(200, np.int64),
                             offsets=FRAME_OFFSETS["long"])
    assert len(long_) < len(short), "a wider window costs usable samples at the episode head"
    assert long_[0][0].shape[0] == len(FRAME_OFFSETS["long"]) * 3


def test_inference_samples_the_same_frames_as_training(tmp_path):
    """The failure this prevents is silent: a stack assembled differently at
    inference than in training still runs, and simply performs worse."""
    offs = FRAME_OFFSETS["wide"]
    vocab = ActionVocab(masks=[0, 1])
    model = BCNet(len(vocab), in_ch=len(offs) * 3)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "meta.json").write_text(json.dumps({
        "vocab_masks": vocab.masks, "vocab_names": vocab.names,
        "modality": "video", "frame_offsets": list(offs),
    }))

    policy = BCPolicy(tmp_path)
    assert policy.offsets == offs

    frames = _frames(120)
    for f in frames:                                   # feed a whole episode
        policy.act(np.repeat(f, 2, axis=0)[:INPUT_HW[0]], temperature=1.0)
    picked = np.stack([policy._stack[-o] for o in policy.offsets])
    got = [int(picked[k, 0, 0, 0]) for k in range(len(offs))]
    assert got == [119 - o + 1 for o in offs], f"inference sampled {got}"


def test_old_checkpoints_keep_working(tmp_path):
    """Checkpoints written before this existed have no offsets recorded."""
    vocab = ActionVocab(masks=[0, 1])
    model = BCNet(len(vocab))
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "meta.json").write_text(json.dumps({
        "vocab_masks": vocab.masks, "vocab_names": vocab.names, "modality": "video",
    }))
    policy = BCPolicy(tmp_path)
    assert policy.offsets == DEFAULT_OFFSETS
    assert policy.span == 4
