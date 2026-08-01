"""Inverse dynamics: recover the action from the frames before and after it.

This is the route to a dataset of human play. No labelled (frame, button) pairs
exist in the world, but thousands of hours of people playing do exist on video,
without the buttons. The VPT recipe (OpenAI, 2022) is to train an inverse
dynamics model on a small labelled set and use it to label the rest.

Our advantage over VPT: they had to hire people to produce labelled pairs,
while an emulator produces them for free and in any quantity.

Accuracy here comes from being NON-CAUSAL — the model sees the future. From a
single frame you cannot tell whether jump is held; from before-and-after it is
trivial. Measured, that is worth about five points over the causal policy.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nes_player.data.reader import Episode
from nes_player.policy.bc import (
    INPUT_HW,
    ActionVocab,
    BCNet,
    device,
    preprocess_frames,
    stack_to_tensor,
)

PAST, FUTURE = 5, 5   # window around the predicted frame; VPT uses 128
WINDOW = PAST + 1 + FUTURE


class IdmDataset(torch.utils.data.Dataset):
    """One sample: the frames around i predict the action pressed at i."""

    def __init__(self, small_frames: np.ndarray, labels: np.ndarray):
        self.frames = small_frames
        self.labels = labels

    def __len__(self) -> int:
        return max(0, len(self.frames) - WINDOW)

    def __getitem__(self, idx: int):
        i = idx + PAST
        window = self.frames[i - PAST : i + FUTURE + 1]
        return stack_to_tensor(window), int(self.labels[i])


def train_idm(episode_dir: str | Path, out_dir: str | Path, epochs: int = 3,
              batch_size: int = 256, lr: float = 3e-4, val_frac: float = 0.1,
              max_episodes: int | None = None, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    ep_dirs: list[Path] = []
    for part in str(episode_dir).split(","):
        root = Path(part)
        if (root / "metadata.json").exists():
            ep_dirs.append(root)
        else:
            ep_dirs.extend(sorted(d for d in root.iterdir()
                                  if (d / "metadata.json").exists()))
    if max_episodes:
        ep_dirs = ep_dirs[:max_episodes]
    eps = [Episode(d) for d in ep_dirs]
    vocab = ActionVocab.from_actions(np.concatenate([e.actions[:, 0] for e in eps]))
    print(f"episodes: {len(eps)}, frames: {sum(len(e) for e in eps)}, "
          f"actions: {len(vocab)}")

    parts = []
    for e in eps:
        parts.append(IdmDataset(preprocess_frames(e.frames[:]), vocab.encode(e.actions[:, 0])))
    n_val = max(1, int(len(parts) * val_frac)) if len(parts) > 1 else 0
    if n_val:
        train_ds = torch.utils.data.ConcatDataset(parts[:-n_val])
        val_ds = torch.utils.data.ConcatDataset(parts[-n_val:])
    else:   # a single episode: hold out its tail
        p = parts[0]
        cut = int(len(p.frames) * (1 - val_frac))
        train_ds = IdmDataset(p.frames[:cut], p.labels[:cut])
        val_ds = IdmDataset(p.frames[cut:], p.labels[cut:])

    dev = device()
    model = BCNet(len(vocab), in_ch=WINDOW * 3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    history = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            loss = nn.functional.cross_entropy(model(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        correct = total = 0
        val_labels = []
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(dev)).argmax(1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
                val_labels.append(y.numpy())
        rec = {"epoch": epoch, "val_acc": correct / total}
        history.append(rec)
        print(rec)

    labels = np.concatenate(val_labels)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    meta = {"kind": "idm", "vocab_masks": vocab.masks, "vocab_names": vocab.names,
            "window": WINDOW, "past": PAST, "future": FUTURE,
            "input_hw": list(INPUT_HW), "episode": str(episode_dir),
            "history": history,
            "val_majority_baseline": float(np.bincount(labels).max() / len(labels))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
