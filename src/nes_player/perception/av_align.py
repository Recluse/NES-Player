"""Contrastive audio-visual alignment: what on screen is making that sound.

A dual encoder. The frame becomes a 10×11 grid of embeddings, a 260 ms mel
window becomes one vector, and InfoNCE pulls the matching pair together with
frame-level similarity taken as the maximum over cells — multiple-instance
style, meaning "the source is somewhere on screen".

The decisive design choice is that negatives come from the SAME episode. Drawn
across episodes, the model learns "level music matches level background"
instead of "sound effect matches event", and scores well while understanding
nothing.

At inference `SoundLocator.sound_map()` returns the similarity of the current
sound to each cell of the frame; its argmax is the likely source.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from nes_player.data.reader import Episode
from nes_player.policy.bc import (
    INPUT_HW,
    MEL_FRAMES,
    MEL_HOP,
    MEL_N,
    AudioEncoder,
    device,
    episode_mel,
    mel_transform,
)

DIM = 64
TAU = 0.07


class AVAlign(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        # Same geometry as BCNet: (3,112,120) -> (dim,10,11)
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, dim, 3, stride=1),
        )
        self.audio = AudioEncoder()
        self.audio_proj = nn.Linear(128, dim)

    def forward(self, frames: torch.Tensor, mels: torch.Tensor):
        g = self.visual(frames)  # (B, dim, h, w)
        g = g / (g.norm(dim=1, keepdim=True) + 1e-8)
        a = self.audio_proj(self.audio(mels))  # (B, dim)
        a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
        return g, a


def _nce_loss(g: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    # sim[i, j] is the max over cells of <audio_i, grid_j>: MIL aggregation
    sim = torch.einsum("id,jdhw->ijhw", a, g).flatten(2).amax(2) / TAU
    labels = torch.arange(len(a), device=a.device)
    return (nn.functional.cross_entropy(sim, labels)
            + nn.functional.cross_entropy(sim.t(), labels)) / 2


def _frame_small(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)


def _mel_window(mel: np.ndarray, offset: int) -> np.ndarray:
    m_end = int(offset // MEL_HOP)
    m = np.zeros((MEL_N, MEL_FRAMES), np.float32)
    seg = mel[:, max(0, m_end - MEL_FRAMES) : m_end]
    if seg.shape[1]:
        m[:, MEL_FRAMES - seg.shape[1]:] = seg
    return m


def train_av_align(
    episode_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 2,
    batch: int = 64,
    steps_per_episode: int = 40,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ep_dirs: list[Path] = []
    for part in str(episode_dir).split(","):
        root = Path(part)
        if (root / "metadata.json").exists():
            ep_dirs.append(root)
        else:
            ep_dirs.extend(sorted(d for d in root.iterdir() if (d / "metadata.json").exists()))
    n_val = max(1, int(len(ep_dirs) * val_frac))
    train_eps, val_eps = ep_dirs[:-n_val], ep_dirs[-n_val:]
    print(f"episodes: {len(train_eps)} train / {len(val_eps)} val")

    dev = device()
    model = AVAlign().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sr = Episode(train_eps[0]).metadata["sample_rate"]

    def episode_batches(d: Path, n_batches: int):
        ep = Episode(d)
        n = len(ep)
        mel, _, _ = episode_mel(ep.audio[:], sr)
        offsets = ep.audio_offsets
        for _ in range(n_batches):
            idx = np.sort(rng.choice(np.arange(30, n), size=min(batch, n - 30), replace=False))
            frames = np.stack([_frame_small(ep.frames[int(i)]) for i in idx])
            f = torch.from_numpy(frames).float().div_(255).permute(0, 3, 1, 2)
            m = torch.from_numpy(np.stack([_mel_window(mel, offsets[int(i)]) for i in idx]))
            yield f.to(dev), m.unsqueeze(1).to(dev)

    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        order = rng.permutation(len(train_eps))
        for k in order:
            for f, m in episode_batches(train_eps[k], steps_per_episode):
                g, a = model(f, m)
                loss = _nce_loss(g, a)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        # Validation: top-1 retrieval of the frame from its sound, within a
        # batch drawn from an episode the model did not train on.
        model.eval()
        hits, total = 0, 0
        with torch.no_grad():
            for d in val_eps:
                for f, m in episode_batches(d, 10):
                    g, a = model(f, m)
                    sim = torch.einsum("id,jdhw->ijhw", a, g).flatten(2).amax(2)
                    hits += int((sim.argmax(1) == torch.arange(len(a), device=dev)).sum())
                    total += len(a)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_top1": hits / total, "chance": 1 / batch}
        history.append(rec)
        print(rec)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    meta = {"kind": "av_align", "dim": DIM, "episode": str(episode_dir),
            "sample_rate": sr, "history": history}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class SoundLocator:
    """Inference: where on screen the current sound comes from."""

    def __init__(self, run_dir: str | Path):
        run = Path(run_dir)
        meta = json.loads((run / "meta.json").read_text())
        # CPU on purpose: this is called from the game thread while the GPU is
        # busy in the brain thread, and concurrent GPU use from two threads
        # segfaults. The ghost predictor hit exactly the same wall.
        self.dev = torch.device("cpu")
        self.model = AVAlign(meta["dim"]).to(self.dev).eval()
        self.model.load_state_dict(torch.load(run / "model.pt", map_location=self.dev))
        self._mel_t = mel_transform(meta["sample_rate"])
        self._ring: list[np.ndarray] = []

    def push_audio(self, pcm: np.ndarray) -> None:
        self._ring.append(pcm)
        need = MEL_FRAMES * MEL_HOP + 512
        while sum(len(c) for c in self._ring[:-1]) > need:
            self._ring.pop(0)

    def sound_map(self, frame_rgb: np.ndarray) -> np.ndarray:
        """A (10, 11) map in [0,1]: how well each cell matches the current sound."""
        if not self._ring:
            return np.zeros((10, 11), np.float32)
        wav = np.concatenate(self._ring).astype(np.float32) / 32768
        mel = torch.log(self._mel_t(torch.from_numpy(wav)) + 1e-5)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)   # training mel is per-episode
        m = torch.zeros(MEL_N, MEL_FRAMES)
        seg = mel[:, -MEL_FRAMES:]
        m[:, MEL_FRAMES - seg.shape[1]:] = seg
        f = torch.from_numpy(_frame_small(frame_rgb)).float().div_(255).permute(2, 0, 1)
        with torch.no_grad():
            g, a = self.model(f.unsqueeze(0).to(self.dev),
                              m.unsqueeze(0).unsqueeze(0).to(self.dev))
            sim = torch.einsum("d,dhw->hw", a[0], g[0]).cpu().numpy()
        lo, hi = sim.min(), sim.max()
        return (sim - lo) / (hi - lo + 1e-8)
