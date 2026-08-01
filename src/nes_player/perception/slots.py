"""Neural slots: an unsupervised decomposition of the frame into K entities.

The appeal over the motion tracker is that this should also see STATIC objects
— bricks, blocks, pipes — which differ from the background in appearance rather
than in movement.

A compact implementation of slot attention (Locatello et al. 2020) with a
spatial broadcast decoder: 56×60 frame -> CNN -> slot attention (K=7, 3
iterations) -> decoder with alpha masks, where a slot's mask is meant to say
"this part of the screen is one thing".

IT DOES NOT WORK on this domain, and the negative result is the point of
keeping it. Reconstruction learns (MSE 0.027 -> 0.012) but the slots split the
frame into positional ripples rather than objects. An NES background is itself
a repeating tile pattern, so a blobby decomposition reconstructs it just as
well as an object-shaped one, and the model has no other gradient to follow.
See docs/experiments.md; the motion tracker remains the production path.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from nes_player.data.reader import Episode
from nes_player.policy.bc import device

K_SLOTS = 7
SLOT_DIM = 48
IN_HW = (56, 60)   # half of INPUT_HW: the decoder costs pixels times slots


def _pos_grid(h: int, w: int) -> torch.Tensor:
    ys = torch.linspace(-1, 1, h)
    xs = torch.linspace(-1, 1, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy, -gx, -gy], 0)  # (4, h, w)


class SlotAttention(nn.Module):
    def __init__(self, dim: int = SLOT_DIM, iters: int = 3):
        super().__init__()
        self.iters = iters
        self.scale = dim ** -0.5
        self.mu = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.norm_in = nn.LayerNorm(dim)
        self.norm_slot = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, feats: torch.Tensor, k: int = K_SLOTS) -> torch.Tensor:
        b, _n, d = feats.shape
        feats = self.norm_in(feats)
        kk, vv = self.to_k(feats), self.to_v(feats)
        slots = self.mu + self.log_sigma.exp() * torch.randn(
            b, k, d, device=feats.device)
        for _ in range(self.iters):
            q = self.to_q(self.norm_slot(slots))
            attn = torch.softmax(
                torch.einsum("bkd,bnd->bkn", q, kk) * self.scale, dim=1)   # slots compete
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
            upd = torch.einsum("bkn,bnd->bkd", attn, vv)
            slots = self.gru(upd.reshape(-1, d), slots.reshape(-1, d)).view(b, k, d)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class SlotAE(nn.Module):
    def __init__(self, dim: int = SLOT_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3 + 4, 32, 5, padding=2), nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2, stride=2), nn.ReLU(),
            nn.Conv2d(32, dim, 5, padding=2), nn.ReLU(),
        )  # (dim, 28, 30)
        self.slot_attn = SlotAttention(dim)
        dh, dw = IN_HW[0] // 4, IN_HW[1] // 4   # broadcast from 14×15, decode ×4
        self.dec_hw = (dh, dw)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim + 4, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 4, 3, padding=1),  # RGB + alpha
        )
        self.register_buffer("pos_enc", _pos_grid(*IN_HW).unsqueeze(0))
        self.register_buffer("pos_dec", _pos_grid(dh, dw).unsqueeze(0))

    def forward(self, x: torch.Tensor):
        b = len(x)
        f = self.enc(torch.cat([x, self.pos_enc.expand(b, -1, -1, -1)], 1))
        slots = self.slot_attn(f.flatten(2).transpose(1, 2))  # (B, K, dim)
        k = slots.shape[1]
        dh, dw = self.dec_hw
        z = slots.reshape(b * k, -1, 1, 1).expand(-1, -1, dh, dw)
        z = torch.cat([z, self.pos_dec.expand(b * k, -1, -1, -1)], 1)
        out = self.dec(z).view(b, k, 4, *IN_HW)
        rgb, alpha = out[:, :, :3], torch.softmax(out[:, :, 3:], dim=1)
        recon = (rgb * alpha).sum(1)
        return recon, alpha.squeeze(2)  # (B,3,H,W), (B,K,H,W)


def _prep(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (IN_HW[1], IN_HW[0]), interpolation=cv2.INTER_AREA)


def train_slots(
    episode_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 3,
    batch: int = 32,
    frames_per_episode: int = 400,
    lr: float = 4e-4,
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
    dev = device()
    model = SlotAE().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        losses = []
        for d in rng.permutation(ep_dirs):
            ep = Episode(Path(d))
            idx = np.sort(rng.choice(np.arange(len(ep)), size=min(frames_per_episode, len(ep)),
                                     replace=False))
            frames = np.stack([_prep(ep.frames[int(i)]) for i in idx])
            for s in range(0, len(frames), batch):
                x = torch.from_numpy(frames[s : s + batch]).float().div_(255)
                x = x.permute(0, 3, 1, 2).to(dev)
                recon, _ = model(x)
                loss = nn.functional.mse_loss(recon, x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        rec = {"epoch": epoch, "recon_mse": float(np.mean(losses))}
        history.append(rec)
        print(rec)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    meta = {"kind": "slots", "k": K_SLOTS, "dim": SLOT_DIM, "in_hw": list(IN_HW),
            "episode": str(episode_dir), "history": history}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class SlotExtractor:
    """Inference: frame -> K alpha masks, upscaled to frame size."""

    def __init__(self, run_dir: str | Path):
        run = Path(run_dir)
        self.dev = device()
        self.model = SlotAE().to(self.dev).eval()
        self.model.load_state_dict(torch.load(run / "model.pt", map_location=self.dev))

    def masks(self, frame_rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(_prep(frame_rgb)).float().div_(255).permute(2, 0, 1)
        with torch.no_grad():
            _, alpha = self.model(x.unsqueeze(0).to(self.dev))
        a = alpha[0].cpu().numpy()  # (K, h, w)
        return np.stack([cv2.resize(m, (frame_rgb.shape[1], frame_rgb.shape[0]),
                                    interpolation=cv2.INTER_LINEAR) for m in a])
