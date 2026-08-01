"""World model v1/v2: action-conditioned dynamics in a latent space.

Frame -> encoder -> z_t; a GRU conditioned on an action embedding predicts
z_{t+1}; a decoder keeps the latent meaningful and renders "dreams", which are
open-loop rollouts of imagination.

THIS VERSION DOES NOT USE ACTIONS. Its advantage over an action-blind baseline
is 1.000 and 0.993 for v1 and v2 — that is, none. It is kept because the
diagnosis was what led to the working model: the latent is dominated by
background and scroll, so the effect of a button press is invisible inside the
MSE of a 256-dimensional vector. The ego-centric model in ego.py is the
successor, and it works.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from nes_player.data.reader import Episode
from nes_player.policy.bc import ActionVocab, device

LATENT = 256
SEQ_LEN = 16
FRAME_HW = (112, 120)   # encoder input
DREAM_HW = (56, 60)     # decoder output


class WorldModel(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.LazyLinear(LATENT),
        )
        self.action_emb = nn.Embedding(n_actions, 32)
        self.dynamics = nn.GRUCell(LATENT + 32, LATENT)
        self.predictor = nn.Linear(LATENT, LATENT)
        # Inverse dynamics: recover the action from a pair of latents. The
        # point is to force the latent to encode action-relevant differences.
        self.inv_head = nn.Linear(2 * LATENT, n_actions)
        self.dec_fc = nn.Linear(LATENT, 64 * 7 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float → (B, LATENT)"""
        return self.encoder(frames)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_fc(z).view(-1, 64, 7, 8)
        return self.decoder(x)[:, :, : DREAM_HW[0], : DREAM_HW[1]]

    def step(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor):
        """One dynamics step, predicting the DELTA of the latent."""
        h = self.dynamics(torch.cat([z, self.action_emb(a)], dim=1), h)
        return h, z + self.predictor(h)


def _frames_tensor(small: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, H, W) float"""
    return torch.from_numpy(small).float().div_(255).permute(0, 3, 1, 2)


def _prep_episode(episode_dir: str | Path):
    ep = Episode(Path(episode_dir))
    actions = ep.actions[:, 0]
    vocab = ActionVocab.from_actions(actions)
    labels = vocab.encode(actions)
    frames = ep.frames[:]
    small = np.empty((len(frames), *FRAME_HW, 3), np.uint8)
    tiny = np.empty((len(frames), *DREAM_HW, 3), np.uint8)
    for i, f in enumerate(frames):
        small[i] = cv2.resize(f, (FRAME_HW[1], FRAME_HW[0]), interpolation=cv2.INTER_AREA)
        tiny[i] = cv2.resize(f, (DREAM_HW[1], DREAM_HW[0]), interpolation=cv2.INTER_AREA)
    return small, tiny, labels, vocab


def train_wm(episode_dir: str | Path, out_dir: str | Path,
             epochs: int = 3, batch: int = 32, lr: float = 3e-4, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    small, tiny, labels, vocab = _prep_episode(episode_dir)
    n = len(small)
    n_val = n // 10
    dev = device()
    model = WorldModel(len(vocab)).to(dev)
    model.encode(torch.zeros(1, 3, *FRAME_HW, device=dev))   # initialise LazyLinear
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    train_starts = np.arange(0, n - n_val - SEQ_LEN - 1, 4)
    history = []
    for epoch in range(epochs):
        np.random.shuffle(train_starts)
        model.train()
        losses = []
        for bi in range(0, len(train_starts), batch):
            starts = train_starts[bi : bi + batch]
            if len(starts) == 0:
                continue
            f = torch.stack([_frames_tensor(small[s : s + SEQ_LEN + 1]) for s in starts])
            t = torch.stack([_frames_tensor(tiny[s : s + SEQ_LEN + 1]) for s in starts])
            a = torch.from_numpy(np.stack([labels[s : s + SEQ_LEN] for s in starts])).long()
            f, t, a = f.to(dev), t.to(dev), a.to(dev)
            B = f.shape[0]
            z = model.encode(f.reshape(-1, 3, *FRAME_HW)).reshape(B, SEQ_LEN + 1, LATENT)
            h = torch.zeros(B, LATENT, device=dev)
            loss_lat, loss_rec, loss_inv = 0.0, 0.0, 0.0
            recon0 = model.decode(z[:, 0])
            loss_rec = nn.functional.mse_loss(recon0, t[:, 0])
            z_in = z[:, 0]
            for k in range(SEQ_LEN):
                h, z_hat = model.step(h, z_in, a[:, k])
                loss_lat = loss_lat + nn.functional.mse_loss(z_hat, z[:, k + 1].detach())
                # Inverse dynamics on a pair of real latents
                logits_a = model.inv_head(torch.cat([z[:, k], z[:, k + 1]], dim=1))
                loss_inv = loss_inv + nn.functional.cross_entropy(logits_a, a[:, k])
                if k % 4 == 3:   # decoding every step is not worth the cost
                    loss_rec = loss_rec + nn.functional.mse_loss(
                        model.decode(z_hat), t[:, k + 1])
                # Scheduled sampling: after the warm-up the dynamics is fed its
                # own predictions, otherwise ignoring the actions stays cheap
                if k >= 3 and np.random.rand() < 0.5:
                    z_in = z_hat
                else:
                    z_in = z[:, k + 1]
            loss = (loss_rec + 0.1 * loss_lat / SEQ_LEN
                    + 0.05 * loss_inv / SEQ_LEN)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        rec = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(rec)
        print(rec)

    # The metric: an open-loop rollout, latent error over the tail (step 8 on),
    # with true actions against shuffled ones
    model.eval()
    val_starts = np.arange(n - n_val, n - SEQ_LEN - 1, 2)
    err_true, err_shuf = [], []
    with torch.no_grad():
        for s in val_starts:
            f = _frames_tensor(small[s : s + SEQ_LEN + 1]).to(dev)
            a = torch.from_numpy(labels[s : s + SEQ_LEN]).long().to(dev)
            a_shuf = a[torch.randperm(SEQ_LEN)]
            z_all = model.encode(f)
            for variant, sink in ((a, err_true), (a_shuf, err_shuf)):
                h = torch.zeros(1, LATENT, device=dev)
                z_roll = z_all[0:1]
                errs = []
                for k in range(SEQ_LEN):
                    h, z_roll = model.step(h, z_roll, variant[k : k + 1])
                    errs.append(float(nn.functional.mse_loss(z_roll, z_all[k + 1 : k + 2])))
                sink.append(float(np.mean(errs[7:])))
    e_true, e_shuf = float(np.mean(err_true)), float(np.mean(err_shuf))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    meta = {
        "vocab_masks": vocab.masks,
        "episode": str(episode_dir),
        "history": history,
        "latent_mse_true_actions": e_true,
        "latent_mse_shuffled_actions": e_shuf,
        "action_advantage": e_shuf / max(e_true, 1e-9),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"true": e_true, "shuffled": e_shuf,
                      "advantage": meta["action_advantage"]}))
    return meta


def dream_strip(run_dir: str | Path, episode_dir: str | Path, start: int,
                steps: int = 12, out_png: str | Path | None = None) -> np.ndarray:
    """A dream: one real frame, then an open-loop rollout on the episode's
    actions. Top row is reality, bottom row is what the model imagined."""
    run = Path(run_dir)
    meta = json.loads((run / "meta.json").read_text())
    vocab = ActionVocab(masks=meta["vocab_masks"])
    dev = device()
    model = WorldModel(len(vocab)).to(dev).eval()
    model.encode(torch.zeros(1, 3, *FRAME_HW, device=dev))
    model.load_state_dict(torch.load(run / "model.pt", map_location=dev))

    small, tiny, labels, _ = _prep_episode(episode_dir)
    with torch.no_grad():
        z = model.encode(_frames_tensor(small[start : start + 1]).to(dev))
        h = torch.zeros(1, LATENT, device=dev)
        dreams = [model.decode(z)[0]]
        for k in range(steps):
            a = torch.tensor([labels[start + k]], device=dev)
            h, z = model.step(h, z, a)
            dreams.append(model.decode(z)[0])
    dream_imgs = [(d.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                  for d in dreams]
    real_imgs = [tiny[start + k] for k in range(steps + 1)]
    top = np.concatenate(real_imgs, axis=1)
    bot = np.concatenate(dream_imgs, axis=1)
    grid = cv2.cvtColor(np.concatenate([top, bot], axis=0), cv2.COLOR_RGB2BGR)
    grid = cv2.resize(grid, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    if out_png:
        cv2.imwrite(str(out_png), grid)
    return grid
