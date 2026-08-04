"""Ego-centric world model: predict where the HERO moves, given the buttons.

The earlier version modelled a latent of the whole screen and ignored actions
entirely. The diagnosis is the reason this file exists: the latent was
dominated by background and scroll, and the effect of a button press — two
pixels of hero displacement — is a grain of sand inside the MSE of a
256-dimensional latent.

So instead: a crop around the controlled object, its velocity and the action ->
(dx, dy) one step ahead, applied recurrently. Quality is measured by rolling
out with true actions against shuffled ones; anything above 1.0 means the model
actually uses what it is told. A by-product is the ghost trajectory drawn on
the dashboard.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nes_player.data.reader import Episode
from nes_player.emulator.controller import BUTTONS
from nes_player.perception.motion import MotionTracker, pick_hero
from nes_player.policy.bc import ActionVocab, device

CROP = 48   # crop around the hero, in source-frame pixels
SEQ = 16


def mask_to_frozen(mask: int) -> frozenset:
    return frozenset(b for i, b in enumerate(BUTTONS) if mask >> i & 1)


def extract_trajectory(episode_dir: str | Path, out_npy: str | Path | None = None):
    """Run the episode through the motion tracker: (N, 3) of cx, cy, valid."""
    ep = Episode(Path(episode_dir))
    frames = ep.frames
    actions = ep.actions[:, 0]
    tracker = MotionTracker()
    out = np.zeros((len(ep), 3), np.float32)
    last = (120.0, 112.0)
    for i in range(len(ep)):
        pressed = mask_to_frozen(int(actions[i]))
        slots = tracker.update(frames[i], pressed)
        best = pick_hero(slots)
        if best is not None and best.ctrl_prob > 0.7 and best.missed == 0:
            last = (best.cx, best.cy)
            out[i] = (best.cx, best.cy, 1.0)
        else:
            out[i] = (*last, 0.0)
    if out_npy:
        np.save(out_npy, out)
    return out


class EgoModel(nn.Module):
    """Hero crop, velocity and action -> one step of position delta."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.LazyLinear(64),
        )
        self.action_emb = nn.Embedding(n_actions, 16)
        self.rnn = nn.GRUCell(64 + 2 + 16, 128)
        self.head = nn.Linear(128, 2)  # (dx, dy)

    def forward_step(self, h, crop_feat, vel, a):
        h = self.rnn(torch.cat([crop_feat, vel, self.action_emb(a)], dim=1), h)
        return h, self.head(h)


def _crop(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = int(np.clip(cx - CROP // 2, 0, w - CROP))
    y0 = int(np.clip(cy - CROP // 2, 0, h - CROP))
    return frame[y0 : y0 + CROP, x0 : x0 + CROP]


def _episode_pack(ep_dir: Path, vocab: ActionVocab):
    """Episode to hero crops, position deltas, validity and actions.
    Full-resolution frames never leave this function."""
    ep = Episode(ep_dir)
    traj_path = ep_dir / "ego_traj.npy"
    if traj_path.exists():
        traj = np.load(traj_path)
    else:
        traj = extract_trajectory(ep_dir, traj_path)
    traj = traj.copy()
    for c in (0, 1):   # median of 3: the box centre jitters
        traj[1:-1, c] = np.median(
            np.stack([traj[:-2, c], traj[1:-1, c], traj[2:, c]]), axis=0)
    frames = ep.frames[:]
    n = len(frames)
    crops = np.stack([_crop(frames[i], traj[i, 0], traj[i, 1]) for i in range(n)])
    del frames
    return {
        "crops": crops,
        "pos": traj[:, :2].astype(np.float32),
        "valid": traj[:, 2] > 0,
        "labels": vocab.encode(ep.actions[:, 0]),
    }


def train_ego(episode_dir: str | Path, out_dir: str | Path,
              epochs: int = 4, batch: int = 64, lr: float = 3e-4, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Comma-separated sources; each is an episode or a directory of them
    ep_dirs: list[Path] = []
    for part in str(episode_dir).split(","):
        root = Path(part)
        if (root / "metadata.json").exists():
            ep_dirs.append(root)
        else:
            ep_dirs.extend(sorted(
                d for d in root.iterdir() if (d / "metadata.json").exists()))
    all_actions = np.concatenate([Episode(d).actions[:, 0] for d in ep_dirs])
    vocab = ActionVocab.from_actions(all_actions)
    packs = []
    for d in ep_dirs:
        print(f"pack {d.name}...", flush=True)
        packs.append(_episode_pack(d, vocab))
    # Windows are (pack, start) where the hero is visible for all SEQ+1 frames
    # and no window crosses an episode boundary
    windows = [(pi, s) for pi, p in enumerate(packs)
               for s in range(len(p["crops"]) - SEQ - 1)
               if p["valid"][s : s + SEQ + 1].all()]
    n_val = max(1, len(windows) // 10)
    train_starts = windows[:-n_val]
    val_starts = windows[-n_val:]
    print(f"episodes: {len(packs)} windows: train={len(train_starts)} val={len(val_starts)}")

    dev = device()
    model = EgoModel(len(vocab)).to(dev)
    model.enc(torch.zeros(1, 3, CROP, CROP, device=dev))  # LazyLinear init
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def batch_tensors(starts):
        crops = np.stack([packs[pi]["crops"][s : s + SEQ] for pi, s in starts])
        pos = np.stack([packs[pi]["pos"][s : s + SEQ + 1] for pi, s in starts])
        acts = np.stack([packs[pi]["labels"][s : s + SEQ] for pi, s in starts])
        c = torch.from_numpy(crops).float().div_(255).permute(0, 1, 4, 2, 3).to(dev)
        d = torch.from_numpy(np.diff(pos, axis=1)).float().to(dev)   # (B,SEQ,2) deltas
        a = torch.from_numpy(acts).long().to(dev)
        return c, d, a

    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        rng.shuffle(train_starts)
        model.train()
        losses = []
        for bi in range(0, len(train_starts), batch):
            starts = train_starts[bi : bi + batch]
            if not len(starts):
                continue
            c, d, a = batch_tensors(starts)
            B = c.shape[0]
            h = torch.zeros(B, 128, device=dev)
            vel = torch.zeros(B, 2, device=dev)
            loss = 0.0
            for k in range(SEQ):
                feat = model.enc(c[:, k])
                h, pred = model.forward_step(h, feat, vel, a[:, k])
                loss = loss + nn.functional.mse_loss(pred, d[:, k])
                # Scheduled sampling: half the steps run on the model's own
                # velocity, or the open-loop rollout falls apart at inference
                if k >= 2 and np.random.rand() < 0.5:
                    vel = pred.detach()
                else:
                    vel = d[:, k]
            loss = loss / SEQ
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        rec = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(rec)
        print(rec)

    # The metric: an open-loop rollout with true against shuffled actions.
    # After the warm-up the crop is frozen — we are imagining — so the model
    # runs on nothing but its recurrent state and the actions.
    model.eval()
    err_t, err_s = [], []
    with torch.no_grad():
        for w in val_starts:
            c, d, a = batch_tensors([w])
            a_shuf = a[:, torch.randperm(SEQ)]
            for acts, sink in ((a, err_t), (a_shuf, err_s)):
                h = torch.zeros(1, 128, device=dev)
                vel = torch.zeros(1, 2, device=dev)
                errs = []
                feat = model.enc(c[:, 0])
                for k in range(SEQ):
                    if k < 4:   # warm up on real crops
                        feat = model.enc(c[:, k])
                    h, pred = model.forward_step(h, feat, vel, acts[:, k])
                    errs.append(float(nn.functional.mse_loss(pred, d[:, k])))
                    vel = pred
                sink.append(float(np.mean(errs[4:])))
    e_t, e_s = float(np.mean(err_t)), float(np.mean(err_s))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    meta = {
        "vocab_masks": vocab.masks,
        "episode": str(episode_dir),
        "history": history,
        "ego_mse_true": e_t,
        "ego_mse_shuffled": e_s,
        "action_advantage": e_s / max(e_t, 1e-9),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"true": e_t, "shuffled": e_s, "advantage": meta["action_advantage"]}))
    return meta


class GhostPredictor:
    """The ghost trajectory: where the hero ends up in K steps if the current
    intention is held."""

    def __init__(self, run_dir: str | Path):
        run = Path(run_dir)
        meta = json.loads((run / "meta.json").read_text())
        self.vocab = ActionVocab(masks=meta["vocab_masks"])
        # CPU on purpose: twelve small sequential GRU steps on the GPU are
        # twelve dispatches of a few milliseconds each, competing with the
        # policy and Grad-CAM for the device. On the CPU it is microseconds.
        self.dev = torch.device("cpu")
        self.model = EgoModel(len(self.vocab)).to(self.dev).eval()
        self.model.enc(torch.zeros(1, 3, CROP, CROP, device=self.dev))
        self.model.load_state_dict(torch.load(run / "model.pt", map_location=self.dev))

    def predict(self, frame_rgb: np.ndarray, cx: float, cy: float,
                vel: tuple[float, float], mask: int, steps: int = 12) -> list[tuple[float, float]]:
        try:
            aid = self.vocab.masks.index(int(mask))
        except ValueError:
            aid = 0
        with torch.no_grad():
            crop = _crop(frame_rgb, cx, cy)
            feat = self.model.enc(
                torch.from_numpy(crop).float().div_(255)
                .permute(2, 0, 1).unsqueeze(0).to(self.dev))
            h = torch.zeros(1, 128, device=self.dev)
            v = torch.tensor([[vel[0], vel[1]]], device=self.dev, dtype=torch.float32)
            a = torch.tensor([aid], device=self.dev)
            pts = []
            x, y = cx, cy
            for _ in range(steps):
                h, pred = self.model.forward_step(h, feat, v, a)
                dx, dy = float(pred[0, 0]), float(pred[0, 1])
                x, y = x + dx, y + dy
                pts.append((x, y))
                v = pred
        return pts
