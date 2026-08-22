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
from nes_player.perception.motion import pick_hero
from nes_player.perception.sprites import SPRITES_VERSION, SpriteTracker
from nes_player.policy.bc import ActionVocab, device

CROP = 48   # crop around the hero, in source-frame pixels
MAX_SCROLL = 6.0   # px per frame; more than this is a scene change
MAX_STEP = 8.0     # px per frame the hero can move; more is the wrong object
# How far ahead the model is trained to imagine, and the planner to plan.
# Sixteen frames was too short for the choice of button to matter: asked of the
# emulator itself, running rather than standing still is worth 1.7 px over 16
# frames, 5.9 over 32, 15.9 over 64, and the spread across the four candidate
# actions grows from 7.7 px to 157.5. Mario is heavy, and inside a quarter of a
# second inertia decides almost everything. The model was not failing to see
# what the buttons do; over that horizon they barely do anything.
SEQ = 48
SWITCH_WEIGHT = 8  # how many times over a window starting at a button change counts


def _sprite_boxes(ep: Episode) -> np.ndarray:
    """The episode's cached sprite table, (N, 64, 3).

    Accepts the older two-column table by zero-filling the tile ids: this only
    needs to know where the objects are. Never replays the run to build one —
    Go-Explore segments start from a restored save state and cannot be replayed
    from their first frame at all.
    """
    for name in (f"sprites.v{SPRITES_VERSION}.npy", "sprites.v1.npy"):
        path = ep.path / name
        if path.exists():
            a = np.load(path)
            if a.shape[-1] == 2:
                a = np.concatenate([a, np.zeros((*a.shape[:2], 1), a.dtype)], -1)
            return a
    raise FileNotFoundError(f"{ep.path} has no cached sprite table")


def mask_to_frozen(mask: int) -> frozenset:
    return frozenset(b for i, b in enumerate(BUTTONS) if mask >> i & 1)


def extract_trajectory(episode_dir: str | Path, out_npy: str | Path | None = None):
    """(N, 4): screen cx, cy, world x, valid.

    Both frames are needed and they are not interchangeable. The crop the model
    looks at is cut from the screen, so it wants screen coordinates. What the
    model must predict is *progress*, and in a scrolling game progress is not
    visible on screen at all: the camera holds the hero near the middle however
    fast he runs, so his screen position stops moving exactly when he is doing
    well. World x is screen position plus the scroll accumulated so far, and
    the tracker has been measuring that scroll every frame and discarding it
    here.

    Positions come from the sprite table, not from pixel differences. Measured
    against the console's own copy of Mario's coordinates, the pixel tracker's
    box scattered by 50 px and its per-frame delta correlated +0.06 with the
    truth; the sprite tracker's box is exact in 96% of frames. That mattered
    more than anything else here: the model is trained on per-frame deltas, the
    real ones have a standard deviation near 1 px, and a target buried under
    several pixels of tracker noise cannot show what a button does.
    """
    ep = Episode(Path(episode_dir))
    frames = ep.frames
    actions = ep.actions[:, 0]
    boxes = _sprite_boxes(ep)
    tracker = SpriteTracker()
    out = np.zeros((len(ep), 4), np.float32)
    last = (120.0, 112.0)
    world = 0.0
    prev_world_x: float | None = None
    gap = 1
    for i in range(len(ep)):
        pressed = mask_to_frozen(int(actions[i]))
        slots = tracker.update(frames[i], pressed, boxes=boxes[i])
        # Scroll is how far the world moved under the camera since last frame;
        # subtracting it turns screen motion into motion through the level.
        # Clamped, because a death or a level change replaces the whole picture
        # and phase correlation reports that as an enormous scroll: unclamped,
        # world x rose on 83% of steps and still finished at -4408, wrecked by
        # a handful of scene cuts. A NES camera cannot outrun the hero, and the
        # hero does about 2.5 px a frame.
        world -= float(np.clip(tracker.scroll_dx, -MAX_SCROLL, MAX_SCROLL))
        best = pick_hero(slots)
        # The remaining 4% of bad boxes are not near-misses, they are the
        # tracker naming a different object: the error jumps from 0 px to 87.
        # Nothing on a NES moves eight pixels in a frame, so a step that large
        # is a change of subject, not a change of position.
        # The budget grows with the gap since the last accepted frame, or one
        # rejection would lock the rest of the episode out.
        moved = (abs(world + best.cx - prev_world_x) / gap
                 if best is not None and prev_world_x is not None else 0.0)
        if best is not None and best.ctrl_prob > 0.7 and best.missed == 0 \
                and moved <= MAX_STEP:
            last = (best.cx, best.cy)
            prev_world_x = world + best.cx
            gap = 1
            out[i] = (best.cx, best.cy, prev_world_x, 1.0)
        else:
            gap += 1
            out[i] = (*last, world + last[0], 0.0)
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
        # Predicting the residual instead — head(h) + vel, so inertia comes free
        # and only the action is left to explain — was tried and changed
        # nothing: action advantage 1.085 against 1.092, and the template
        # ordering got worse, with `wait` moving to first place. The model's
        # indifference to the buttons is not caused by the shape of the target.
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
    # v2 put world x alongside the screen position; v3 measures both from the
    # sprite table instead of pixel differences. Every version is the same
    # array of the same shape holding different numbers, which is exactly what
    # a shape check cannot catch.
    traj_path = ep_dir / "ego_traj.v3.npy"
    if traj_path.exists():
        traj = np.load(traj_path)
    else:
        traj = extract_trajectory(ep_dir, traj_path)
    traj = traj.copy()
    for c in (0, 1, 2):   # median of 3: the box centre jitters
        traj[1:-1, c] = np.median(
            np.stack([traj[:-2, c], traj[1:-1, c], traj[2:, c]]), axis=0)
    frames = ep.frames[:]
    n = len(frames)
    crops = np.stack([_crop(frames[i], traj[i, 0], traj[i, 1]) for i in range(n)])
    del frames
    return {
        "crops": crops,
        # The target is world x and screen y: horizontal progress is invisible
        # on screen, vertical is not scrolled in these levels.
        "pos": traj[:, [2, 1]].astype(np.float32),
        "valid": traj[:, 3] > 0,
        "labels": vocab.encode(ep.actions[:, 0]),
    }


def train_ego(episode_dir: str | Path, out_dir: str | Path,
              epochs: int = 4, batch: int = 64, lr: float = 3e-4, seed: int = 0,
              branches: str | Path | None = None) -> dict:
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
    n_episode_packs = len(packs)
    if branches:
        from nes_player.world_model.counterfactual import load_packs

        # Each branch is its own pack, so no window can straddle two of them —
        # which matters more here than for episodes, since consecutive branches
        # are the same moment replayed with a different button.
        for part in str(branches).split(","):
            packs.extend(load_packs(part, vocab))
        print(f"branches: {len(packs) - n_episode_packs}", flush=True)
    # Windows are (pack, start) where the hero is visible for all SEQ+1 frames
    # and no window crosses an episode boundary
    def windows_of(lo, hi):
        return [(pi, s) for pi in range(lo, hi)
                for s in range(len(packs[pi]["crops"]) - SEQ - 1)
                if packs[pi]["valid"][s : s + SEQ + 1].all()]

    windows = windows_of(0, n_episode_packs)
    n_val = max(1, len(windows) // 10)
    train_starts = windows[:-n_val]
    val_starts = windows[-n_val:]
    # Branch windows never go in the validation set: the four branches of one
    # moment share their first frame, so a branch in val has three near-copies
    # in train. The measurements that judge this model are outside it anyway.
    train_starts = train_starts + windows_of(n_episode_packs, len(packs))
    # Where the button changes is the only place in the data that says what a
    # button does. Actions are held for four to sixteen frames and a window is
    # sixteen, so most windows show one action throughout, and the model can
    # satisfy them all with "carry on as before". Measured on such a model:
    # from a standstill it ranked doing nothing above running, and running
    # against left was the single contrast it got right, 14 times out of 14.
    # These windows are the same data, weighted so the answer is visible.
    held = set(val_starts)
    switch = [w for w in windows if w not in held and w[1] > 0
              and packs[w[0]]["labels"][w[1]] != packs[w[0]]["labels"][w[1] - 1]]
    train_starts = train_starts + switch * (SWITCH_WEIGHT - 1)
    print(f"episodes: {len(packs)} windows: train={len(train_starts)} "
          f"val={len(val_starts)} switch={len(switch)}")

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
            # One crop, then imagination — the same regime the planner uses,
            # which asks for a whole trajectory from the frame in front of it
            # and never gets another look. Training used to hand over a fresh
            # crop at every step, so the model leant on it and, when it was
            # taken away at inference, the 48-step rollout collapsed towards a
            # constant: it answered within ±8 px for every action where the
            # game spans -42 to +29.
            feat = model.enc(c[:, 0])
            for k in range(SEQ):
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

    # The metric: an open-loop rollout with true against shuffled actions, from
    # one crop, exactly as the planner asks for it.
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
                    h, pred = model.forward_step(h, feat, vel, acts[:, k])
                    errs.append(float(nn.functional.mse_loss(pred, d[:, k])))
                    vel = pred
                sink.append(float(np.mean(errs)))
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
