"""Behavioural cloning: a CNN over frames, optionally over sound as well.

The action vocabulary is built from the data — the distinct button masks that
actually occur — rather than from all 256 combinations, most of which no player
ever presses.

Two details here are load-bearing and easy to get wrong. Validation is the tail
of an episode, never a random split: consecutive frames are near-duplicates and
a random split leaks the answer. And accuracy is always reported next to the
majority-class baseline, because on a game where one button covers 88% of
frames, 90% accuracy means nothing.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from nes_player import provenance
from nes_player.data.reader import Episode
from nes_player.emulator.controller import BUTTONS, resolve_conflicts

FRAME_STACK = 4
INPUT_HW = (112, 120)   # half of the canonical 224×240

# How far back each frame in the stack is taken from, newest last. Four
# consecutive frames span 67 ms, which is enough to see a velocity and nothing
# else — the policy is effectively memoryless. A player holds far more than
# that: where the enemy came from, that this door was already tried, which way
# the level was going.
#
# Spacing the samples geometrically buys time depth at no extra cost: the same
# number of channels, the same network, a window tens of times longer. It is
# not memory in the proper sense — a recurrent state or a map would be that —
# but it is the difference between seeing a moment and seeing a manoeuvre.
FRAME_OFFSETS = {
    "short": (4, 3, 2, 1),               # 67 ms, the original stack
    "wide": (32, 16, 8, 4, 2, 1),        # 0.53 s
    "long": (128, 64, 32, 16, 8, 4, 2, 1),   # 2.1 s
    # Ten times `long`. Two costs that the shorter presets do not have: the
    # first 1280 frames of every episode produce no training sample, which is
    # 36% of a 60-second recording, and the first convolution grows to 33 input
    # channels. Whether a convolution can use a frame from 21 seconds ago is the
    # open question — it has no way to know that the thing in it is the same
    # thing it is looking at now.
    "epic": (1280, 640, 320, 160, 80, 40, 20, 10, 5, 2, 1),   # 21.3 s
}
DEFAULT_OFFSETS = FRAME_OFFSETS["short"]

# Audio: a log-mel window of about 260 ms ending at the moment of the decision
MEL_HOP = 160
MEL_N = 32
MEL_FRAMES = 52   # 52 × 160 / 32040 is about 260 ms


def mel_transform(sample_rate: int):
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=512, hop_length=MEL_HOP, n_mels=MEL_N)


def episode_log_mel(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """The episode's whole audio track as an unnormalised (MEL_N, T) log-mel."""
    wav = torch.from_numpy(audio.astype(np.float32) / 32768)
    mel = mel_transform(sample_rate)(wav)
    return torch.log(mel + 1e-5).numpy()


def mel_moments(mels: list[np.ndarray]) -> tuple[float, float]:
    """Mean and standard deviation over several log-mels, pooled by size.

    Averaging per-episode means and per-episode standard deviations — which is
    what this used to do — is not the statistic of the pooled data unless every
    episode is the same length and has the same spread. Neither holds.
    """
    n = sum(m.size for m in mels)
    mean = sum(float(m.sum()) for m in mels) / n
    var = sum(float(((m - mean) ** 2).sum()) for m in mels) / n
    return mean, float(np.sqrt(var))


def episode_mel(audio: np.ndarray, sample_rate: int):
    """Log-mel normalised by the episode's own statistics.

    Kept for the AV-align model, which is trained per episode and has no
    train/validation split to leak across. Behavioural cloning must not use it:
    normalising a validation episode by its own statistics tells the network
    about audio it has not heard yet, and normalising each training episode by
    its own puts them all on different scales from the one used at play time.
    """
    mel = episode_log_mel(audio, sample_rate)
    mean, std = float(mel.mean()), float(mel.std())
    return (mel - mean) / (std + 1e-6), mean, std


ATTN_HW = (10, 11)   # shape of BCNet's last conv output at INPUT_HW


def _attn_cache_name(source: str, leads: tuple[int, ...]) -> str:
    from nes_player.perception.motion import TRACKER_VERSION
    from nes_player.perception.sprites import SPRITES_VERSION

    # The version is in the filename on purpose. Checking only the shape, as
    # this did, meant that improving the tracker left every dataset holding
    # masks from the old one — same shape, still loaded, silently teaching the
    # model to look where the tracker used to be wrong.
    suffix = "" if leads == (0,) else ".lead" + "-".join(str(v) for v in leads)
    if source == "oam":
        return f"attn_mask.oam.v{SPRITES_VERSION}{suffix}.npy"
    return f"attn_mask.v{TRACKER_VERSION}{suffix}.npy"


def episode_attn_masks(ep: Episode, lead: int | tuple[int, ...] = 0,
                       source: str = "tracker") -> np.ndarray:
    """Masks of "objects are here"; cached on disk.

    Two sources. `tracker` infers objects from motion in the pixels, which is
    what the agent itself has to do and is therefore honest but wrong sometimes
    — on Double Dragon it latched onto a piece of background and the instinct
    policy spent three minutes jumping at it. `oam` reads the console's own
    sprite table, which is exact and free in every NES game. Supervision is
    allowed to cheat: the target says where to look, the network still has to
    find it in pixels at play time (spec §3).

    These supervise attention. Left alone, behavioural cloning puts only 12.5%
    of its attention inside object boxes against 13.8% for a uniform gaze — it
    keys on background that correlates with the action just as well as the enemy
    does. The boxes are downsampled to the conv feature grid and used as a
    target for the spatial softmax of the activations.

    With `lead` above zero the target is where each object *will be* in that
    many frames, extrapolated from the velocity the tracker already computes.
    A player aims at where the enemy is going, not where it is; this is the
    cheapest possible way to ask the network to do the same — no model, just
    cx + vx·lead.

    `lead` may be several values, in which case the target is their union. That
    matters: a single lead *replaces* now with later, and measuring it showed
    the cost of doing so — an agent watching where the enemy will be stops
    hitting the enemy in front of it. A union marks both places and leaves the
    network to weight them, which is the difference between prescribing a
    strategy and offering one.
    """
    leads = tuple(sorted({lead} if isinstance(lead, int) else set(lead)))
    cache = ep.path / _attn_cache_name(source, leads)
    if cache.exists():
        m = np.load(cache)
        if m.shape[1:] == ATTN_HW:
            return m

    if source == "oam":
        if leads != (0,):
            raise ValueError(
                "attention lead needs per-object velocity, and the sprite table "
                "gives positions only — matching sprites between frames is a "
                "tracker again. Use --attn-source tracker for a lead.")
        from nes_player.perception.sprites import episode_sprites, sprite_mask

        boxes = episode_sprites(ep)
        hw = ep.frames.shape[1:3]
        out = np.stack([sprite_mask(b, hw, ATTN_HW) for b in boxes])
        np.save(cache, out)
        return out

    from nes_player.perception.motion import MotionTracker

    tracker = MotionTracker()
    frames = ep.frames
    h_full, w_full = frames.shape[1:3]
    out = np.zeros((frames.shape[0], *ATTN_HW), np.uint8)
    for i in range(frames.shape[0]):
        f = frames[i]
        full = np.zeros(f.shape[:2], np.uint8)
        for s in tracker.update(f, frozenset()):
            bx, by, w, h = s.bbox
            for lv in leads:
                # Extrapolate, then clip rather than drop: an object heading off
                # screen is still worth watching at the edge it is leaving by.
                x = max(0, min(int(round(bx + s.vx * lv)), w_full - w))
                y = max(0, min(int(round(by + s.vy * lv)), h_full - h))
                full[y : y + h, x : x + w] = 255
        small = cv2.resize(full, (ATTN_HW[1], ATTN_HW[0]), interpolation=cv2.INTER_AREA)
        out[i] = small > 16   # a cell counts once an object covers about 6% of it
    np.save(cache, out)
    return out


def _build_attn_cache(ep_dirs: list[Path], lead=0, source: str = "tracker") -> None:
    """Fill the attention cache for several episodes at once, before training.

    Built lazily one episode at a time this is two minutes each — over an hour
    for a directory of thirty-odd, on one core, with the rest of the machine
    idle and no output to say what is happening. The episodes are independent,
    so they simply go in parallel.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    leads = tuple(sorted({lead} if isinstance(lead, int) else set(lead)))
    name = _attn_cache_name(source, leads)
    todo = [d for d in ep_dirs if not (d / name).exists()]
    if not todo:
        return
    workers = min(len(todo), max(1, (os.cpu_count() or 4) - 2))
    print(f"attention masks ({source}): building {len(todo)} on {workers} workers",
          flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(_attn_worker,
                                       [(str(d), lead, source) for d in todo]), 1):
            print(f"  masks {i}/{len(todo)}", flush=True)


def _attn_worker(arg) -> int:
    path_str, lead, source = arg
    return int(episode_attn_masks(Episode(Path(path_str)), lead, source).shape[0])


def mask_name(mask: int) -> str:
    names = [b for i, b in enumerate(BUTTONS) if mask >> i & 1]
    return "+".join(names) if names else "NOOP"


def mask_to_pressed(mask: int) -> frozenset[str]:
    """A vocabulary index turned into buttons a hand could actually press.

    Every network policy decodes through here, which is why the conflict
    resolver lives here rather than in each of them. Old checkpoints carry
    LEFT+RIGHT in their vocabulary — 24 of ours do — because it was cloned from
    TAS movies that record it; they can still be loaded, they just cannot
    produce it any more.
    """
    return resolve_conflicts(b for i, b in enumerate(BUTTONS) if mask >> i & 1)


def normalise_mask(mask: int) -> int:
    """The same bitmask with impossible directions removed."""
    pressed = mask_to_pressed(mask)
    return sum(1 << i for i, b in enumerate(BUTTONS) if b in pressed)


@dataclass
class ActionVocab:
    masks: list[int]   # index -> button bitmask

    @classmethod
    def from_actions(cls, actions: np.ndarray) -> ActionVocab:
        return cls(masks=sorted({normalise_mask(int(m)) for m in np.unique(actions)}))

    @property
    def names(self) -> list[str]:
        return [mask_name(m) for m in self.masks]

    def encode(self, actions: np.ndarray) -> np.ndarray:
        # Labels are normalised the same way the vocabulary was, so a TAS frame
        # holding LEFT+RIGHT becomes the closest thing a player could have done
        # instead of an unreachable class of its own.
        lut = {m: i for i, m in enumerate(self.masks)}
        return np.asarray([lut[normalise_mask(int(a))] for a in actions],
                          dtype=np.int64)

    def __len__(self) -> int:
        return len(self.masks)


class BCNet(nn.Module):
    def __init__(self, n_actions: int, in_ch: int = FRAME_STACK * 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.body(torch.zeros(1, in_ch, *INPUT_HW)).shape[1]
        self.head = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def preprocess_frames(frames: np.ndarray) -> np.ndarray:
    """Resize the frames, keeping colour: on the NES it carries information."""
    out = np.empty((len(frames), *INPUT_HW, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = cv2.resize(f, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
    return out


def stack_to_tensor(stack: np.ndarray) -> torch.Tensor:
    """(FRAME_STACK, H, W, 3) uint8 → (FRAME_STACK*3, H, W) float32 [0,1]."""
    t = torch.from_numpy(stack).float().div_(255)
    return t.permute(0, 3, 1, 2).reshape(-1, *stack.shape[1:3])


class AudioEncoder(nn.Module):
    """A small encoder for the log-mel window: 2D CNN to a 128-dim embedding."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, 1, MEL_N, MEL_FRAMES)).shape[1]
        self.proj = nn.Linear(n, 128)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(m))


class BCNetAV(nn.Module):
    """Multimodal: visual conv and audio conv concatenated before the head."""

    def __init__(self, n_actions: int, in_ch: int = FRAME_STACK * 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.body(torch.zeros(1, in_ch, *INPUT_HW)).shape[1]
        self.audio = AudioEncoder()
        self.head = nn.Sequential(
            nn.Linear(n_flat + 128, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.body(v), self.audio(m)], dim=1))


class EpisodeBCDataset(torch.utils.data.Dataset):
    """Sample i: the frames at i-offset predict the action taken at frame i."""

    def __init__(self, small_frames: np.ndarray, labels: np.ndarray, attn=None,
                 offsets: tuple[int, ...] = DEFAULT_OFFSETS):
        self.frames = small_frames
        self.labels = labels
        self.attn = attn   # mask of the last visible frame, or None
        self.offsets = offsets
        self.span = max(offsets)

    def __len__(self) -> int:
        return len(self.frames) - self.span

    def __getitem__(self, idx: int):
        i = idx + self.span
        x = stack_to_tensor(self.frames[[i - o for o in self.offsets]])
        if self.attn is None:
            return x, int(self.labels[i])
        return x, int(self.labels[i]), torch.from_numpy(self.attn[i - 1].astype(np.float32))


class EpisodeBCDatasetAV(EpisodeBCDataset):
    """Adds the log-mel window ending at the start of frame i. The mel and the
    offsets belong to the WHOLE episode; a dataset segment is a slice of the
    offsets, so that windows never straddle an episode boundary."""

    def __init__(self, small_frames, labels, mel: np.ndarray, audio_offsets: np.ndarray,
                 attn=None, offsets: tuple[int, ...] = DEFAULT_OFFSETS):
        super().__init__(small_frames, labels, attn, offsets)
        self.mel = mel
        self.audio_offsets = audio_offsets

    def __getitem__(self, idx: int):
        x, *rest = super().__getitem__(idx)
        i = idx + self.span
        m_end = int(self.audio_offsets[i] // MEL_HOP)
        m = np.zeros((MEL_N, MEL_FRAMES), np.float32)
        seg = self.mel[:, max(0, m_end - MEL_FRAMES) : m_end]
        if seg.shape[1]:
            m[:, MEL_FRAMES - seg.shape[1]:] = seg
        return x, torch.from_numpy(m).unsqueeze(0), *rest


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_bc(
    episode_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    use_audio: bool = False,
    seed: int = 0,
    init_from: str | Path | None = None,
    max_episodes: int | None = None,
    attn: float = 0.0,
    attn_lead: int | tuple[int, ...] = 0,
    attn_source: str = "tracker",
    memory: str = "short",
) -> dict:
    offsets = FRAME_OFFSETS[memory]
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Comma-separated sources; each is an episode or a directory of episodes
    ep_dirs: list[Path] = []
    for part in str(episode_dir).split(","):
        root = Path(part)
        if (root / "metadata.json").exists():
            ep_dirs.append(root)
        else:
            ep_dirs.extend(sorted(
                d for d in root.iterdir() if (d / "metadata.json").exists()))
    if max_episodes:
        ep_dirs = ep_dirs[:max_episodes]
    eps = [Episode(d) for d in ep_dirs]
    print(f"episodes: {len(eps)}, frames: {sum(len(e) for e in eps)}")
    if attn:
        _build_attn_cache(ep_dirs, attn_lead, attn_source)
    all_actions = np.concatenate([e.actions[:, 0] for e in eps])
    vocab = ActionVocab.from_actions(all_actions)
    sr = eps[0].metadata["sample_rate"]
    mel_stats = None

    # Audio normalisation is decided by the training data alone, and the same
    # numbers are then used for validation and, from the checkpoint, at play
    # time. Previously there were three different distributions: each episode
    # normalised by its own statistics (validation included, which is a look at
    # its own future), and inference by an average of all of them.
    n_val_eps = max(1, int(len(eps) * val_frac))
    raw_mels: list[np.ndarray] = []
    if use_audio:
        raw_mels = [episode_log_mel(e.audio[:], sr) for e in eps]
        if len(eps) == 1:
            # One episode: the split is by time, so cut the log-mel at the same
            # place. Columns are MEL_HOP audio samples apart.
            cut = int(eps[0].audio_offsets[len(eps[0]) - int(len(eps[0]) * val_frac)])
            train_mels = [raw_mels[0][:, : max(1, cut // MEL_HOP)]]
        else:
            train_mels = raw_mels[:-n_val_eps]
        mel_mean, mel_std = mel_moments(train_mels)
        mel_stats = {"mean": mel_mean, "std": mel_std, "sample_rate": sr}
        raw_mels = [(m - mel_mean) / (mel_std + 1e-6) for m in raw_mels]

    def episode_ds(e: Episode, k: int):
        small = preprocess_frames(e.frames[:])
        labels = vocab.encode(e.actions[:, 0])
        masks = episode_attn_masks(e, attn_lead, attn_source) if attn else None
        if not use_audio:
            return EpisodeBCDataset(small, labels, masks, offsets), labels
        return (EpisodeBCDatasetAV(small, labels, raw_mels[k], e.audio_offsets,
                                   masks, offsets), labels)

    if len(eps) == 1:   # one episode: validation is its last 10%
        small = preprocess_frames(eps[0].frames[:])
        labels = vocab.encode(eps[0].actions[:, 0])
        masks = episode_attn_masks(eps[0], attn_lead, attn_source) if attn else None
        n_val = int(len(small) * val_frac)
        n_train = len(small) - n_val
        mtr = masks[:n_train] if attn else None
        mva = masks[n_train:] if attn else None
        if use_audio:
            off = eps[0].audio_offsets
            train_ds = EpisodeBCDatasetAV(
                small[:n_train], labels[:n_train], raw_mels[0], off[: n_train + 1],
                mtr, offsets)
            val_ds = EpisodeBCDatasetAV(
                small[n_train:], labels[n_train:], raw_mels[0], off[n_train:],
                mva, offsets)
        else:
            train_ds = EpisodeBCDataset(small[:n_train], labels[:n_train], mtr, offsets)
            val_ds = EpisodeBCDataset(small[n_train:], labels[n_train:], mva, offsets)
        val_labels = labels[n_train:]
        n_val_count = n_val
    else:   # many: validation is the last 10% of episodes, taken whole
        parts = [episode_ds(e, k) for k, e in enumerate(eps)]
        train_ds = torch.utils.data.ConcatDataset([d for d, _ in parts[:-n_val_eps]])
        val_ds = torch.utils.data.ConcatDataset([d for d, _ in parts[-n_val_eps:]])
        val_labels = np.concatenate([lbl for _, lbl in parts[-n_val_eps:]])
        n_val_count = len(val_labels)
    if use_audio:
        model = BCNetAV(len(vocab), in_ch=len(offsets) * 3)
    else:
        model = BCNet(len(vocab), in_ch=len(offsets) * 3)
    if init_from:   # transfer: body and audio encoder from the base, new heads —
        # the target game has a different action vocabulary
        base = torch.load(Path(init_from) / "model.pt", map_location="cpu")
        own = model.state_dict()
        moved = [k for k, v in base.items()
                 if k in own and own[k].shape == v.shape and not k.startswith("head")]
        own.update({k: base[k] for k in moved})
        model.load_state_dict(own)
        print(f"transferred {len(moved)} tensors from {init_from}")
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    dev = device()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    acts: dict = {}
    if attn:   # attention supervision needs the last conv activations
        model.body[4].register_forward_hook(lambda m_, i_, o: acts.__setitem__("a", o))

    def _forward(batch):
        if use_audio:
            x, m, y, *rest = batch
            logits = model(x.to(dev), m.to(dev))
        else:
            x, y, *rest = batch
            logits = model(x.to(dev))
        return logits, y.to(dev), rest[0].to(dev) if rest else None

    def _attn_loss(target):
        """Cross-entropy of the activations' spatial softmax against the mask."""
        a = acts["a"].relu().mean(1).flatten(1)  # (B, h*w)
        logp = torch.log_softmax(a, dim=1)
        t = target.flatten(1)
        tsum = t.sum(1)
        valid = tsum > 0   # frames with no objects teach nothing
        if not valid.any():
            return None
        ce = -(t / tsum.clamp(min=1).unsqueeze(1) * logp).sum(1)
        return ce[valid].mean()

    history = []
    # Keep the best epoch, not the last one. Past a handful of epochs training
    # accuracy keeps climbing while validation turns over, and saving whatever
    # the final epoch happened to be throws away the useful model in favour of
    # a memorised one. (Validation accuracy ranks how faithfully the demos are
    # cloned, not how well the agent plays — measured: a 25-point gap between
    # two checkpoints produced play that was indistinguishable. So this picks
    # the best clone, which is the most this loss can be asked for.)
    best_acc, best_epoch, best_state = -1.0, -1, None
    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for batch in train_dl:
            logits, y, amask = _forward(batch)
            loss = nn.functional.cross_entropy(logits, y)
            if attn and amask is not None:
                al = _attn_loss(amask)
                if al is not None:
                    loss = loss + attn * al
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                logits, y, _ = _forward(batch)
                v_correct += int((logits.argmax(1) == y).sum())
                v_total += len(y)
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_acc": correct / total,
            "val_acc": v_correct / v_total,
        }
        history.append(rec)
        if rec["val_acc"] > best_acc:
            best_acc, best_epoch = rec["val_acc"], epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(rec)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(),
               out / "model.pt")
    if best_epoch >= 0 and best_epoch != epochs - 1:
        print(f"kept epoch {best_epoch} (val_acc {best_acc:.3f}), "
              f"not the last one ({history[-1]['val_acc']:.3f})")
    majority = float(np.bincount(val_labels).max() / n_val_count)
    meta = {
        "vocab_masks": vocab.masks,
        "vocab_names": vocab.names,
        "episode": str(episode_dir),
        "frame_stack": len(offsets),
        "frame_offsets": list(offsets),   # a checkpoint remembers its own geometry
        "memory": memory,
        "input_hw": list(INPUT_HW),
        "modality": "av" if use_audio else "video",
        "best_epoch": best_epoch,
        "attn": attn,
        "attn_source": attn_source,
        "attn_lead": list(attn_lead) if not isinstance(attn_lead, int) else attn_lead,
        "mel_stats": mel_stats,
        "history": history,
        "val_majority_baseline": majority,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    provenance.write(out, config={k: v for k, v in meta.items()
                                  if k not in ("history", "vocab_masks",
                                               "vocab_names", "mel_stats")},
                     episodes=ep_dirs, game=eps[0].metadata.get("game"))
    return meta


class BCPolicy:
    """Inference: keeps the frame stack, returns an action and its distribution."""

    def __init__(self, run_dir: str | Path):
        run = Path(run_dir)
        meta = json.loads((run / "meta.json").read_text())
        self.vocab = ActionVocab(masks=meta["vocab_masks"])
        self.dev = device()
        self.modality = meta.get("modality", "video")
        # Checkpoints written before dilated stacks existed have no offsets and
        # are four consecutive frames by definition.
        self.offsets = tuple(meta.get("frame_offsets", DEFAULT_OFFSETS))
        self.span = max(self.offsets)
        in_ch = len(self.offsets) * 3
        if self.modality == "av":
            self.model = BCNetAV(len(self.vocab), in_ch=in_ch).to(self.dev).eval()
            st = meta["mel_stats"]
            self._mel_t = mel_transform(st["sample_rate"])
            self._mel_mean, self._mel_std = st["mean"], st["std"]
            self._audio_ring: list[np.ndarray] = []
            self._mel_cache = None
        else:
            self.model = BCNet(len(self.vocab), in_ch=in_ch).to(self.dev).eval()
        self.model.load_state_dict(torch.load(run / "model.pt", map_location=self.dev))
        self._stack: list[np.ndarray] = []
        self.last_features: np.ndarray | None = None   # (8, h, w), for the dashboard

    def reset(self) -> None:
        self._stack.clear()
        if self.modality == "av":
            self._audio_ring.clear()

    def push_audio(self, pcm: np.ndarray) -> None:
        """Call every frame to accumulate the PCM window; a no-op for video-only."""
        if self.modality != "av":
            return
        self._audio_ring.append(pcm)
        need = MEL_FRAMES * MEL_HOP + 512
        while sum(len(c) for c in self._audio_ring[:-1]) > need:
            self._audio_ring.pop(0)

    def _current_mel(self, cache_ok: bool = False) -> torch.Tensor:
        if cache_ok and self._mel_cache is not None:
            return self._mel_cache
        if self._audio_ring:
            wav = np.concatenate(self._audio_ring).astype(np.float32) / 32768
        else:
            wav = np.zeros(MEL_FRAMES * MEL_HOP, np.float32)
        mel = torch.log(self._mel_t(torch.from_numpy(wav)) + 1e-5)
        mel = (mel - self._mel_mean) / (self._mel_std + 1e-6)
        m = torch.zeros(MEL_N, MEL_FRAMES)
        seg = mel[:, -MEL_FRAMES:]
        m[:, MEL_FRAMES - seg.shape[1]:] = seg
        self._mel_cache = m.unsqueeze(0).unsqueeze(0).to(self.dev)
        return self._mel_cache

    def _input_tensor(self, frame_rgb: np.ndarray, advance: bool) -> torch.Tensor:
        """The network's visual input for this frame.

        One helper for both callers on purpose. When `compute_cam` built its own
        stack it hard-coded four consecutive frames, so every checkpoint wider
        than `short` was handed 12 channels where it wanted 18, 24 or 33. The
        GUI calls it every frame inside a blanket `except`, so the attention
        overlay was silently dead on three of the four memory presets.

        `advance` distinguishes deciding, which consumes the frame, from
        looking at the same frame again between decisions.
        """
        stack = self._extend(frame_rgb)
        if advance:
            self._stack = stack
        picked = np.stack([stack[-o] for o in self.offsets])
        return stack_to_tensor(picked).unsqueeze(0).to(self.dev)

    def _extend(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        small = cv2.resize(frame_rgb, (INPUT_HW[1], INPUT_HW[0]),
                           interpolation=cv2.INTER_AREA)
        # The ring holds every frame back to the furthest offset, but only the
        # sampled ones reach the network. At the start of an episode there is
        # no history, so the oldest frame available stands in for the ones
        # before it.
        stack = self._stack + [small]
        if len(stack) < self.span:
            stack = [stack[0]] * (self.span - len(stack)) + stack
        return stack[-self.span:]

    # ---------- observe / decide, for the frame-indexed evaluator ----------

    def observe(self, frame_rgb: np.ndarray, audio_pcm: np.ndarray | None = None) -> None:
        """Take in one emulator frame. Cheap: no network runs here.

        Separating this from `decide` is what makes the memory presets mean
        what they say. The offsets are frame counts — `long` reaches 128 frames
        back, which the comment calls 2.1 seconds — but the history used to be
        filled only when a decision was taken, and decisions were taken on a
        wall clock at 15 Hz. So one entry in the history covered about four
        emulator frames in realtime and an unpredictable number headless, and
        `long` was really reaching 8.5 seconds, or whatever the machine's load
        made it that day.
        """
        self._stack = self._extend(frame_rgb)
        if audio_pcm is not None:
            self.push_audio(audio_pcm)
        if self.modality == "av":
            self._mel_cache = None   # the window moved; recompute on demand

    def decide(self, temperature: float = 1.0):
        """Choose an action from what has been observed. Returns (pressed, ranked)."""
        if not self._stack:
            raise RuntimeError("decide() before any observe()")
        picked = np.stack([self._stack[-o] for o in self.offsets])
        x = stack_to_tensor(picked).unsqueeze(0).to(self.dev)
        m = self._current_mel() if self.modality == "av" else None
        with torch.no_grad():
            logits = (self.model(x, m) if m is not None else self.model(x))[0]
        return self._sample(logits, temperature)

    def _sample(self, logits: torch.Tensor, temperature: float):
        probs = torch.softmax(logits / temperature, dim=0).cpu().numpy()
        idx = int(np.random.choice(len(probs), p=probs / probs.sum()))
        ranked = sorted(zip(self.vocab.names, probs.tolist(), strict=True),
                        key=lambda t: -t[1])
        return mask_to_pressed(self.vocab.masks[idx]), ranked

    def act(self, frame_rgb: np.ndarray, temperature: float = 1.0, with_cam: bool = False):
        """Returns (pressed, ranked) and optionally a Grad-CAM map in [0,1]."""
        x = self._input_tensor(frame_rgb, advance=True)

        m = self._current_mel() if self.modality == "av" else None
        cam = None
        if with_cam:
            logits, cam = self._forward_with_cam(x, m)
        else:
            with torch.no_grad():
                logits = (self.model(x, m) if m is not None else self.model(x))[0]

        pressed, ranked = self._sample(logits, temperature)
        return (pressed, ranked, cam) if with_cam else (pressed, ranked)

    def _forward_with_cam(self, x: torch.Tensor, mel: torch.Tensor | None = None):
        acts: dict = {}
        hook = self.model.body[4].register_forward_hook(   # the last conv layer
            lambda m, i, o: acts.__setitem__("a", o))
        with torch.enable_grad():
            logits = (self.model(x, mel) if mel is not None else self.model(x))[0]
            a = acts["a"]
            grad = torch.autograd.grad(logits[int(logits.argmax())], a)[0]
        hook.remove()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        heat = torch.relu((weights * a).sum(1))[0].detach()
        heat = heat / (heat.max() + 1e-8)
        # The eight most active channels of the last conv, shown live on the panel
        a0 = a[0].detach()
        top = a0.mean(dim=(1, 2)).topk(8).indices
        feats = a0[top]
        feats = feats / (feats.amax(dim=(1, 2), keepdim=True) + 1e-8)
        self.last_features = feats.cpu().numpy()
        return logits.detach(), heat.cpu().numpy()

    def compute_cam(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Grad-CAM for the current frame without advancing the stack or picking
        an action, so it can be called every frame between decisions."""
        x = self._input_tensor(frame_rgb, advance=False)
        m = self._current_mel(cache_ok=True) if self.modality == "av" else None
        _, cam = self._forward_with_cam(x, m)
        return cam
