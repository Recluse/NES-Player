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

from nes_player.data.reader import Episode
from nes_player.emulator.controller import BUTTONS

FRAME_STACK = 4
INPUT_HW = (112, 120)   # half of the canonical 224×240

# Audio: a log-mel window of about 260 ms ending at the moment of the decision
MEL_HOP = 160
MEL_N = 32
MEL_FRAMES = 52   # 52 × 160 / 32040 is about 260 ms


def mel_transform(sample_rate: int):
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=512, hop_length=MEL_HOP, n_mels=MEL_N)


def episode_mel(audio: np.ndarray, sample_rate: int):
    """The episode's whole audio track to (MEL_N, T) log-mel, plus its statistics."""
    wav = torch.from_numpy(audio.astype(np.float32) / 32768)
    mel = mel_transform(sample_rate)(wav)
    mel = torch.log(mel + 1e-5)
    mean, std = float(mel.mean()), float(mel.std())
    return ((mel - mean) / (std + 1e-6)).numpy(), mean, std


ATTN_HW = (10, 11)   # shape of BCNet's last conv output at INPUT_HW


def episode_attn_masks(ep: Episode) -> np.ndarray:
    """Masks of "objects are here", from the motion tracker; cached on disk.

    These supervise attention. Left alone, behavioural cloning puts only 12.5%
    of its attention inside object boxes against 13.8% for a uniform gaze — it
    keys on background that correlates with the action just as well as the enemy
    does. The boxes are downsampled to the conv feature grid and used as a
    target for the spatial softmax of the activations.
    """
    cache = ep.path / "attn_mask.npy"
    if cache.exists():
        m = np.load(cache)
        if m.shape[1:] == ATTN_HW:
            return m
    from nes_player.perception.motion import MotionTracker

    tracker = MotionTracker()
    frames = ep.frames
    out = np.zeros((frames.shape[0], *ATTN_HW), np.uint8)
    for i in range(frames.shape[0]):
        f = frames[i]
        full = np.zeros(f.shape[:2], np.uint8)
        for s in tracker.update(f, frozenset()):
            x, y, w, h = s.bbox
            full[y : y + h, x : x + w] = 255
        small = cv2.resize(full, (ATTN_HW[1], ATTN_HW[0]), interpolation=cv2.INTER_AREA)
        out[i] = small > 16   # a cell counts once an object covers about 6% of it
    np.save(cache, out)
    return out


def mask_name(mask: int) -> str:
    names = [b for i, b in enumerate(BUTTONS) if mask >> i & 1]
    return "+".join(names) if names else "NOOP"


def mask_to_pressed(mask: int) -> frozenset[str]:
    return frozenset(b for i, b in enumerate(BUTTONS) if mask >> i & 1)


@dataclass
class ActionVocab:
    masks: list[int]   # index -> button bitmask

    @classmethod
    def from_actions(cls, actions: np.ndarray) -> ActionVocab:
        return cls(masks=sorted(int(m) for m in np.unique(actions)))

    @property
    def names(self) -> list[str]:
        return [mask_name(m) for m in self.masks]

    def encode(self, actions: np.ndarray) -> np.ndarray:
        lut = {m: i for i, m in enumerate(self.masks)}
        return np.asarray([lut[int(a)] for a in actions], dtype=np.int64)

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
    """Sample i: frames [i-4..i-1] predict player 0's action at frame i."""

    def __init__(self, small_frames: np.ndarray, labels: np.ndarray, attn=None):
        self.frames = small_frames
        self.labels = labels
        self.attn = attn   # mask of the last visible frame, or None

    def __len__(self) -> int:
        return len(self.frames) - FRAME_STACK

    def __getitem__(self, idx: int):
        i = idx + FRAME_STACK
        x = stack_to_tensor(self.frames[i - FRAME_STACK : i])
        if self.attn is None:
            return x, int(self.labels[i])
        return x, int(self.labels[i]), torch.from_numpy(self.attn[i - 1].astype(np.float32))


class EpisodeBCDatasetAV(EpisodeBCDataset):
    """Adds the log-mel window ending at the start of frame i. The mel and the
    offsets belong to the WHOLE episode; a dataset segment is a slice of the
    offsets, so that windows never straddle an episode boundary."""

    def __init__(self, small_frames, labels, mel: np.ndarray, audio_offsets: np.ndarray,
                 attn=None):
        super().__init__(small_frames, labels, attn)
        self.mel = mel
        self.offsets = audio_offsets

    def __getitem__(self, idx: int):
        x, *rest = super().__getitem__(idx)
        i = idx + FRAME_STACK
        m_end = int(self.offsets[i] // MEL_HOP)
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
) -> dict:
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
    all_actions = np.concatenate([e.actions[:, 0] for e in eps])
    vocab = ActionVocab.from_actions(all_actions)
    sr = eps[0].metadata["sample_rate"]
    mel_stats = None
    stat_acc: list[tuple[float, float]] = []

    def episode_ds(e: Episode):
        small = preprocess_frames(e.frames[:])
        labels = vocab.encode(e.actions[:, 0])
        masks = episode_attn_masks(e) if attn else None
        if not use_audio:
            return EpisodeBCDataset(small, labels, masks), labels
        mel, mm, ms = episode_mel(e.audio[:], sr)   # normalised per episode
        stat_acc.append((mm, ms))
        return EpisodeBCDatasetAV(small, labels, mel, e.audio_offsets, masks), labels

    if len(eps) == 1:   # one episode: validation is its last 10%
        small = preprocess_frames(eps[0].frames[:])
        labels = vocab.encode(eps[0].actions[:, 0])
        masks = episode_attn_masks(eps[0]) if attn else None
        n_val = int(len(small) * val_frac)
        n_train = len(small) - n_val
        mtr = masks[:n_train] if attn else None
        mva = masks[n_train:] if attn else None
        if use_audio:
            mel, mm, ms = episode_mel(eps[0].audio[:], sr)
            stat_acc.append((mm, ms))
            off = eps[0].audio_offsets
            train_ds = EpisodeBCDatasetAV(
                small[:n_train], labels[:n_train], mel, off[: n_train + 1], mtr)
            val_ds = EpisodeBCDatasetAV(
                small[n_train:], labels[n_train:], mel, off[n_train:], mva)
        else:
            train_ds = EpisodeBCDataset(small[:n_train], labels[:n_train], mtr)
            val_ds = EpisodeBCDataset(small[n_train:], labels[n_train:], mva)
        val_labels = labels[n_train:]
        n_val_count = n_val
    else:   # many: validation is the last 10% of episodes, taken whole
        n_val_eps = max(1, int(len(eps) * val_frac))
        parts = [episode_ds(e) for e in eps]
        train_ds = torch.utils.data.ConcatDataset([d for d, _ in parts[:-n_val_eps]])
        val_ds = torch.utils.data.ConcatDataset([d for d, _ in parts[-n_val_eps:]])
        val_labels = np.concatenate([lbl for _, lbl in parts[-n_val_eps:]])
        n_val_count = len(val_labels)
    if use_audio:
        mel_stats = {"mean": float(np.mean([a for a, _ in stat_acc])),
                     "std": float(np.mean([b for _, b in stat_acc])),
                     "sample_rate": sr}
        model = BCNetAV(len(vocab))
    else:
        model = BCNet(len(vocab))
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
        print(rec)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    majority = float(np.bincount(val_labels).max() / n_val_count)
    meta = {
        "vocab_masks": vocab.masks,
        "vocab_names": vocab.names,
        "episode": str(episode_dir),
        "frame_stack": FRAME_STACK,
        "input_hw": list(INPUT_HW),
        "modality": "av" if use_audio else "video",
        "attn": attn,
        "mel_stats": mel_stats,
        "history": history,
        "val_majority_baseline": majority,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class BCPolicy:
    """Inference: keeps the frame stack, returns an action and its distribution."""

    def __init__(self, run_dir: str | Path):
        run = Path(run_dir)
        meta = json.loads((run / "meta.json").read_text())
        self.vocab = ActionVocab(masks=meta["vocab_masks"])
        self.dev = device()
        self.modality = meta.get("modality", "video")
        if self.modality == "av":
            self.model = BCNetAV(len(self.vocab)).to(self.dev).eval()
            st = meta["mel_stats"]
            self._mel_t = mel_transform(st["sample_rate"])
            self._mel_mean, self._mel_std = st["mean"], st["std"]
            self._audio_ring: list[np.ndarray] = []
            self._mel_cache = None
        else:
            self.model = BCNet(len(self.vocab)).to(self.dev).eval()
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

    def act(self, frame_rgb: np.ndarray, temperature: float = 1.0, with_cam: bool = False):
        """Returns (pressed, ranked) and optionally a Grad-CAM map in [0,1]."""
        small = cv2.resize(frame_rgb, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
        self._stack.append(small)
        if len(self._stack) < FRAME_STACK:
            self._stack = [small] * FRAME_STACK
        self._stack = self._stack[-FRAME_STACK:]
        x = stack_to_tensor(np.stack(self._stack)).unsqueeze(0).to(self.dev)

        m = self._current_mel() if self.modality == "av" else None
        cam = None
        if with_cam:
            logits, cam = self._forward_with_cam(x, m)
        else:
            with torch.no_grad():
                logits = (self.model(x, m) if m is not None else self.model(x))[0]

        probs = torch.softmax(logits / temperature, dim=0).cpu().numpy()
        idx = int(np.random.choice(len(probs), p=probs / probs.sum()))
        ranked = sorted(zip(self.vocab.names, probs.tolist(), strict=True),
                        key=lambda t: -t[1])
        pressed = mask_to_pressed(self.vocab.masks[idx])
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
        small = cv2.resize(frame_rgb, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
        stack = (self._stack or [small])[-(FRAME_STACK - 1):] + [small]
        while len(stack) < FRAME_STACK:
            stack.insert(0, stack[0])
        x = stack_to_tensor(np.stack(stack)).unsqueeze(0).to(self.dev)
        m = self._current_mel(cache_ok=True) if self.modality == "av" else None
        _, cam = self._forward_with_cam(x, m)
        return cam
