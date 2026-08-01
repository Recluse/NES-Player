"""Writing an episode to disk (spec §10.5).

episode/
├── metadata.json      game, ROM sha256, source, sample rate, versions
├── frames.zarr        frames (N, H, W, 3) uint8 zstd; audio (S,) int16 zstd
├── audio_offsets.npy  (N+1,) int64: frame i owns audio[off[i]:off[i+1]]
├── actions.npy        (N, players) uint8, a BUTTONS bitmask
└── preview.mp4        optional, written by the viewer
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from nes_player.emulator.adapter import EmulatorObservation
from nes_player.emulator.controller import BUTTONS

FRAME_CHUNK = 256


def buttons_mask(pressed: frozenset[str]) -> int:
    mask = 0
    for i, name in enumerate(BUTTONS):
        if name in pressed:
            mask |= 1 << i
    return mask


@dataclass
class EpisodeWriter:
    out_dir: Path
    metadata: dict[str, Any]
    _frames: Any = None
    _audio: Any = None
    _frame_buf: list[np.ndarray] = field(default_factory=list)
    _audio_buf: list[np.ndarray] = field(default_factory=list)
    _audio_offsets: list[int] = field(default_factory=lambda: [0])
    _actions: list[list[int]] = field(default_factory=list)
    _n: int = 0

    def append(
        self, obs: EmulatorObservation, pressed_per_player: tuple[frozenset[str], ...]
    ) -> None:
        if self._frames is None:
            h, w, c = obs.frame_rgb.shape
            self.out_dir.mkdir(parents=True, exist_ok=True)
            root = zarr.open_group(str(self.out_dir / "episode.zarr"), mode="w")
            self._frames = root.create_array(
                "frames", shape=(0, h, w, c), chunks=(FRAME_CHUNK, h, w, c),
                dtype="uint8", compressors=zarr.codecs.ZstdCodec(level=3))
            self._audio = root.create_array(
                "audio", shape=(0,), chunks=(obs.sample_rate * 60,), dtype="int16",
                compressors=zarr.codecs.ZstdCodec(level=3))
        self._frame_buf.append(obs.frame_rgb)
        self._audio_buf.append(obs.audio_pcm)
        self._audio_offsets.append(self._audio_offsets[-1] + len(obs.audio_pcm))
        self._actions.append([buttons_mask(p) for p in pressed_per_player])
        self._n += 1
        if len(self._frame_buf) >= FRAME_CHUNK:
            self._flush()

    def _flush(self) -> None:
        if self._frame_buf:
            self._frames.append(np.stack(self._frame_buf))
            self._audio.append(np.concatenate(self._audio_buf))
            self._frame_buf.clear()
            self._audio_buf.clear()

    def close(self) -> None:
        if self._frames is None:
            raise ValueError("empty episode")
        self._flush()
        np.save(self.out_dir / "actions.npy", np.asarray(self._actions, dtype=np.uint8))
        np.save(
            self.out_dir / "audio_offsets.npy", np.asarray(self._audio_offsets, dtype=np.int64)
        )
        meta = dict(self.metadata, frames=self._n, buttons=list(BUTTONS))
        (self.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
