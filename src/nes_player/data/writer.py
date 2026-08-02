"""Writing an episode to disk (spec §10.5).

episode/
├── metadata.json      game, ROM sha256, source, sample rate, versions
├── frames.zarr        frames (N, H, W, 3) uint8 zstd; audio (S,) int16 zstd
├── audio_offsets.npy  (N+1,) int64: frame i owns audio[off[i]:off[i+1]]
├── actions.npy        (N, players) uint8, a BUTTONS bitmask
└── preview.mp4        optional, written by the viewer
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from nes_player.emulator.adapter import EmulatorObservation
from nes_player.emulator.controller import BUTTONS

FRAME_CHUNK = 256
SCHEMA_VERSION = 1


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

    @property
    def _staging(self) -> Path:
        """Where frames land while the episode is still being written.

        The final directory used to be created on the first frame, while
        actions, offsets and metadata were only written by `close()`. A quit
        from the viewer returns before `close()`, and so does an exception —
        leaving a directory that looks like an episode, is missing its labels
        and its last buffered frames, and is picked up by the next training run
        as if it were fine. Staging plus an atomic rename means a directory
        either does not exist or is complete.
        """
        return self.out_dir.with_name(self.out_dir.name + ".partial")

    def append(
        self, obs: EmulatorObservation, pressed_per_player: tuple[frozenset[str], ...]
    ) -> None:
        if self._frames is None:
            h, w, c = obs.frame_rgb.shape
            if self._staging.exists():
                shutil.rmtree(self._staging)     # an earlier attempt that died
            self._staging.mkdir(parents=True, exist_ok=True)
            root = zarr.open_group(str(self._staging / "episode.zarr"), mode="w")
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
        """Finish the episode and publish it under its real name.

        Safe to call twice, and safe to call from a `finally` — a half-written
        episode is discarded rather than published.
        """
        if self._frames is None:
            self.abandon()
            raise ValueError("empty episode")
        self._flush()
        np.save(self._staging / "actions.npy", np.asarray(self._actions, dtype=np.uint8))
        np.save(self._staging / "audio_offsets.npy",
                np.asarray(self._audio_offsets, dtype=np.int64))
        meta = dict(self.metadata, frames=self._n, buttons=list(BUTTONS),
                    schema=SCHEMA_VERSION, complete=True)
        (self._staging / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self._staging.rename(self.out_dir)   # atomic: same filesystem, one call
        self._frames = None

    def abandon(self) -> None:
        """Throw away what was written. For the error path."""
        self._frames = None
        if self._staging.exists():
            shutil.rmtree(self._staging, ignore_errors=True)

    def __enter__(self) -> EpisodeWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None or self._n == 0:
            self.abandon()
        else:
            self.close()
