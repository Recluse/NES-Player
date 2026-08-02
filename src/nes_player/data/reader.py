"""Reading an episode written by EpisodeWriter."""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import zarr


@dataclass
class Episode:
    path: Path

    @cached_property
    def metadata(self) -> dict[str, Any]:
        return json.loads((self.path / "metadata.json").read_text())

    @cached_property
    def _root(self) -> zarr.Group:
        return zarr.open_group(str(self.path / "episode.zarr"), mode="r")

    @property
    def frames(self) -> zarr.Array:
        return self._root["frames"]

    @property
    def audio(self) -> zarr.Array:
        return self._root["audio"]

    @cached_property
    def actions(self) -> np.ndarray:
        return np.load(self.path / "actions.npy")

    @cached_property
    def audio_offsets(self) -> np.ndarray:
        return np.load(self.path / "audio_offsets.npy")

    def __len__(self) -> int:
        return int(self.metadata["frames"])

    def frame_audio(self, i: int) -> np.ndarray:
        return self.audio[self.audio_offsets[i] : self.audio_offsets[i + 1]]

    def validate(self) -> None:
        """Refuse a damaged episode here, naming the path and the invariant.

        Without this a truncated directory fails much later, inside a DataLoader
        worker, as an index error with no idea which episode caused it. Episodes
        written before the schema existed have no `complete` marker and are
        checked on their contents alone — the invariants are what matter, the
        marker only makes the check cheap.
        """
        n = len(self)
        checks = [
            (self.frames.shape[0] == n, f"frames {self.frames.shape[0]} != {n}"),
            (self.actions.shape[0] == n, f"actions {self.actions.shape[0]} != {n}"),
            (len(self.audio_offsets) == n + 1,
             f"audio offsets {len(self.audio_offsets)} != {n + 1}"),
            (self.frames.ndim == 4 and self.frames.shape[3] == 3,
             f"frames have shape {self.frames.shape}, expected (N, H, W, 3)"),
        ]
        off = self.audio_offsets
        if len(off):
            checks += [
                (off[0] == 0, f"audio offsets start at {off[0]}, not 0"),
                (bool(np.all(np.diff(off) >= 0)), "audio offsets go backwards"),
                (off[-1] == self.audio.shape[0],
                 f"last audio offset {off[-1]} != audio length {self.audio.shape[0]}"),
            ]
        broken = [why for ok, why in checks if not ok]
        if broken:
            raise ValueError(f"{self.path} is not a complete episode: "
                             + "; ".join(broken))
