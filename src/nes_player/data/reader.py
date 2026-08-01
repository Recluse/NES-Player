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
