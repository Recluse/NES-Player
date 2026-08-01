"""Emulator adapter interface (spec §10.1)."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from nes_player.emulator.controller import ControllerState


@dataclass
class EmulatorObservation:
    frame_rgb: np.ndarray  # (H, W, 3) uint8
    audio_pcm: np.ndarray  # (n_samples,) int16 mono
    frame_index: int
    sample_rate: int
    done_hint: bool
    debug: dict[str, Any] | None  # never fed to the policy


class EmulatorAdapter(Protocol):
    def reset(self, seed: int | None = None) -> EmulatorObservation: ...
    def step(self, action: ControllerState) -> EmulatorObservation: ...
    def save_state(self) -> bytes: ...
    def load_state(self, state: bytes) -> None: ...
    def close(self) -> None: ...
