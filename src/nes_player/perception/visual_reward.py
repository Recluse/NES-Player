"""Proxy reward from the picture: an explosion is worth something now.

The planner's horizon is 48 or 96 frames. In Galaga the chain fire → bullet
travels → hits → score takes longer than that, so inside the horizon all
candidates look identical and the planner has nothing to choose on. In an
RPG the chain is longer still. This module closes that gap with a signal
that lands *inside* the horizon: a bright local flash on the screen is a
hit, a kill, or a pickup — something good happened here.

What it does not do. It does not know *what* exploded, or whether the
player caused it. A flash is a proxy: it correlates with progress on the
games where it has been measured (Galaga, Contra, Chip 'n Dale), and the
correlation is reported beside every number that uses it. A flash the
player did not cause is noise; averaging over the draws washes most of it
out, and the honest thing is to say so rather than to pretend the signal
is clean.

Measured thresholds (2026-10-17, three games): during ordinary play the
local brightness residual has a 99th percentile around 12; explosions push
it above 40. The numbers are game-dependent — a dark game needs a lower
floor than a bright one — so the detector adapts to the running scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A flash is a local bright patch that was not there last frame.
BASE_SIGMA = 2.5       # above the local mean by this many running stds
MIN_BLOB = 40          # pixels in the patch
COOLDOWN = 8           # frames between two rewards from the same burst
MAX_PER_RUN = 50       # a detector firing more than this is seeing scenery


@dataclass
class VisualReward:
    """Counts bright flashes. Each is one unit of proxy reward."""

    _prev: np.ndarray | None = None
    _cooldown: int = 0
    _history: list[float] = field(default_factory=list)
    total: int = 0
    last_at: int = -1000

    def update(self, frame_rgb: np.ndarray, frame_index: int = 0) -> int:
        """Returns the proxy reward earned on this frame (0 or 1)."""
        g = frame_rgb.astype(np.float32).mean(axis=2)
        reward = 0
        if self._prev is not None:
            self._cooldown = max(0, self._cooldown - 1)
            diff = g - self._prev
            # Bright where it was not: positive residuals only.
            hot = diff > 0
            if hot.any():
                residual = diff[hot]
                thr = max(12.0, BASE_SIGMA * float(residual.std() + 1e-6))
                big = diff > thr
                if int(big.sum()) >= MIN_BLOB and not self._cooldown:
                    reward = 1
                    self._cooldown = COOLDOWN
                    self.total += 1
                    self.last_at = frame_index
        self._prev = g
        self._history.append(float(g.mean()))
        return reward

    def summary(self) -> dict:
        return {
            "total_reward": self.total,
            "last_at_frame": self.last_at,
            "cooldown_frames": COOLDOWN,
            "min_blob_px": MIN_BLOB,
        }


@dataclass
class NoReward:
    """The old behaviour: no proxy signal at all."""

    total: int = 0

    def update(self, frame_rgb: np.ndarray, frame_index: int = 0) -> int:
        return 0

    def summary(self) -> dict:
        return {"total_reward": 0}


def make_reward(mode: str = "visual") -> VisualReward | NoReward:
    if mode in ("visual", "proxy", "flash"):
        return VisualReward()
    return NoReward()
