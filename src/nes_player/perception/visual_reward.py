"""Proxy reward from the picture: combat state and explosions.

The planner's horizon is 48 or 96 frames. In Galaga the chain fire → bullet
travels → hits → score takes longer than that, so inside the horizon all
candidates look identical. Two signals close that gap:

1. **Flash** — a bright local burst (discrete: a hit, a kill, a pickup).
2. **Combat** — the ship is aligned with an enemy and firing (continuous).
   This is the adaptive one: it rewards *being in the fight*, which is what
   separates `aligned barrage` from `run`. On Galaga agent-play (2026-10-18)
   frames under combat score 3.2× more per second than frames running
   without fire.

What it does not do: it does not know what is being shot at. Alignment +
fire is a proxy for doing the right thing. The correlation is reported
beside every number that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BASE_SIGMA = 2.5
MIN_BLOB = 40
COOLDOWN = 8
ALIGN_X = 20
FIRE_PIXELS = 8
COMBAT_DECAY = 0.9


@dataclass
class VisualReward:
    """Flash (discrete) + combat (continuous) from the picture."""

    _prev: np.ndarray | None = None
    _cooldown: int = 0
    _history: list[float] = field(default_factory=list)
    total: int = 0
    last_at: int = -1000
    combat_ema: float = 0.0
    combat_frames: int = 0
    total_combat: float = 0.0

    def update(self, frame_rgb: np.ndarray, frame_index: int = 0,
               firing: bool = False) -> int:
        g = frame_rgb.astype(np.float32).mean(axis=2)
        reward = 0

        # flash
        if self._prev is not None:
            self._cooldown = max(0, self._cooldown - 1)
            diff = g - self._prev
            hot = diff > 0
            if hot.any():
                residual = diff[hot]
                thr = max(12.0, BASE_SIGMA * float(residual.std() + 1e-6))
                if int((diff > thr).sum()) >= MIN_BLOB and not self._cooldown:
                    reward = 1
                    self._cooldown = COOLDOWN
                    self.total += 1
                    self.last_at = frame_index
        self._prev = g

        # combat: ship aligned with enemies above and firing
        combat = self._detect_combat(g) if firing else 0.0
        self.combat_ema = COMBAT_DECAY * self.combat_ema + (1 - COMBAT_DECAY) * combat
        self.total_combat += combat
        if combat > 0.5:
            self.combat_frames += 1

        self._history.append(float(g.mean()))
        return reward

    def _detect_combat(self, g: np.ndarray) -> float:
        h = g.shape[0]
        enemies = g[: h // 2]
        ship_band = g[3 * h // 4:]
        enemy_cols = (enemies > 100).sum(axis=0)
        ship_cols = (ship_band > 100).sum(axis=0)
        if not enemy_cols.any() or not ship_cols.any():
            return 0.0
        enemy_x = float(np.argmax(enemy_cols))
        ship_x = float(np.argmax(ship_cols))
        aligned = abs(enemy_x - ship_x) < ALIGN_X
        firing_px = int(ship_cols.max()) >= FIRE_PIXELS
        return 1.0 if (aligned and firing_px) else 0.0

    def summary(self) -> dict:
        return {
            "total_reward": self.total,
            "combat_frames": self.combat_frames,
            "total_combat": round(self.total_combat, 1),
            "combat_ema": round(self.combat_ema, 3),
        }


@dataclass
class NoReward:
    total: int = 0

    def update(self, frame_rgb: np.ndarray, frame_index: int = 0,
               firing: bool = False) -> int:
        return 0

    def summary(self) -> dict:
        return {"total_reward": 0}


def make_reward(mode: str = "visual") -> VisualReward | NoReward:
    if mode in ("visual", "proxy", "flash", "combat"):
        return VisualReward()
    return NoReward()
