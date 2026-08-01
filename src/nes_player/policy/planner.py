"""Model-predictive control on top of the ego world model.

Each time it replans: imagine a few behaviour templates 16 frames ahead through
the ego model, score each one (progress to the right, minus collisions with
objects known to be dangerous), execute the first frames of the best, replan.
Runs on the CPU in microseconds.

Worth knowing before trusting it: on an early, weak world model this planner
lost badly to the plain reactive policy. It only started winning after the
fourth iteration of the model. A planner is exactly as good as what it plans in.
"""

from dataclasses import dataclass

import numpy as np

from nes_player.world_model.ego import (  # noqa: F401 (CROP documents the geometry)
    CROP,
    GhostPredictor,
)

RUN = frozenset({"B", "RIGHT"})
JUMP = frozenset({"A", "B", "RIGHT"})
NOOP: frozenset = frozenset()
LEFT = frozenset({"LEFT"})

# Templates: (name, a 16-step button sequence)
TEMPLATES = [
    ("run", [RUN] * 16),
    ("jump now", [JUMP] * 10 + [RUN] * 6),
    ("jump later", [RUN] * 4 + [JUMP] * 10 + [RUN] * 2),
    ("wait", [NOOP] * 16),
    ("back off", [LEFT] * 8 + [RUN] * 8),
]


@dataclass
class Plan:
    name: str
    pressed: frozenset
    score: float
    traj: list


class EgoPlanner:
    def __init__(self, ghost: GhostPredictor):
        self.ghost = ghost
        self.last: Plan | None = None

    def _mask(self, pressed: frozenset) -> int:
        from nes_player.emulator.controller import BUTTONS

        return sum(1 << k for k, b in enumerate(BUTTONS) if b in pressed)

    def plan(self, frame_rgb, ctrl, danger: list, screen_w: int = 240) -> Plan:
        """ctrl is the hero's slot; danger is [(cx, cy, vx, vy)] for each threat."""
        import torch

        best: Plan | None = None
        with torch.no_grad():
            for name, seq in TEMPLATES:
                # Roll the template out: the same steps as GhostPredictor.predict,
                # but with the action changing at each one.
                m = self.ghost.model
                feat = m.enc(torch.from_numpy(
                    self.ghost_crop(frame_rgb, ctrl.cx, ctrl.cy)).float()
                    .div_(255).permute(2, 0, 1).unsqueeze(0))
                h = torch.zeros(1, 128)
                v = torch.tensor([[ctrl.vx, ctrl.vy]], dtype=torch.float32)
                x, y = ctrl.cx, ctrl.cy
                traj = []
                for pressed in seq:
                    aid_mask = self._mask(pressed)
                    try:
                        aid = self.ghost.vocab.masks.index(aid_mask)
                    except ValueError:
                        aid = 0
                    h, pred = m.forward_step(h, feat, v, torch.tensor([aid]))
                    x, y = x + float(pred[0, 0]), y + float(pred[0, 1])
                    traj.append((x, y))
                    v = pred
                score = self._score(traj, ctrl, danger, screen_w)
                p = Plan(name, seq[0], score, traj)
                if best is None or p.score > best.score:
                    best = p
        self.last = best
        return best

    @staticmethod
    def ghost_crop(frame, cx, cy):
        from nes_player.world_model.ego import _crop

        return _crop(frame, cx, cy)

    @staticmethod
    def _score(traj, ctrl, danger, screen_w) -> float:
        dx_total = traj[-1][0] - ctrl.cx   # progress to the right
        collision = 0.0
        for k, (px, py) in enumerate(traj):
            for (ex, ey, evx, evy) in danger:
                # threats are extrapolated linearly
                d = np.hypot(px - (ex + evx * k), py - (ey + evy * k))
                if d < 16:
                    collision += (16 - d) * (1.0 - k / len(traj))   # sooner is scarier
        off_screen = max(0.0, traj[-1][0] - (screen_w - 8)) + max(0.0, 8 - traj[-1][0])
        return float(dx_total - 6.0 * collision - 2.0 * off_screen)
