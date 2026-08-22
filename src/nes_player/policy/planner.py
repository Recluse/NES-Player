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
    SEQ,
    GhostPredictor,
)

RUN = frozenset({"B", "RIGHT"})
JUMP = frozenset({"A", "B", "RIGHT"})
NOOP: frozenset = frozenset()
LEFT = frozenset({"LEFT"})

# Templates: (name, a SEQ-step button sequence). The horizon is the model's,
# because a plan longer than the model was trained to imagine is extrapolation.
# The jump lengths are physical rather than scaled: A held longer than about
# ten frames only raises the arc, it does not lengthen the plan.
H = SEQ
TEMPLATES = [
    ("run", [RUN] * H),
    ("jump now", [JUMP] * 10 + [RUN] * (H - 10)),
    ("jump later", [RUN] * 12 + [JUMP] * 10 + [RUN] * (H - 22)),
    ("wait", [NOOP] * H),
    ("back off", [LEFT] * 12 + [RUN] * (H - 12)),
]


# How many frames of a chosen plan are executed before it is reconsidered.
# Replanning at every tick and emitting only the first press means a jump is
# never made: clearing a pipe needs A held for ten to sixteen frames, so
# "jump now" has to win three or four elections in a row, each judged from a
# mid-air crop unlike anything the ranking was measured on. Two of the five
# templates also start with the same press, so at one frame of commitment the
# menu is four distinct buttons rather than five.
COMMIT = 16


@dataclass
class Plan:
    name: str
    pressed: frozenset
    score: float
    traj: list
    seq: list | None = None


class EgoPlanner:
    def __init__(self, ghost: GhostPredictor):
        self.ghost = ghost
        self.last: Plan | None = None
        self._at = COMMIT     # frames consumed of the committed plan

    def step(self, frame_rgb, ctrl, danger: list, screen_w: int = 240,
             scroll_dx: float = 0.0, repeat: int = 4) -> Plan:
        """The plan to act on now, replanning only when the commitment runs out.

        `repeat` is how many frames the caller will hold the returned press for,
        which is how far along the committed sequence this call moves.
        """
        if self.last is None or self._at >= COMMIT:
            self.plan(frame_rgb, ctrl, danger, screen_w, scroll_dx)
            self._at = 0
        held = self.last.seq[min(self._at, len(self.last.seq) - 1)]
        self._at += repeat
        return Plan(self.last.name, held, self.last.score, self.last.traj,
                    self.last.seq)

    def _mask(self, pressed: frozenset) -> int:
        from nes_player.emulator.controller import BUTTONS

        return sum(1 << k for k, b in enumerate(BUTTONS) if b in pressed)

    def plan(self, frame_rgb, ctrl, danger: list, screen_w: int = 240,
             scroll_dx: float = 0.0) -> Plan:
        """ctrl is the hero's slot; danger is [(cx, cy, vx, vy, risk)] per object.

        `scroll_dx` is the camera's motion this frame, needed to turn a
        threat's screen velocity into its motion through the level — the
        same correction the hero's own prediction already carries.
        """
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
                score = self._score(traj, ctrl, danger, screen_w, scroll_dx)
                p = Plan(name, seq[0], score, traj, seq)
                if best is None or p.score > best.score:
                    best = p
        self.last = best
        return best

    @staticmethod
    def ghost_crop(frame, cx, cy):
        from nes_player.world_model.ego import _crop

        return _crop(frame, cx, cy)

    @staticmethod
    def _score(traj, ctrl, danger, screen_w, scroll_dx: float = 0.0) -> float:
        """Progress through the level, minus running into things.

        Everything here is in the hero's own frame: how far he gets, and where
        threats end up relative to him. Absolute screen positions cannot be
        used because the camera moves — the previous version scored
        `traj[-1][0] - ctrl.cx`, the change in *screen* x, and the camera holds
        the hero near the middle however fast he runs. Measured, that made
        every template score negative while the camera scrolled and ranked
        running below jumping, so the planner jumped 200 times against 38 runs.
        It was picking the least-negative number in a coordinate system that
        had stopped meaning anything.
        """
        # The rollout starts at ctrl.cx and adds predicted deltas, so this is
        # their sum — displacement through the level, now that the model
        # predicts world rather than screen movement.
        dx_total = traj[-1][0] - ctrl.cx
        collision = 0.0
        n = len(traj)
        for k, (px, py) in enumerate(traj):
            # The hero's displacement from where he started, in world terms.
            hx = px - ctrl.cx
            for (ex, ey, evx, evy, risk) in danger:
                # A threat's own motion through the level is its screen motion
                # less the camera's, the same correction applied to the hero.
                rel_x = (ex - ctrl.cx) + (evx - scroll_dx) * k - hx
                rel_y = (ey - ctrl.cy) + evy * k - (py - ctrl.cy)
                d = np.hypot(rel_x, rel_y)
                if d < 16:
                    # Weighted by how often this thing has killed us, rather
                    # than by a verdict that never fires: a Goomba's tally is
                    # 5 deaths in 133 contacts, and 0.038 of a collision is a
                    # truer statement about it than either "danger" or "safe".
                    collision += risk * (16 - d) * (1.0 - k / n)   # sooner is scarier
        # The off-screen penalty is gone with the screen frame it belonged to:
        # the camera follows the hero, so he cannot walk off the edge, and the
        # term only ever punished him for making progress.
        return float(dx_total - 6.0 * collision)
