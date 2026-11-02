"""Hierarchical subgoals: decompose a long goal into horizon-sized steps.

The Metroid class of games has action-result windows of hours. A fixed-horizon
planner cannot see the consequence, and every candidate scores the same.
Hierarchical subgoals break "reach the boss" into room-level steps where each
step's reward fits inside the horizon.

The mechanism is progressive target scheduling: the planner gets a target_px
that advances as intermediate thresholds are reached. Each subgoal's reward is
inside the horizon; the chain spans the hours.

    from nes_player.policy.hierarchical import SubgoalChain
    chain = SubgoalChain(thresholds=[100, 500, 1000, 2000, 5000])
    target = chain.current_target(position=0)    # 100
    chain.update(position=120)                    # reached 100, target -> 500
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubgoalChain:
    """A chain of position thresholds that the planner walks through.

    thresholds: sorted list of progress values. Each is a subgoal.
    When the player's progress reaches a threshold, the chain advances
    and the next subgoal becomes the target.

    The value function rewards reaching the current subgoal within the
    horizon, not the final distant goal. The chain connects the steps.
    """

    thresholds: list[float] = field(default_factory=list)
    _index: int = 0
    _history: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.thresholds = sorted(set(self.thresholds))

    @property
    def current_target(self) -> float:
        """The subgoal the planner should aim at now."""
        if self._index >= len(self.thresholds):
            return float("inf")  # all subgoals reached
        return self.thresholds[self._index]

    @property
    def progress(self) -> float:
        """Fraction of subgoals reached."""
        return self._index / max(1, len(self.thresholds))

    @property
    def done(self) -> bool:
        return self._index >= len(self.thresholds)

    def update(self, position: float) -> bool:
        """Advance the chain if position reached the current subgoal.

        Returns True if the chain advanced.
        """
        if self.done:
            return False
        if position >= self.current_target:
            self._history.append((position, self.current_target))
            self._index += 1
            return True
        return False

    def shaped_value(self, position: float, horizon_px: float) -> float:
        """Shaped reward: distance to the current subgoal, not the final goal.

        For a subgoal within horizon_px, the planner can see the consequence.
        The shape is: negative distance to current target, plus a bonus if
        the target is within horizon.
        """
        if self.done:
            return 0.0
        target = self.current_target
        dist = max(0.0, target - position)
        # If the target is within horizon, the planner can reach it
        if dist <= horizon_px:
            return 1.0 - dist / max(1.0, horizon_px)
        # Otherwise: reward closing the distance (potential-based shaping)
        return -dist / max(1.0, target)

    def remaining(self) -> list[float]:
        return self.thresholds[self._index:]

    def reset(self) -> None:
        self._index = 0
        self._history.clear()


def make_uniform_chain(max_position: float, n_subgoals: int = 8) -> SubgoalChain:
    """Uniform subgoals from 0 to max_position."""
    step = max_position / n_subgoals
    return SubgoalChain(thresholds=[step * (i + 1) for i in range(n_subgoals)])


def make_manual_chain(thresholds: list[float]) -> SubgoalChain:
    """Subgoals from the manual (e.g., room boundaries, boss positions)."""
    return SubgoalChain(thresholds=thresholds)
