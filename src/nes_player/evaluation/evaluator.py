"""One deterministic, frame-indexed way to make an agent play.

Everything measured in this project used to go through the play loop, where a
background thread decided on a wall clock at 15 Hz while the emulator ran as
fast as it could. Two consequences, both of which invalidate comparisons rather
than merely blur them:

- The frame stack advanced once per DECISION, not once per frame. So `long`,
  documented as reaching 128 frames back — 2.1 seconds — actually reached 128
  decisions back, which in realtime is about 8.5 seconds and headless is
  whatever the machine's load made it. The memory presets did not measure the
  windows their names claim, and the same checkpoint fed different inputs on
  different machines.
- The planner replanned every `repeat` EMULATOR frames while the policy decided
  on the clock, so "BC versus BC plus planner" compared two schedules as well
  as two agents.

Here observation happens on every frame, decisions happen on a fixed grid of
frame indices, and the action is held between them. Realtime is a sleep at the
end of the loop and changes nothing about the result. The decision indices are
returned with the result so that two arms can be shown to have had the same
opportunities, rather than assumed to have.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FRAME_SECONDS = 1 / 60.098814      # NTSC NES


@dataclass
class RunResult:
    frames: int
    actions: list[frozenset[str]]          # what was held on each frame
    decision_frames: list[int]             # where a new action was chosen
    decisions: list[frozenset[str]]        # what was chosen there
    sources: list[str]                     # "policy" or whoever overrode it
    metrics: dict[str, Any] = field(default_factory=dict)

    def trace(self) -> list[tuple[int, tuple[str, ...]]]:
        """A comparable summary: (frame, buttons) at every decision."""
        return [(f, tuple(sorted(a)))
                for f, a in zip(self.decision_frames, self.decisions, strict=True)]


def evaluate(
    env,
    policy,
    frames: int = 3000,
    *,
    action_repeat: int = 4,
    temperature: float = 1.0,
    seed: int = 0,
    idle_start: int = 0,
    realtime: bool = False,
    override: Callable[[int, Any, frozenset[str]], tuple[frozenset[str], str] | None]
    | None = None,
    on_frame: Callable[[int, Any, frozenset[str]], None] | None = None,
    strip: Iterable[str] = ("START", "SELECT"),
) -> RunResult:
    """Play one episode synchronously and report exactly what happened.

    `override` is how a planner joins in: it is offered the same decision ticks
    as the policy and nothing else, so its contribution is measurable on its own
    rather than tangled with a different cadence. `on_frame` is for metrics —
    it is called after the step and cannot change the action.
    """
    obs = env.reset(seed=seed)
    for _ in range(idle_start):
        obs = env.step_buttons([frozenset()])

    if hasattr(policy, "reset"):
        policy.reset()
    strip = set(strip)
    held: frozenset[str] = frozenset()
    result = RunResult(frames=0, actions=[], decision_frames=[], decisions=[],
                       sources=[])

    for i in range(frames):
        t0 = time.monotonic()
        policy.observe(obs.frame_rgb, getattr(obs, "audio_pcm", None))
        if i % action_repeat == 0:
            held, _ = policy.decide(temperature=temperature)
            source = "policy"
            if override is not None:
                alt = override(i, obs, held)
                if alt is not None:
                    held, source = alt
            result.decision_frames.append(i)
            result.decisions.append(held)
            result.sources.append(source)
        action = held - strip
        result.actions.append(action)
        obs = env.step_buttons([action])
        if on_frame is not None:
            on_frame(i, obs, action)
        if realtime:
            # The only thing a clock is allowed to do here: pace the display.
            time.sleep(max(0.0, FRAME_SECONDS - (time.monotonic() - t0)))

    result.frames = frames
    return result


class InstinctAdapter:
    """`observe`/`decide` around the instinct policy, so it runs in the same harness.

    The instincts think on every frame by design — the tracker needs every one
    — so `decide` returns what the last `observe` already worked out. Holding it
    for `action_repeat` frames is then the evaluator's business, not theirs.
    """

    def __init__(self, policy, feedback, wants_ram: bool = False, env=None):
        self.policy, self.feedback = policy, feedback
        self.wants_ram, self.env = wants_ram, env
        self._pressed: frozenset[str] = frozenset()
        self.slots: list = []
        self.verdicts: dict = {}

    def reset(self) -> None:
        self.policy.reset()

    def observe(self, frame_rgb: np.ndarray, audio_pcm=None) -> None:
        ram = self.env._env.get_ram() if self.wants_ram else None
        fb = self.feedback.update(frame_rgb, None)
        self._pressed, self.slots, self.verdicts = self.policy.step(
            frame_rgb, fb.score, fb.died, ram)

    def decide(self, temperature: float = 1.0):
        return self._pressed, []
