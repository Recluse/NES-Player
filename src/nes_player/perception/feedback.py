"""Where the agent learns that something good or bad just happened.

The instinct policy needs two facts to build its object memory: the score went
up, and we died. From those it labels objects `reward` and `danger`, and those
labels choose buttons. So this is not telemetry — it is an **input**, and where
it comes from decides whether the claim "pixels and sound only" is true.

It used to come from emulator memory. Measured: replaying identical frames with
the numbers scrambled changed 51% of the actions, first divergence at frame 832.
The claim was false, and the datasets recorded by that policy contain
RAM-conditioned actions.

Three sources now, named rather than implied:

- `NoFeedback` — nothing. The default, and the only one that makes "pixels and
  sound only" a true statement today. The object memory still clusters what it
  sees; it just never learns that a particular object killed us, so `danger`
  and `reward` never appear.
- `PrivilegedFeedback` — reads the emulator. Legitimate for a teacher that is
  meant to see more than its student, and for measurement. Never the default.
- `PixelFeedback` — reads the panel on the screen, the way a person does. This
  is where the honest signal should come from, and it does not work well enough
  yet to be the default. Measured against the memory on 4000 frames:

      Double Dragon   score 2 vs 178      agreement 12%   deaths 0 vs 0
      Super Mario     score 0 vs 10       agreement 26%   deaths 0 vs 3

  On Double Dragon the digit reader locks onto the health bar and returns
  eighteen groups that all read as 9. On Mario the number of lives is not drawn
  during play at all, so a death is not visible on the panel by construction —
  it has to come from the picture instead.

All three produce the same small record, so the policy cannot tell which it has
— which is the point: the fence is at the boundary, not scattered through the
decision code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from nes_player.perception.text import HudReader

FIT_FRAMES = 240       # frames the digit reader needs before it can read
LIVES_MAX = 9          # a group of digits this small is a counter, not a score
UNREADABLE = -1        # HudReader's marker for a glyph it does not know


class Feedback(NamedTuple):
    score: int         # accumulated, from the start of the episode
    died: bool         # a life was lost since the previous frame


@dataclass
class NoFeedback:
    """No consequences reported at all. What "pixels only" costs, honestly."""

    privileged = False

    def update(self, frame_rgb: np.ndarray, debug: dict | None = None) -> Feedback:
        return Feedback(0, False)


@dataclass
class PrivilegedFeedback:
    """From emulator memory. Exact, and a cheat — say so out loud."""

    privileged = True
    _score0: int | None = None
    _lives: int | None = None
    _frame: int = 0

    def update(self, frame_rgb: np.ndarray, debug: dict | None) -> Feedback:
        d = debug or {}
        self._frame += 1
        if self._score0 is None and self._frame > 200:
            self._score0 = d.get("score", 0)
        score = max(0, d.get("score", 0) - self._score0) if self._score0 is not None else 0
        died = False
        lives = d.get("lives")
        if lives is not None and lives <= 90:      # above that it is uninitialised
            died = self._lives is not None and lives < self._lives
            self._lives = lives
        return Feedback(score, died)


@dataclass
class PixelFeedback:
    """From the panel on the screen, like a person reading it.

    Which number on the panel is the score is not something we are told, so it
    is worked out from behaviour: the score is the group of digits that only
    ever goes up. A timer counts down, lives change rarely and by one, and a
    level number is small. Only "never decreases, and has increased at least
    once" fits a score, and it fits it in any game.

    A death is a small counter that drops by exactly one. That is what a lives
    display does, and reading it off the screen is what a person does too.
    """

    privileged = False
    hud: HudReader = field(default_factory=HudReader)
    _buf: list = field(default_factory=list)
    _prev: list[int] = field(default_factory=list)
    _drops: list[int] = field(default_factory=list)   # per group: times it fell
    _rises: list[int] = field(default_factory=list)   # per group: times it rose
    _score_group: int | None = None
    _score0: int | None = None
    _score: int = 0

    def update(self, frame_rgb: np.ndarray, debug: dict | None = None) -> Feedback:
        if not self.hud.groups:
            # Collect first, fit once. Fitting needs to see the digits change,
            # which is why it cannot be done from a single frame.
            self._buf.append(frame_rgb.copy())
            if len(self._buf) >= FIT_FRAMES:
                self.hud.fit(self._buf)
                self._buf.clear()
            return Feedback(self._score, False)

        values = self.hud.read(frame_rgb)
        if len(values) != len(self._prev):
            self._prev = values
            self._drops = [0] * len(values)
            self._rises = [0] * len(values)
            return Feedback(self._score, False)

        died = False
        for i, (now, before) in enumerate(zip(values, self._prev, strict=True)):
            if now == UNREADABLE or before == UNREADABLE:
                continue
            if now > before:
                self._rises[i] += 1
            elif now < before:
                self._drops[i] += 1
                if before <= LIVES_MAX and before - now == 1:
                    died = True
        self._prev = values

        if self._score_group is None:
            # The first group that has gone up and never come down.
            for i in range(len(values)):
                if self._rises[i] > 0 and self._drops[i] == 0:
                    self._score_group = i
                    break
        if self._score_group is not None:
            v = values[self._score_group]
            if v != UNREADABLE:
                if self._score0 is None:
                    self._score0 = v
                # Never let it go backwards: one misread digit would otherwise
                # look like the score collapsing, and a collapse would be
                # mislabelled as "that object cost us points".
                self._score = max(self._score, v - self._score0)
        return Feedback(self._score, died)


SOURCES = {"strict": NoFeedback, "privileged": PrivilegedFeedback,
           "pixel": PixelFeedback}


def make_feedback(mode: str = "strict"):
    """`strict` unless something explicitly asks to see more."""
    if mode not in SOURCES:
        raise ValueError(f"unknown feedback mode {mode!r}; "
                         f"known: {', '.join(sorted(SOURCES))}")
    return SOURCES[mode]()
