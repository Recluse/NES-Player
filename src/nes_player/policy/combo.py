"""Small abilities, switchable one by one, so they can be combined and measured.

None of these wins on its own. That is the point: an agent that clears a level
does not do it by one large correct idea, it does it by a dozen small ones that
each look like noise in isolation. Deleting a part because it measured -60 over
six seeds throws away a brick; keeping it silently on throws away the ability to
tell what is doing what. So each is a named switch, off by default, and there is
a sweep that measures combinations rather than single changes.

What is here and what it measured alone, against the plain network:

    pit_jump      a hole is coming, so jump          -60  (2W/4L)
    stuck_help    hand over when nothing moves      -274  (2W/4L)

`stuck_help` looks the weakest and has the clearest reason to be weak: the agent
is not stuck when it dies, so the rule fires late or not at all. It stays
because "rarely applicable" is not the same as "wrong", and on a game with real
dead ends it may be the one that matters.

Everything observes every frame whichever is steering. The instinct tracker
needs unbroken frames to keep object identities, and the network's frame stack
is only meaningful if it is continuous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nes_player.perception.feedback import make_feedback
from nes_player.perception.motion import pick_hero
from nes_player.perception.terrain import gap_ahead
from nes_player.policy.bc import BCPolicy
from nes_player.policy.instinct import PIT_JUMP_AT, PIT_LOOKAHEAD, InstinctPolicy

JUMP_FRAMES = 26      # frames an imposed jump keeps the direction it was given


@dataclass
class Abilities:
    """Which small helpers are switched on. All off is the plain network."""

    pit_jump: bool = False
    stuck_help: bool = False

    @classmethod
    def parse(cls, spec: str) -> Abilities:
        """"pit_jump+stuck_help", or "none"."""
        if spec in ("", "none"):
            return cls()
        names = {f.strip() for f in spec.split("+") if f.strip()}
        known = set(cls().__dict__)
        unknown = names - known
        if unknown:
            raise ValueError(f"unknown abilities {sorted(unknown)}; "
                             f"known: {sorted(known)}")
        return cls(**{n: True for n in names})

    def __str__(self) -> str:
        on = [k for k, v in self.__dict__.items() if v]
        return "+".join(on) if on else "none"


class ComboPolicy:
    """The network, plus whichever helpers are switched on."""

    def __init__(self, checkpoint: str | Path, game: str, abilities: Abilities,
                 knowledge_path: str | Path | None = None,
                 feedback: str = "visual") -> None:
        self.bc = BCPolicy(checkpoint)
        self.abilities = abilities
        self.needs_instinct = abilities.stuck_help
        self.instinct = InstinctPolicy(
            knowledge_path=knowledge_path or f"runs/knowledge/{game}.json")
        self.feedback = make_feedback(feedback)
        self.offsets = self.bc.offsets
        self.vocab = self.bc.vocab
        self.reset()

    def reset(self) -> None:
        self.bc.reset()
        self.instinct.reset()
        self._instinct_pressed: frozenset[str] = frozenset()
        self._pit: int | None = None
        self._jump_left = 0
        self.fired = dict.fromkeys(self.abilities.__dict__, 0)

    def push_audio(self, pcm: np.ndarray) -> None:
        self.bc.push_audio(pcm)

    def observe(self, frame_rgb: np.ndarray, audio_pcm: np.ndarray | None = None) -> None:
        self.bc.observe(frame_rgb, audio_pcm)
        fb = self.feedback.update(frame_rgb)
        self._instinct_pressed, slots, _ = self.instinct.step(
            frame_rgb, fb.score, fb.died)
        self._pit = None
        if self.abilities.pit_jump:
            hero = pick_hero(slots)
            if hero is not None:
                d = gap_ahead(frame_rgb, hero.cx, 1, PIT_LOOKAHEAD)
                if d is not None and d <= PIT_JUMP_AT:
                    self._pit = d

    def decide(self, temperature: float = 1.0):
        pressed, ranked = self.bc.decide(temperature=temperature)
        if self._jump_left > 0:
            self._jump_left -= 1
            return (pressed | {"A", "B", "RIGHT"}) - {"LEFT"}, ranked
        if self.abilities.pit_jump and self._pit is not None:
            self._jump_left = JUMP_FRAMES
            self.fired["pit_jump"] += 1
            return (pressed | {"A", "B", "RIGHT"}) - {"LEFT"}, ranked
        if self.abilities.stuck_help and self.instinct._stuck():
            self.fired["stuck_help"] += 1
            return self._instinct_pressed, ranked
        return pressed, ranked

    def act(self, frame_rgb: np.ndarray, temperature: float = 1.0):
        self.observe(frame_rgb)
        return self.decide(temperature=temperature)
