"""NES controller state: 8 buttons, physically impossible combos rejected."""

from collections.abc import Iterable
from dataclasses import dataclass

BUTTONS = ("A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

# A d-pad is one physical rocker: pressing two opposite directions is not
# something a hand can do. TAS movies record them anyway — the emulator accepts
# the bits — and cloning those recordings taught 24 of our checkpoints action
# vocabularies containing LEFT+RIGHT. The archive keeps the raw combination;
# nothing that reaches a controller may.
OPPOSITES = (("LEFT", "RIGHT"), ("UP", "DOWN"))


def resolve_conflicts(pressed: Iterable[str]) -> frozenset[str]:
    """Drop opposite directions, keeping neither.

    Neither, rather than one of them: which one a player "meant" is a guess,
    and on the games where this appears the combination is a TAS trick whose
    effect is closer to no direction than to either. Dropping both is the one
    choice that never invents an intent.
    """
    out = set(pressed)
    for a, b in OPPOSITES:
        if a in out and b in out:
            out -= {a, b}
    return frozenset(out)


@dataclass(frozen=True)
class ControllerState:
    a: bool = False
    b: bool = False
    select: bool = False
    start: bool = False
    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False

    def __post_init__(self) -> None:
        if self.left and self.right:
            raise ValueError("LEFT+RIGHT is physically impossible on a NES d-pad")
        if self.up and self.down:
            raise ValueError("UP+DOWN is physically impossible on a NES d-pad")

    @classmethod
    def from_mask(cls, mask: int) -> ControllerState:
        """Bit i of mask corresponds to BUTTONS[i]."""
        return cls(**{name.lower(): bool(mask >> i & 1) for i, name in enumerate(BUTTONS)})

    def to_mask(self) -> int:
        mask = 0
        for i, name in enumerate(BUTTONS):
            if getattr(self, name.lower()):
                mask |= 1 << i
        return mask

    def pressed(self) -> tuple[str, ...]:
        return tuple(name for name in BUTTONS if getattr(self, name.lower()))

    def to_retro_array(self, retro_buttons: list[str | None]) -> list[int]:
        """Map to a libretro buttons array (order taken from env.buttons)."""
        return buttons_to_retro_array(self.pressed(), retro_buttons)


def buttons_to_retro_array(
    pressed: Iterable[str], retro_buttons: list[str | None]
) -> list[int]:
    """Raw mapping with no physical-combination check: TAS movies do press L+R."""
    active = set(pressed)
    return [1 if b in active else 0 for b in retro_buttons]
