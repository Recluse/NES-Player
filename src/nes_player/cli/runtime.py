"""Small pieces of per-frame bookkeeping shared by `play` and `explore`.

Both commands run the same outer loop — step the emulator, watch the screen,
keep a short log of recent sounds — and both had their own copy of it. These
are those copies, deduplicated. Nothing here knows about policies or rendering.
"""

import numpy as np

RECENT_SOUNDS = 7        # how many sound events the side panel shows
UNPAUSE_FRAMES = 2       # how long to hold START when unpausing


class PauseWatchdog:
    """Detects a frozen screen and asks for a START press.

    A perfectly identical frame for a second or two almost always means the
    game is paused — usually because the agent pressed START itself. Comparing
    raw frames is enough: real NES gameplay is never pixel-identical for that
    long, not even on a still title screen with a blinking prompt.
    """

    def __init__(self, frames: int) -> None:
        self.frames = frames         # how many identical frames count as "paused"
        self._prev: np.ndarray | None = None
        self._frozen = 0

    def push(self, frame_rgb: np.ndarray) -> bool:
        """Feed the newest frame. Returns True when START should be pressed."""
        if self._prev is not None and np.array_equal(frame_rgb, self._prev):
            self._frozen += 1
            if self._frozen >= self.frames:
                self._frozen = 0
                self._prev = frame_rgb
                return True
        else:
            self._frozen = 0
        self._prev = frame_rgb
        return False


class SoundLog:
    """The last few audio events, with their age in frames.

    Ages matter for attribution: the death jingle plays roughly 120 frames
    *before* the lives counter in memory drops, so a death is matched against
    sounds heard in the recent past, not against whatever is playing right now.
    """

    def __init__(self, keep: int = RECENT_SOUNDS) -> None:
        self.keep = keep
        self.events: list[list] = []   # [cluster_id, age_in_frames, times_heard]

    def add(self, cluster_id: int, heard: int = 0) -> None:
        self.events.insert(0, [cluster_id, 0, heard])
        del self.events[self.keep:]

    def tick(self) -> None:
        """Advance every event's age by one frame."""
        for e in self.events:
            e[1] += 1

    def within(self, max_age: int) -> list[int]:
        """Cluster ids of events heard no longer than `max_age` frames ago."""
        return [e[0] for e in self.events if e[1] < max_age]


class PingLog:
    """Sound-source pings from the contrastive audio↔frame model.

    Each ping is a point in normalised frame coordinates that fades out; the
    viewer draws the surviving ones over the picture.
    """

    def __init__(self, ttl: int = 45, keep: int = 4) -> None:
        self.ttl = ttl
        self.keep = keep
        self.pings: list[list] = []    # [x01, y01, age_in_frames]

    def add(self, x01: float, y01: float) -> None:
        self.pings.append([x01, y01, 0])

    def tick(self) -> None:
        for p in self.pings:
            p[2] += 1
        self.pings[:] = [p for p in self.pings if p[2] <= self.ttl][-self.keep:]


class Thoughts:
    """Bounded log of what the agent is doing, shown in the side panel.

    Bounded because episodes can run for hours in `--loop`; an unbounded list
    is a slow memory leak in a process that is supposed to stream all day.
    """

    def __init__(self, limit: int = 200, keep: int = 100) -> None:
        self.limit, self.keep = limit, keep
        self.lines: list[str] = []

    def add(self, line: str) -> None:
        self.lines.append(line)
        if len(self.lines) > self.limit:
            del self.lines[:-self.keep]

    def __iter__(self):
        return iter(self.lines)

    def __add__(self, other: list[str]) -> list[str]:
        return self.lines + other


def action_entropy(ranked: list[tuple[str, float]]) -> float:
    """Normalised entropy of the policy's action distribution, 0..1.

    1 means "no idea", 0 means "certain". Plotted over time it shows whether
    the model is actually deciding or just drifting.
    """
    probs = np.asarray([p for _, p in ranked])
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum() / np.log(len(ranked)))
