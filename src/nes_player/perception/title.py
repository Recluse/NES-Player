"""Title-screen detection from pixels alone (spec §10.1).

The agent has no access to emulator memory, so "are we back on the title
screen?" has to be answered from the picture. We fingerprint the logo band at
the start of an episode and compare every later frame against it.
"""

import numpy as np

LOGO_BAND = slice(32, 120)   # rows where NES title logos live
SIG_SIZE = (32, 16)          # fingerprint resolution, width x height
STATIC_EPS = 1.5             # mean abs difference below which a frame is "still"
MATCH_EPS = 6.0              # ...and below which two frames are "the same screen"


class TitleWatch:
    """Remembers the title screen and recognises a return to it.

    The `capture` guard is the important part. Without it a run started from a
    savestate (`--state default`) memorises an ordinary gameplay frame as the
    "title", then treats every similar frame as a return to the menu and keeps
    pressing START — which in-game means pausing. That is exactly what happened
    on Double Dragon: a third of the frames in a two-minute recording were
    frozen.
    """

    def __init__(self) -> None:
        self.sig: np.ndarray | None = None
        self._prev: np.ndarray | None = None   # previous candidate, for the static check

    @staticmethod
    def _sig(frame_rgb: np.ndarray) -> np.ndarray:
        import cv2

        logo = cv2.cvtColor(frame_rgb[LOGO_BAND], cv2.COLOR_RGB2GRAY)
        return cv2.resize(logo, SIG_SIZE).astype(np.float32)

    def capture(self, frame_rgb: np.ndarray) -> bool:
        """Memorise this frame as the title screen — only if it is STILL.

        Returns whether the frame was accepted.
        """
        sig = self._sig(frame_rgb)
        static = self._prev is not None and float(np.abs(sig - self._prev).mean()) < STATIC_EPS
        self._prev = sig
        if static:
            self.sig = sig
        return static

    def at_title(self, frame_rgb: np.ndarray) -> bool:
        """True if the current frame looks like the memorised title screen."""
        if self.sig is None:
            return False
        return float(np.abs(self._sig(frame_rgb) - self.sig).mean()) < MATCH_EPS


class TitleTracker:
    """`TitleWatch` plus the timing rules shared by `play` and `explore`.

    Both commands do the same two things every frame: sample the title screen
    early in the episode, then watch for a return to it once the run is under
    way. They only differed in how long they held START afterwards, so that is
    left to the caller.
    """

    CAPTURE_FROM, CAPTURE_TO = 40, 400   # frame window for sampling the title
    CAPTURE_EVERY = 10
    CHECK_FROM = 400                     # only look for a return after this frame
    CHECK_EVERY = 30
    PRESS_COOLDOWN = 150                 # frames between two START presses

    def __init__(self) -> None:
        self.watch = TitleWatch()
        self.last_press = -1000

    def step(self, frame_rgb: np.ndarray, i: int, *, in_game: bool,
             from_power_on: bool) -> bool:
        """Advance one frame. Returns True when START should be pressed.

        `from_power_on` must be False when the episode started from a savestate:
        there is no title screen to sample, and sampling gameplay would poison
        the fingerprint. `in_game` (a controllable object is on screen) vetoes
        the press, because START during play is the pause button.
        """
        if (from_power_on and self.watch.sig is None
                and self.CAPTURE_FROM <= i <= self.CAPTURE_TO and i % self.CAPTURE_EVERY == 0):
            self.watch.capture(frame_rgb)
            return False
        if (self.watch.sig is not None and i > self.CHECK_FROM and i % self.CHECK_EVERY == 0
                and i - self.last_press > self.PRESS_COOLDOWN
                and not in_game
                and self.watch.at_title(frame_rgb)):
            self.last_press = i
            return True
        return False
