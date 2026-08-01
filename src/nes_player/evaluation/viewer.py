"""The live window, and recording a reference video.

Holding 60 fps takes a specific arrangement. The worker thread — emulation,
perception, rendering and audio — runs flat out, paced by a blocking PCM write.
The main thread does nothing but `imshow` and `waitKey`, which cost 10-25 ms on
macOS, and always shows the newest frame available.

Two things were learned the hard way and are worth not rediscovering. A Python
callback inside the audio realtime thread crackles, because it waits on the GIL
held by our own threads; only a blocking write from a thread we own works. And
pacing with a fixed `waitKey` delay adds to the processing time instead of
absorbing it, slowing everything down.

In-window controls: q, Esc, the close button or STOP to quit; r or RESTART to
restart the episode.
"""

import threading
import time
from pathlib import Path

import cv2

from nes_player.emulator.adapter import EmulatorObservation
from nes_player.evaluation.dashboard import (
    BTN_MUTE,
    BTN_RESTART,
    BTN_STOP,
    BTN_TOGGLE_BOXES,
    BTN_TOGGLE_CAM,
    PAD_HITS,
    Dashboard,
)

HINT_HOLD = 12   # frames a button stays held after a click on the drawn pad

WINDOW = "NES Player"


class Viewer:
    """`show()` is called from the worker thread and returns the user command —
    None, 'quit' or 'restart' — that the GUI thread left for it."""

    def __init__(self, window: bool = False, video_out: str | Path | None = None,
                 throttle: bool = False, fps: float = 60.1, title: str = "NES Player",
                 scale: float = 1.0):
        self.window = window
        self.throttle = throttle
        self.fps = fps
        self.scale = scale   # 1.5 gives a 1920×1080 window
        self.dashboard = Dashboard(title=title)
        self.writer: cv2.VideoWriter | None = None
        self._video_out = str(video_out) if video_out else None
        self._window_ready = False
        self._audio: list = []   # per-frame PCM, written to .wav alongside the video
        self._sample_rate = 32040
        self._speaker = None     # live sound, only under --realtime
        self._deadline: float | None = None
        self._lock = threading.Lock()
        self._latest = None      # newest rendered frame, for the GUI thread
        self._pending: str | None = None   # command from the GUI to the worker
        self.muted = False
        self._loop_t: list[float] = []     # timings for the fps counters
        self._gui_t: list[float] = []
        self.show_cam = True
        self.show_boxes = True
        self._human: dict[str, int] = {}   # pad clicks: button -> frames remaining
        self._ring = bytearray()           # ring buffer feeding the audio thread
        self._ring_lock = threading.Lock()
        self._audio_stop = False

    # ---------- worker thread ----------

    def fps_info(self) -> str:
        def rate(ts):
            if len(ts) < 2:
                return 0.0
            return (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)
        return f"{rate(self._loop_t):.0f} loop / {rate(self._gui_t):.0f} gui"

    def show(
        self,
        obs: EmulatorObservation,
        pressed: tuple[frozenset[str], ...],
        info: dict[str, str] | None = None,
        thoughts: list[str] | None = None,
        action_probs: list[tuple[str, float]] | None = None,
        slots: list | None = None,
        verdicts: dict | None = None,
        heatmap=None,
        entropy_hist: list[float] | None = None,
        features=None,
        gallery: list | None = None,
        audio_events: list | None = None,
        ghost: list | None = None,
        sound_pings: list | None = None,
    ) -> str | None:
        self._loop_t.append(time.monotonic())
        del self._loop_t[:-120]
        if len(pressed) > 1:
            info = {"P2": "+".join(sorted(pressed[1])) or "-"} | (info or {})
        if not self.show_boxes:
            slots = None
        if not self.show_cam:
            heatmap = None
        img = self.dashboard.render(
            obs, pressed[0], info, thoughts, action_probs, slots, verdicts,
            heatmap, entropy_hist,
            toggles={"cam": self.show_cam, "boxes": self.show_boxes},
            features=features, gallery=gallery, audio_events=audio_events, ghost=ghost,
            sound_pings=sound_pings)
        # Upscaling on the CPU is only for the recording; the window is scaled
        # by the system compositor for free
        if self._video_out or self.writer is not None:
            vid = img
            if self.scale != 1.0:
                vid = cv2.resize(img, None, fx=self.scale, fy=self.scale,
                                 interpolation=cv2.INTER_LINEAR)
            if self.writer is None:
                # avc1 is h264 through the platform encoder, hardware where available
                self.writer = cv2.VideoWriter(
                    self._video_out, cv2.VideoWriter_fourcc(*"avc1"),
                    self.fps, (vid.shape[1], vid.shape[0]))
                if not self.writer.isOpened():   # no h264 encoder: software fallback
                    self.writer = cv2.VideoWriter(
                        self._video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps, (vid.shape[1], vid.shape[0]))
            self.writer.write(vid)
            self._audio.append(obs.audio_pcm)
            self._sample_rate = obs.sample_rate

        if self._speaker is None and self.throttle:
            try:
                import sounddevice

                self._sample_rate = obs.sample_rate
                # Blocking write: the realtime path feeds the C audio code with
                # no Python in it. The callback variant crackled, because the
                # callback waited on the GIL held by our own threads.
                self._speaker = sounddevice.OutputStream(
                    samplerate=obs.sample_rate, channels=1, dtype="int16",
                    latency="low")
                self._speaker.start()
                threading.Thread(target=self._audio_feeder, daemon=True).start()
            except Exception as e:   # no audio device: carry on silently
                print(f"audio disabled: {e}")
                self._speaker = False
        if self._speaker:
            pcm = obs.audio_pcm
            if self.muted:
                import numpy as np

                pcm = np.zeros_like(pcm)
            with self._ring_lock:
                self._ring += pcm.tobytes()
            # Pacing: keep the buffer 50-90 ms full — minimum audible latency
            # with enough slack to absorb a slow frame
            target = int(0.12 * self._sample_rate) * 2   # 120 ms: the AV model and
            # the ghost predictor are hungrier for the GIL
            while True:
                with self._ring_lock:
                    backlog = len(self._ring)
                if backlog <= target:
                    break
                time.sleep(0.004)
        elif self.throttle:
            # No audio device: pace the loop ourselves
            now = time.monotonic()
            if self._deadline is None:
                self._deadline = now
            if self._deadline > now:
                time.sleep(self._deadline - now)
            self._deadline = max(self._deadline + 1 / self.fps, now - 0.05)

        if self.window:
            with self._lock:
                self._latest = img
        cmd, self._pending = self._pending, None
        return cmd

    def _audio_feeder(self) -> None:
        """Dedicated thread: pours the ring buffer into the device, blocking."""
        import numpy as np

        while not self._audio_stop:
            with self._ring_lock:
                take = min(len(self._ring) - len(self._ring) % 2, 4096)
                chunk = bytes(self._ring[:take]) if take > 0 else b""
                del self._ring[:take]
            if chunk:
                try:
                    self._speaker.write(np.frombuffer(chunk, dtype=np.int16))
                except Exception:
                    return
            else:
                time.sleep(0.003)

    # ---------- main thread ----------

    def gui_pump(self) -> None:
        if not self._window_ready:
            # WINDOW_NORMAL: show 720p and let the system compositor scale it to
            # HD for free, instead of a CPU resize sixty times a second
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            from nes_player.evaluation.dashboard import H as _H
            from nes_player.evaluation.dashboard import W as _W

            cv2.resizeWindow(WINDOW, int(_W * self.scale), int(_H * self.scale))
            cv2.setMouseCallback(WINDOW, self._on_mouse)
            self._window_ready = True
        with self._lock:
            img = self._latest
        if img is not None:
            cv2.imshow(WINDOW, img)
            self._gui_t.append(time.monotonic())
            del self._gui_t[:-120]
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self._pending = "quit"
        elif key == ord("r"):
            self._pending = "restart"
        elif key == ord("m"):
            self.muted = not self.muted
        elif key == ord("c"):
            self.show_cam = not self.show_cam
        elif key == ord("b"):
            self.show_boxes = not self.show_boxes
        elif img is not None and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self._pending = "quit"

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        from nes_player.evaluation.dashboard import H as _H
        from nes_player.evaluation.dashboard import W as _W

        try:   # window coordinates to frame coordinates, via the real geometry
            _, _, rw, rh = cv2.getWindowImageRect(WINDOW)
            if rw > 0 and abs(rw - _W) > 2:
                x, y = x * _W / rw, y * _H / rh
        except Exception:
            x, y = x / self.scale, y / self.scale

        def hit(btn):
            return btn[0] <= x <= btn[2] and btn[1] <= y <= btn[3]

        if hit(BTN_MUTE):
            self.muted = not self.muted
        elif hit(BTN_TOGGLE_CAM):
            self.show_cam = not self.show_cam
        elif hit(BTN_TOGGLE_BOXES):
            self.show_boxes = not self.show_boxes
        elif hit(BTN_STOP):
            self._pending = "quit"
        elif hit(BTN_RESTART):
            self._pending = "restart"
        else:
            for name, box in PAD_HITS.items():
                if hit(box):
                    self._human[name] = HINT_HOLD   # a hint from the human
                    break

    def human_buttons(self) -> frozenset[str]:
        """Buttons "pressed" by clicking the pad; called once per frame."""
        if not self._human:
            return frozenset()
        out = frozenset(self._human)
        self._human = {b: n - 1 for b, n in self._human.items() if n > 1}
        return out

    def close(self) -> None:
        self._audio_stop = True
        if self._speaker:
            self._speaker.stop()
            self._speaker.close()
        if self.writer is not None:
            self.writer.release()
            if self._audio:
                import wave

                import numpy as np

                with wave.open(self._video_out + ".wav", "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self._sample_rate)
                    w.writeframes(np.concatenate(self._audio).tobytes())
        if self._window_ready:
            cv2.destroyAllWindows()


def run_with_gui(viewer: Viewer, loop_fn) -> None:
    """Windowed mode: loop_fn in a background thread, GUI on the main one —
    macOS insists that the window live on the main thread."""
    if not viewer.window:
        loop_fn()
        return
    error: list[BaseException] = []

    def _wrapped():
        try:
            loop_fn()
        except BaseException as e:  # noqa: BLE001 — re-raised on the main thread
            error.append(e)

    t = threading.Thread(target=_wrapped, daemon=True)
    t.start()
    while t.is_alive():
        viewer.gui_pump()
    t.join()
    if error:
        raise error[0]
