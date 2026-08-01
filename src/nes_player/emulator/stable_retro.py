"""Headless stable-retro backend for the NES."""

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from nes_player.emulator.adapter import EmulatorObservation
from nes_player.emulator.controller import ControllerState, buttons_to_retro_array


def rom_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# The agent's canonical frame — what a television would have shown. The PPU
# draws 256×240, but the edges disappeared behind the bezel of a CRT, and MANY
# GAMES LEAVE GARBAGE EXACTLY THERE: a dirty left column, fragments of tiles
# along the top. Cropping is not cosmetic — without it the motion tracker picks
# that garbage up as moving objects.
#
# Cores also disagree: our fceumm build already crops to 240×224 while nestopia
# and mesen return the full 256×240. Normalising here is what lets a trained
# model run on a core it has never seen.
CANON_H, CANON_W = 224, 240
CANON_SAMPLE_RATE = 32040   # cores differ here too; nestopia gives 48 kHz


def normalize_frame(frame: np.ndarray, viewport: str = "tv") -> np.ndarray:
    """Any core's frame to the canonical 240×224 ('tv'), or untouched ('raw')."""
    if viewport == "raw":
        return frame
    h, w = frame.shape[:2]
    if (h, w) == (CANON_H, CANON_W):
        return frame
    dy, dx = (h - CANON_H) // 2, (w - CANON_W) // 2
    if dy < 0 or dx < 0:   # smaller than canonical: hand it back rather than crop
        return frame
    return frame[dy : dy + CANON_H, dx : dx + CANON_W]


def resample_pcm(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Plain linear resampling. Mel features are insensitive to the difference,
    and without it a model breaks when the core changes: 32040 against 48000 Hz."""
    if src_rate == dst_rate or len(pcm) == 0:
        return pcm
    n_out = max(1, round(len(pcm) * dst_rate / src_rate))
    x = np.linspace(0, len(pcm) - 1, n_out, dtype=np.float64)
    return np.interp(x, np.arange(len(pcm)), pcm.astype(np.float64)).astype(np.int16)


class StableRetroAdapter:
    """One game instance. `game` is a stable-retro integration id, e.g. 'SuperMarioBros-Nes'."""

    def __init__(
        self,
        game: str,
        state: str | None = None,
        integration_dir: str | Path | None = None,
        include_debug: bool = False,
        players: int = 1,
        viewport: str = "tv",
        core: str | None = None,
    ):
        """viewport: 'tv' gives the canonical 240×224 frame with the garbage
        edges removed, identical across cores; 'raw' gives whatever the core
        returned. core: which emulation core to use, see emulator/cores.py."""
        if core:
            from nes_player.emulator import cores

            cores.use(core)
        import retro  # local import: heavy, and lets unit tests run without it

        inttype = retro.data.Integrations.ALL
        if integration_dir is not None:
            retro.data.Integrations.add_custom_path(str(Path(integration_dir).resolve()))
            inttype = retro.data.Integrations.CUSTOM_ONLY

        # state=None means a real power-on boot, not the integration's DEFAULT
        # savestate. Passing 'default' asks for that savestate explicitly, which
        # some games need: Double Dragon's title screen cannot be passed from
        # power-on by any button sequence at all.
        if state == "default":
            state = retro.State.DEFAULT
        kwargs: dict[str, Any] = {
            "inttype": inttype,
            "render_mode": None,
            "state": state if state is not None else retro.State.NONE,
            "players": players,
            # The default Actions.FILTERED silently drops START and L+R, which
            # breaks TAS replay in a way that looks like a desync.
            "use_restricted_actions": retro.Actions.ALL,
        }
        self._env = retro.make(game, **kwargs)
        self.players = players
        self._buttons: list[str | None] = self._env.buttons
        self._frame_index = 0
        self._include_debug = include_debug
        self.viewport = viewport
        self.core_sample_rate = int(self._env.em.get_audio_rate())
        # Callers always see the canonical rate, whatever the core runs at.
        self.sample_rate = (self.core_sample_rate if viewport == "raw"
                            else CANON_SAMPLE_RATE)
        self.rom_path = retro.data.get_romfile_path(game, inttype)
        self.rom_sha256 = rom_sha256(self.rom_path)
        # Snapshot of the power-on state: em.reset() without a state is a soft
        # reset that leaves RAM intact, so a reproducible reset means restoring
        # this snapshot instead.
        self._poweron_state: bytes | None = (
            self._env.em.get_state() if state is None else None
        )

    def _observe(self, frame: np.ndarray, done: bool, info: dict[str, Any]) -> EmulatorObservation:
        audio = self._env.em.get_audio()  # (n, 2) int16
        mono = audio.astype(np.int32).mean(axis=1).astype(np.int16)
        mono = resample_pcm(mono, self.core_sample_rate, self.sample_rate)
        frame = normalize_frame(frame, self.viewport)
        return EmulatorObservation(
            frame_rgb=frame,
            audio_pcm=mono,
            frame_index=self._frame_index,
            sample_rate=self.sample_rate,
            done_hint=done,
            debug=dict(info) if self._include_debug else None,
        )

    def reset(self, seed: int | None = None) -> EmulatorObservation:
        frame, info = self._env.reset(seed=seed)
        if self._poweron_state is not None:
            self._env.em.set_state(self._poweron_state)
            frame = self._env.em.get_screen()
        self._frame_index = 0
        return self._observe(frame, done=False, info=info)

    def step(self, action: ControllerState) -> EmulatorObservation:
        return self.step_buttons([action.pressed()])

    def step_buttons(self, pressed_per_player: list[Iterable[str]]) -> EmulatorObservation:
        """Raw buttons with no combination check — TAS movies do press L+R."""
        arr: list[int] = []
        for p in range(self.players):
            pressed = pressed_per_player[p] if p < len(pressed_per_player) else ()
            arr += buttons_to_retro_array(pressed, self._buttons)
        frame, _reward, terminated, truncated, info = self._env.step(arr)
        self._frame_index += 1
        return self._observe(frame, done=terminated or truncated, info=info)

    def save_state(self) -> bytes:
        return self._env.em.get_state()

    def load_state(self, state: bytes) -> None:
        self._env.em.set_state(state)

    def close(self) -> None:
        self._env.close()
