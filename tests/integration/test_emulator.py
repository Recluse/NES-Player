"""Needs a ROM imported into stable-retro (`retro.import <dir-with-roms>`). Skips otherwise."""

import numpy as np
import pytest

retro = pytest.importorskip("retro")

from nes_player.emulator.controller import ControllerState
from nes_player.emulator.stable_retro import StableRetroAdapter

GAME = "SuperMarioBros-Nes-v0"


def _rom_available() -> bool:
    try:
        retro.data.get_romfile_path(GAME, retro.data.Integrations.ALL)
        return True
    except (FileNotFoundError, KeyError):
        return False


pytestmark = pytest.mark.skipif(not _rom_available(), reason=f"ROM for {GAME} not imported")


@pytest.fixture
def env():
    adapter = StableRetroAdapter(GAME)
    yield adapter
    adapter.close()


def test_boot_and_step(env):
    obs = env.reset(seed=0)
    assert obs.frame_rgb.shape == (224, 240, 3) or obs.frame_rgb.shape[2] == 3
    assert obs.frame_rgb.dtype == np.uint8
    assert obs.sample_rate > 0
    obs2 = env.step(ControllerState(right=True))
    assert obs2.frame_index == 1
    assert obs2.audio_pcm.dtype == np.int16
    assert len(obs2.audio_pcm) > 0
    assert obs2.debug is None  # policy must not see telemetry


def test_deterministic_replay(env):
    actions = [ControllerState(right=True)] * 30 + [ControllerState(right=True, a=True)] * 30

    def run() -> np.ndarray:
        env.reset(seed=0)
        frame = None
        for a in actions:
            frame = env.step(a).frame_rgb
        return frame

    assert np.array_equal(run(), run())


def test_save_load_state(env):
    env.reset(seed=0)
    for _ in range(60):
        env.step(ControllerState(right=True))
    snapshot = env.save_state()
    after = env.step(ControllerState()).frame_rgb
    for _ in range(30):
        env.step(ControllerState(a=True))
    env.load_state(snapshot)
    assert np.array_equal(env.step(ControllerState()).frame_rgb, after)
