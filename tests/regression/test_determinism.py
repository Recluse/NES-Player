"""Golden hashes of a fixed run, plus TAS action identity.

These break when the emulation core, the power-on snapshot or the FM2 parser
changes — which is a signal to re-examine every trained model and dataset, not
just to update the hashes. They are what caught a third-party core silently
replacing the default one for every game, a failure that produced no error
anywhere and would have degraded models in silence.

Requires an imported Super Mario Bros. ROM; otherwise the test skips.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

retro = pytest.importorskip("retro")

from nes_player.emulator.controller import ControllerState
from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.tas.fm2 import parse_fm2

GAME = "SuperMarioBros-Nes-v0"
ROOT = Path(__file__).parents[2]
TAS = ROOT / "rnd/tas/happylee_mars608-smb-warpless.fm2"

# Golden values: stable-retro 1.0.1 with fceumm, power-on snapshot
FRAME_SHA = "9e93657e4212a4a31c7236ca340b94e77f408455bc2cbc51bb59decaf62a8919"
AUDIO_SHA = "b7861eddd4f1abacf9b305d99d03a00f38246b47c5fe2d5f98acc42de1cae866"
TAS_SHA = "dc078d9abaf3c1811a836552483a9ef71fed58b07130545d9f179df46edfd49b"
TAS_FRAMES = 67117


def _rom_available() -> bool:
    try:
        retro.data.get_romfile_path(GAME, retro.data.Integrations.ALL)
        return True
    except (FileNotFoundError, KeyError):
        return False


@pytest.mark.skipif(not _rom_available(), reason=f"ROM for {GAME} not imported")
def test_rollout_frame_and_audio_hashes():
    em = StableRetroAdapter(GAME)
    em.reset()
    audio = []
    obs = None
    for i in range(120):
        pressed = set()
        if i >= 40:
            pressed.add("RIGHT")
        if i % 30 < 8:
            pressed.add("A")
        obs = em.step(ControllerState(frozenset(pressed)))
        audio.append(obs.audio_pcm)
    em.close()
    assert hashlib.sha256(obs.frame_rgb.tobytes()).hexdigest() == FRAME_SHA
    assert hashlib.sha256(np.concatenate(audio).tobytes()).hexdigest() == AUDIO_SHA


@pytest.mark.skipif(not TAS.exists(), reason="rnd/tas movie not present")
def test_tas_actions_identity():
    m = parse_fm2(TAS)
    canon = ";".join("+".join(sorted(p[0])) for p in m.inputs)
    assert len(m.inputs) == TAS_FRAMES
    assert hashlib.sha256(canon.encode()).hexdigest() == TAS_SHA
