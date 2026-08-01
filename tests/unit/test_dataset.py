import numpy as np

from nes_player.data.reader import Episode
from nes_player.data.writer import EpisodeWriter
from nes_player.emulator.adapter import EmulatorObservation


def _obs(i: int) -> EmulatorObservation:
    rng = np.random.default_rng(i)
    return EmulatorObservation(
        frame_rgb=rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
        audio_pcm=rng.integers(-100, 100, (533,), dtype=np.int16),
        frame_index=i,
        sample_rate=32040,
        done_hint=False,
        debug=None,
    )


def test_roundtrip(tmp_path):
    w = EpisodeWriter(out_dir=tmp_path / "ep", metadata={"game": "test", "sample_rate": 32040})
    observations = [_obs(i) for i in range(300)]
    for i, obs in enumerate(observations):
        pressed = (frozenset({"A"}) if i % 2 else frozenset(), frozenset())
        w.append(obs, pressed)
    w.close()

    e = Episode(tmp_path / "ep")
    assert len(e) == 300
    assert e.frames.shape == (300, 16, 16, 3)
    assert e.actions.shape == (300, 2)
    assert e.actions[1, 0] == 1   # the A bit of the mask
    assert e.actions[0, 0] == 0
    assert np.array_equal(e.frames[7], observations[7].frame_rgb)
    assert np.array_equal(e.frame_audio(7), observations[7].audio_pcm)
    assert e.audio.shape[0] == 300 * 533
