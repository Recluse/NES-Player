"""An episode directory is either complete or absent — never half-written.

The writer used to create the final directory on the first frame and write the
labels only at the end, so a quit or an exception left something that looked
like an episode, had no actions and no metadata, and was loaded by the next
training run without complaint.
"""

import numpy as np
import pytest

from nes_player.data.reader import Episode
from nes_player.data.writer import EpisodeWriter
from nes_player.emulator.adapter import EmulatorObservation


def _obs(i: int) -> EmulatorObservation:
    return EmulatorObservation(
        frame_rgb=np.full((16, 16, 3), i % 256, np.uint8),
        audio_pcm=np.zeros(534, np.int16),
        frame_index=i,
        sample_rate=32040,
        done_hint=False,
        debug=None,
    )


def _write(out, n: int) -> EpisodeWriter:
    w = EpisodeWriter(out_dir=out, metadata={"game": "Test-Nes-v0", "sample_rate": 32040})
    for i in range(n):
        w.append(_obs(i), (frozenset({"RIGHT"}),))
    return w


def test_a_finished_episode_is_complete_and_valid(tmp_path):
    out = tmp_path / "ep001"
    _write(out, 300).close()
    ep = Episode(out)
    ep.validate()
    assert len(ep) == 300
    assert ep.metadata["complete"] is True


def test_an_abandoned_episode_leaves_nothing_behind(tmp_path):
    out = tmp_path / "ep001"
    _write(out, 300).abandon()
    assert not out.exists()
    assert not out.with_name("ep001.partial").exists()


def test_an_exception_mid_recording_publishes_nothing(tmp_path):
    out = tmp_path / "ep001"
    with pytest.raises(RuntimeError):
        with EpisodeWriter(out_dir=out, metadata={"game": "Test-Nes-v0",
                                                  "sample_rate": 32040}) as w:
            for i in range(300):
                w.append(_obs(i), (frozenset(),))
            raise RuntimeError("the emulator fell over")
    assert not out.exists()


def test_frames_beyond_the_last_flush_are_not_lost(tmp_path):
    """300 frames is one full 256-frame chunk plus a partial one."""
    out = tmp_path / "ep001"
    _write(out, 300).close()
    ep = Episode(out)
    assert ep.frames.shape[0] == 300
    assert ep.actions.shape[0] == 300


def test_validate_names_the_broken_invariant(tmp_path):
    out = tmp_path / "ep001"
    _write(out, 300).close()
    np.save(out / "actions.npy", np.zeros((11, 1), np.uint8))
    ep = Episode(out)
    with pytest.raises(ValueError, match="actions 11 != 300"):
        ep.validate()


def test_audio_offsets_cover_the_whole_audio(tmp_path):
    out = tmp_path / "ep001"
    _write(out, 300).close()
    ep = Episode(out)
    assert ep.audio_offsets[0] == 0
    assert ep.audio_offsets[-1] == ep.audio.shape[0]
    assert len(ep.frame_audio(7)) == 534
