"""Audio normalisation is decided by the training data and nothing else.

Three distributions used to exist at once: every training episode normalised by
its own statistics, every validation episode by its own — which is a look at
audio the model has not heard yet — and inference by an average of all of them,
validation included. The features a window got therefore depended on which of
the three paths produced it.
"""

import numpy as np
import pytest

from nes_player.policy.bc import episode_log_mel, mel_moments

SR = 32040


def _tone(seconds: float, hz: float, amp: float, seed: int = 0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    rng = np.random.default_rng(seed)
    wav = amp * np.sin(2 * np.pi * hz * t) + 0.01 * rng.standard_normal(len(t))
    return (wav * 32767).clip(-32768, 32767).astype(np.int16)


def test_moments_pool_by_size_not_by_episode():
    """A short loud episode must not count as much as a long quiet one."""
    long_quiet = np.full((32, 1000), -5.0, np.float32)
    short_loud = np.full((32, 10), 5.0, np.float32)
    mean, _ = mel_moments([long_quiet, short_loud])
    naive = (-5.0 + 5.0) / 2          # the average of the two episode means
    assert mean == pytest.approx(-4.90, abs=0.01)
    assert abs(mean - naive) > 4.0


def test_moments_match_numpy_on_the_concatenation():
    a = episode_log_mel(_tone(0.5, 440, 0.3, seed=1), SR)
    b = episode_log_mel(_tone(0.3, 880, 0.6, seed=2), SR)
    mean, std = mel_moments([a, b])
    both = np.concatenate([a, b], axis=1)
    assert mean == pytest.approx(float(both.mean()), rel=1e-6)
    assert std == pytest.approx(float(both.std()), rel=1e-5)


def test_a_windows_features_do_not_depend_on_later_audio():
    """The audit's acceptance criterion, directly.

    Append more audio to a validation episode; windows that already existed
    must come out identical. Under per-episode normalisation they did not.
    """
    head = _tone(0.4, 440, 0.3, seed=3)
    tail = _tone(0.4, 120, 0.9, seed=4)          # louder and lower: shifts the stats
    train = [episode_log_mel(_tone(0.6, 300, 0.4, seed=5), SR)]
    mean, std = mel_moments(train)               # fixed by the training data

    short = (episode_log_mel(head, SR) - mean) / (std + 1e-6)
    long = (episode_log_mel(np.concatenate([head, tail]), SR) - mean) / (std + 1e-6)
    n = short.shape[1] - 2                       # ignore the frames the tail bleeds into
    assert np.allclose(short[:, :n], long[:, :n], atol=1e-5)


def test_per_episode_normalisation_would_have_failed_that():
    """The same check against the old behaviour, so the test has teeth."""
    head = _tone(0.4, 440, 0.3, seed=3)
    tail = _tone(0.4, 120, 0.9, seed=4)

    def own_stats(mel):
        return (mel - mel.mean()) / (mel.std() + 1e-6)

    short = own_stats(episode_log_mel(head, SR))
    long = own_stats(episode_log_mel(np.concatenate([head, tail]), SR))
    n = short.shape[1] - 2
    assert not np.allclose(short[:, :n], long[:, :n], atol=1e-5)
