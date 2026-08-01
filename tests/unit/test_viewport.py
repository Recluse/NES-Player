"""Canonical frame and audio: the agent sees the same thing on any core."""

import numpy as np

from nes_player.emulator.stable_retro import (
    CANON_H,
    CANON_SAMPLE_RATE,
    CANON_W,
    normalize_frame,
    resample_pcm,
)


def test_full_frame_is_cropped_to_tv_view():
    """A core returning the full 256×240 gets 8 px cropped from each side.
    Many games leave garbage there, which the tracker took for moving objects."""
    full = np.zeros((240, 256, 3), np.uint8)
    full[:8] = full[-8:] = 255          # garbage along the top and bottom
    full[:, :8] = full[:, -8:] = 255    # and down the sides
    full[100, 100] = 42                 # a marker inside the picture
    out = normalize_frame(full)
    assert out.shape == (CANON_H, CANON_W, 3)
    assert out.max() == 42, "the edge garbage must be cropped away"
    assert out[100 - 8, 100 - 8, 0] == 42, "the content must not shift"


def test_already_canonical_frame_untouched():
    frame = np.random.default_rng(0).integers(0, 255, (CANON_H, CANON_W, 3), dtype=np.uint8)
    assert np.array_equal(normalize_frame(frame), frame)


def test_raw_viewport_keeps_core_output():
    full = np.zeros((240, 256, 3), np.uint8)
    assert normalize_frame(full, viewport="raw").shape == (240, 256, 3)


def test_smaller_frame_is_not_padded():
    small = np.zeros((200, 200, 3), np.uint8)
    assert normalize_frame(small).shape == (200, 200, 3)


def test_audio_resampled_to_canonical_rate():
    # A single bump rather than a sine wave: a periodic signal has several
    # near-equal peaks, and rounding to int16 changes which one is the maximum
    t = np.linspace(-3, 3, 800)
    src = (np.exp(-(t**2)) * 10000).astype(np.int16)
    out = resample_pcm(src, 48000, CANON_SAMPLE_RATE)
    assert abs(len(out) - 800 * CANON_SAMPLE_RATE / 48000) <= 1
    assert out.dtype == np.int16
    # The shape survives: the bump sits at about the same relative time
    assert abs(int(out.argmax()) / len(out) - int(src.argmax()) / len(src)) < 0.02


def test_audio_untouched_when_rate_matches():
    pcm = np.arange(100, dtype=np.int16)
    assert np.array_equal(resample_pcm(pcm, CANON_SAMPLE_RATE, CANON_SAMPLE_RATE), pcm)
