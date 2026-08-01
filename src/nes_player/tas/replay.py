"""Deterministic replay of an .fm2 movie (spec §10.5, §13.1)."""

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from nes_player.emulator.adapter import EmulatorObservation
from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.tas.fm2 import CMD_SOFT_RESET, FM2Movie, parse_fm2


def verify_rom_matches(movie: FM2Movie, rom_path: str | Path) -> None:
    want = movie.rom_md5
    if want is None:   # no checksum in the movie: nothing to verify against
        print("warning: movie has no romChecksum, ROM match unverified")
        return
    data = Path(rom_path).read_bytes()
    body = data[16:] if data[:4] == b"NES\x1a" else data
    got = hashlib.md5(body).digest()
    if got != want:
        raise ValueError(
            f"ROM does not match the movie: md5 {got.hex()} != {want.hex()} "
            f"(movie was recorded against '{movie.header.get('romFilename')}')"
        )


def _probe_offset(game, movie, integration_dir, offset: int, probe_frames: int) -> float:
    """How far the game has moved away from its starting screen at this offset.

    Divergence from the FIRST frame, not frame-to-frame activity. Many title
    screens animate by themselves — Metroid's stars twinkle — and by activity a
    run stuck on the title looked exactly as healthy as one that was playing.
    """
    players = max(movie.players, 1)
    env = StableRetroAdapter(game, integration_dir=integration_dir, players=players)
    try:
        env.reset(seed=0)
        for _ in range(max(0, -offset)):   # emulator behind: idle frames to catch up
            env.step_buttons([frozenset()] * players)
        first, score, n = None, 0.0, 0
        start = max(0, offset)   # emulator ahead: skip the start of the movie
        for i in range(start, min(start + probe_frames, len(movie.inputs))):
            if movie.commands[i] & CMD_SOFT_RESET:
                break
            small = env.step_buttons(list(movie.inputs[i])).frame_rgb[::8, ::8].astype(np.int16)
            if first is None:
                first = small
            elif i % 10 == 0:
                score += float(np.abs(small - first).mean())
                n += 1
    finally:
        env.close()
    return score / max(n, 1)


def find_boot_offset(game: str, movie: FM2Movie, integration_dir: str | Path | None = None,
                     probe_frames: int = 600, max_offset: int = 6,
                     good_enough: float = 25.0) -> int:
    """Find the movie's start offset relative to our first step.

    The offset differs per movie and is SIGNED: positive means the emulator has
    already consumed frames and the start of the movie should be skipped;
    negative means the opposite, and idle frames come first. Searching only
    positive offsets swallowed an early START, which is why Metroid and Bad
    Dudes sat on their title screens for entire runs.

    The right offset is the one where the screen comes alive.
    """
    best, best_score = 0, -1.0
    # Typical offsets first, so the search usually exits early
    for off in [2, 0, 1, 3, 5, -1, -2, 4, 6, -3, -4, -5, -6]:
        if abs(off) > max_offset:
            continue
        score = _probe_offset(game, movie, integration_dir, off, probe_frames)
        if score > best_score:
            best, best_score = off, score
        if best_score >= good_enough:   # clearly off the starting screen
            break
    return best


def replay_frames(
    game: str,
    movie_path: str | Path,
    integration_dir: str | Path | None = None,
    max_frames: int | None = None,
    include_debug: bool = False,
    boot_offset: int | None = None,
    core: str | None = None,
) -> Iterator[tuple[EmulatorObservation, tuple[frozenset[str], ...]]]:
    """Frame by frame: (the observation AFTER applying that frame's buttons, the buttons).

    boot_offset=None searches for the offset automatically.
    """
    movie = parse_fm2(movie_path)
    env = StableRetroAdapter(
        game,
        integration_dir=integration_dir,
        players=max(movie.players, 1),
        include_debug=include_debug,
        core=core,
    )
    verify_rom_matches(movie, env.rom_path)
    # retro.make and reset consume a few emulated frames before our first step,
    # so the movie timeline runs ahead of our frame counter by that much.
    if boot_offset is None:
        env.close()
        boot_offset = find_boot_offset(game, movie, integration_dir)
        env = StableRetroAdapter(game, integration_dir=integration_dir,
                                 players=max(movie.players, 1),
                                 include_debug=include_debug, core=core)
    players = max(movie.players, 1)
    start = max(0, boot_offset)
    for i in range(start):
        if any(movie.inputs[i]):
            print(f"warning: skipping the press at frame {i} (offset {boot_offset})")
    try:
        env.reset(seed=0)
        for _ in range(max(0, -boot_offset)):   # emulator behind: idle frames
            env.step_buttons([frozenset()] * players)
        n = len(movie.inputs) if max_frames is None else min(start + max_frames,
                                                             len(movie.inputs))
        for i in range(start, n):
            # A reset on frame 0 is already covered by our power-on boot;
            # a later one is not supported.
            if movie.commands[i] & CMD_SOFT_RESET:
                raise NotImplementedError(f"soft reset at frame {i} is not supported")
            yield env.step_buttons(list(movie.inputs[i])), movie.inputs[i]
    finally:
        env.close()


def run_replay(
    game: str,
    movie_path: str | Path,
    integration_dir: str | Path | None = None,
    max_frames: int | None = None,
    on_frame: Callable[[EmulatorObservation, tuple[frozenset[str], ...]], None] | None = None,
) -> dict:
    """Replay the whole movie and return a summary."""
    frames = 0
    audio_samples = 0
    last: np.ndarray | None = None
    for obs, _pressed in replay_frames(game, movie_path, integration_dir, max_frames):
        frames += 1
        audio_samples += len(obs.audio_pcm)
        last = obs.frame_rgb
        if on_frame is not None:
            on_frame(obs, _pressed)
    return {
        "frames": frames,
        "audio_samples": audio_samples,
        "last_frame_sha1": hashlib.sha1(last.tobytes()).hexdigest() if last is not None else None,
    }
