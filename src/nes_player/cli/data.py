"""Commands that turn TAS movies into frames and datasets (spec §19)."""

import argparse
import time


def cmd_tas_replay(args: argparse.Namespace) -> None:
    """Replay an .fm2 movie through the emulator and report the throughput."""
    from nes_player.evaluation.viewer import Viewer
    from nes_player.tas.replay import run_replay

    viewer = Viewer(window=args.window, video_out=args.video_out, throttle=args.realtime)
    t0 = time.monotonic()
    try:
        summary = run_replay(
            args.game, args.movie,
            integration_dir=args.integrations,
            max_frames=args.max_frames,
            on_frame=viewer.show if (args.window or args.video_out) else None,
        )
    finally:
        viewer.close()
    dt = time.monotonic() - t0
    print(f"frames={summary['frames']} audio_samples={summary['audio_samples']} "
          f"fps={summary['frames'] / dt:.0f} last_frame_sha1={summary['last_frame_sha1']}")


def cmd_dataset_build(args: argparse.Namespace) -> None:
    """Record a TAS replay into a Zarr episode: frames, audio and buttons."""
    from pathlib import Path

    from nes_player.data.writer import EpisodeWriter
    from nes_player.tas.replay import replay_frames

    writer = EpisodeWriter(
        out_dir=Path(args.out),
        metadata={
            "game": args.game,
            "movie": Path(args.movie).name,
            "source": "tasvideos.org",
        },
    )
    t0 = time.monotonic()
    n = 0
    for obs, pressed in replay_frames(
        args.game, args.movie, integration_dir=args.integrations, max_frames=args.max_frames
    ):
        if n == 0:
            writer.metadata["sample_rate"] = obs.sample_rate
        writer.append(obs, pressed)
        n += 1
    writer.close()
    print(f"episode: {args.out} frames={n} fps={n / (time.monotonic() - t0):.0f}")
