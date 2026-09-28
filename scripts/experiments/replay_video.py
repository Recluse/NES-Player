"""Replay a recorded button sequence from a savestate and write it as video.

    uv run python scripts/experiments/replay_video.py runs/vice/geboss3/boss.json \
        --then-start 2600 --out runs/vice/boss_to_3-1.mp4

--then-start N: after the recording, run N more frames with no buttons, so
the video ends on whatever comes next. Cinemas play in full: on a recording
nothing is skipped (owner's rule). --skip B taps B through them instead; never
START, which is pause once play resumes.
"""

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--game", default="ViceProjectDoom-Nes-v0")
    ap.add_argument("--then-start", type=int, default=0)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--skip", choices=("none", "B"), default="none")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rec = json.loads(Path(args.recording).read_text())
    env = StableRetroAdapter(args.game, include_debug=True, state="default",
                             integration_dir=None)
    env.reset(seed=0)
    env.load_state(Path(rec["load_state"]).read_bytes())
    seq = [frozenset(b) for b in rec["actions"]]
    tail = args.then_start

    h, w = 224 * args.scale, 240 * args.scale
    raw = args.out + ".video.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", "60.0988", "-i", "-", "-c:v", "h264_videotoolbox",
         "-b:v", "6M", raw], stdin=subprocess.PIPE)
    pcm = []
    room0 = None

    def buttons():
        # The first cut pulsed START through the cinema and on into 3-1,
        # where START is pause. Now nothing is pressed unless asked, and a
        # skip taps B only until the room byte moves.
        nonlocal room0
        yield from seq
        room0 = int(env._env.get_ram()[168])
        for i in range(tail):
            moved = int(env._env.get_ram()[168]) != room0
            yield (frozenset({"B"}) if args.skip == "B" and not moved
                   and i > 400 and i % 60 in (0, 1) else frozenset())

    n = 0
    for b in buttons():
        n += 1
        o = env.step_buttons([b])
        f = np.repeat(np.repeat(o.frame_rgb, args.scale, 0), args.scale, 1)
        ff.stdin.write(np.ascontiguousarray(f).tobytes())
        pcm.append(o.audio_pcm)
    ff.stdin.close()
    ff.wait()
    wav = args.out + ".wav"
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(env.sample_rate)
        wf.writeframes(np.concatenate(pcm).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", args.out],
                   check=True)
    Path(raw).unlink()
    Path(wav).unlink()
    ram = env._env.get_ram()
    print(f"{n} frames -> {args.out}; room {int(ram[168])}, "
          f"boss {int(ram[644])}, hp {int(ram[640])}, lives {int(ram[866])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
