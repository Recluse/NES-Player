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
    ap.add_argument("--dashboard", default="",
                    help="a clone checkpoint: render through the full dashboard "
                         "(objects, memory, the clone's attention and action "
                         "probabilities on the same frames, sounds) like the "
                         "planner's videos, instead of the bare picture")
    ap.add_argument("--label", default="replay of a search's buttons",
                    help="what the dashboard's arm line says drove the pad")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.dashboard:
        return dashboard(args)

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


def dashboard(args) -> int:
    """The same replay through the planner videos' dashboard.

    The buttons are the recording's; everything the panels draw is computed
    live from the frames — the tracker's boxes, the object memory, audio
    events, and a clone looking at the same picture (its attention map and
    what it would have pressed), which is labelled as such.
    """
    from nes_player.cli.runtime import PingLog, SoundLog, action_entropy
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.evaluation.viewer import Viewer
    from nes_player.perception.audio_events import AudioEventDetector
    from nes_player.perception.memory import ObjectMemory
    from nes_player.perception.sprites import SpriteTracker, sprite_boxes
    from nes_player.policy.bc import BCPolicy

    rec = json.loads(Path(args.recording).read_text())
    env = StableRetroAdapter(args.game, include_debug=True, state="default",
                             integration_dir=None)
    env.reset(seed=0)
    env.load_state(Path(rec["load_state"]).read_bytes())
    policy = BCPolicy(args.dashboard)
    view = Viewer(video_out=args.out, fps=60.1, title="NES Player")
    tracker, memory = SpriteTracker(), ObjectMemory()
    ears, sounds, pings = AudioEventDetector(env.sample_rate), SoundLog(), PingLog()
    seq = [frozenset(b) for b in rec["actions"]]
    room0 = None
    ranked, entropy_hist, cam, verdicts = None, [], None, None
    # no extra step before the loop: one idle frame desynced the replay
    # (it ended in room 109 with the boss alive)
    n = 0
    total = len(seq) + args.then_start
    while n < total:
        if n < len(seq):
            b = seq[n]
        else:
            if room0 is None:
                room0 = int(env._env.get_ram()[168])
            i = n - len(seq)
            moved = int(env._env.get_ram()[168]) != room0
            b = (frozenset({"B"}) if args.skip == "B" and not moved and i > 400
                 and i % 60 in (0, 1) else frozenset())
        obs = env.step_buttons([b])
        slots = tracker.update(obs.frame_rgb, b,
                               boxes=sprite_boxes(env._env.get_ram()))
        d0 = obs.debug or {}
        verdicts = memory.update(obs.frame_rgb, slots, n,
                                 int(d0.get("score", 0) or 0), False)
        if n % 4 == 0:
            _, ranked = policy.act(obs.frame_rgb, 1.0)
            entropy_hist.append(action_entropy(ranked))
            del entropy_hist[:-500]
            cam = policy.compute_cam(obs.frame_rgb)
        for ev in ears.push(obs.audio_pcm, n):
            sounds.add(ev.cluster_id, ears.clusters[ev.cluster_id].heard)
        sounds.tick()
        pings.tick()
        ram = env._env.get_ram()
        pressed_now = "+".join(sorted(b)) or "-"
        view.show(obs, (b,), info={
            "arm": args.label, "room": str(int(ram[168])),
            "hero P": f"{int(ram[640])}/20", "boss E": f"{int(ram[644])}/20",
            "frame": f"{n:,}"},
            thoughts=[f"pad: {pressed_now}",
                      "buttons: the search's recording",
                      "attention + probabilities:",
                      "  the clone, watching the same frames"],
            action_probs=ranked, slots=slots, verdicts=verdicts, heatmap=cam,
            entropy_hist=entropy_hist, features=policy.last_features,
            gallery=[(c.proto, c.verdict, c.cluster_id, c.seen)
                     for c in sorted(memory.clusters, key=lambda c: -c.seen)[:8]],
            audio_events=[(e[0], e[1], ears.clusters[e[0]].verdict)
                          for e in sounds.events],
            sound_pings=pings.pings)
        n += 1
    view.close()
    ram = env._env.get_ram()
    print(f"{n} frames -> {args.out}; room {int(ram[168])}, "
          f"boss {int(ram[644])}, hp {int(ram[640])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
