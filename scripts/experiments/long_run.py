"""Play one game for a long time and report how far it actually got.

The duels run 3000 frames because that is enough to compare two checkpoints.
It is not enough to answer "does it finish anything" — Double Dragon's first
mission is several minutes of walking and fighting, and 3000 frames is fifty
seconds. This runs five minutes by default, or until the run is over.

What it watches, all of it either from pixels or from the lives counter, never
from a level address that most games do not expose anyway:

- lives, so a death and a game over are actual events rather than guesses;
- accumulated camera scroll, which in a walking game is distance covered;
- scene cuts, meaning a frame that shares almost nothing with the one before —
  a new area, a transition screen, or a game over;
- a filmstrip of frames on a fixed interval, so the run can be looked at
  instead of only summarised. Every measurement mistake this project has made
  was found by looking.

    uv run python scripts/experiments/long_run.py runs/bc_dd_long --minutes 5
    uv run python scripts/experiments/long_run.py instinct --runs 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

GAME = "DoubleDragon-Nes-v0"
STATE = "default"
CUT_DIFF = 46.0        # mean abs pixel change that counts as a new scene
CUT_COOLDOWN = 30      # frames; a cut takes a few frames to finish
LIVES_SANE = 90        # anything above this is the uninitialised value


class Recorder:
    """Frames straight to disk, sound to a wav, muxed at the end.

    Frames are written as they arrive rather than collected: five minutes of
    240x256 RGB is 3.3 GB in memory and 60 MB as h264. Audio is small enough
    to hold (~30 MB) and has to be one continuous buffer for the mux anyway.
    """

    def __init__(self, path: Path, fps: float, scale: int):
        self.path, self.fps, self.scale = path, fps, scale
        self.raw = path.with_suffix(".rec.mp4")
        self.writer = None
        self.audio: list = []
        self.rate = 48000
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, frame_rgb, pcm, rate: int) -> None:
        import cv2

        h, w = frame_rgb.shape[:2]
        size = (w * self.scale, h * self.scale)
        if self.writer is None:
            # avc1 is h264 through the platform encoder; mp4v is the fallback
            self.writer = cv2.VideoWriter(
                str(self.raw), cv2.VideoWriter_fourcc(*"avc1"), self.fps, size)
            if not self.writer.isOpened():
                self.writer = cv2.VideoWriter(
                    str(self.raw), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        self.writer.write(cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                                     size, interpolation=cv2.INTER_NEAREST))
        self.audio.append(pcm)
        self.rate = rate

    def close(self) -> None:
        import subprocess
        import wave

        if self.writer is None:
            return
        self.writer.release()
        wav = self.path.with_suffix(".wav")
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.rate)
            f.writeframes(np.concatenate(self.audio).astype(np.int16).tobytes())
        ok = subprocess.call(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.raw),
             "-i", str(wav), "-c:v", "copy", "-c:a", "aac", str(self.path)]) == 0
        if ok:
            self.raw.unlink()
            wav.unlink()
        else:
            self.raw.replace(self.path)   # no ffmpeg: silent video beats none
        print(f"video: {self.path}", flush=True)


def run(agent: str, frames: int, seed: int, temperature: float,
        strip_every: int, out_dir: Path | None,
        video_out: Path | None = None, video_scale: int = 3,
        no_gate: bool = False, perception: str = "motion",
        feedback_mode: str = "strict") -> dict:
    import cv2

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.improve import VisualProgress

    env = StableRetroAdapter(GAME, include_debug=True, state=STATE)
    if agent == "instinct":
        from nes_player.perception.feedback import make_feedback
        from nes_player.policy.instinct import InstinctPolicy

        policy = InstinctPolicy(knowledge_path=f"runs/knowledge/{GAME}.json",
                                perception=perception)
        feedback = make_feedback(feedback_mode)
        policy.curiosity_needs_progress = not no_gate
        act = None
    else:
        from nes_player.policy.bc import BCPolicy

        policy = BCPolicy(agent)
        act = policy.act

    progress = VisualProgress()
    obs = env.reset(seed=seed)
    for _ in range(seed * 37):
        obs = env.step_buttons([frozenset()])

    score0, lives_now, deaths, cuts = None, None, 0, []
    game_over_at, prev_gray, cut_block = None, None, 0
    strip, best_score = [], 0
    rec = Recorder(video_out, 60.1, video_scale) if video_out else None
    for i in range(frames):
        d = obs.debug or {}
        if score0 is None and i > 200:
            score0 = d.get("score", 0)
        score = max(0, d.get("score", 0) - score0) if score0 is not None else 0
        best_score = max(best_score, score)

        lv = d.get("lives")
        if lv is not None and lv <= LIVES_SANE:
            if lives_now is not None and lv < lives_now:
                deaths += 1
                if lv < 0 and game_over_at is None:
                    game_over_at = i
            lives_now = lv

        if act is None:
            ram = env._env.get_ram() if perception == "sprites" else None
            fb = feedback.update(obs.frame_rgb, d)
            pressed, _, _ = policy.step(obs.frame_rgb, fb.score, fb.died, ram)
        else:
            policy.push_audio(obs.audio_pcm)
            pressed, _ = act(obs.frame_rgb, temperature=temperature)
        pressed = pressed - {"START", "SELECT"}
        obs = env.step_buttons([pressed])
        progress.update(obs.frame_rgb)
        if rec is not None:
            rec.add(obs.frame_rgb, obs.audio_pcm, obs.sample_rate)

        gray = cv2.cvtColor(obs.frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if prev_gray is not None:
            cut_block = max(0, cut_block - 1)
            if not cut_block and float(np.abs(gray - prev_gray).mean()) > CUT_DIFF:
                cuts.append(i)
                cut_block = CUT_COOLDOWN
        prev_gray = gray

        if out_dir is not None and i % strip_every == 0:
            strip.append((i, obs.frame_rgb.copy()))
        if game_over_at is not None and i > game_over_at + 600:
            break   # the run is over; the rest is a title screen

    env.close()
    if rec is not None:
        rec.close()
    if out_dir is not None and strip:
        out_dir.mkdir(parents=True, exist_ok=True)
        cols = min(6, len(strip))
        rows = (len(strip) + cols - 1) // cols
        h, w = strip[0][1].shape[:2]
        sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
        for k, (fi, fr) in enumerate(strip):
            r, c = divmod(k, cols)
            cv2.putText(fr, str(fi), (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = fr
        name = ("instinct" if agent == "instinct" else Path(agent).name)
        cv2.imwrite(str(out_dir / f"{name}_seed{seed}.jpg"),
                    cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])

    return {"score": best_score, "deaths": deaths, "lives_left": lives_now,
            "game_over_at": game_over_at, "frames": i + 1,
            "progress": round(progress.total, 1), "scene_cuts": len(cuts),
            "cut_frames": cuts[:12]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", help="a checkpoint directory, or the word 'instinct'")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--strip-every", type=int, default=1200)
    ap.add_argument("--frames-dir", default=None,
                    help="write a filmstrip per run so the runs can be looked at")
    ap.add_argument("--video-out", default=None,
                    help="mp4 with sound; with --runs > 1 the seed is appended")
    ap.add_argument("--video-scale", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None,
                    help="run this single seed instead of 0..runs-1")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--feedback", choices=("strict", "privileged", "pixel"),
                    default="strict",
                    help="instinct only: where 'something good/bad happened' comes from")
    ap.add_argument("--perception", choices=("motion", "sprites"), default="motion",
                    help="instinct only: where objects come from — inferred from "
                         "pixels, or read from the console's sprite table")
    ap.add_argument("--no-curiosity-gate", action="store_true",
                    help="instinct only: the pre-fix behaviour, for an ablation")
    args = ap.parse_args()

    frames = int(args.minutes * 60 * 60.1)
    out = Path(args.frames_dir) if args.frames_dir else None
    seeds = ([args.seed] if args.seed is not None
             else list(range(args.first_seed, args.first_seed + args.runs)))
    rows = []
    for seed in seeds:
        vid = None
        if args.video_out:
            vid = Path(args.video_out)
            if len(seeds) > 1:
                vid = vid.with_name(f"{vid.stem}_seed{seed}{vid.suffix}")
        r = run(args.agent, frames, seed, args.temperature, args.strip_every, out,
                vid, args.video_scale, args.no_curiosity_gate, args.perception,
                args.feedback)
        r["seed"] = seed
        rows.append(r)
        print(json.dumps(r), flush=True)

    finished = [r for r in rows if r["game_over_at"] is None]
    print()
    print(f"{args.agent}: {args.minutes:.0f} min × {len(rows)} runs")
    print(f"  survived the whole run: {len(finished)}/{len(rows)}")
    print(f"  score       {[r['score'] for r in rows]}")
    print(f"  deaths      {[r['deaths'] for r in rows]}")
    print(f"  distance    {[r['progress'] for r in rows]}")
    print(f"  scene cuts  {[r['scene_cuts'] for r in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
