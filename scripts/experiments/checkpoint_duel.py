"""Compare Double Dragon checkpoints by playing, not by validation accuracy.

Validation accuracy answers "how predictable is the policy this model cloned",
which is a different question from "how well does it play". A policy that stands
in a corner repeating one manoeuvre is trivially predictable and scores well; a
policy that reacts to where the enemies are does not.

So the checkpoints are made to play. Both arms run in one process on the same
seeds, each seed starting the model at a different point in the level, since
these models are near-deterministic at low temperature.

    uv run python scripts/experiments/dd_checkpoint_duel.py \
        runs/bc_dd_attn2 runs/bc_dd_attn3 --runs 8
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

GAME = "DoubleDragon-Nes-v0"
STATE = "default"
WARMUP_FRAMES = 1200   # budget for getting past an intro before giving up


def run(checkpoint: str, frames: int, seed: int, temperature: float,
        game: str = GAME, state: str | None = STATE,
        integrations: str | None = None, start_pulses: int = 0) -> dict:
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.improve import VisualProgress

    env = StableRetroAdapter(game, include_debug=True, state=state,
                             integration_dir=integrations)
    policy = BCPolicy(checkpoint)
    progress = VisualProgress()
    obs = env.reset(seed=seed)
    for _ in range(seed * 37):
        obs = env.step_buttons([frozenset()])
    # Get into the game, and confirm that we did. Without this the agent sits
    # through the attract demo, which on Gradius plays itself: the screen
    # scrolls, the metric fills with numbers, and none of them belong to any
    # model. Pressing START a fixed number of times is not enough either —
    # depending on where the intro is, the press lands on nothing, and the run
    # silently becomes another demo recording that scores "survived the whole
    # episode" because it never played.
    started = start_pulses == 0
    if start_pulses:
        watch = VisualProgress()
        for j in range(WARMUP_FRAMES):
            obs = env.step_buttons([frozenset({"START"}) if j % 45 < 3 else frozenset()])
            watch.update(obs.frame_rgb)
            lv = (obs.debug or {}).get("lives")
            if abs(watch.total) > 12 or (lv is not None and 0 < lv <= 9):
                started = True
                break
    score0, attacks = None, 0
    lives0, lives_now, deaths, survived = None, None, 0, frames
    for i in range(frames):
        d = obs.debug or {}
        if score0 is None and i > 100:
            score0 = d.get("score", 0)
        score = max(0, d.get("score", 0) - score0) if score0 is not None else 0
        # Lives, where the map has them, is the metric that means something in a
        # game whose screen scrolls by itself: camera scroll there measures the
        # passage of time and nothing about the player.
        lv = d.get("lives")
        if lv is not None and 0 <= lv <= 90:
            if lives0 is None and i > 200:
                lives0 = lv
            if lives_now is not None and lv < lives_now:
                deaths += 1
                if deaths == 1:
                    survived = i
            lives_now = lv
        policy.push_audio(obs.audio_pcm)
        pressed, _ = policy.act(obs.frame_rgb, temperature=temperature)
        pressed = pressed - {"START", "SELECT"}   # START pauses; the menus are behind us
        attacks += "B" in pressed
        obs = env.step_buttons([pressed])
        progress.update(obs.frame_rgb)
    env.close()
    return {"score": score, "progress": round(progress.total, 1), "attacks": attacks,
            "deaths": deaths, "survived": survived, "started": started}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--integrations", default=None)
    ap.add_argument("--start-pulses", type=int, default=0,
                    help="START presses before handing over; games booted from "
                         "power-on need these to leave the title screen")
    args = ap.parse_args()

    out: dict[str, list[dict]] = {c: [] for c in args.checkpoints}
    for seed in range(args.runs):
        for ckpt in args.checkpoints:
            r = run(ckpt, args.frames, seed, args.temperature,
                    args.game, args.state or None, args.integrations,
                    args.start_pulses)
            r["seed"] = seed
            out[ckpt].append(r)
            print(json.dumps({"checkpoint": ckpt, **r}), flush=True)

    print()
    # A seed where any arm failed to start the game is not a comparison; it is
    # three recordings of a title screen. Drop the seed for everyone, so the
    # arms stay paired.
    bad = {r["seed"] for rs in out.values() for r in rs if not r["started"]}
    if bad:
        print(f"dropping {len(bad)} seed(s) where the game never started: {sorted(bad)}")
        out = {k: [r for r in rs if r["seed"] not in bad] for k, rs in out.items()}
        if not any(out.values()):
            print("no usable runs")
            return 1
    base = out[args.checkpoints[0]]
    # Pick the metric the game actually reports: a shmup scrolls by itself, so
    # its "progress" is a clock, and only survival says anything.
    key = "survived" if all(r["score"] == 0 for rs in out.values() for r in rs) else "score"
    print(f"comparing on: {key}")
    for ckpt, rows in out.items():
        vals = [r[key] for r in rows]
        diffs = [r[key] - b[key] for r, b in zip(rows, base, strict=True)]
        spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{Path(ckpt).name:16} {key} {statistics.mean(vals):7.1f} ±{spread:6.1f}  "
              f"deaths {statistics.mean(r['deaths'] for r in rows):4.1f}  "
              f"progress {statistics.mean(r['progress'] for r in rows):7.1f}  "
              f"attack frames {statistics.mean(r['attacks'] for r in rows):5.0f}  "
              f"vs base {statistics.mean(diffs):+7.1f} "
              f"({sum(d > 0 for d in diffs)}W/{sum(d < 0 for d in diffs)}L)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
