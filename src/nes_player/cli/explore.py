"""`nes-player explore` — the instinct policy plays by itself (spec §10).

This is where training data comes from. No demonstrations, no TAS movies: the
instinct policy calibrates the controls in-game, hunts for progress, jumps when
stuck and pokes at unfamiliar objects. Headless it runs at about a thousand
frames per second, so a few hundred episodes take minutes, and behavioural
cloning learns from those.

Optionally a trained network can ride along as an `--observer`: it watches
without touching the controls, and its attention map and action probabilities
fill the side panel — "what would the network have pressed here".
"""

import argparse
import threading
import time

from nes_player.cli.runtime import (
    UNPAUSE_FRAMES,
    PauseWatchdog,
    SoundLog,
    Thoughts,
    action_entropy,
)
from nes_player.perception.title import TitleTracker

OBSERVER_HZ = 10
FROZEN_FRAMES = 45           # identical frames that count as a pause
CTRL_PROB = 0.7              # confidence above which a tracked object is "me"
PROMPT_EVERY = 60            # frames between screen-prompt reads
DEATH_SOUND_LOOKBACK = 140   # the death jingle precedes the RAM lives counter
REWARD_SOUND_LOOKBACK = 40
START_PULSES = (60, 61, 240, 241, 420, 421, 600, 601, 780, 781)


class Observer:
    """A BC policy watching the instincts play, on its own thread.

    Same reasoning as the brain thread in `play`: neural work never runs inside
    the game loop, because its spikes tear the audio.
    """

    def __init__(self, checkpoint: str) -> None:
        from nes_player.policy.bc import BCPolicy

        self.policy = BCPolicy(checkpoint)
        self.frame = None
        self.ranked: list = []
        self.cam = None
        self._stop = False
        self._fails = 0
        self.last_error: str | None = None

    def start(self) -> None:
        threading.Thread(target=self._work, daemon=True).start()

    def stop(self) -> None:
        self._stop = True

    def push_audio(self, pcm) -> None:
        self.policy.push_audio(pcm)

    @property
    def last_features(self):
        return self.policy.last_features

    def _work(self) -> None:
        while not self._stop:
            t0 = time.monotonic()
            frame = self.frame
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                _, self.ranked, self.cam = self.policy.act(frame, with_cam=True)
                self._fails = 0
            except Exception as e:
                # Same reasoning as the policy thread in play.py: keep going,
                # but say so. A permanent failure here looks exactly like an
                # observer that has nothing to show.
                self._fails += 1
                self.last_error = f"{type(e).__name__}: {e}"
                if self._fails in (1, 10, 100) or self._fails % 1000 == 0:
                    print(f"observer failed {self._fails}x: {self.last_error}",
                          flush=True)
                time.sleep(0.05)
            time.sleep(max(0.0, 1 / OBSERVER_HZ - (time.monotonic() - t0)))


def cmd_explore(args: argparse.Namespace) -> None:
    from nes_player.data.writer import EpisodeWriter
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.evaluation.viewer import Viewer, run_with_gui
    from nes_player.perception.audio_events import AudioEventDetector
    from nes_player.perception.feedback import make_feedback
    from nes_player.policy.instinct import InstinctPolicy

    env = StableRetroAdapter(args.game, integration_dir=args.integrations,
                             include_debug=True, state=args.state, core=args.core)
    viewer = Viewer(window=args.window, video_out=args.video_out, throttle=args.realtime,
                    title=f"{args.game} | instinct explorer", scale=1.5 if args.hd else 1.0)
    policy = InstinctPolicy(knowledge_path=f"runs/knowledge/{args.game}.json")
    ears = AudioEventDetector(env.sample_rate)
    sounds = SoundLog()
    max_frames = args.max_frames or (10**12 if args.loop else 3600)

    observer = Observer(args.observer) if args.observer else None
    if observer:
        observer.start()

    def _loop() -> None:
        episode = 0
        while True:
            episode += 1
            thoughts = Thoughts()
            thoughts.add(f"episode {episode}")
            obs = env.reset(seed=0)
            policy.reset()
            writer = None
            if args.record:
                from pathlib import Path

                out = Path(args.record) / f"{args.game}_ep{episode:03d}"
                writer = EpisodeWriter(out_dir=out, metadata={
                    "game": args.game, "source": "instinct-explore",
                    "sample_rate": env.sample_rate})
            pause = PauseWatchdog(FROZEN_FRAMES)
            title = TitleTracker()
            game_over = False
            prev_score = 0
            feedback = make_feedback(args.feedback)
            last_reason = ""
            unpause_left = 0
            title_press_at = -1000
            screen_prompt: str | None = None
            entropy_hist: list[float] = []
            last_entropy_ranked: list = []
            restart = quitting = False

            for i in range(max_frames):
                # Two separate readings of the same frame, and the difference
                # matters. `feedback` is an INPUT: it tells the policy that
                # something good or bad happened, and those labels pick buttons.
                # Its source is chosen by --feedback and is `strict` by default.
                # `d` below is for the display and the run's own bookkeeping —
                # game over, the printed line — and never reaches the policy.
                d = obs.debug or {}
                fb = feedback.update(obs.frame_rgb, d)
                lives = d.get("lives", 0)
                game_over = lives is not None and lives < 0

                pressed, slots, verdicts = policy.step(obs.frame_rgb, fb.score, fb.died)
                in_game = any(s.ctrl_prob > CTRL_PROB for s in slots)

                if i % PROMPT_EVERY == 0:
                    from nes_player.perception.text import find_prompt

                    screen_prompt = find_prompt(obs.frame_rgb)
                    if screen_prompt:
                        thoughts.add(f"f{i}: screen says PRESS {screen_prompt}")

                # Obey the screen when it names a button, but only outside the
                # game: in a two-player game that prompt belongs to player two,
                # and START during play is the pause button.
                if screen_prompt and i % 24 < 2 and not in_game:
                    pressed = frozenset({screen_prompt})
                elif (args.state is None and i in START_PULSES
                        or (game_over and i % 180 < 2) or i - title_press_at < 2):
                    pressed = frozenset({"START"})
                if unpause_left > 0:
                    pressed, unpause_left = frozenset({"START"}), unpause_left - 1
                human = viewer.human_buttons()
                if human:
                    pressed = human

                obs = env.step_buttons([pressed])
                if writer is not None:
                    writer.append(obs, (pressed,))

                if pause.push(obs.frame_rgb):
                    unpause_left = UNPAUSE_FRAMES
                    thoughts.add(f"f{i}: screen frozen — pressing START")

                for ev in ears.push(obs.audio_pcm, i):
                    sounds.add(ev.cluster_id)
                    if ev.is_new:
                        thoughts.add(f"f{i}: heard NEW sound #{ev.cluster_id}")
                sounds.tick()
                if len(entropy_hist) > 2000:
                    del entropy_hist[:-500]

                if fb.died:
                    for cid in ears.attribute(sounds.within(DEATH_SOUND_LOOKBACK), "death"):
                        thoughts.add(f"f{i}: sound #{cid} = DANGER (death jingle)")
                if fb.score > prev_score:
                    for cid in ears.attribute(sounds.within(REWARD_SOUND_LOOKBACK), "reward"):
                        thoughts.add(f"f{i}: sound #{cid} = REWARD (score sound)")
                prev_score = fb.score

                if title.step(obs.frame_rgb, i, in_game=in_game,
                              from_power_on=args.state is None):
                    title_press_at = i
                    thoughts.add(f"f{i}: title screen again — pressing START")

                if policy.last_reason != last_reason:
                    last_reason = policy.last_reason
                    thoughts.add(f"f{i}: {last_reason}")

                if args.window or args.video_out:
                    if observer is not None:
                        observer.frame = obs.frame_rgb
                        observer.push_audio(obs.audio_pcm)
                        ranked = observer.ranked
                        if ranked and ranked is not last_entropy_ranked:
                            last_entropy_ranked = ranked
                            entropy_hist.append(action_entropy(ranked))
                    cmd = viewer.show(
                        obs, (pressed,),
                        info={"game": args.game, "mode": f"instinct/{policy.mode}",
                              "episode": str(episode), "fps": viewer.fps_info(),
                              "score": str(fb.score),
                              **({"observer": "BC watches"} if observer else {})},
                        thoughts=thoughts + policy.knowledge.lines(),
                        slots=slots, verdicts=verdicts,
                        action_probs=observer.ranked if observer else None,
                        entropy_hist=entropy_hist if observer else None,
                        heatmap=observer.cam if observer and viewer.show_cam else None,
                        features=(observer.last_features
                                  if observer and viewer.show_cam else None),
                        gallery=[(c.proto, c.verdict, c.cluster_id, c.seen)
                                 for c in sorted(policy.memory.clusters,
                                                 key=lambda c: -c.seen)[:8]],
                        audio_events=[(e[0], e[1], ears.clusters[e[0]].verdict)
                                      for e in sounds.events],
                    )
                    if cmd == "quit":
                        # Break rather than return: the frames recorded so far
                        # are real data and deserve to be finished properly.
                        # Returning here used to skip close() entirely, leaving
                        # a directory with no actions and no metadata that the
                        # next training run happily picked up.
                        quitting = True
                        break
                    if cmd == "restart":
                        restart = True
                        break

            if writer is not None:
                try:
                    writer.close()
                    print(f"recorded: {writer.out_dir}")
                except ValueError:
                    pass          # empty episode, nothing to write
            print(f"episode={episode} debug={obs.debug or {}}")
            if quitting or not (args.loop or restart):
                break

    try:
        run_with_gui(viewer, _loop)
    finally:
        if observer:
            observer.stop()
        viewer.close()
        env.close()
