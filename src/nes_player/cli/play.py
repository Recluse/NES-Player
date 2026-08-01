"""`nes-player play` — run a trained behavioural-cloning policy (spec §19).

Threading matters here. The game loop must hold 60 fps with clean audio, and a
neural forward pass is far too spiky to sit inside it, so the loop is pure CPU
(~4 ms/frame) and everything on the GPU — the policy, Grad-CAM, the ghost world
model — runs in a separate "brain" thread at about 15 Hz. Audio output owns a
third thread. Merging any two of them tears the sound.
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from nes_player.cli.runtime import (
    UNPAUSE_FRAMES,
    PauseWatchdog,
    PingLog,
    SoundLog,
    Thoughts,
    action_entropy,
)
from nes_player.perception.title import TitleTracker

BRAIN_HZ = 15                # policy decisions per second
HUD_FIT_FRAMES = 240         # frames collected before the digit reader trains
PROMPT_EVERY = 15            # brain ticks between "does the screen ask for a button?"
FROZEN_FRAMES = 90           # identical frames that count as a pause
CTRL_PROB = 0.7              # confidence above which a tracked object is "me"
DEATH_TAG_WINDOW = 90        # frames after a death in which sounds mean "danger"
REWARD_TAG_WINDOW = 20       # ...and after a score bump, "reward"
DEATH_SOUND_LOOKBACK = 140   # the death jingle precedes the RAM lives counter
START_PULSES = (60, 61, 240, 241, 420, 421, 600, 601, 780, 781)


def _resolve_jump_hold(args: argparse.Namespace) -> int:
    """How many frames to hold A for a jump.

    Taken from the knowledge base that `explore` calibrates in-game, with a
    floor: the measured height is noisy, and on Super Mario Bros. a 32-frame
    hold reaches x=2136 against 643 for no hold at all.
    """
    if args.jump_hold is not None:
        return args.jump_hold
    # ponytail: floor 32 — refine once hold duration becomes part of training
    jump_hold = 32
    kpath = Path(f"runs/knowledge/{args.game}.json")
    if kpath.exists():
        import json

        heights = json.loads(kpath.read_text()).get("jump_height", {})
        if heights:
            jump_hold = max(int(max(heights, key=lambda k: heights[k])), 32)
    return jump_hold


class JumpShaper:
    """Turns the policy's momentary "A" into a held jump of a fixed length.

    The policy decides at 15 Hz and cannot express duration, but jump height on
    the NES is duration. Two rules keep this honest:

    * the hold starts on the *rising edge* of the model's A, never while it is
      already held — otherwise the hold restarts itself and A is never released;
    * a mandatory release window follows every hold, so the next jump is a new
      press rather than a continuation of the last one.

    A press therefore lasts `hold + 1` frames: the edge frame plus the hold.
    """

    RELEASE_FRAMES = 8

    def __init__(self, hold: int) -> None:
        self.hold = hold
        self._hold_left = 0
        self._release_left = 0
        self._prev_a = False

    def apply(self, pressed: frozenset[str]) -> frozenset[str]:
        model_a = "A" in pressed
        if self._release_left > 0:
            self._release_left -= 1
            out = pressed - {"A"}
        elif self._hold_left > 0:
            self._hold_left -= 1
            out = pressed | {"A"}
            if self._hold_left == 0:
                self._release_left = self.RELEASE_FRAMES
        elif model_a and not self._prev_a:
            self._hold_left = self.hold
            out = pressed | {"A"}
        else:
            out = pressed - {"A"}
        self._prev_a = model_a
        return out


class Brain:
    """The GPU half of the agent, running off the game loop.

    The game thread writes `frame` and reads the latest decision; nothing waits
    on anything. A dropped or repeated decision is invisible at 60 fps, whereas
    a blocked game thread is audible immediately.
    """

    def __init__(self, policy, viewer, args, hud, ghost_model) -> None:
        self.policy, self.viewer, self.args = policy, viewer, args
        self.hud, self.ghost_model = hud, ghost_model
        self.frame = None
        self.pressed: frozenset[str] = frozenset()
        self.ranked: list = []
        self.cam = None
        self.ego = None
        self.ghost = None
        self.prompt: str | None = None
        self.seq = 0
        self.hud_buf: list = []
        self._ticks = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def _work(self) -> None:
        while not self._stop:
            t0 = time.monotonic()
            frame = self.frame
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                self._think(frame)
            except Exception:
                # A transient failure here must not kill the run: the game loop
                # keeps playing with the previous decision.
                time.sleep(0.05)
            time.sleep(max(0.0, 1 / BRAIN_HZ - (time.monotonic() - t0)))

    def _think(self, frame) -> None:
        self.pressed, self.ranked = self.policy.act(frame, temperature=self.args.temperature)
        self.seq += 1
        if self.args.cam and self.viewer.show_cam:
            self.cam = self.policy.compute_cam(frame)
        # The digit reader trains once per run, here rather than in the game loop.
        if not self.hud.groups and len(self.hud_buf) >= HUD_FIT_FRAMES:
            self.hud.fit(self.hud_buf[:HUD_FIT_FRAMES])
            self.hud_buf.clear()
        # Roughly once a second: is the screen asking for a specific button?
        # Reading it beats pulsing START blindly.
        self._ticks += 1
        if self._ticks % PROMPT_EVERY == 0:
            from nes_player.perception.text import find_prompt

            self.prompt = find_prompt(frame, self.hud)
        if self.ghost_model is not None and self.ego is not None:
            frame_e, cx, cy, vx, vy, mask = self.ego
            self.ghost = self.ghost_model.predict(frame_e, cx, cy, (vx, vy), mask, steps=12)


def _auto_start_buttons(args, brain, i, in_game, game_over,
                        pressed: frozenset[str], thoughts: Thoughts) -> frozenset[str]:
    """Decide what to press while the game is still in its menus.

    Three rules, in order of trust: obey the screen if it names a button; fall
    back to a series of START pulses for games whose intro swallows the first
    one; and never press START once we are actually playing, because there it
    is the pause button. That last clause is what a two-player game like Double
    Dragon needs — its "PRESS START" prompt is addressed to player two and sits
    on screen for the whole level.
    """
    if brain.prompt and not in_game and i % 24 < 2:
        if i % 120 < 2:
            thoughts.add(f"f{i}: screen says PRESS {brain.prompt} — pressing")
        pressed = frozenset({brain.prompt})
    elif args.state is None and not in_game and i in START_PULSES:
        # Games with an intro swallow the first START, so send a series of them.
        # A stray pause is cleared by the frozen-screen watchdog.
        pressed = frozenset({"START"})
    else:
        pressed = pressed - {"START", "SELECT"}
    # Game over (lives below zero) means we are back in the menus.
    if game_over and i % 180 < 2 and not in_game:
        pressed = frozenset({"START"})
    return pressed


def cmd_play(args: argparse.Namespace) -> None:
    from nes_player.emulator.controller import BUTTONS
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.evaluation.viewer import Viewer, run_with_gui
    from nes_player.perception.audio_events import AudioEventDetector
    from nes_player.perception.memory import ObjectMemory
    from nes_player.perception.motion import MotionTracker
    from nes_player.perception.text import HudReader
    from nes_player.policy.bc import BCPolicy

    policy = BCPolicy(args.checkpoint)
    env = StableRetroAdapter(args.game, integration_dir=args.integrations,
                             include_debug=True, state=args.state, core=args.core)
    viewer = Viewer(window=args.window, video_out=args.video_out, throttle=args.realtime,
                    title=f"{args.game} | BC policy", scale=1.5 if args.hd else 1.0)
    tracker = MotionTracker()
    memory = ObjectMemory()          # outlives episodes: object knowledge accrues all stream
    ears = AudioEventDetector(env.sample_rate)   # likewise
    hud = HudReader()
    sounds = SoundLog()
    pings = PingLog()
    jump = JumpShaper(_resolve_jump_hold(args))
    max_frames = args.max_frames or (10**12 if args.loop else 3600)

    ghost_model, planner = None, None
    if args.ghost and Path(args.ghost).exists():
        from nes_player.world_model.ego import GhostPredictor

        ghost_model = GhostPredictor(args.ghost)
        if args.planner:
            from nes_player.policy.planner import EgoPlanner

            planner = EgoPlanner(ghost_model)

    locator = None
    if args.sound_loc and Path(args.sound_loc).exists():
        from nes_player.perception.av_align import SoundLocator

        locator = SoundLocator(args.sound_loc)

    brain = Brain(policy, viewer, args, hud, ghost_model)
    brain.start()

    def _loop() -> None:
        episode = 0
        while True:
            episode += 1
            thoughts = Thoughts()
            thoughts.add(f"episode {episode}")
            obs = env.reset(seed=0)
            policy.reset()
            pause = PauseWatchdog(FROZEN_FRAMES)
            title = TitleTracker()
            ranked: list = []
            score0, lives_prev, prev_score = None, None, 0
            death_tag_until, reward_tag_until = -1, -1
            unpause_left, game_over = 0, False
            entropy_hist: list[float] = []
            curiosity_plan: list[tuple[frozenset, int]] = []
            curiosity_cooldown = 0
            seen_seq = -1
            plan_pressed = None
            best_x = 0
            slots: list = []   # from the previous frame; needed before tracker.update
            restart = False

            for i in range(max_frames):
                brain.frame = obs.frame_rgb
                policy.push_audio(obs.audio_pcm)   # the AV model hears; video-only is a no-op
                if locator:
                    locator.push_audio(obs.audio_pcm)

                for ev in ears.push(obs.audio_pcm, i):
                    sounds.add(ev.cluster_id, ears.clusters[ev.cluster_id].heard)
                    if locator:
                        sm = locator.sound_map(obs.frame_rgb)
                        py_, px_ = np.unravel_index(int(sm.argmax()), sm.shape)
                        pings.add((px_ + 0.5) / sm.shape[1], (py_ + 0.5) / sm.shape[0])
                    if ev.is_new:
                        thoughts.add(f"f{i}: heard NEW sound #{ev.cluster_id}")
                    # Meaning of sounds: the death and score jingles play *after*
                    # their event, so tag whatever arrives inside a forward window.
                    if i <= death_tag_until:
                        for cid in ears.attribute([ev.cluster_id], "death"):
                            thoughts.add(f"f{i}: sound #{cid} = DANGER (death jingle)")
                    elif i <= reward_tag_until:
                        for cid in ears.attribute([ev.cluster_id], "reward"):
                            thoughts.add(f"f{i}: sound #{cid} = REWARD (score sound)")
                sounds.tick()
                pings.tick()

                if (not hud.groups and i > 120 and i % 4 == 0
                        and len(brain.hud_buf) < HUD_FIT_FRAMES):
                    brain.hud_buf.append(obs.frame_rgb.copy())
                if len(entropy_hist) > 2000:
                    del entropy_hist[:-500]

                ranked = brain.ranked
                if brain.seq != seen_seq and ranked:
                    seen_seq = brain.seq
                    entropy_hist.append(action_entropy(ranked))

                pressed = jump.apply(brain.pressed)
                if plan_pressed is not None:      # MPC overrides the policy, A shaping included
                    pressed = plan_pressed
                if curiosity_plan:                # ...and curiosity overrides everything
                    buttons, left = curiosity_plan[0]
                    pressed = buttons
                    curiosity_plan[0] = (buttons, left - 1)
                    if left <= 1:
                        curiosity_plan.pop(0)

                in_game = any(s.ctrl_prob > CTRL_PROB for s in slots)
                if args.auto_start:
                    pressed = _auto_start_buttons(args, brain, i, in_game, game_over,
                                                  pressed, thoughts)
                if unpause_left > 0:
                    pressed, unpause_left = frozenset({"START"}), unpause_left - 1
                human = viewer.human_buttons()    # a click on the drawn pad wins over everything
                if human:
                    pressed = human
                    if i % 12 == 0:
                        thoughts.add(f"f{i}: human hint {'+'.join(sorted(human))}")

                obs = env.step_buttons([pressed])

                if args.auto_start and title.step(obs.frame_rgb, i, in_game=in_game,
                                                  from_power_on=args.state is None):
                    unpause_left = UNPAUSE_FRAMES
                    thoughts.add(f"f{i}: title screen again — pressing START")
                if pause.push(obs.frame_rgb):
                    unpause_left = UNPAUSE_FRAMES
                    thoughts.add(f"f{i}: screen frozen — looks like pause, pressing START")

                # Emulator memory is read here for *evaluation only* — the policy
                # above never sees it (spec §3).
                d = obs.debug or {}
                x_now = d.get("xscroll", d.get("xscrollHi", 0) * 256 + d.get("xscrollLo", 0))
                if x_now < 6000:
                    best_x = max(best_x, x_now)
                if score0 is None and i > 100:
                    score0 = d.get("score", 0)
                score = max(0, d.get("score", 0) - score0) if score0 is not None else 0
                died = lives_prev is not None and d.get("lives", 0) < lives_prev
                lives = d.get("lives", 0)
                was_over, game_over = game_over, lives is not None and lives < 0
                if game_over and not was_over:
                    thoughts.add(f"f{i}: GAME OVER — will re-enter via START")
                if was_over and not game_over:
                    score0 = None                 # new game, new score baseline
                lives_prev = lives

                slots = tracker.update(obs.frame_rgb, pressed)
                verdicts = memory.update(obs.frame_rgb, slots, i, score, died)
                top = [s for s in slots if s.ctrl_prob > CTRL_PROB]

                if planner is not None and i % args.repeat == 0:
                    if top:
                        # Only CONFIRMED threats: phantom ones (clouds, blocks)
                        # made the planner flinch and lose ground.
                        dangers = [(sl.cx, sl.cy, sl.vx, sl.vy) for sl in slots
                                   if sl is not top[0]
                                   and verdicts.get(sl.slot_id) == "danger"]
                        pl = planner.plan(obs.frame_rgb, top[0], dangers)
                        plan_pressed = pl.pressed
                        if i % 28 == 0:
                            thoughts.add(f"f{i}: plan '{pl.name}' score={pl.score:.0f}")
                    else:
                        plan_pressed = None       # hero not visible — let BC drive
                if top:
                    t0s = top[0]
                    mask = sum(1 << k for k, b in enumerate(BUTTONS) if b in pressed)
                    brain.ego = (obs.frame_rgb, t0s.cx, t0s.cy, t0s.vx, t0s.vy, mask)

                # Curiosity: an unknown object overhead is worth head-butting.
                curiosity_cooldown -= 1
                if top and not curiosity_plan and curiosity_cooldown <= 0:
                    ctrl = top[0]
                    for s in slots:
                        if s is ctrl or verdicts.get(s.slot_id) != "unknown" or s.missed > 0:
                            continue
                        dx, dy = s.cx - ctrl.cx, ctrl.cy - s.cy
                        # Ahead and above: jump early, running inertia carries us there.
                        if 10 < dx < 55 and 12 < dy < 90:
                            curiosity_plan = [(frozenset({"A"}), jump.hold), (frozenset(), 10)]
                            curiosity_cooldown = 120
                            thoughts.add(f"f{i}: curious about #{s.slot_id} overhead — jumping")
                            break

                if died:
                    thoughts.add(f"f{i}: DIED — memorizing the culprit")
                    death_tag_until = i + DEATH_TAG_WINDOW
                    for cid in ears.attribute(sounds.within(DEATH_SOUND_LOOKBACK), "death"):
                        thoughts.add(f"f{i}: sound #{cid} = DANGER (death jingle)")
                elif i % 30 == 0:
                    t = f"f{i}: {ranked[0][0]} p={ranked[0][1]:.2f}" if ranked else f"f{i}"
                    if top:
                        t += f" | ctrl #{top[0].slot_id}"
                    thoughts.add(t)
                if score > prev_score:
                    reward_tag_until = i + REWARD_TAG_WINDOW
                prev_score = score

                if args.window or args.video_out:
                    cam = brain.cam if args.cam and viewer.show_cam else None
                    cmd = viewer.show(
                        obs, (pressed,),
                        info={"game": args.game, "mode": "BC policy",
                              "episode": str(episode), "fps": viewer.fps_info(),
                              "temp": str(args.temperature), "score": str(score),
                              "scroll dx": f"{tracker.scroll_dx:+.1f}",
                              "hud read": (" ".join(str(v) for v in hud.read(obs.frame_rgb))
                                           if hud.groups else "learning digits...")},
                        thoughts=thoughts + memory.summary(4), action_probs=ranked,
                        slots=slots, verdicts=verdicts,
                        heatmap=cam, entropy_hist=entropy_hist,
                        features=policy.last_features if viewer.show_cam else None,
                        gallery=[(c.proto, c.verdict, c.cluster_id, c.seen)
                                 for c in sorted(memory.clusters, key=lambda c: -c.seen)[:8]],
                        audio_events=[(e[0], e[1], ears.clusters[e[0]].verdict)
                                      for e in sounds.events],
                        ghost=brain.ghost if top else None,
                        sound_pings=pings.pings,
                    )
                    if cmd == "quit":
                        print(f"episode={episode} stopped by user")
                        return
                    if cmd == "restart":
                        restart = True
                        break

            print(f"episode={episode} best_x={best_x} debug={obs.debug or {}}")
            if not (args.loop or restart):
                break

    try:
        run_with_gui(viewer, _loop)
    finally:
        brain.stop()
        viewer.close()
        env.close()
