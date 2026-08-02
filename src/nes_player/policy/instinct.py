"""The instinct policy: calibrate the controls, then explore (spec §10.10).

No neural network and no emulator memory — pixels only, through the motion
tracker and the object memory. This is what produces training data for games
that have no demonstrations, and it is also the agent that turns out to play an
unfamiliar game better than a transferred network does.

CALIBRATE runs a probe protocol: running speed under RIGHT and B+RIGHT, then
jump height for A held 4, 12, 24 and 32 frames. The result is a per-game
knowledge file.

EXPLORE is a handful of rules:

- run in the direction that makes progress (B+RIGHT);
- stuck — the scroll stopped while RIGHT is held — jump with the calibrated
  hold; after three failures, back off and jump with a run-up;
- curiosity: an unfamiliar object nearby, including a flashing block overhead,
  is worth walking to and jumping into;
- an object already known to be dangerous is closing in, so jump over it.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nes_player.emulator.controller import resolve_conflicts
from nes_player.perception.memory import ObjectMemory
from nes_player.perception.motion import MotionTracker, Slot

NOOP: frozenset[str] = frozenset()
RIGHT = frozenset({"RIGHT"})
RUN = frozenset({"B", "RIGHT"})
JUMP_RUN = frozenset({"A", "B", "RIGHT"})
LEFT = frozenset({"LEFT"})

# Beat-em-ups: when an enemy walks up, turn to face it and hit, rather than
# running past to the right. The vertical threshold is narrow — you can only
# land a hit on something in your own lane.
ENGAGE_DX = 44        # horizontal distance at which a fight starts
ENGAGE_DY = 20        # vertical spread that counts as the same lane
HIT_DX = 26           # closer than this and a strike connects
SURROUNDED_DX = 52    # enemies on both sides within this: retreat
BACK_OFF = 24         # frames of retreat

# Damage ACCUMULATES, so an enemy has to be finished rather than poked once.
# Measured on Double Dragon by score over 3000 frames, the crude approach won:
#   strike the nearest continuously          124
#   + lock the target for 150 frames         110  (sticks to one knocked away)
#   + press/release rhythm                   110  (loses frames to the gaps)
#   original (one strike, then move on)       92
# Target locking is left switched off rather than deleted: on games with larger
# sprites, where the tracker does not merge the two fighters, it may well win.
TARGET_HOLD = 1       # frames to keep a target; >1 enables locking
TARGET_KEEP_DX = 70   # a locked target stays ours until it gets this far away
ATTACK_PRESS = 5      # frames holding the attack button...
ATTACK_RELEASE = 0    # ...and releasing it; 0 means hit continuously
FACING_MIN_DX = 10    # closer than this the blobs merge and the sign of dx lies

# The fight is TWO-dimensional: until you stand at the enemy's depth your
# strikes pass through empty air. Enemies are sought in a wide band and
# approached vertically; strikes only happen once aligned.
APPROACH_DY = 64      # within this band an enemy is reachable on foot

# Screen column below which the hero has nowhere left to retreat to. Walking
# into a wall produces no scroll, which the stuck detector reads as being stuck,
# which triggers another retreat into the same wall.
LEFT_EDGE = 28
WALL_EVIDENCE = 15    # frames of fruitless LEFT before concluding there is a wall

WAIT_MIN = 120        # minimum frames to wait for gameplay to begin
WAIT_MAX = 1800       # fuse: past this, calibrate with whatever we have

# (name, buttons, frames) — the calibration protocol
CALIBRATION_STEPS = [
    ("settle", NOOP, 40),
    ("run:RIGHT", RIGHT, 60),
    ("settle", NOOP, 30),
    ("run:B+RIGHT", RUN, 60),
    ("settle", NOOP, 30),
    ("jump:4", frozenset({"A"}), 4), ("land:4", NOOP, 55),
    ("jump:12", frozenset({"A"}), 12), ("land:12", NOOP, 55),
    ("jump:24", frozenset({"A"}), 24), ("land:24", NOOP, 55),
    ("jump:32", frozenset({"A"}), 32), ("land:32", NOOP, 55),
]


@dataclass
class Knowledge:
    run_speed: dict[str, float] = field(default_factory=dict)    # buttons -> px/frame
    jump_height: dict[str, float] = field(default_factory=dict)  # A hold -> px

    def best_jump_hold(self) -> int:
        # A physical prior beats the measurement here: on the NES a longer hold
        # always jumps higher, while the measured heights are noisy — the
        # tracker loses the hero mid-flight on a running jump — so an argmax
        # over them picks badly.
        if not self.jump_height:
            return 28
        return max(int(k) for k in self.jump_height)

    def lines(self) -> list[str]:
        out = [f"run {k}: {v:.2f} px/f" for k, v in self.run_speed.items()]
        out += [f"jump A hold {k}f: {v:.0f} px" for k, v in self.jump_height.items()]
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))


class InstinctPolicy:
    """Same interface as BCPolicy, plus its own tracker and object memory."""

    def __init__(self, knowledge_path: str | Path | None = None,
                 attack_button: str = "B", perception: str = "motion"):
        # "motion" infers objects from the pixels, which is what the agent has
        # to do for real. "sprites" reads the console's sprite table: exact
        # positions, no inference. Swapping them isolates perception as a
        # variable — see perception/sprites.py.
        self.perception = perception
        self.tracker = self._new_tracker()
        self.memory = ObjectMemory()
        self.knowledge = Knowledge()
        self.knowledge_path = Path(knowledge_path) if knowledge_path else None
        self.attack_button = attack_button   # which button strikes; found by probing
        self._pressed = NOOP
        self.reset()

    def reset(self) -> None:
        self._frame = 0
        self._cal_step = 0
        self._cal_left = CALIBRATION_STEPS[0][2]
        self._samples: dict[str, list[float]] = {}
        self._jump_start_cy: float | None = None
        self._scroll_hist: list[float] = []
        self._stuck_jumps = 0
        self._enemies_near = False    # enemies nearby means "stuck" does not count
        # Off = the pre-fix behaviour, kept so the change can be ablated rather
        # than argued about. See _explore_step.
        self.curiosity_needs_progress = True
        self._target_id: int | None = None   # finish one enemy, not all of them once
        self._target_left = 0
        self._attack_phase = 0
        self._facing: str | None = None      # last confident direction to the enemy
        self._wall_frames = 0                # consecutive frames of LEFT doing nothing
        self._plan: list[tuple[frozenset[str], int]] = []   # queue of (buttons, frames)
        self._cal_started = False
        self.tracker = self._new_tracker()
        if hasattr(self, "memory"):   # reset() also runs from __init__
            self.memory.begin_episode()
        # Knowledge already gathered, from an earlier episode or from disk:
        # calibrating again would waste the first thousand frames of every run.
        if self.knowledge.jump_height:
            self.mode = "explore"
            self.last_reason = "knowledge loaded — exploring"
        else:
            self.mode = "calibrate"
            self.last_reason = "calibration start"

    # ---------- one frame ----------

    def _new_tracker(self):
        if self.perception == "sprites":
            from nes_player.perception.sprites import SpriteTracker

            return SpriteTracker()
        return MotionTracker()

    def step(self, frame_rgb: np.ndarray, score: int = 0, died: bool = False,
             ram: np.ndarray | None = None):
        """Returns (pressed, slots, verdicts); the last two are for rendering.

        `ram` is only read when perception is "sprites", and only to locate
        objects. The policy never reads game variables from it (spec §3).
        """
        slots = (self.tracker.update(frame_rgb, self._pressed, ram)
                 if self.perception == "sprites"
                 else self.tracker.update(frame_rgb, self._pressed))
        verdicts = self.memory.update(frame_rgb, slots, self._frame, score, died)
        best = max(slots, key=lambda s: s.ctrl_prob, default=None)
        self._scroll_hist.append(self.tracker.scroll_dx)
        if self.mode == "calibrate" and not self._cal_started:
            # Wait for the controls to RESPOND, not for a timer: press RIGHT and
            # watch for a slot that correlates with it. Cutscenes, menus and
            # attract demos never correlate, so calibration cannot start on them.
            if (self._frame >= WAIT_MIN and best is not None
                    and best.ctrl_prob > 0.65) or self._frame > WAIT_MAX:
                self._cal_started = True
                self.last_reason = "control responds — calibrating"
                pressed = self._calibrate_step(best)
            else:
                self.last_reason = "waiting for control response"
                pressed = RIGHT if self._frame % 20 < 12 else NOOP
        elif self.mode == "calibrate":
            # No confidence threshold during calibration: in the first seconds
            # the only thing moving under our own input is us.
            pressed = self._calibrate_step(best)
        else:
            ctrl = best if best is not None and best.ctrl_prob >= 0.55 else None
            pressed = self._explore_step(slots, ctrl, verdicts)
            pressed = self._unstick_from_left_wall(pressed, best)
        # One resolver on the way out. The unstick guard turns LEFT into RIGHT
        # while a plan may still be holding LEFT, and a hand cannot press both.
        self._pressed = pressed = resolve_conflicts(pressed)
        self._frame += 1
        return pressed, slots, verdicts

    def _unstick_from_left_wall(self, pressed: frozenset[str],
                                ctrl: Slot | None) -> frozenset[str]:
        """Stop pressing LEFT once it has demonstrably stopped doing anything.

        Guarding the one escalation branch that walks left was not enough: the
        surrounded rule, curiosity and the manoeuvre queue all issue LEFT too,
        and any of them can hold the agent against the edge. This is the single
        place every action passes through, so the guard belongs here.

        It waits for evidence rather than assuming a wall. Only after LEFT has
        been held for `WALL_EVIDENCE` frames with the hero pinned at the edge
        and the world refusing to move does it conclude there is nothing there,
        drop the press and clear whatever plan produced it.
        """
        if ctrl is None or ctrl.ctrl_prob <= 0.7 or "LEFT" not in pressed:
            self._wall_frames = 0
            return pressed
        if ctrl.cx >= LEFT_EDGE or abs(self.tracker.scroll_dx) > 0.4:
            self._wall_frames = 0
            return pressed
        self._wall_frames += 1
        if self._wall_frames < WALL_EVIDENCE:
            return pressed
        self._plan.clear()
        self.last_reason = "left edge does not give — turning around"
        return (pressed - {"LEFT"}) | {"RIGHT"}

    # ---------- calibration ----------

    def _calibrate_step(self, ctrl: Slot | None) -> frozenset[str]:
        name, buttons, dur = CALIBRATION_STEPS[self._cal_step]
        elapsed = dur - self._cal_left
        if name.startswith("run:") and ctrl is not None and elapsed > dur // 2:
            # World velocity = screen velocity + camera velocity. The camera
            # moving right gives a negative dx, hence the subtraction.
            self._samples.setdefault(name, []).append(ctrl.vx - self.tracker.scroll_dx)
        if name.startswith("jump:") and self._jump_start_cy is None and ctrl is not None:
            self._jump_start_cy = ctrl.cy
        if name.startswith(("jump:", "land:")) and ctrl is not None \
                and self._jump_start_cy is not None:
            height = self._jump_start_cy - ctrl.cy
            key = name.split(":")[1]
            self._samples[f"h{key}"] = [max(height, *(self._samples.get(f"h{key}", [0])))]

        self._cal_left -= 1
        if self._cal_left <= 0:
            if name.startswith("land:"):
                self._jump_start_cy = None
            self._cal_step += 1
            if self._cal_step >= len(CALIBRATION_STEPS):
                self._finish_calibration()
                return NOOP
            self._cal_left = CALIBRATION_STEPS[self._cal_step][2]
            self.last_reason = f"calibrating: {CALIBRATION_STEPS[self._cal_step][0]}"
        return buttons

    def _finish_calibration(self) -> None:
        for key in ("run:RIGHT", "run:B+RIGHT"):
            if self._samples.get(key):
                self.knowledge.run_speed[key.split(":")[1]] = float(
                    np.mean(self._samples[key]))
        for hold in ("4", "12", "24", "32"):
            if self._samples.get(f"h{hold}"):
                self.knowledge.jump_height[hold] = float(self._samples[f"h{hold}"][0])
        if self.knowledge_path:
            self.knowledge.save(self.knowledge_path)
        self.mode = "explore"
        self.last_reason = "calibration done -> exploring"

    # ---------- exploration ----------

    def _stuck(self) -> bool:
        # Stuck means almost no frames scrolled noticeably. Counting frames
        # rather than summing the scroll matters: phase correlation is noisy on
        # flashing sprites, and a sum smears the threshold away.
        recent = self._scroll_hist[-40:]
        moving = sum(1 for v in recent if abs(v) > 0.4)
        return len(recent) == 40 and moving <= 3

    def _explore_step(self, slots, ctrl, verdicts) -> frozenset[str]:
        # Fighting outranks every manoeuvre, so it is checked BEFORE the plan
        # queue. Otherwise a long retreat from the stuck logic runs for 135
        # frames while an enemy stands next to us, unhit.
        if ctrl is not None:
            engage = self._engage_step(slots, ctrl)
            if engage is not None:
                return engage
        if self._plan:
            buttons, left = self._plan[0]
            self._plan[0] = (buttons, left - 1)
            if left <= 1:
                self._plan.pop(0)
            return buttons

        hold = self.knowledge.best_jump_hold()
        # Curiosity is only curiosity while it is getting somewhere. Measured on
        # Double Dragon it latched onto a fixed blob overhead and jumped at it
        # for 3912 of 4200 frames — and because curiosity is checked above the
        # stuck escalation, the escalation never got a turn: the agent stood in
        # one spot for three minutes, alive only because nothing came to kill
        # it. Being demonstrably stuck disqualifies curiosity, and once the
        # escalation gets the world moving again curiosity is allowed back.
        stuck = self._stuck() and not self._enemies_near
        gate = stuck and self.curiosity_needs_progress
        if ctrl is not None:
            # Danger ahead: jump over it.
            for s in slots:
                if s is ctrl or verdicts.get(s.slot_id) != "danger":
                    continue
                dx = s.cx - ctrl.cx
                if 0 < dx < 48 and (s.vx < 0 or abs(dx) < 30):
                    self._plan = [(JUMP_RUN, hold), (RUN, 12)]
                    self.last_reason = f"danger #{s.slot_id} ahead — jumping over"
                    return self._take_from_plan()
            # Curiosity: something unfamiliar overhead is worth jumping into.
            for s in (() if gate else slots):
                if s is ctrl or verdicts.get(s.slot_id) != "unknown":
                    continue
                dx, dy = s.cx - ctrl.cx, ctrl.cy - s.cy
                if abs(dx) < 20 and 10 < dy < 80:
                    self._plan = [(frozenset({"A"}), hold), (NOOP, 20)]
                    self.last_reason = f"curious: #{s.slot_id} overhead — trying to reach"
                    return self._take_from_plan()
                if 14 <= dx < 60 and abs(dy) < 60 and s.vx == 0:
                    self.last_reason = f"curious: heading to #{s.slot_id}"
                    return RUN if dx > 0 else LEFT

        # In a beat-em-up the camera holds still while enemies are alive. That
        # is not being stuck, it is the game working as designed.
        if stuck:
            self._stuck_jumps += 1
            self._scroll_hist.clear()   # fresh window after a manoeuvre
            if self._stuck_jumps <= 3:
                self._plan = [(JUMP_RUN, hold), (RUN, 16)]
                self.last_reason = f"stuck — jump hold {hold}f (try {self._stuck_jumps})"
            elif ctrl is not None and ctrl.cx < LEFT_EDGE:
                # Already against the left wall. Backing off left is what the
                # escalation would normally do, and there it does nothing at
                # all: no movement, so still "stuck", so back off again — the
                # agent pins itself to the edge for the rest of the episode.
                # The way on is always to the right.
                self._plan = [(RUN, 90), (JUMP_RUN, max(hold, 24)), (RUN, 40)]
                self.last_reason = "stuck at the left edge — the way on is right"
            else:
                # Escalate: back off and jump with a run-up. The retreat is kept
                # short and followed by a long run forward, because in a
                # beat-em-up the level only advances while you walk right — a
                # long walk left was eating 854 of every 3000 frames.
                back = min(30 * (self._stuck_jumps - 3), 60)
                self._plan = [(LEFT, back), (NOOP, 8), (RUN, back * 2),
                              (JUMP_RUN, max(hold, 24)), (RUN, 40)]
                self.last_reason = f"still stuck — back off {back}f, running jump"
            return self._take_from_plan()

        if abs(self.tracker.scroll_dx) > 0.3:
            self._stuck_jumps = 0
        self.last_reason = "running right (progress)"
        return RUN

    def _engage_step(self, slots, ctrl: Slot) -> frozenset[str] | None:
        """Beat-em-up behaviour: face the nearest enemy and hit it, or back away
        when caught between two. Returns None when there is no fight.

        Anything moving, uncontrolled and in range counts as an enemy. Waiting
        for the `danger` verdict would not do: that verdict only appears after
        something has killed us, and by then the fight is over.
        """
        near = []
        for s in slots:
            if s is ctrl or s.small or s.ctrl_prob > 0.7:
                continue
            dx, dy = s.cx - ctrl.cx, s.cy - ctrl.cy
            if abs(dy) <= APPROACH_DY and abs(dx) <= max(ENGAGE_DX, SURROUNDED_DX):
                near.append((dx, s))
        self._enemies_near = bool(near)
        if not near:
            return None
        self._plan.clear()   # a fight cancels whatever manoeuvre was running

        # "Surrounded" is judged on the NARROW lane only. An enemy at a
        # different depth cannot reach us, so retreating from it is pointless —
        # with the wide band this rule fired constantly and the score fell by
        # half.
        lane = [(d, s) for d, s in near if abs(s.cy - ctrl.cy) <= ENGAGE_DY]
        left_side = [d for d, _ in lane if d < 0 and abs(d) <= SURROUNDED_DX]
        right_side = [d for d, _ in lane if d >= 0 and d <= SURROUNDED_DX]
        if left_side and right_side:   # pinned: retreat towards the safer side
            away = "RIGHT" if min(abs(d) for d in left_side) < min(right_side) else "LEFT"
            self._plan = [(frozenset({away}), BACK_OFF)]
            self.last_reason = f"surrounded ({len(near)} enemies) — backing off {away}"
            return self._take_from_plan()

        # Stay on the chosen target. Switching to whoever is nearest each frame
        # means hitting everyone once and killing nobody.
        target = None
        if self._target_id is not None and self._target_left > 0:
            for d, s in near:
                if s.slot_id == self._target_id and abs(d) <= TARGET_KEEP_DX:
                    target, dx = s, d
                    break
        if target is None:
            dx, target = min(near, key=lambda t: abs(t[0]))
            self._target_id, self._target_left = target.slot_id, TARGET_HOLD
            self._attack_phase = 0
        self._target_left -= 1

        if abs(dx) > ENGAGE_DX:
            return None
        # At contact range the hero's blob and the enemy's merge into one and
        # the sign of dx becomes noise, so hold the last confident direction —
        # otherwise the agent turns around and punches the air behind it.
        if abs(dx) >= FACING_MIN_DX:
            self._facing = "RIGHT" if dx > 0 else "LEFT"
        facing = self._facing or ("RIGHT" if dx > 0 else "LEFT")
        # Different depth: line up first, or the strike passes through air.
        dy = target.cy - ctrl.cy
        if abs(dy) > ENGAGE_DY:
            vert = "DOWN" if dy > 0 else "UP"
            buttons = {vert}
            if abs(dx) > HIT_DX:
                buttons.add(facing)
            self.last_reason = f"aligning with #{target.slot_id} ({vert})"
            return frozenset(buttons)
        if abs(dx) <= HIT_DX:
            # Press/release rhythm: in many games a strike registers on the
            # button's rising edge, and a held B hits once and then stops.
            # (Measured on Double Dragon it lost, so ATTACK_RELEASE is 0 —
            # kept because it is the right behaviour for games that need it.)
            self._attack_phase = (self._attack_phase + 1) % (ATTACK_PRESS + ATTACK_RELEASE)
            hit = self._attack_phase < ATTACK_PRESS
            self.last_reason = (f"finishing #{target.slot_id} — "
                                f"{'hit' if hit else 'wind-up'} {facing}")
            return frozenset({self.attack_button, facing}) if hit else frozenset({facing})
        self.last_reason = f"closing in on #{target.slot_id} ({facing})"
        return frozenset({facing})

    def _take_from_plan(self) -> frozenset[str]:
        buttons, left = self._plan[0]
        self._plan[0] = (buttons, left - 1)
        if left <= 1:
            self._plan.pop(0)
        return buttons
