# Perception

Everything here works without emulator memory: pixels and PCM only. Debug RAM
is read in the training loop and in evaluation — as reward, as ground truth for
checking a reading — and never by the policy.

## MotionTracker (`perception/motion.py`)

Per frame:

1. **Camera scroll** — `cv2.phaseCorrelate` between consecutive frames, with the
   HUD band excluded. The camera moving right gives dx < 0.
2. **Compensated difference** — shift the previous frame by the scroll, take
   `absdiff`, and what is left is what moved relative to the background.
   Connected components give bounding boxes: 24..3000 px normally, plus a
   separate branch for compact 4..24 px blobs so that Contra's bullets survive.
3. **Tracking** — greedy matching by centre distance under 28 px, giving stable
   ids and an EMA velocity. A stalled object is kept as a ghost at its last
   position; the controlled one for up to 300 frames.
3a. **Clinches** — when two tracks claim the same blob, which is what happens
   the moment two fighters touch, the blob is cut between their predicted
   positions rather than awarded to the nearer one. Without this the loser
   ghosts at its last position and the winner's centre slides into the gap, so
   "which way is the enemy" becomes noise. Attack frames on Double Dragon rise
   from 254 to 462; the score does not measurably move. See
   [experiments.md](experiments.md).
4. **Which one is me** — correlate the sign of the slot's **world** velocity
   (screen velocity minus the scroll) with LEFT/RIGHT being pressed. The result
   is `ctrl_prob`, an EMA through a sigmoid.

Subtracting the scroll is the part that matters. When the camera follows the
hero his screen velocity is close to zero, and without the correction the agent
cannot recognise itself at all.

Known limits: objects that never move are invisible (an unanimated brick),
flashing ones are detected whether or not they are objects (? blocks, the HUD),
and water or a busy background adds noise.

## ObjectMemory (`perception/memory.py`)

- Sprite clusters: a 16×16 grey patch, an EMA prototype, matched by L2 distance.
- A contact is the controlled slot coming within 22 px of another slot.
- Outcomes: a score change inside a 45-frame window counts as `score_gain`; a
  death counts as `deaths`.
- The verdict is `danger` (deaths followed contacts), `reward` (points did) or
  `unknown`.
- It outlives episodes. During a long stream the agent accumulates knowledge
  about the enemies it keeps meeting rather than relearning them every life.

## AudioEventDetector (`perception/audio_events.py`)

- Onsets: log-mel spectral flux (n_fft 512, hop 160, 32 mel bands) above
  `median + 2.6·MAD` over a running 120-column window, which adapts quickly once
  the music stops. Cooldown about 60 ms.
- Clusters: the onset's mel column as a signature, matched at L2 < **1.5**. At
  4.0 the death jingle merged with the background music.
- Meaning: `attribute(ids, kind)` counts deaths and rewards; a verdict is issued
  after at least 2 events and a share above 0.25 of the times that sound was
  heard.
- Timing: in Super Mario Bros. the lives counter in RAM decrements about **120
  frames after** the contact, by which time the death jingle has already
  finished. Sounds are therefore attributed through a backward window of 140
  frames, with a forward window of 90 for games where the jingle follows the
  counter instead.

## SoundLocator (`perception/av_align.py`) — where a sound came from

A contrastive dual encoder. The frame becomes a 10×11 grid of embeddings, a
260 ms mel window becomes one vector, and InfoNCE pulls them together with
similarity taken as the maximum over cells (multiple-instance style). Negatives
are drawn **only from within the same episode**; drawn across episodes the model
learns "level music ↔ level background" instead of sound effects.

Trained with `nes-player train-av`; `--sound-loc` switches it on during play,
drawing a ring where the sound is believed to originate.

On Super Mario Bros.: retrieving the right frame from a sound among 64
candidates succeeds **25.4%** of the time against a 1.6% chance level. Hitting
blocks localises correctly. Jumps drift into the sky, which is fair — a jump has
no source on screen. Inference runs on the CPU because the GPU belongs to the
brain thread.

## HudReader (`perception/text.py`) — reading numbers off the screen

Digits are learned **without labels and without RAM**. The NES draws text as
8×8 tiles, so a glyph is an exact 64-bit signature; the only unknown is which
signature means what.

1. Find the text cells: almost always non-empty, they change, and they take at
   most 14 distinct values.
2. Find the ring: in the lowest digit of any counter, transitions form a cycle
   of length 10 — 0→1→…→9→0. The "most frequent successor" graph is built **per
   cell**, because a global one mixes counters that run in opposite directions
   (a timer counting down, a score counting up) and degenerates.
3. Find zero: by shape, not by frequency. A frequency window misses — by the
   time a few seconds of Super Mario Bros. have passed the timer reads "39x" and
   9 is the most common digit. Zero is instead identified topologically: it is
   the glyph with a **closed hole** inside it. In the Castlevania font the 2 is
   heavier than the 0, so picking by ink volume chose wrong.
4. Find the direction: 1 is the thinnest glyph in any font, so of zero's two
   neighbours in the ring the one with less ink is the 1. The ring alone cannot
   tell a counter going up from one going down.

Checked against RAM (`scripts/experiments/hud_read_check.py`, Super Mario Bros.)
the timer reads with **correlation 1.000 and 95.8% exact matches**; coins and
score digits are right too. Reading costs 0.022 ms per frame because only the
learned cells are touched. The reader trains itself in the first twenty seconds
of a run.

### Letters and menu prompts

Letters have no dynamics to derive them from, so the alphabet is supplied as a
prior — the same way a person arrives at a game already knowing how to read. The
atlas is bootstrapped by `scripts/build_font_atlas.py`: take screens whose text
is known ("1 PLAYER GAME" in Super Mario Bros., "PRESS START TO PLAY" in
Battletoads) and align the non-empty cells of a line onto its characters. Twenty
four characters from seven lines, all matching one to one. Digits are mixed in
from whatever the HudReader worked out for this game.

Glyphs match exactly by signature, falling back to the nearest within 5 bits;
beyond that it starts confusing C with O and 0 with D. An unknown glyph reads as
`?`.

`find_prompt(frame)` looks for "PRESS/PUSH/HIT … START/SELECT" and returns the
button, so the agent presses what the screen asks for instead of pulsing START
blindly. Two guards against false positives:

- lines that mention a second player ("PLAYER 2 PRESS START") are ignored;
- a prompt is obeyed only while there is **no controllable object** on screen.
  If we are already playing, the message is addressed to a partner, and START
  in-game is the pause button.

Fonts the atlas has not seen yet (Gradius, Balloon Fight) still read as `?`;
extending it is a matter of adding sources to the script.

## Neural slots (`perception/slots.py`) — negative result

A slot-attention autoencoder, K=7, spatial broadcast decoder. Reconstruction
learns (MSE 0.027 → 0.012) but the decomposition is not objects: the slots split
the frame into positional ripples. That is not a bug in the training so much as
a property of the domain — an NES background is itself a repeating tile pattern,
and a blobby decomposition describes it just as well as an object-shaped one
would. The method has no other gradient to follow.

The motion tracker remains the production path. Ideas for a second attempt are
in [experiments.md](experiments.md).

## InstinctPolicy (`policy/instinct.py`)

A playable policy with no training at all. Three phases:

1. **wait** — pulse RIGHT and wait for the controls to *respond*, meaning a slot
   with `ctrl_prob > 0.65`, with a 1800-frame fuse. Cutscenes, menus and attract
   demos never respond, so calibration does not start on them.
2. **calibrate** — a probe protocol: RIGHT and B+RIGHT for 60 frames each to
   measure running speed from world velocity, then A held for 4/12/24/32 frames
   to measure jump height from the change in cy. The result is written to
   `runs/knowledge/<game>.json` and not measured again. Height measurement is
   noisy, so the hold is chosen by the physical prior "longer is higher" rather
   than by the raw numbers.
3. **explore** — a handful of rules:
   - run in the direction that makes progress (B+RIGHT);
   - stuck (fewer than 3 of the last 40 frames scrolled noticeably — a sum is
     useless here because phase correlation is noisy) → jump with the calibrated
     hold; after three failures escalate the back-off, 45 → 90 → 135 frames, and
     jump with a run-up;
   - curiosity: an unfamiliar object ahead and above → jump early;
   - a `danger` object closing in → jump over it;
   - **beat-em-ups**: an enemy in the same lane (|dy| ≤ 20 px) → close in and
     strike while holding the direction, which turns the hero to face it;
     surrounded → back away from the nearer one; an enemy at another depth
     (±64 px) → line up vertically first, because the fight is two-dimensional.
     Engagement is evaluated before the manoeuvre plan and clears it, and while
     enemies are close "stuck" does not count — during a fight the camera is not
     supposed to move.

It keeps its own MotionTracker and ObjectMemory.


---

## Objects from the sprite table

The NES draws everything that moves as sprites, and the picture processing unit
is told where they are by a 256-byte table it receives by direct memory access
from CPU page `$0200` every frame: 64 sprites of four bytes each — y, tile,
attributes, x. Practically every game on the machine uses that transfer, so
reading page 2 gives exact positions of every on-screen object **in any game**,
with no per-game memory map to write.

```python
oam = ram[0x200:0x200 + 4 * 64].reshape(64, 4)
visible = oam[oam[:, 0] < 0xEF]        # 0xEF and above parks a sprite off-screen
```

`SpriteTracker` presents these through the same interface as `MotionTracker`, so
the instinct policy can be run on either without changing a line of its logic —
which is what isolates perception as a variable.

Two limits, both real:

- **This is privileged.** It is for supervision, for measurement, and for a
  teacher that is meant to see more than its student. A policy that plays does
  not read it.
- **Sprites are objects, not the world.** Floors, pits, spikes and walls are
  background tiles and do not appear in the table at all. In Super Mario Bros.
  the most lethal thing on the screen is invisible to it.

Recorded episodes do not store RAM and do not need to: every episode replays
frame-exactly from its recorded actions, so the table can be recovered
afterwards. The recovery verifies itself — the replay is compared with the
stored frames every 600 frames and raises rather than returning a plausible
table from a different run.

## Where "something good or bad happened" comes from

The object memory turns contacts into `danger` and `reward` verdicts using two
facts: the score went up, and we died. Those verdicts choose buttons, so the
facts are an **input**, and their source decides whether the pixels-and-sound
claim is true. `perception/feedback.py` names three sources and defaults to
`strict`, which supplies neither fact. See [cli.md](cli.md#feedback) for the
measurements behind that default.


## Remembering which object is which

A cluster is an averaged 16×16 grey crop, matched by nearest prototype. Two
numbers in that sentence were measured rather than chosen, and one of them had
been wrong since the beginning:

- **the threshold, 55.** Over pairs of sightings that are certainly the same
  object against pairs that are certainly not, that is where the two
  distributions separate best. It was 28, which files an animating sprite as a
  new object every few frames.
- **the death window, 260 frames.** The lives counter does not move when the
  hero is hit; it moves when the death animation has finished and the level has
  restarted, 213 frames later on Super Mario Bros. Points still use 45 frames,
  because the score moves the instant it moves.

An object is only credited with a death if it is the **last** thing touched. A
window wide enough to cover the counter's lag holds several contacts at once,
and blaming all of them makes half the screen look lethal.
