# Experiment log

Newest first. Negative results are here on the same footing as positive ones —
in this project roughly half of what was learned came from working out why
something failed. Every entry names the script that reproduces it.

---

## Splitting a clinch: the perception fix works, the score does not move

The long-standing open item: at contact range a beat-em-up's hero and enemy
become one connected component, the greedy match hands it to whichever track is
nearer, the other ghosts at its last position, and the sign of
`enemy − hero` becomes noise. `_split_detection` now cuts the blob between the
two predicted positions.

Measured on Double Dragon, both arms in one process, 12 runs of 3000 frames.
Each run starts the policy at a different point in the level — the instinct
policy is deterministic, so seeding the emulator alone produces twelve
byte-identical runs and an n of 1 wearing the clothes of an n of 12. The first
attempt at this measurement did exactly that and reported a confident 119
against 139.

| Arm | Score | Progress | Attack frames | Alignment frames | Paired vs base |
|---|---|---|---|---|---|
| merged (before) | 103.5 ± 26.6 | 7.0 | 254 | 466 | — |
| split, lane 20 (default) | 109.2 ± 23.8 | 21.4 | **435** | 668 | +5.8 (5W/6L) |
| split, lane 24 | 118.5 ± 21.5 | 9.0 | 438 | 625 | +15.0 (7W/5L) |
| split, lane 28 | 113.8 ± 19.9 | 13.8 | 462 | 503 | +10.3 (5W/5L) |
| split, lane 32 | 113.2 ± 29.1 | 8.1 | **480** | 477 | +9.8 (8W/4L) |

**The score difference is not established, and neither is any lane width.**
Every split arm sits 6 to 15 points above the baseline, which looks like a
trend until the paired win/loss column is read: the best of them is 8 wins to 4
losses, which a fair coin produces about one time in five. The ordering by mean
(24 > 28 ≈ 32 > 20) does not match the ordering by win rate (32 > 24 > 28 = 20),
and that mismatch is what noise looks like. With five arms on the table, the
best-looking one is expected to look good by chance; picking it would be
measuring the sampling error and calling it a result.

The second metric did not rescue the comparison either. Camera scroll was added
on the theory that a beat-em-up level only advances once the enemies are down,
but over 3000 frames on Double Dragon it barely advances at all, and the values
(7.0, 21.4, 9.0, 13.8, 8.1) track nothing. It is a reasonable metric for a
scrolling game and the wrong one here.

**What is established is the perception fix itself.** Frames in which the agent
is striking a tracked enemy rise from 254 to 462, an 82% increase, which is a
direct count rather than a downstream proxy. The tracker now keeps two objects
where it kept one.

**The interesting part is the side effect.** Alignment frames jump from 466 to
668 once splitting is on. The depth-alignment rule fires whenever the enemy is
more than 20 px away vertically — and until now `dy` at contact range was
garbage, so the rule mostly did not fire at all. With an accurate `dy` it fires
constantly and the agent spends its time repositioning by a few pixels instead
of hitting. Loosening the lane to 28 px brings it back to 503.

In other words the policy's threshold had been tuned, unknowingly, against
broken perception. That is worth remembering for every other constant in
`instinct.py`: several of them were chosen while the tracker was merging
fighters.

The default lane stays at 20 px. Changing a constant on an unproven +15 would
be exactly the mistake this log exists to prevent — and the two "clever" ideas
that lost earlier on this same game were lost the same way. The split itself
stays on: object counts feed the planner's threat list, contact attribution in
the object memory and the attention masks used in training, all of which want
two objects to be two objects regardless of what this one policy scores.

What the sweep does establish is that **this metric cannot resolve differences
of this size**. A ±25 spread on a mean of 110 needs roughly 50 runs to see a
15-point effect, not 12. Either the next attempt budgets for that, or it finds
a tighter measurement — the honest candidate being a direct count of how often
the agent strikes in the wrong direction, which needs ground-truth positions
that Double Dragon's memory map does not currently expose.

Reproduce with `scripts/experiments/clinch_ab.py`.

---

## Comparing emulation cores on third-party TAS movies

The movies were recorded in FCEUX and BizHawk while we play on fceumm, so
perhaps a different core would hold them longer. Nestopia, QuickNES and Mesen
were plugged in and measured.

Share of frozen frames over 6000 frames — lower is better, ~100% means stuck:

| Movie | fceumm | nestopia | quicknes |
|---|---|---|---|
| Dungeon Magic | **82%** | 90% | 93% |
| New Ghostbusters II | **32%** | **32%** | 33% |
| Bad Dudes | 100% | 100% | 100% |
| Werewolf | **27%** | 70% | 68% |
| Metroid | **94%** | 95% | **94%** |
| Transformers | **58%** | 70% | 72% |
| Kid Dracula | **3%** | 14% | 9% |
| Excitebike | **23%** | 25% | 25% |
| Donkey Kong | 83% | 84% | **47%** |
| Blues Brothers | 92% | 92% | **50%** |

**There is no single winner.** fceumm is best or tied on eight movies, but on
Donkey Kong and Blues Brothers quicknes lasts twice as long — precisely where
fceumm looked hopeless. The conclusion is to pick the core per movie rather than
once and for all.

Mesen kills the process with no message: it wants frontend services our minimal
harness does not provide.

**Two mistakes along the way, both the same mistake — treating the absence of an
error as success:**

1. Dropping a third-party core with its json into the shared directory silently
   replaced fceumm for **every** game. Caught by the golden frame and audio
   hashes in the regression tests.
2. Rewriting the switch to patch the core table from Python, then reporting that
   it worked. The suspicion only arrived when four supposedly different
   emulators returned identical numbers to the percent: the core is chosen in
   C++, which never reads that table. The working mechanism is
   `RetroEmulator.load_core_info` in process memory, verified by frame hash —
   fceumm `f75958f3`, nestopia `ea9e146b`, quicknes `995c84f0` after the same
   300 button presses.

Reproduce with `scripts/experiments/core_compare.py`.

---

## Emulator recordings as a data source: survey

Instead of video, use **emulator recordings**: the buttons are already in them,
so no inverse dynamics is needed, and replay runs headless thousands of times
faster than video.

**Scale, counted honestly:** 933 NES publications on TASVideos, 763 distinct
games, **240 hours** of frame-accurate input. Formats: 71% `.fm2` (we read it),
21% `.fcm` (old FCEU), the rest BizHawk. A warning for anyone querying that API:
it silently ignores `pageNumber` and returns the same page every time. The
working parameter is `currentPage`; getting this wrong inflates the count
eightfold.

**Yield on a sample of 12 movies:** all 12 downloaded, **10 of 12 found their
ROM** by checksum. Then the real problem starts.

**The bottleneck is not volume, it is synchronisation.** Our core lags on
different frames than FCEUX does, so movies drift. Of the 10 pairs, Excitebike,
Blues Brothers and Donkey Kong were visually confirmed to play; New Ghostbusters
runs about 6000 frames and dies; Metroid and Bad Dudes sit on the menu forever.

Fixed along the way:

- **unpadded base64** in `romChecksum` crashed the parser on real files; it is
  padded now, and a movie without a checksum warns instead of failing;
- **the start offset differs per movie and is signed** — the emulator can run
  ahead of the movie or behind it. Only positive offsets were searched before,
  which swallowed an early START;
- **choosing the offset by screen activity was fooled by animated title
  screens** (Metroid's stars twinkle): a stuck run looked no worse than a
  playing one. The measure is now divergence from the *starting* frame rather
  than frame-to-frame activity.

The automatic desync detector is still unreliable in both directions: "share of
frozen frames" misfires on games with small sprites over a static background
(Donkey Kong reads 94% frozen while playing normally) and on animated menus. The
honest position is that **a yield figure cannot be quoted yet**.

The practical route is not to chase full synchronisation but to take the
**verified prefix** of each movie — replay it, find where it breaks, keep the
beginning. Even a minute each from hundreds of games is a large and varied
corpus, and it answers the "the model will just memorise the TAS" risk better
than whole movies would: short fragments from many places teach
situation → action rather than one memorised route.

---

## Inverse dynamics model — the key to a "gamer dataset"

There is no labelled (frame, button) dataset in the world, but there are
thousands of hours of human play on video with no buttons attached. The recipe
from VPT (OpenAI, 2022) is to train an inverse dynamics model on a small
labelled set and use it to label the mountain of video.

**Our advantage over VPT: they had to hire people to produce labelled pairs; an
emulator produces them for free, in any quantity.**

Implemented in `policy/idm.py`, `nes-player train-idm`. The model is
non-causal — it sees a ±5 frame window around the frame it is predicting. From a
single frame you cannot tell whether jump is held; from before-and-after it is
trivial.

Same data, same split (Super Mario Bros. TAS plus 11 instinct episodes):

| Model | What it sees | Validation accuracy |
|---|---|---|
| majority baseline | — | 0.450 |
| BC (causal) | 4 past frames | 0.552 |
| **IDM (non-causal)** | ±5 frames around | **0.603** |

Non-causality is worth 5.1 points, so the mechanism works. It is nowhere near
good enough to label with yet — VPT reaches 90%+ with a 128-frame window and a
transformer. The route is a wider window, more data, and an architecture of its
own instead of reusing `BCNet`.

---

## Zero-shot: meeting a game for the first time

Protocol: a game that appeared in **no** training run; no demonstrations; the
agent gets START pulses for the menus and nothing else. Three agents — random
buttons, instincts (which calibrate on the spot), and the multi-game BC base
with no fine-tuning. Three runs of 3600 frames.
Script: `scripts/experiments/zero_shot.py`.

The metric is **the score the agent read off the screen itself** — digits
learned without labels, see [perception.md](perception.md). Verified against
RAM: 700 = 700, and on a scoreless run 44 of 44 zeros matched.

| Game | Random | Instincts | Base, zero-shot |
|---|---|---|---|
| **Castlevania** | 0 points | **700 points** | 0 points |
| Bubble Bobble | 0 | 0 | 0 |
| 1943 | counter not recognised | — | — |

Horizontal progress on Castlevania per 1000 frames: instincts **213.8**, random
126.1, base **9.7**.

The main finding, and it was not the expected one: **zero-shot transfer of a
trained policy does not work — the base plays worse than random mashing** (9.7
against 126 on progress, zero points for both). The model confidently presses
what was right in its own games, and on a new one that actively gets in the way.
What works is the instincts: calibrating the controls in place yields both
points and progress.

Meanwhile few-shot transfer of the *same* base does work (+7–8 points, see
below). So **features transfer and behaviour does not.**

Honest limits: on Bubble Bobble (single screen) nobody scored; on 1943 the
reader found no non-decreasing multi-digit counter. Horizontal scroll is a
meaningless metric for those two — vertical scrolling and a static screen — so
the progress row covers Castlevania only.

---

## Beat-em-up instincts, and what they revealed about Double Dragon

In a beat-em-up you need to walk backwards and strike towards the enemy rather
than run right. Implemented in `_engage_step` (`policy/instinct.py`): the
nearest non-controlled object in the same lane (|dy| ≤ 20 px) is the target;
farther than `HIT_DX` walk towards it, closer than that strike while holding the
direction so the hero turns to face it; enemies on both sides within 52 px means
back away from the nearer one. Small blobs (bullets) are excluded.

Two bugs found by watching the game, not by tests:

- **a long plan blocked the fight** — the 135-frame retreat from the "stuck"
  logic prevented striking an enemy who had walked up. Engagement is now checked
  *before* the manoeuvre plan and clears it;
- **false "stuck"** — in a beat-em-up the camera is not supposed to move while
  enemies are alive. The stuck counter no longer increments when enemies are
  near.

Measured on Double Dragon over 3000 frames: 252 frames closing in, 202 striking,
195 backing away when surrounded — all zero before the fix.

### Finishing an enemy off: three ideas, the simplest won

Damage accumulates, so an enemy must be finished rather than poked once.
Measured by score over 3000 frames on Double Dragon:

| Variant | Score |
|---|---|
| original (one strike, long "stuck" retreat) | 92 |
| lock the target for 150 frames + press/release rhythm | 110 |
| **strike the nearest continuously + short retreat** | **124** |

Both clever ideas hurt. Locking sticks to an enemy who has been knocked away;
rhythm loses frames to the gaps between presses. Brute force wins — keep hitting
whoever is closest.

The bigger win came from somewhere else entirely: the agent was spending **854
of 3000 frames** on the Mario manoeuvre "stuck → retreat 135 frames → jump with
a run-up". In a beat-em-up the level only moves forward, so the retreat was
shortened to 60 frames and followed by a long run forward.

Movement in depth: the fight is two-dimensional, and until you stand at the same
depth as the enemy your strikes miss. Enemies are now sought in a wide band
(±64 px) and approached vertically, with strikes only once aligned: **124 → 134
points**, at a cost of 647 of 3000 frames spent lining up. One detail matters —
the "surrounded" rule stayed on the *narrow* band. With the wide one it counted
enemies at other depths as an encirclement, the agent retreated continuously,
and the score fell to 66.

Sprite overlap was the open problem here: at contact range the motion tracker
merged hero and enemy into one blob, the sign of the direction became noise, and
the agent hit backwards. Holding the last confident direction was added and
changed nothing (110 against 110). It has since been addressed in the tracker
itself, by splitting the merged blob — see the entry at the top of this log,
including what that did and did not improve.

### Double Dragon: the dataset was garbage

It turned out the game had **never actually started**: all 41 episodes of
`datasets/explore_dd` were the static title screen. Its title cannot be passed
from power-on by anything — START, START+A+B, waiting for the attract demo were
all tried — because the stable-retro integration expects its own savestate.
A `--state default` flag was added.

The transfer result for Double Dragon was retracted (see below). The other six
datasets were checked by eye and contain real gameplay.

The lesson is about metrics, not about Double Dragon: accuracy of 0.42 does not
distinguish "playing badly" from "having learned a still picture". It was found
only by looking at the frames.

---

## Reading numbers off the screen without labels

Implemented without a neural network and without labels; the algorithm is in
[perception.md](perception.md). Digits are learned from the **dynamics** of
counters — the transition ring of the lowest digit — with zero anchored by shape
and the ring's direction settled by the thinness of the 1.

Validated against RAM (Super Mario Bros., learned blind on 60% of frames and
checked on the rest): the timer reads with **r = 1.000 and 95.8% exact matches**,
in 100% of frames. In the live window the panel shows "4 2 391" against a HUD
reading "score 000400, coins ×02, TIME 391" — score digit, coins and timer all
correct.

Three traps:

- a transition graph built over **all** cells degenerates, because the timer
  counts down while the score counts up. The ring has to be found per cell.
- anchoring zero by a 60-frame frequency window misses: a few seconds into Super
  Mario Bros. the timer already reads "39x", 9 becomes the most common glyph, and
  every reading came out one too high — visible on the panel as 448 instead of
  337. Zero is now found topologically, as the glyph with a closed hole.
- scanning all 840 cells every frame is too expensive at 60 fps; at inference
  only the learned cells are read, 0.022 ms per frame.

Script: `scripts/experiments/hud_read_check.py`.

---

## Progress from pixels instead of telemetry, and visual self-imitation

The goal is to estimate progress without reading score or coordinates from RAM.
Implementation: accumulated camera scroll via phase correlation, with jumps over
16 px discarded as scene changes (`VisualProgress` in `policy/improve.py`).

**Validated against ground truth** (`scripts/experiments/visual_progress_check.py`,
Super Mario Bros., 8 rollouts): the pixel estimate against true `xscroll` from
debug RAM gives **Pearson r = 0.872**. A visual reward can stand in for
telemetry.

Applied as `improve --visual`, self-imitation never opens RAM at all. On BT&DD
(6 rounds × 8 rollouts) mean progress went **100.3 → 114.2** (+14%), maximum
111.7 → 118.1. The mechanism works.

But in absolute terms the policy still loses to random mashing on that game
(30.2 against 42.2 progress per 1000 frames), and self-imitation collapsed it
into a deterministic one — all six measurement runs returned an identical
result. Diagnosis: mode collapse, since retraining on the top 2 of 8 rollouts
round after round narrows the distribution. Untried remedies: an entropy bonus,
a wider top slice, mixing the original demonstrations back in.

Fixed in passing: self-imitation silently broke on AV checkpoints (`model(x)`
against a `(v, m)` signature) and fed the policy no sound during rollouts. Mel
windows are now written alongside frames and used in retraining.

---

## BT&DD milestone: a verified prefix instead of FCEUX

FCEUX 2.6.6 (Qt build) is **unusable** as a capture reference: `--soundrecord`
silently creates no file, and movie playback reliably kills the process around
frame 300 — with a Lua `frameadvance` loop, with `registerafter`, and with no Lua
at all. The scripts are kept in the repo in case another build behaves.

We managed without a reference, because a desync is visible in the game itself.
The BT&DD TAS replayed in fceumm plays level 1 plausibly to about frame 5280; by
5400 there is a continue screen, then the title. A conservative prefix of **4798
frames (80 s)** was taken, with frames, audio and the buttons of *both* players.

Metrics without RAM (`scripts/experiments/btdd_survival.py`): progress is
accumulated camera scroll per 1000 frames, health is the cyan HUD bars, survival
is frames until the continue screen. A control on the TAS itself showed the
metric works — the expert scores **164.6** progress against 42.2 for random, 4×.

| Agent (6 runs) | Progress/1000 | Health |
|---|---|---|
| TAS (expert, control) | **164.6** | 342 |
| random | 42.2 | 303 |
| BC on the TAS prefix, t=0.3 | 27.5 | 251 |

Diagnosis, in three parts. (a) In a beat-em-up random mashing is a strong
baseline — it constantly attacks and holds right. (b) Eighty seconds of data is
too little and the model overfits (val 0.51 against a majority of 0.60). (c) The
real problem is a **mode mismatch**: the TAS coordinates *two* players and the
policy plays alone, so it imitates behaviour that assumes a partner. Training on
23 instinct episodes did not help, because the instincts are close to random on
this game themselves.

The first three runs showed the opposite picture. That was noise, and it is where
the rule "never measure BT&DD with fewer than 6 runs" comes from.

---

## Transfer between games — confirmed

Protocol: a `BCNetAV` base trained on Super Mario Bros. (TAS) plus Contra (16
instinct episodes), val 0.893. Few-shot on a **new** game, Ice Climber: 3
episodes train, 1 val, 3 epochs, identical seed. Scratch against transfer, where
transfer means `--init-from`: the convolutional body and audio encoder come from
the base and the heads are retrained.

| Epoch | Scratch | Transfer |
|---|---|---|
| 1 | 0.570 | **0.704** |
| 2 | 0.671 | **0.757** |
| 3 | 0.697 | **0.771** |

**Transfer after one epoch beats scratch after three.** Final gap +7.4 points.

Repeated on seeds 1–3 with the same data and protocol, transfer wins **4 of 4
seeds and 12 of 12 per-epoch comparisons**, and on every seed transfer after one
epoch is at least scratch after three. Mean final gap +6.6 points
(0.705 → 0.769).

Extended across genres (base v2: Super Mario Bros. + 40 Contra episodes, val
0.963; targets get 3 few-shot episodes, 3 seeds, final validation accuracy):

| Target | Genre | Scratch (mean) | Transfer (mean) | Transfer wins |
|---|---|---|---|---|
| Ice Climber | vertical platformer | 0.705 | **0.769** | 4/4 seeds |
| Battle City | top-down tanks | 0.559 | **0.596** | 3/3 |
| ~~Double Dragon~~ | ~~beat-em-up~~ | ~~0.421~~ | ~~0.479~~ | **retracted, bad data** |

**The Double Dragon row is retracted.** Its dataset was a recording of the static
title screen — see above. The model learned a motionless image, so "transfer wins
3/3" there means nothing. The remaining datasets were checked by eye.

Without Double Dragon: **transfer wins 7 of 7 finals**, 29 of 30 per-epoch
comparisons, the single loss being one epoch on Battle City seed 1. Transfer
survives a change of genre. Absolute values on the new games are lower because
instinct demonstrations are noisier there.

---

## Held-out transfer protocol — the strict version

Three games seen by neither the base nor any hyperparameter tuning: Gradius (a
shoot-em-up), Balloon Fight (an aerial arcade game) and Battletoads (a
beat-em-up). Data is fresh instinct demonstrations. Few-shot 3+1 episodes, 3
epochs, seeds 1–3. Final validation accuracy, mean:

| Game | Majority | Scratch | Transfer (av base) | Transfer (attn base) |
|---|---|---|---|---|
| Gradius | 0.449 | 0.707 | 0.776 | **0.787** |
| Battletoads | 0.388 | 0.594 | **0.671** | 0.607 |
| Balloon Fight | 0.403 | **0.464** | 0.429 | 0.459 |

Transfer is confirmed on 2 of 3 held-out games (+6.9 and +7.7 points, winning on
every seed) but it is **not universal**: on Balloon Fight transfer *hurts*, by
3.5 points. Its aerial physics is alien to the base features — train accuracy
0.85 against val 0.44 is severe overfitting on three episodes, and a pretrained
body overfits faster than a random one.

The attention-supervised base transfers better on Gradius and worse on
Battletoads, so neither base is a general winner.

Notes: Excitebike was excluded because the instincts never start the race, so the
data is static; Battletoads gets through character select on START pulses by
itself, and the first ~1500 frames of an episode are intro and title.

Postscript: on **full** datasets (33–37 episodes) all three clone confidently —
Gradius **0.872**, Balloon Fight **0.944**, Battletoads **0.702**. So the Balloon
Fight failure above is a few-shot data shortage, not an unclonable game.

Script: `scripts/experiments/heldout_transfer.py`.

---

## Contra without a TAS — the full instinct-to-model loop

Dataset: 56 episodes × 3600 frames (~200k) from `explore --record`, collected
headless at about 1000 fps, so minutes of wall clock. `BCNetAV`, 3 epochs, val =
the last 6 episodes. **Validation accuracy 0.973** against a majority baseline of
0.377.

The instincts were cloned successfully, weaknesses included — the model also
jumps at the water. The first complete "instincts → data → model" cycle on a game
with no synchronised TAS.

---

## MPC planner — a negative result, then a rematch

MPC over the ego world model, 5 templates × 16 steps, scored by progress minus
collisions. `best_x` over 3600 frames, 3 runs: BC alone **726**; planner v1
(threats = danger ∪ moving) 575; v2 (confirmed danger only) 251.

Planning on a model with an action advantage of 1.09 **hurts**: template
predictions are unreliable and the plan is systematically worse than the reactive
policy. This is a classic model-based RL failure.

### Rematch with ego v4 — the criterion is met

The ego model was retrained on 43 episodes, advantage 1.09 → **1.19**. Same duel,
`best_x` over 3600 frames, 3 runs: BC alone **328 / 201 / 202**; planner with
ego v4 **1040 / 1005 / 1323**. The planner's worst run is three times farther
than BC's best, with no overlap.

Which confirms the earlier finding from the other side: **planning lives and dies
by the quality of the world model.** The absolute BC numbers are below the
earlier 726 because the measurement came after small blobs were added to the
tracker; the comparison within each pair is fair.

Script: `scripts/experiments/duel_best_x.py`.

---

## Attention supervision from tracker boxes

The suspicion that the model was looking at the wrong things was confirmed by
measurement: only **12.5%** of the Grad-CAM mass fell inside the motion tracker's
object boxes, against 13.8% for a uniform gaze. The model was looking at scenery
— shortcut learning, exactly what behavioural cloning rewards when the background
correlates with the action as well as the enemy does.

The fix is cross-entropy between the spatial softmax of the last conv layer and
an object mask, cached in `attn_mask.npy`.

| Model | CAM-in-box | Validation accuracy | best_x (3 runs) |
|---|---|---|---|
| `bc_smb_av` (base) | 0.125 | 0.694 | 334 / 202 / 333 |
| attn 0.1, 4 epochs | 0.130 | 0.672 | — |
| **attn 3.0, 6 epochs** | **0.215** | **0.708** | 204 / 323 / 324 |

A weight of 0.1 does nothing; 3.0 gives 1.7× the base focus, above chance, with
accuracy no worse and play a draw.

Scaled to multi-game checkpoints (Contra validation episodes, chance 0.104):

| Model | CAM-in-box | Validation accuracy |
|---|---|---|
| `bc_base41_av` | 0.088 | 0.963 |
| `bc_base41_attn` (weight 3.0) | 0.365 | 0.898 — too expensive |
| **`bc_base41_attn1`** (weight 1.0) | **0.380** | **0.944** |
| `bc_contra_av` | 0.104 (= chance) | 0.973 |
| **`bc_contra_attn`** (weight 1.0) | **0.423** | 0.943 |

On large datasets the best weight is 1.0; 3.0 costs 6.5 points of accuracy with
no gain in focus.

Script: `scripts/experiments/cam_focus.py`.

---

## Contrastive sound↔frame

Dual encoder: the frame becomes a 10×11 grid of embeddings, a 260 ms mel window
becomes a vector, InfoNCE with similarity taken as the maximum over cells. The
decisive design choice is that negatives come **only from the same episode** —
across episodes the model learns "level music ↔ level background" instead of
sound effects.

Trained on the Super Mario Bros. TAS plus 38 explore episodes, 2 epochs: top-1
retrieval of the right frame from a sound among 64 candidates is **0.254**
against a chance level of 0.016, a factor of 16.

Visually, block-hit sounds localise onto the row of bricks and question blocks;
jumps drift into the sky. That last one is a fair weakness rather than a bug —
a jump has no source on screen, it "sounds like" the hero's trajectory.

---

## Neural slots v1 — negative result

A slot-attention autoencoder (Locatello 2020, K=7, spatial broadcast decoder,
56×60 frames). Reconstruction learns — MSE 0.027 → 0.0123 over 8 epochs, then
plateaus — but the decomposition is **not** into objects. Each slot takes a
lattice of blobs spread across the whole frame; Mario, Goombas and blocks are
never isolated.

Diagnosis: the classic weakness of vanilla slot attention on tiled textures. The
Super Mario Bros. background is itself a repeating pattern, so a blobby
decomposition reconstructs it as well as an object-shaped one would, and the
model has no other gradient to follow.

The motion tracker remains the production perception path. Routes to a v2:
condition on motion (reconstruct the frame difference or the flow rather than
pixels), a transformer decoder in the style of SLATE, or slots restricted to the
motion region.

---

## World model, first iterations

Goal: latent dynamics that use actions, beating an action-blind baseline. Metric:
a 16-step open-loop rollout, MSE of the tail latent, true against shuffled
actions.

- v1 (GRU + reconstruction + latent MSE): advantage 1.000 — actions ignored.
- v2 (+ delta prediction, 50% scheduled sampling, inverse dynamics 0.05): 0.993.

Diagnosis worth keeping: the frame latent is dominated by background and scroll,
and the effect of an action — two pixels of hero displacement — is
indistinguishable inside the MSE of a 256-dimensional latent.

**v3, ego-centric** (`world_model/ego.py`): a 48×48 crop around the hero plus
velocity and action → GRU → (dx, dy). The hero's trajectory is extracted offline
by the motion tracker.

- v3.0 (raw trajectory, stride 2, 4 epochs): advantage **1.063**
- v3.1 (+ median-smoothed trajectory, stride 1, scheduled sampling of velocity,
  6 epochs): advantage **1.093**, loss 4.0 → 2.0
- v4 (43 episodes): advantage **1.19**

---

## Audio ablation

Task: action prediction on `datasets/smb_warps_tas` (17,866 frames). Identical
split (last 10% for validation), 4 epochs, AdamW 3e-4, batch 256, seeds 0–3.
`BCNet` (video) against `BCNetAV` (video + 260 ms log-mel).

| Seed | Video only | Video + audio |
|---|---|---|
| 0 | 0.510 | 0.694 |
| 1 | 0.580 | 0.619 |
| 2 | 0.565 | 0.616 |
| 3 | 0.673 | 0.695 |
| **mean** | **0.582** | **0.656** |

Audio wins on 4 of 4 seeds, mean gain about 7 points. The tail's majority
baseline is 0.717, and the best epoch of both variants runs into it — the tail of
this episode is monotonous — which is why the comparison uses the final epoch.

---

## Self-imitation on Super Mario Bros.

Eight rounds × 8 rollouts, reward = progress + Δscore − 300×death, retraining on
the best third. Mean progress 335 → 472, maximum 407 → 639, and the final round
took no deaths.

---

## Jump hold

Progress over 3600 frames, 3 runs: no hold [643, 515, 643], hold 24 [686, 725,
612], hold 32 [680, **2136**, 1020]. A floor of 32 was adopted.

The 2136 outlier is the point: with a long enough hold the agent occasionally
clears a section it otherwise cannot, and the distribution becomes bimodal rather
than merely shifted.
