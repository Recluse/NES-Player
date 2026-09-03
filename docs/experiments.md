# Experiment log

Newest first. Negative results are here on the same footing as positive ones —
in this project roughly half of what was learned came from working out why
something failed. Every entry names the script that reproduces it.

---

## A frame stack that reaches seconds: first real signal

Four consecutive frames span 67 ms. That is enough to read a velocity and
nothing else, which makes the policy effectively memoryless — a poor model of a
player, who remembers where the enemy came from and which way the level was
going. Sampling the stack geometrically instead of consecutively buys time depth
at almost no cost: `wide` is six frames at 1,2,4,8,16,32 back (0.53 s) and
`long` is eight reaching 128 back (2.1 s).

Trained on the same Double Dragon episodes, same recipe. Validation accuracy
barely moves — 0.618, 0.656, 0.625 — which by now is expected to say nothing.
Ten paired runs of play:

| Window | Score | Progress | Attack frames |
|---|---|---|---|
| short, 67 ms | 142.7 ± 33.0 | 22.3 | 1340 |
| wide, 0.53 s | 153.1 ± 30.3 | 28.5 | 1330 |
| long, 2.1 s | 155.5 ± 49.9 | **56.8** | 1499 |

Paired against the short window, per seed:

| Arm | Metric | Mean diff | t | Bootstrap P(>0) |
|---|---|---|---|---|
| wide | score | +10.4 | 0.58 | 0.72 |
| wide | progress | +6.3 | 0.67 | 0.74 |
| long | score | +12.8 | 0.68 | 0.76 |
| long | **progress** | **+34.5** | **2.25** | **0.996** |

Every row but the last is the same coin flip every measurement in this log has
produced. The last one is not: 2.5× the progress, medians 55.9 against 21.6 so
it is not one outlier dragging a mean, and a bootstrap that keeps the difference
positive 996 times in 1000.

It is also the metric that *should* respond. Progress is how far the agent moved
the level along, which is exactly what needs memory — to go forward you have to
know you have already been to the left. Score in a beat-em-up accrues from
flailing, which is why it stays silent.

Two cautions, both earned today. This is one game and ten runs, and this session
has already produced a +15 that was a measurement bug and a rising training
curve that was measuring one memorised moment. And the win is on the secondary
metric while the headline one shrugs.

## Leading the target: better accuracy, worse play, and a mechanism for why

A player shoots where the enemy is going, not where it is. The tracker already
computes a velocity per object, so the cheapest possible version of that idea is
to aim the attention target ahead of them: `--attn-lead N` extrapolates every
box N frames forward before it becomes the supervision target. No model, one
multiplication.

Accuracy went up, on both settings, and with a suggestive shape — the leading
models start *below* the baseline and finish above it, as a harder target
should:

| Attention target | Accuracy by epoch |
|---|---|
| where objects are | 0.431 → 0.515 → 0.625 |
| 15 frames ahead | 0.415 → 0.527 → **0.652** |
| 30 frames ahead | 0.412 → 0.534 → 0.641 |

Play says the opposite, and clearly:

| Model | Score | Attack frames | Paired | t |
|---|---|---|---|---|
| no lead | 151.6 ± 38.8 | 1432 | — | — |
| lead 15 | 132.9 ± 49.3 | 1269 | −18.7 (3W/7L) | −1.13 |
| lead 30 | 115.9 ± 21.0 | 1170 | **−35.7 (3W/7L)** | **−2.44** |

The 30-frame lead is significantly *worse* — t = −2.44, the same magnitude that
made the memory result convincing, pointing the other way. And the damage scales
with the lead, which is what a real effect looks like rather than noise.

The attack-frame column explains it. Leading costs 163 and 262 frames of
attacking respectively, t = −2.8 and −4.2, far stronger than the score signal.
The agent is looking at empty floor where the enemy will be and not hitting the
enemy that is in front of it. In a beat-em-up contact is now; the useful lead is
zero, and a quarter-second lead is enough to miss.

This is the clearest instance yet of the pattern this log keeps recording: **the
better-cloning model is the worse player.** Leading makes the demonstrated
behaviour easier to predict — the target moves smoothly instead of jittering
with the tracker — while making the actual decisions worse.

### Offering the lead instead of imposing it

The fault above is in how the question was asked, not in the idea. A single
lead value *replaces* now with later. Marking both — `--attn-lead 0 15` unions
the current and extrapolated boxes into one target — leaves the network to
weight them.

That change removes the damage entirely:

| Attention target | Score vs plain | t | Attack frames | t |
|---|---|---|---|---|
| forced +15 | −18.7 | −1.13 | **−163** | **−2.82** |
| forced +30 | **−35.7** | **−2.44** | **−262** | **−4.24** |
| offered 0 and +15 | +2.3 | 0.15 | −22 | −0.41 |
| offered 0, +15, +30 | +4.0 | 0.21 | **+210** | **+3.66** |

Score is a draw in both offered variants, so **the lead adds nothing to a
beat-em-up** — the expected answer, since contact happens now. But the attack
column is not a draw. Forcing the lead cost 163 and 262 frames of attacking;
offering it costs 22, and offering three horizons *gains* 210 (t = 3.66).

So the network does exactly what it was given the option to do: it keeps
attending to the enemy in front of it, and the extra marks neither help nor get
in the way. Given a choice it declines the lead; given no choice it obeyed and
played worse. That distinction — between prescribing a strategy and offering
one — is the transferable part of this experiment.

One caveat on the three-horizon variant: its accuracy is the worst of the set
(0.606) and its score spread the widest (±69). Painting three boxes per object
covers enough of the screen that "look here" stops being a constraint, so its
+210 attack frames are more likely loosened supervision than insight.

The idea is not closed. Leading should pay where a projectile takes real time
to arrive, which needs a shoot-em-up whose measurement can resolve anything —
Gradius currently cannot, since every model there dies to the same wave.

---

### Ten times wider is worse, and the 2.1 s window wins again

Asked for ten times `long`: `epic` reaches 1280 frames back, 21.3 seconds,
eleven frames and 33 input channels. It is the worst of the four, and worst from
the first epoch — 0.336 against 0.421–0.433 — so it is not undertrained, it is
being fed something harder to learn from.

| Window | Span | Accuracy by epoch |
|---|---|---|
| short | 4 | 0.421 → 0.554 → 0.618 |
| wide | 32 | 0.433 → 0.553 → **0.656** |
| long | 128 | 0.431 → 0.515 → 0.625 |
| epic | 1280 | 0.336 → 0.468 → **0.533** |

In play, a fresh duel of short against long against epic:

| Model | Score | Mean diff | t | Bootstrap P(>0) |
|---|---|---|---|---|
| short | 124.8 ± 17.9 | — | — | — |
| long | 159.3 ± 41.9 | **+34.5** | **2.44** | **0.999** |
| epic | 128.9 ± 27.4 | +4.1 | 0.40 | 0.65 |

**This is the second independent win for the 2.1 s window**, and on the other
metric: the first duel had it ahead on progress (+34.5, t=2.25) with score
silent; this one has it ahead on score (+34.5, t=2.44) with progress silent.
Two separate runs, two different metrics, same direction, both past t=2.2. The
2.1 s window is now the strongest result in this log.

`epic` is nothing, and its failure is informative. Twenty-one seconds ago has no
pixel in common with now, so nine of the eleven channels carry noise that
dilutes the two that carry signal — and a third of every episode is lost to
having no history that far back. A convolution has no way to know that the
object in a frame from 21 seconds ago is the same object it is looking at now;
it sees eleven unrelated pictures. Past a couple of seconds, more frames is the
wrong mechanism. Carrying a *conclusion* forward — a recurrent state, or the
object memory that already exists — is the right one.

The second game was chosen as a different genre — a horizontal shoot-em-up
instead of a beat-em-up. On validation accuracy it agrees, and more cleanly than
Double Dragon did: 0.866, 0.885, **0.915** against a 0.449 baseline, monotone in
window width, with the long model ahead from the first epoch (0.828 against
0.797) rather than catching up at the end.

The play measurement says nothing at all:

| Window | Survived | Deaths | Paired |
|---|---|---|---|
| short | 617.4 ± 837.2 | 2.7 | — |
| wide | 620.4 ± 836.1 | 2.7 | +3.0 (9W/0L) |
| long | 619.6 ± 836.4 | 2.7 | +2.2 (8W/1L) |

Three frames out of six hundred. The 9W/0L looks impressive and is not: every
arm dies at frame 352–356 of gameplay, a spread of four frames, to the same
scripted first wave. The models are uniformly bad here in exactly the same way,
so the measurement has no resolution — winning by three frames ten times is one
observation about tie-breaking, not about skill.

**So the hypothesis is neither confirmed nor refuted.** Double Dragon showed a
real effect on progress; Gradius agrees on the proxy metric and cannot speak on
the real one. Confirming this properly needs a game where the models differ
enough to be separated — or many more Double Dragon runs.

Getting even this far required fixing the harness three times, each fault
producing confident numbers about nothing:

- it never pressed START, so on Gradius it measured the attract demo, which
  plays itself. All three checkpoints returning identical progress to the
  decimal is what gave that away;
- "progress = camera scroll" is a clock in an auto-scrolling game, not an
  achievement. The metric is now chosen from what the game reports;
- a fixed number of START presses lands wrong on some seeds, and a run that
  never started reports surviving the whole episode. The start is now confirmed
  by the world moving, and an unstarted seed is dropped for every arm.

**What this is not.** Two seconds of frames is not "holding the level in your
head". A convolutional stack cannot associate a thing seen now with the same
thing seen 128 frames ago; it sees eight independent pictures. Real memory needs
a recurrent state carrying a *conclusion* rather than pixels, or an explicit map.
That the 2.1 s window helps at all suggests the ceiling here is not the idea but
the mechanism.

---

## Self-imitation improved its own numbers and not its play

With cloning apparently at a ceiling — two Double Dragon models 25 accuracy
points apart played identically — the next thing to try is an objective tied to
play rather than to imitation. Self-imitation with the pixel reward, six rounds
of eight rollouts.

Two faults surfaced before it could run at all. `improve` had no `--state`, so
on a game whose title screen cannot be passed from power-on it would have
recorded eight rollouts of the title, scored every one of them zero and
fine-tuned on the result without complaint. That is precisely how the discarded
Double Dragon dataset came to exist. A round in which nothing moves and every
reward is identical now raises instead.

The round log then looked encouraging:

| Round | Mean progress | Best |
|---|---|---|
| 0 | 6.0 | 27.4 |
| 3 | 27.3 | 104.9 |
| 5 | 20.5 | 33.1 |

And the duel against the model it started from said otherwise:

| Model | Score | Progress | Attack frames | Paired |
|---|---|---|---|---|
| `bc_dd_attn3` | 146.1 ± 21.9 | 28.7 | 1366 | — |
| `bc_dd_si` | 161.0 ± 42.3 | 23.6 | 925 | +14.9 (5W/4L) |

Five wins to four is a coin, the spread doubled, and — the detail that gives it
away — **the quantity being optimised came out lower**. Camera scroll was the
reward, and the trained model scrolls less at evaluation than the model it was
trained from.

The cause is one line: `run_rollout` began every rollout with `env.reset(seed=0)`
and nothing else. On a game started from a savestate that makes all eight
rollouts in a round the same initial condition, differing only by sampling
noise. Six rounds of improving at one moment of one level, measured at that same
moment, produce a rising graph that means very little; evaluated anywhere else
it is gone.

Rollouts now begin after a different number of idle frames. Which is the same
correction the measurement harness needed earlier in the day for the same
reason — a deterministic setup repeated N times looks like N samples and is one.

### Rerun with varied starts: it does not work either

The round numbers immediately fell by a factor of three, which is the honest
version of the same measurement: 2.5, 2.5, 6.2, 7.3, 9.3, 5.6 against the
previous 6.0 through 20.5. Still rising to round four, then falling back.

The duel is unambiguous:

| Model | Score | Progress | Attack frames | Paired |
|---|---|---|---|---|
| `bc_dd_attn3` | 149.1 ± 20.2 | 22.8 | 1472 | — |
| `bc_dd_si2` | 137.3 ± 39.2 | 15.8 | 1044 | **−11.8 (3W/7L)** |

Worse on score, seven losses to three, spread doubled again, and — for the
second time — the reward being optimised comes out lower at evaluation than the
model it started from. Camera scroll 15.8 against 22.8.

**Self-imitation does not improve this policy on this game.** Both runs are
reported; neither checkpoint is adopted; the default stays `bc_dd_attn2`.

What the two attempts together say about why is more useful than the result.
Six rounds of eight rollouts is 48 episodes of experience, and the evaluation
spread is ±20 to ±40 points. An effect small enough to hide in that is also
small enough that fine-tuning on the top two rollouts per round will chase noise
rather than skill — and the doubled spread in both runs is what chasing noise
looks like. The mode collapse seen earlier on BT&DD is the same mechanism at a
later stage.

So the next attempt at this needs to change the budget, not the hyperparameters:
far more rollouts per round, a wider kept slice, and the original demonstrations
mixed back in to stop the distribution narrowing. Tuning `keep_frac` or the
temperature against a ±30 measurement would just be the lane sweep again.

---

## Validation accuracy is not a measure of play

Once the tracker and the instincts were fixed, the Double Dragon dataset was
collected again — the existing model had been cloned from demonstrations
produced by the broken versions — and a new model trained on it, same recipe,
same hyperparameters, 35 fresh episodes.

It came out far worse on the metric:

| Model | Data | Validation accuracy | Majority baseline |
|---|---|---|---|
| `bc_dd_attn2` | old instincts | **0.864** | 0.318 |
| `bc_dd_attn3` | fixed instincts | **0.618** | 0.326 |

Twenty-five points is not a small difference, and the majority baselines are
almost identical, so it is not that one dataset is more monotonous than the
other.

Then both were made to play, 10 paired runs of 3000 frames:

| Model | Score | Progress | Attack frames | Paired |
|---|---|---|---|---|
| `bc_dd_attn2` | 149.4 ± 37.1 | 18.2 | 876 | — |
| `bc_dd_attn3` | 146.2 ± 40.9 | 25.9 | 1366 | −3.2 (3W/7L) |

**They play the same.** A 25-point gap in validation accuracy corresponds to no
difference in play at all.

The explanation is that the metric answers a different question than the one it
is usually read as answering. Validation accuracy measures how *predictable* the
demonstrated policy is from four frames. Old instincts stood in a corner
repeating one manoeuvre, which clones beautifully. Fixed instincts close on
enemies, align by depth, strike and move on — behaviour that depends on the
situation, and is correspondingly harder to guess.

This was already visible in the table without being understood: Super Mario
Bros. sits at 0.708 against a 0.717 baseline and is the strongest model there.
This experiment is the same phenomenon isolated, with the play measurement
attached.

Consequences worth stating plainly:

- **The accuracy column in the model table ranks cloneability, not skill.** It
  is still worth reporting — against its baseline it says whether anything was
  learned at all — but a higher number does not mean a better player.
- **Retraining on better demonstrations did not produce a better player**, at
  least not measurably at n=10 with a ±40 spread. The default checkpoint is
  therefore left as it was; there is no evidence on which to change it.
- The one clear difference is that the new model attacks far more often, 876
  frames against 1366, which is what the fixed instincts taught it.

Reproduce with `scripts/experiments/dd_checkpoint_duel.py`.

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

### Pinned against the left wall

Watching the recording afterwards turned up a separate fault the score had been
hiding. Once the enemies were down the agent walked into the left edge and
stayed there for the rest of the episode.

The mechanism is a loop between two correct components. The escalation for being
stuck is to back off left and jump with a run-up; against a wall backing off
produces no movement, so no scroll, so the stuck detector fires again, so the
agent backs off into the same wall. Neither part is wrong on its own — the
detector correctly reports no progress, and the manoeuvre is correct in open
ground. The fault is applying a manoeuvre where it is physically impossible.

Guarding that one branch cut the frames spent at the edge from 657 to 423 per
4000 and left the agent in the corner for a tenth of the episode, because three
other rules also issue LEFT: the surrounded rule, curiosity, and whatever is
already queued in the manoeuvre plan. Plugging one hole of four.

The guard now sits in `step()`, the single point every action passes through,
and waits for evidence rather than assuming a wall: LEFT held for 15 frames with
the hero at the edge and the world not moving. Then it drops the press and
clears the plan that produced it.

| | Pinned frames per 4000 | Score |
|---|---|---|
| no guard | 657 | 145.7 ± 28.6 |
| guard on one branch | 423 | 147.9 ± 35.8 |
| guard at the single choke point | **388** | 152.7 ± 40.1 |

Score moves by +7 with four wins and three losses, which is noise again and is
not the claim. The claim is the 41% drop in time spent stuck in a corner, and
that one is a direct count.

Worth noting how this was found: not by a metric, but by watching two minutes of
video. The score barely registers an agent standing still in a corner, because
standing still is not much worse than the flailing it replaced.

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

### Re-measured after the schedule confound was removed

That rematch spawned `nes-player play` twice, and there the policy decided on a
wall clock at 15 Hz while the planner replanned every `--repeat` **emulator**
frames. So the two arms differed in schedule as well as in agent, and a busier
machine would have changed the comparison — which makes "three times farther"
a number that cannot be attributed to the planner.

Repeated through the synchronous evaluator: one process, both arms, the planner
offered exactly the policy's decision ticks and no others, and the decision
indices compared afterwards rather than assumed equal. Eight paired seeds,
3000 frames:

| | best_x |
|---|---|
| BC | 267, 483, 323, 300, 322, 344, 404, 260 → **337.9** |
| BC + planner | 612, 1010, 1008, 1005, 513, 609, 383, 784 → **740.5** |

**+402.6, t = +4.50, 7W/1L.** The planner does help, and the effect survives
the fix — but it is about 2.2×, not the 3–4× the confounded measurement
suggested. The earlier figure should not be quoted.

Script: `scripts/experiments/planner_duel.py`.

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


---

## Attention supervision: the tracker was mostly wrong

The attention targets came from the motion tracker. Measured against the
console's own sprite table on one episode:

| | |
|---|---|
| of the cells the tracker calls an object, actually a sprite | **31%** |
| of real objects, found by the tracker | **57%** |
| objects per frame: tracker / sprite table | 10.9 / 2.9 |

Eight of every eleven "objects" were background. `--attn-source oam` reads the
sprite table instead. Existing datasets did not need re-recording: every episode
replays frame-exactly from its recorded actions (checked — mean absolute
difference 0.000 at frames 0, 1, 10, 100, 600, 1800 and 3599), so the table can
be recovered from a run that never stored it.

Duelled at 10 paired seeds against the tracker baseline: score +28.2 (t = +1.35,
5W/5L), progress −24.0 (t = −1.40). No effect either way. The change is kept
because it fixes a real defect, not because it won.

## Epochs do not buy play

Same data, same everything, 3 epochs against 40:

| | validation | score in play |
|---|---|---|
| 3 epochs | 0.567 | 148.8 |
| 40 epochs | **0.981** | 117.2 (t = −0.14) |

A forty-one point gap in how faithfully the demonstrations are cloned, and no
difference at all in how well the agent plays. An earlier pair with a 25-point
gap gave the same answer. Behavioural cloning is bounded by whoever produced the
data, and the data comes from the instinct policy, which does not finish a
level.

## Perception versus decision

The instinct policy, unchanged, given exact object positions instead of inferred
ones. Eight paired seeds, three minutes each:

| | sprite table | motion tracker | |
|---|---|---|---|
| score | 416.9 | 274.6 | **+142, t = +2.42, 6W/2L** |
| distance | 88.3 | 121.6 | −33, t = −1.40 |
| deaths | 2.0 | 0.9 | **+1.1, t = +3.81, 6W/0L** |

Perception was a real bottleneck **for fighting** and not for getting anywhere:
seeing the enemies properly makes the agent fight better and die more, and does
not make it advance. That separates the two failures — what it sees, and what it
decides — and says a teacher that merely sees more will reproduce this result
rather than improve on it.

## A privileged teacher, trained on results

A small network over 36 numbers from the sprite table — the hero, the six
nearest objects **relative to the hero**, and the camera — cloned from the same
episodes (validation 0.877) and then improved by keeping the best four of twelve
rollouts each round.

**Double Dragon: no improvement over ten rounds.** The reason is visible in the
numbers rather than guessable: distance varied from −8 to +18 across every round
while score varied from 98 to 137 and a death costs 300, so progress supplied
about a tenth of the selection signal. In a beat-em-up the camera holds while
enemies are alive, so every rollout advances about the same distance and there
is nothing for selection to climb. This is not fixable with weights: the game
does not let distance grow without killing the enemies first.

**Super Mario Bros.: it learns.** Twelve rounds, six held-out seeds evaluated
every round:

| | reward | deaths |
|---|---|---|
| first four rounds | −29.0 | 2.54 |
| last four rounds | **+236.8** | **1.33** |

Deaths halved. And the shape of the learning is legible: by round 7 it had
stopped throwing itself at everything at the cost of distance (deaths 0.67, x
down from 773 to 526), and by round 10 it had the distance back and kept the
deaths down (x 739, deaths 1.33). Not "became cautious" — began to tell where it
can charge and where it cannot.

The difference between the two games is not the network. Mario publishes its own
x position and level number, which differ between policies by hundreds; Double
Dragon has no such axis.

## What the fence cost

Turning off the memory-derived feedback channel entirely, eight paired seeds:

| | strict | privileged | |
|---|---|---|---|
| score | 227.4 | 243.1 | −15.8, t = −0.40 |
| distance | 100.9 | 125.2 | −24.3, t = −0.63 |
| deaths | 0.8 | 0.6 | +0.1, t = +0.42 |

Nothing significant. Half of all actions depended on that channel and the result
did not change, which means it was moving behaviour without improving it.

## Metrics that lied

Every one of these looked sensible and was wrong. They are listed because the
pattern is the finding: a plausible number is not a measurement.

| Metric | What it actually was | How it was caught |
|---|---|---|
| "scene cuts" | deaths — the respawn flash | RAM byte `$06B1` (lives) stepped on exactly those frames, three times over two runs |
| "distance" | the sum of twitching | holding RIGHT for 900 frames from the start moves the camera −4.9 px: it is locked while enemies live |
| "finished a level" | walked to the next screen | inferred from a tileset change; a real level change alters the gameplay, and it had not |
| validation accuracy | cloneability, not skill | 0.567 against 0.981 played identically |
| a self-improvement curve | different starting points | rollout seeds differed per round, so dips were seeds with deaths |
| held-out evaluation | zero, every round | the start offset `seed × 37` idled seed 901 for 33,000 frames, into the attract demo |


---

## Four fixes to the object memory, and what they were worth

The instinct policy could not learn that anything was dangerous on Super Mario
Bros. — 60 contacts in a run, zero danger labels. Four separate causes, found by
printing the game state frame by frame rather than by reasoning about the code.

**The lives counter lags by 213 frames.** The hero touches a Goomba at frame
291; the counter moves at 504, after the dying animation and the level restart.
The attribution window was 45 frames for both score and death, so every death
arrived long after its cause had expired. Two windows now: 45 for points, which
appear immediately, and 260 for a death. With the wider window several contacts
are in flight at once, so a death is credited to the **last** thing touched
rather than to all of them — the wide-window version of a bug already fixed
once.

**Knowledge was written and never read.** `runs/knowledge/<game>.json` was saved
after every calibration and loaded by nothing, while `reset` contained a branch
saying "knowledge loaded — exploring" that could not be reached. Every run of
every game therefore spent its first 512 frames standing still, re-measuring its
own jump height. On Double Dragon that is 17% of a three-minute experiment; on
Mario it is fatal, because the first Goomba arrives at frame 504 while the agent
is still on the calibration protocol.

**The recognition threshold was half of what it should be.** Measured over pairs
of sightings that are certainly the same object — one track, at most eight
frames apart — against pairs that are certainly not, the 16×16 grey crop
separates them best at a distance of **55**. The code used 28. That is tight
enough to file the same sprite as a new object whenever it animates, which is
where 251 clusters in one run of a ten-object game came from. At 55 it is 108.

**Fighting outranked avoiding.** An object that has killed us and never once
paid for being hit is an obstacle, not an opponent, and the fight rules ran
first, so the agent walked into the same Goomba on every life. The distinction
comes from the memory rather than from knowing the game: a Double Dragon thug
kills and also gives points when struck, so it is worth engaging.

Super Mario Bros., distance over two minutes, eight paired seeds:

| | distance |
|---|---|
| privileged, before these fixes | 556.9 |
| strict, after | 674.5 |
| privileged, after | **1070.7** |

**+513.8 against the same arm before, t = +3.05, eight seeds out of eight.**
Score appears on six of eight seeds where it had been zero everywhere.

Double Dragon, three minutes, eight seeds: score 310.2 against 227.4
(+82.9, t = +1.13, 6W/2L). The right direction, not significant — as expected,
since what limits that game is not what it remembers.

### A method note

Four measurements before these said "no effect", and all four were taken with
the feedback channel at its `strict` default, where the machinery being fixed is
switched off by construction. The fix was being tested with the thing it fixes
disabled.

## Object recognition: two richer descriptors, both worse

The 16×16 grey crop includes the scenery behind the object, so in principle the
same Goomba against a hill and against a brick wall are two clusters. Two
replacements were built and measured against it.

**Masking the crop with the tracker's motion mask** changed nothing at all —
eight paired seeds identical to the digit — and pushed the cluster count to the
256 cap. A frame difference marks where the object *was* as well as where it
*is*, so it outlines a smear rather than a sprite.

**Separating the sprite from the background by colour** — the ring around a
box is background by construction on a tile machine, so the colours in it are
the background's — then describing what is left by a 12×12 silhouette and a
64-bin palette histogram, canonicalised for mirroring so that a character facing
left and right is one character.

| descriptor | balanced accuracy |
|---|---|
| 16×16 grey crop | **0.849** |
| background removed, silhouette + palette | 0.674 |

The richer one accepts far too much (same-object 0.84, different-object 0.51):
with the background gone, everything on one screen has the same few colours and
a coarse silhouette. Deleted.

The ground truth is what made this readable. Scored against "any two sightings
of one track", both descriptors look mediocre and close together — 0.659 and
0.690 — because a track that runs long enough drifts onto a different object,
and the descriptor gets charged for the tracker's mistake. Restricting to
sightings at most eight frames apart separates the two answers.


---

## Earning the death signal instead of taking it

Once the object memory worked, the observation fence had a price: on Super Mario
Bros. the agent covered 674.5 with no feedback against 1070.7 while reading the
emulator. Lowering the fence was not an option, so the signal had to come from
somewhere a player can see.

Measuring first. Every candidate signal was logged per frame across 41,400
frames of two games, with the emulator's lives counter kept as ground truth and
nothing else:

| | ordinary play | at a death |
|---|---|---|
| mean absolute frame change | p99 ≈ 20, max 26.8 | 98 – 141 |

A death is two cuts, not one — the screen goes black and then the level is
rebuilt — and the two are separated by what follows them:

| | screen after the cut |
|---|---|
| death | mean 0.0 |
| restart | mean 141.0 (Mario), 98.4 (Double Dragon) |

So the rule is: the picture was replaced, and replaced by nothing.

| | real deaths | detected | false alarms |
|---|---|---|---|
| 5 runs, 34,200 frames, two games | 12 | **12** | **0** |

The cut is late — 4 frames behind the lives counter on Mario, 129 on Double
Dragon, and the counter is itself 213 behind the hit — so the attribution window
went to 400 frames, which is measured rather than padded.

Super Mario Bros., distance over two minutes, eight paired seeds:

| | distance | |
|---|---|---|
| `strict` | 674.5 | |
| `visual` | **1083.1** | +408.7 against strict, t = +2.24 |
| `privileged` | 1070.7 | `visual` is +12.5 against it, t = +0.22 |

The honest signal recovers the whole benefit and is statistically
indistinguishable from reading the machine. Double Dragon is unchanged
(score 328.5 against 310.2, t = +0.56), as expected on a game where the answer
is to hit things rather than avoid them.

What it still cannot do: read the score, so `reward` labels never form; and tell
a death from finishing a level, since both replace the screen and neither game
reached the end of one while this was measured.


## Better data, worse validation, better play

The Mario datasets had been recorded by the instinct policy as it was before any
of today's fixes — the one that stood still calibrating for 512 frames, walked
into the first Goomba and did not recognise it next time. Cloning is bounded by
whoever produced the data, so the data was re-recorded with the improved policy.

The first attempt made things **worse**, and interestingly so. The new model
survived all 3000 frames of every run with zero deaths, which the duel picked as
its metric because nobody scored, and it looked like a win. It was not: distance
−0.1, and pressing bare `B` in 91% of frames without ever leaving the start.

The cause was a change of mine that had measured as harmless. Mid-air steering —
letting a running jump drop its direction to get out of the way — never won a
measurement on the policy itself (+12.6 distance, t = +0.09), and I kept it
because it seemed right. Dropping the direction produces `A+B` and bare `B`:

| actions with no direction | |
|---|---|
| old dataset | 2% |
| with steering | **33%** |
| after removing it | 7% |

A third of the dataset had no direction in it, and behavioural cloning found the
common denominator and held `B`. **A rule can be harmless for the hand-written
policy and lethal for whatever learns from it.** The policy keeps following its
plan regardless; the network only sees the buttons.

Steering was removed, the data re-recorded, and the same model retrained:

| trained on | score | distance | deaths | validation |
|---|---|---|---|---|
| old data | 3.0 | 920.0 | 2.0 | **0.941** |
| new data | **24.0** | **1385.7** | **1.3** | 0.568 |

Score +21.0, winning on 8 of 10 paired seeds; distance +51%.

And the clearest statement yet of what validation accuracy is worth here: the
model at **0.568 plays substantially better than the model at 0.941**. Not
equally, as on Double Dragon earlier — better, by a wide margin. The old data is
easy to predict precisely because a quarter of it is the same pointless retreat,
and its majority-class baseline says so: 0.451 against 0.335. Choosing a
checkpoint by validation accuracy would have chosen the one that walks into a
wall.


### Measured on the game's own axis

Camera scroll is a stand-in for progress on games that do not report their own.
Mario does report it, so `scripts/experiments/mario_reach.py` asks the game.
Eight paired seeds, 4000 frames:

| trained on | reach into 1-1 | deaths |
|---|---|---|
| old data | 702 (21%) | 3.0 |
| new data | **1164** (36%, best run 1888) | **1.6** |

+462, winning 7 of 8 seeds. Neither finishes the level.

The first version of that script cut each run short: it stopped as soon as the
level number changed, and the level number moves during a death as well as at
the end of a level, so every run ended at its first death. It reported
1164 as 755 until that was removed.


## Terrain: the one lethal thing that is not a sprite

Everything the perception layer finds moves. The motion tracker finds it because
it moves; the console's sprite table lists it because the hardware draws it as a
sprite. A hole in the floor is neither, so to every part of this project a pit
does not exist — which is why the agent walks into one at running speed. The
owner put the diagnosis better than any instrument had: *it does not stand at
the pit and it is not stuck, it just never tries to jump*.

`perception/terrain.py` finds the floor without a model or a memory. A level is
tiles over a flat background, and the background is whatever fills the top of
the screen, so a column whose lower band is that colour has nothing to stand on.
Overworld or cave, blue or black, the same test works: it asks "is there
anything here", not "is this a floor". Verified live — the detector watches a
pit's left edge scroll in at columns 226, 224, 221, 219, 216, 214.

One guard was needed: a death fades the screen to black, black matches the
background by definition, and the detector reported a hole across the whole
width. More than 90% empty now means "not looking at a level".

### Two ways of combining the network with the instincts, both worse

| | reach | |
|---|---|---|
| network alone | 1130 | |
| hand over when stuck | 856 | −274, 2W/4L |
| network alone | 1188 | |
| override when a pit is ahead | 1127 | −60, 2W/4L |

The first failed for the reason the owner had already given: an agent that is
never stuck cannot be helped by a rule that waits for it to be stuck. The
second is sound and fires correctly — it simply has almost nothing to do.

### Where the deaths actually are

Eight seeds, 22 deaths, checking whether a hole was even on screen in the two
seconds before each:

| | |
|---|---|
| deaths with a hole visible | 5 |
| deaths with no hole anywhere | **17** |

And they are not scattered. Sixteen of the twenty-two happen at two places:

    200, 201, 201, 202, 203, 203, 204, 204     the first Goomba
    702, 702, 702, 702, 702, 702, 702, 702     the pipes
    789, 1007, 1007, 1295, 1313, 1411

Eight deaths on one coordinate is not bad luck, it is something the policy
cannot do. Looking at those frames: the hero clears the first pipe and lands
among two Goombas standing between the pipes.

So terrain perception is real and correct, and it is not the bottleneck. That is
worth writing down as plainly as a win would be: three hours of work produced a
capability the project needed and no improvement in play, because the thing
being fixed was not the thing that was wrong.

### Keeping the bricks

Both combinations were deleted on the grounds that they measured worse, and that
was the wrong call — an agent that finishes a level does not get there by one
large correct idea but by a dozen small ones that each look like noise alone.
They are back in `policy/combo.py` as named switches, off by default, with a
sweep that measures combinations instead of single changes:

| | reach | |
|---|---|---|
| network alone | 1188 | |
| `pit_jump` | 1127 | −60, 2W/4L |
| `stuck_help` | 1103 | −85, 3W/4L |
| both | 973 | −215, 2W/6L |

Still all worse, and the pair is worse than either — but now that is a
measurement of a combination rather than a reason to throw a part away, and the
next small ability can be measured against every subset instead of only against
the plain network.

### The agent does kill Goombas

It stomps them, which is a partial skill worth building on rather than a total
failure to route around. Counting the score changes over six seeds:

| score change | count | meaning |
|---|---|---|
| +10 | 7 | one Goomba stomped |
| +20 | 9 | two hundred points — a chain, or a shell |

Sixteen scoring events against twenty-two deaths in comparable runs. The first
attempt to count this returned **zero**, because the integration stores the
score divided by ten and a stomp was being looked for as +100. A measurement in
the wrong unit reads exactly like an absence.

## Six ways of measuring the wrong thing

A day spent on the state teacher produced two working mechanisms and one firm
negative, and most of it went on discovering that the instruments were lying.
The six are listed together because they are the same mistake wearing six
costumes, and each was found only because a number looked too tidy.

**The console playing itself.** Self-improvement reported `eval_progress`
702.0 — the same figure to the pixel, on six different seeds, for five rounds
running. A frame-by-frame trace explained it: the agent died three times at
the first Goomba, ran out of lives, and the attract-mode demo took over, which
walks and scrolls and scores by itself. Every frame after that was measured as
progress and trained on as the policy's own choices. 702 is where the demo
parks. `cli/play.py` already had the rule; the measurement paths did not use
it. With the demo excluded the pixel models fall — `bc_smb_new` 1188 → 1098,
`bc_smb_all` 1092 → 925 — and the weaker one was flattered more, because it
reached the demo sooner.

**A reward that paid for dying.** Progress was the sum of forward pixels, so
dying at 700 and walking back to 700 scored 1400, and a life cost 300 against
a ceiling the policy could not pass. Twelve rounds of that bought deaths 3.0 →
1.3 with progress falling 773 → 585: it had learned to stand still. Progress
is now the furthest point reached, and a life costs 100.

**Unseeded sampling.** The policy draws its action from its own softmax, and
the rollout never seeded it, so "the same seeds every round" meant the same
starting point and a different run. Removing the demo made the baseline go
*up*, 693.8 → 801.7, which is impossible for a change that can only subtract —
that was the noise, not the change.

**The wrong operating point.** `_evaluate` sharpened to temperature 0.35
regardless of how the rollouts ran, a default carried over from Double Dragon.
Same weights, same six held-out seeds, only the temperature moving:

| T | reached |
|---|---|
| 0.35 | 335.8 |
| 0.6 | 908.3 |
| 1.0 | **1779.5** |
| 1.2 | 1008.3 |

Five times the reach at the temperature the rollouts already used. A day of
"the loop is not learning" was a policy trained at one operating point and
graded at another.

**Six seeds is not a sample.** With the temperature fixed the loop still chose
badly, because it picks the best of many noisy measurements and the maximum of
noisy numbers is biased upward by construction. One run selected a round that
beat the incumbent by 5 on six seeds and lost to it by 237 on thirty-two. The
held-out set is twenty-four now, with `eval_every` to keep the cost bearable.

**Verifying on the selection set.** The last one was self-inflicted while
checking the others. A forty-round run selected round 27, and a paired test
said it beat its starting point by +176.4 (t=+2.02). Twenty-four of those
forty seeds were the set the round had been *selected* on. Repeated on seeds
the selection never saw:

| | selection seeds | clean seeds |
|---|---|---|
| progress | +176.4, t=+2.02 | **−10.3, t=−0.09** |
| clean | −25.6, t=−0.25 | −102.8, t=−0.97 |

The whole effect was contamination.

### What was actually learned

Two mechanisms, both confirmed by behaviour rather than by a summary statistic.

**Enemies were a labelling problem, not a perception problem.** The teacher
died at the first Goomba with the Goomba plainly in its input — `present 1`,
closing from dx +0.44 to +0.06 at the same height — while its running-jump
probability sat at 0.02 and plain `RIGHT` rose instead: it was letting go of
the run button and walking into it. The reason is in the data. Jump rate by
situation, against each dataset's own baseline:

| | our demos | expert TAS |
|---|---|---|
| pit ahead | +20.4 pts | +7.8 pts |
| enemy ahead | **+1.2 pts** | **+11.5 pts** |

Our own recorded play does not jump at enemies at all, and the clone
reproduced that faithfully — a +20.4 lift in the data became +23.1 in the
network, a +1.2 lift became about +2. Mixing the expert run in at a fifth of
the weight moved the jump probability at contact from 0.02 to about 0.5 and
cut deaths by a quarter. It did **not** change distance (t=+0.35 and t=−1.15),
which is what the counting showed instead: first deaths at the Goomba halved,
14 → 7 of 32 runs, and moved forward to the next obstacle rather than
disappearing, 7 → 11 at x 600–799. Five runs finished with no deaths at all,
against none before. Averages could not see this because the next obstacle is
four hundred pixels along.

**Stalls were a physics problem.** Of 117 real stalls across 32 runs — frozen
x that ended with the agent moving again, so not death animations — 90 were at
x 600–799, up to 1089 frames motionless while pressing run-and-jump. Jump
height on the NES is how long A is held, and this policy resamples every frame,
so it releases A partway up and the jump dies there. `JumpShaper` had existed
for the pixel models since the beginning and `mario_reach.py` even names the
symptom in a comment — an agent without it "cannot clear the first pipe of
1-1". The teacher never had it. Holding for 32 frames cut stalls 114 → 33.

### Self-imitation, six configurations, nothing

Elite buffer persisted across rounds, best-round checkpointing, corrected
temperature, jump shaper inside the rollouts, twenty-four held-out seeds,
forty rounds. No configuration beat its own pretrained starting point on seeds
it had not been selected on. The mechanism is not mysterious: cloning one's own
sampled actions pulls the policy toward its own mean harder than reward-ranked
selection pulls it forward. Recorded as a property of the method here rather
than as parameters not yet found.

The champion remains a pretrain: `state_smb_pre4`, with expert data mixed in
and the jump held, at 1278.5 over forty clean seeds against the best pixel
student's 1098.

## One pretrain is a noisy artifact

The status-bar sprite was being mistaken for the player in 4.0% of frames — a
parked sprite that scores exactly 1.00 for control, ties with Mario, and wins
the tie-break in `max(slots, key=ctrl_prob)`. Six call sites took that plain
maximum. Fixing it is obviously right regardless of what it buys: those frames
measured every object position from a fixed point in the interface.

What it bought, measured, is nothing — and finding that out produced the more
useful result. Retraining on corrected data scored 386 below the old model at
t=−2.47, which reads as a regression. Three pretrains on *identical* data,
differing only in the torch seed, say otherwise:

| pretrain | reached | clean | val_acc |
|---|---|---|---|
| corrected, seed 0 | 964.3 | 634.1 | 0.604 |
| corrected, seed 1 | 1246.4 | 906.6 | 0.612 |
| corrected, seed 2 | 1159.5 | 903.6 | 0.612 |
| old data | 1350.5 | 924.9 | 0.605 |

**282 points of spread from the seed alone.** The apparent regression is barely
larger than that, and the old model is itself one draw; against the corrected
mean of 1123 it comes to z≈1.35. Nothing.

Two things follow. Every single-pretrain comparison on this page is
underpowered, including expert-data mixing — that conclusion ("no change in
distance") happens to survive, but it was never demonstrable with one training
run per arm. The behavioural evidence is what carries those findings: jump
probability at contact moving 0.02 → 0.5, first deaths at the Goomba halving
and relocating, stalls falling 114 → 33. Counts of specific events, not means
of a noisy aggregate.

And validation accuracy is flat at 0.604–0.612 across the three while play
varies by 282. It measures cloneability, not skill, which this project already
knew and keeps rediscovering.

The method to use from here is three pretrains per arm against three, not one
against one. Three times the cost, and the alternative is chasing noise.

## Reaching further, and what that did not buy

Six configurations of self-imitation had failed, and the measurements said why:
on one life the agent reached x 650-720 of a 3266-pixel level and died, so the
rest of the game was territory it had never entered. Nothing that reweights
frames it already has can teach what is not in them.

Go-Explore keeps an archive of places reached, returns to one by restoring the
emulator rather than replaying it, and explores onward. Sixty iterations of
uniformly random button-holds took the frontier to 2352 — nearly twice what the
trained policy averages — in about a minute. A longer run cleared **1-1 and 1-2
back to back with no deaths** and entered 1-3, verified by replaying the found
sequence from a cold start and reading WORLD 1-3 off the screen.

Three of its own faults were found by that clear, each because a number looked
wrong rather than by inspection:

- The frontier sat at 3130 for fourteen hundred iterations and was not stuck.
  It had reached the flagpole on iteration 100; the explore segment was 300
  frames, shorter than flag-descent-castle-next-screen, so every segment ended
  inside the celebration.
- The cell key was (level, camera x). The camera stops before a level ends, so
  every state on the last stretch collapsed into one cell.
- `pick` rated cells against the global maximum x, and x restarts at zero in a
  new level — so the first cells of a freshly opened level scored below every
  cell of the one just finished, and the search abandoned each frontier the
  moment it reached it. After entering 1-2 the archive had gained five cells in
  two hundred iterations. Fixing this got 1-2 finished and 1-3 started.

### Coverage was not the bottleneck

The obvious next move is to train on the newly reachable ground. The search
wrote 300 segments — all of 1-1, all of 1-2, part of 1-3 — against existing
datasets that live entirely in x 0-720. Three pretrains per arm, forty held-out
seeds no selection had touched:

| | reached | clean | levels in 120 runs |
|---|---|---|---|
| first fifth only | **1123.4** | **815** | 2 |
| plus whole level | 887.6 | 464 | 0 |

Worse, and on `clean` — distance before the first death — clearly so, 815 to
464 against arm spreads of 282 and 383.

The reason was predictable and was predicted before the run: Go-Explore presses
buttons at random, so a surviving segment is a *lucky* random one, not a
skilled one. Cloning it teaches randomness that happened to work once. Being
able to reach a place is not the same as having something worth imitating
there, and the data has to be good rather than merely well-located.

That points at what Go-Explore actually prescribes for its second phase, which
is not cloning the exploration: start the agent near the end of a found
trajectory and walk the starting point backwards as it succeeds, so it learns
its *own* actions in those places. Not attempted yet.

### The backward curriculum: local skill, global loss

Go-Explore's own phase two is not cloning the exploration. It puts the agent a
few frames from the end of a found trajectory, lets it play to the finish
itself, and moves the start back when it wins often enough — so the actions
learned are the agent's own, and the trajectory only ever supplies starting
positions.

It works at what it does. From nothing, the agent learned to finish the last
**19.3%** of a two-level route by itself, rung by rung, with a real difficulty
gradient: 8/8 on the first rungs, then 6/8, then 4/8 where a rung it had failed
at 3/8 became passable after training on the previous one's wins. It stalled at
level 1-2, x≈2188, and six further passes could not clear it.

Measured on the actual game, from the start, it is a clear regression:

| | pre4 | after curriculum | |
|---|---|---|---|
| reached | 1350.5 | 867.7 | −482.8, t=−3.44, 12W/28L |
| clean | 924.9 | 526.1 | −398.8, t=−2.70, 6W/26L |

Twenty-two passes of gradient steps on one narrow stretch deep inside 1-2,
against an evaluation that plays from the beginning of 1-1. The policy
specialised where it was trained and lost where it was not; mixing the original
demonstrations back in at half the batch did not anchor it.

That is three attempts in a row — self-imitation, whole-level data, and this —
where training that moves the data away from the opening of the game makes
performance at the opening worse, in one case while genuinely improving at the
thing it trained on. The common factor is worth stating as the next hypothesis
rather than a conclusion: the teacher is a 256-wide MLP over 38 numbers, and it
may simply be too small to hold both the opening of 1-1 and the end of 1-2 at
once. That is testable by widening it before trying any of these again.

### It is not capacity

Three methods lost ground at the opening whenever training moved away from it,
so the teacher being too small to hold both ends of the game was the obvious
suspect: a 256-wide MLP over 38 numbers is not much. Running the same two arms
at four times the width says no.

| | width 256 | width 1024 |
|---|---|---|
| first fifth only | 1123.4 (spread 282) | 1166.4 (spread 293) |
| plus whole level | 887.6 (spread 383) | 965.8 (spread 486) |
| gap | −235.8 | −200.6 |

Neither arm moved beyond its own seed spread and the gap is unchanged. With
three seeds an arm this cannot prove a null, so the claim is only that there is
no evidence widening helps — but there is certainly none, and the hypothesis is
dropped rather than tuned.

What survives is the original reading: Go-Explore data is random play that
happened to live, and no amount of model is going to make imitating it produce
skill. The curriculum's regression is likewise about which frames it saw for
twenty-two passes, not about how many weights it had.

Recording the width in the checkpoint was needed to run this at all — it had
been a default argument nobody stored, so every loader rebuilt 256 and a wider
model could not have been reloaded.

## Telling the policy what an object is: blocked on the descriptor

The teacher's state gives six anonymous objects — dx, dy, velocity, present. A
coin, a Goomba, a pipe top and a platform are the same numbers under that, so
it cannot tell what to avoid from what to land on. The object memory now
produces real danger verdicts, so the obvious move is to hand them over.

Three things had to be built first, and they were worth building. The memory
could not be saved at all, despite a docstring claiming it outlives episodes,
so every session relearned the same enemies from the same deaths. Asking it a
question used to teach it — `update` counts a sighting, which a policy checking
"is that dangerous" must not do. And the checkpoint did not record its game, so
the lookup would have quietly returned nothing.

The verdict rule also needed tightening, and the archive shows why: a cluster
touched 307 times with 5 deaths (0.02) sat beside one touched 29 times with 8
(0.28). The first is something brushed past constantly, the second is an enemy,
and the old rule — one contact, one death — called both dangerous. Two deaths
minimum plus a rate of 0.15 keeps the second and drops the first.

Then the feature failed, for a reason worth recording. With nine well-evidenced
danger clusters the flag still came on in **98% of frames**, and no match
threshold fixes it:

| match distance | frames with a threat flagged |
|---|---|
| 55 | 98% |
| 30 | 98% |
| 20 | 98% |
| 12 | 0% |

There is nothing in between. Genuine matches and everything else sit at the
same distance, because the descriptor is a 16×16 grey crop *including
background*, the background is brick, and brick is everywhere. The clusters
themselves are sound — six of the nine prototypes are visibly the same dark
blob on brick, which is a Goomba — so the memory has learned a real enemy. It
simply cannot answer "is **this** object that enemy".

A flag that is on 98% of the time carries no information, so the flags were
taken back out rather than trained on: the cache rebuild and three pretrains
would have measured nothing. The plumbing stays, including `verdict_of`, which
matches without counting a sighting.

This is the same descriptor already known to be the weak part — two richer ones
were tried and both scored worse at clustering. The task here is different
though: not "are these two sightings the same thing" but "which known thing is
this", and the crop is much worse at the second. Cropping to the sprite tiles
rather than the tracker's box is the untried option.

### The console already names the parts

The crop fails because it is a picture of a place, not a description of a
thing. The sprite table is four bytes per entry — y, **tile index**, attributes,
x — and the tile index is the game's own name for a visual component.
`sprite_boxes` discards it on its first line, keeping only x and y.

Clustering visible sprites by position and keeping their tiles, over 1500
frames of running and jumping through 1-1:

    0xff                          cy 24 always, ignores input   status bar
    0x70,0x71 / 0x72,0x73         cy 188/200, ignores the jump   Goomba
    0x32,0x33,0x34,0x35           cy moves with the jump         Mario
    0x36..0x39                    cy moves with the jump         Mario, next frame

Forty-nine distinct signatures over those frames, against 256 saturated pixel
clusters on comparable data, and the three things above are cleanly separable —
which the crop could not do at any threshold. Identity is exact rather than a
distance, and there is no background in it.

Two caveats found in the same measurement, before building anything on it.
Mario's signature *changes as he animates*, so a whole-set signature is not
stable for one object; the tile has to be the atom, with an object's identity
being the tiles it is made of. And NES games reuse tile slots per level, so
identity is reliable within a level and not necessarily across a game.

Settled afterwards by cropping the sprites out of a frame and looking at them,
which is not an inference and should have come first. Tiles 0x32-0x3C are drawn
at Mario's position, 0x70-0x73 at the Goomba's, 0xFF is the transparent
sprite-0 timing marker, and 0xFC is a blank padding tile that shows whatever is
behind it — so it appears in almost every object and is useless as an
identifier.

An intermediate measurement said those Goomba tiles sat inside the hero's own
object in 43.8% of frames, and that was read as the identification being wrong.
It was not. Restricted to frames where Mario is demonstrably on screen, the
hero is a Goomba in **0 of 202**. The 43.8% is entirely frames where Mario is
absent — dying, respawning, between levels — where a stale enemy slot keeps
enough control score to be chosen by default. That is a real fault, and a
different one: `pick_hero` should decline when the player is not there rather
than return the best of what remains.

A flag based on tile identity marks 41.8% of object sightings here against the
crop's 98% of frames — not the same denominator, so not a clean comparison, and
the honest fire-rate test still has to be run on the same measure once the
tiles reach the episode caches. They are not there yet: the caches store (x, y)
only, so this needs the sprite tables rewritten before it can be trained on.

## Five ways of seeing better, and none of them helped

Identifying which object the player controls was wrong in 26.9% of frames. The
cause is a coupling that is invisible in the code: control was scored only
sideways, as velocity times the pressed direction, and that evidence dries up
when the player is not going anywhere — with RIGHT held, Mario's world velocity
is +0.155 and the camera is barely moving. This agent stalls constantly, so the
true hero earned almost nothing, his score decayed, and any drifting enemy
overtook him. Perception needed the agent to move, and the agent was stuck.

Jumping does not need him to be going anywhere, and nothing else on screen goes
up because A was pressed. Scoring that too:

| | hero correct while the player is on screen |
|---|---|
| sideways only | 66.5% (80.4 / 57.2 / 61.8) |
| plus jump response | **88.3%** (84.7 / 95.1 / 85.1) |

Three other things were tried first and all were worse: keeping the same object
until it disappears (28.6%), the same with a margin before switching (29.9%),
and the same again gated on tiles (29.9%, byte-identical, the tile filter never
fired). What redirected the search was counting the ceiling — the tracker finds
Mario in 3323 of 3330 frames and he simply fails the confidence threshold in
911 of them, so the chooser was never the problem.

Retrained on the corrected features, three pretrains against three:

| | reached |
|---|---|
| broken hero | 1123.4 (spread 282) |
| corrected hero | 968.7 (spread 533) |

No difference (t≈−0.84). Measured at inference only it looked slightly worse
still, which it had to: the model had been trained on states measured from the
wrong object, so changing perception underneath it is a distribution it never
saw.

### The pattern

That is five independent improvements to what the agent perceives or how much
it has seen, and not one of them changed how well it plays:

| | result | effect on play |
|---|---|---|
| pit and floor detection | works, and the network uses it | none |
| richer object descriptors | both worse than a grey crop | none |
| whole-level data coverage | 300 segments across 1-1, 1-2, 1-3 | worse |
| four times the network width | trains fine | none |
| hero identification 66.5% → 88.3% | measured, holds on three seeds | none |

The one intervention that did change behaviour was about labels — mixing an
expert run in moved the jump probability at an enemy from 0.02 to 0.5 — and
even that left distance unmoved.

This is the founding hypothesis of `state_teacher.py`, arrived at again from
five directions: "the teacher's advantage cannot be that it sees more; the
agent above saw everything and still stood still". Perception is not the
bottleneck and has not been for some time. What remains is what the agent does
with what it sees, and the only large measured win in this project belongs to
that side of the line — the planner, at +403 distance and t=+4.50.

The hero fix is kept regardless. It is a correctness result on its own terms,
everything downstream reads those object positions, and it costs nothing.

## The world model has no world in it

The planner is the one large measured win this project has — +403 distance,
t=+4.50 — and it stopped helping once the world model got stronger. That was
recorded as a curiosity. It has an explanation.

`EgoPlanner` scores a behaviour template by the hero's predicted change in
**screen** x over sixteen frames. The camera follows the player, so his screen
position is held near the middle no matter how fast he runs, and screen
displacement stops describing progress. Rolling every template through the
model with the trained policy driving:

| template | camera scrolling | camera still |
|---|---|---|
| run | −17.23 | −3.48 |
| jump now | −8.85 | +4.30 |
| jump later | −8.23 | +4.70 |
| wait | −36.51 | −9.90 |
| back off | −43.76 | −22.87 |

While the camera scrolls every option predicts moving backwards, the best
available score is negative, and `run` ranks *below* both jumps. The planner
duly jumps: 200 jump choices against 38 runs over the same rollouts. It is not
choosing to jump, it is choosing the least-negative number in a coordinate
system that has gone blind.

And it goes blind exactly when the agent is doing well. Standing still, the
scores are positive and ordered sensibly; running with the camera in tow, they
invert. A better model predicts the screen-lock more accurately — that is,
predicts closer to zero for everything — which is why improving the model
removed the planner's advantage instead of increasing it.

The root is one line in `ego.py`: trajectories are extracted as `(cx, cy)`,
the hero's position **on screen**. Everything downstream inherits it. Progress
in a scrolling game lives in world coordinates, which are screen position plus
accumulated scroll, and the scroll is already measured every frame by the
tracker — it is simply thrown away at this point.

Not fixed here: it means re-extracting the trajectories, retraining the ego
model, and re-measuring the planner. Recorded because "the planner stopped
helping" has stood as an unexplained fact for some time, and it is not a
mystery, it is a coordinate system.

### Fixing the coordinates, and finding the real problem underneath

The world model now predicts progress through the level rather than movement
across the screen: `ego.py` keeps both frames, because the crop it looks at is
cut from the screen while the thing it must predict is invisible there.

One trap on the way. World x rose on 83% of steps and still finished an episode
at −4408: a death or level change replaces the picture and phase correlation
reports it as an enormous scroll, so a few scene cuts swamped the sum. Clamped
to 6 px a frame, world x runs 48 → 1508 across an episode whose screen x never
leaves 47..143 — fifteen hundred pixels of progress the model could not see.

The coordinate fault is genuinely fixed:

| template | before (screen) | after (world) |
|---|---|---|
| run | −17.23 | +30.12 |
| jump now | −8.85 | +35.58 |
| wait | −36.51 | +32.19 |
| back off | −43.76 | +12.76 |

Nothing predicts backwards any more and `back off` is properly worst. But
`wait` outranks `run`, and measured against the plain policy on identical
decision ticks the planner is far worse than useless:

    best_x   bc 1058.2   bc+planner 553.9   −504.4, t=−3.44, 2W/6L

Because the model barely uses the actions at all. Its action advantage — how
much worse a rollout gets when the actions are shuffled — is 1.09, and it was
1.19 in screen coordinates. In world terms almost all of the next step is
inertia, so a model can drive its loss down by extrapolating velocity and never
learning what a button does.

Making the head predict the residual, `head(h) + vel`, so inertia is free and
only the action remains to explain, changed nothing: advantage 1.085, and the
ordering got worse, with `wait` taking first place outright. Reverted. The
indifference is not caused by the shape of the target.

So the planner cannot work yet, and the reason is not the planner. A search
over actions is worthless on a model that predicts nearly the same future
whichever action it is given, and this one does — in both coordinate systems,
with either target. Before planning is worth revisiting, the model has to be
shown to depend on the buttons: an advantage of 1.09 over shuffled actions is
close to not using them.

That also puts a question mark over the +403 the planner once measured. It is
recorded here as obtained with a weak model that happened to help; on the
current model, with the coordinates now correct, the planner loses by 504.

## Six things wrong with one experiment

The model was shown to depend on the buttons. It took four repairs, and the
game did not care about any of them.

**The target was noise.** `extract_trajectory` built world x from the pixel
motion tracker, whose hero box scatters 50 px against the console's own copy of
Mario's position; per-frame the target correlated +0.21 with the truth at an
error of 4.5 px, against a signal whose standard deviation is 1.0 px. The
sprite tracker was already in the repository, already exact — 0.0 px error up
to the 90th percentile, and the remaining 4% are not near misses but the
tracker naming a different object, an error that jumps straight to 87 px.
Switching to it, and refusing any step larger than 8 px as a change of subject
rather than a change of position, took the per-frame correlation to +0.72 at
0.84 px, the 16-frame correlation to +0.99, and the fraction of frames with a
usable hero from 84% to 99%.

**The horizon was too short for the question.** Asked of the emulator itself,
holding B+RIGHT rather than nothing is worth 1.7 px over 16 frames, 5.9 over
32, 15.9 over 64; the spread across the four candidate actions grows from 7.7
px to 157.5. Mario is heavy and a quarter of a second is mostly inertia. The
model was not failing to see what the buttons do — over that horizon they
barely do anything. `SEQ` went to 48.

**Training was a regime that does not exist.** Every step of the training
rollout was handed a fresh crop from the frame it was predicting, while the
planner has one frame and must imagine forward. With the crop frozen, the
48-step rollout collapsed towards a constant: ±8 px for every action where the
game spans −42 to +29. Training on one crop and imagination — which is also
48× cheaper in encoder calls — moved the action advantage from 1.03 to **2.39**
and the open-loop rollout error from 3.68 to 0.94. From a standstill the model
finally ranked the four actions in the right order, and its spread across them
matched the emulator's 57.8 px against 66.7.

**Four fifths of the counterfactual data was the game playing itself.** The
branch collector saves a state, replays each candidate action, and restores; out
of lives, SMB runs its own attract-mode demo, and 488 of 604 branch points came
back byte-identical across all four actions. Four copies of one trajectory
carrying four different action labels is not neutral data, it is supervision
that the buttons do nothing — the loss is smallest exactly at the
action-averaged prediction. With a `game_over` guard the dead fraction fell to
18.8%, and the four actions separated for the first time: over 55 frames,
noop +19.1, left −23.9, run +40.3, jump +43.9, where before they had all been
+13 give or take half a pixel.

**The duel was measuring through the wrong sensor.** `planner_duel.py` tracked
with `MotionTracker` while every target the model learned came from the sprite
table, and with `CROP` at 48 px a 50 px centring error decides whether the crop
contains the hero at all. It also took the first slot above a confidence
threshold rather than the best one — SMB parks a static sprite in the status
bar that scores 1.00 — and never passed the camera scroll.

**And the planner had no idea anything could kill it.** `memory.update` was
called with `died=False`, a literal. The object memory learns which things are
dangerous only from being told the hero died, so no cluster was ever blamed,
no verdict ever reached "danger", and the collision half of the score was dead:
zero threats offered across 432 replans, in every duel ever run. What the
planner has been maximising all along is `dx_total` and nothing else — a
controller whose entire objective is "go right fast", in a game about not
dying. Wiring lives through detects the deaths (3 in 3000 frames) but the
verdict still needs two deaths blamed on one visual cluster, which a single
3000-frame run cannot supply; the memory has to be built up and persisted
across runs before the term can fire.

The claim that `_score` mixes world and screen coordinates was checked and is
false: the initial offset is taken at one instant, and both bodies move in
world terms from there.

The scoreboard, all on identical decision ticks against the same reactive
policy at 1058.2:

| planner arm | best_x |
| --- | --- |
| the model that ignored actions | 553.9 |
| model retrained on a clean target | 531.1 |
| plus the sensor fixed in the duel | 577.9 |
| plus 16 frames of plan commitment | 638.5 |
| plus clean counterfactual branches | 579.4 |
| plus the death signal wired through | 590.1 |

Every measurement of the model improved, several of them by more than a
factor of two, and the game metric moved from 553.9 to 638.5 at best — still
40% below simply doing what the policy says. This is the case the literature
warns about directly: lower prediction error does not buy better control. The
model now knows Mario's physics on average and still picks the same best action
as the emulator only 10 times in 27, because it is shown a 48×48 square around
the hero, and the pit he is about to run into is not in it.

Plan commitment is the one change that moved anything on its own, and it is
worth keeping for the reason it was made: replanning every four frames while
emitting only the first press means A is never held the ten to sixteen frames a
jump needs. One seed reached 1005 where every previous run of every previous
configuration stopped at 628.

### A Goomba is a reward

The danger verdict was wired up and still never fired, so the memory was built
over 40,000 frames instead of 3,000: 32 deaths, 10 restarts, 149 clusters. The
three clusters with the most deaths against them all came back **"reward"**.

That is not a bug, it is Super Mario Bros. Stomping a Goomba is worth 100
points and it kills only when the stomp is missed, so it is touched far more
often than it kills — 5 deaths in 133 contacts, 4 in 362, 4 in 192. Neither
`DANGER_DEATHS` nor `DANGER_RATE` is met, the score test is, and the most
dangerous object in the game is filed under things that pay. Out of 149
clusters exactly one earned "danger".

The category was the wrong question. A planner does not need to know whether
something is dangerous; it needs a number to weigh a collision by, and the
tally already holds one. `risk_of` returns deaths over contacts, every object
is now offered to the planner rather than only confirmed threats, and the
collision term is scaled by it: a Goomba weighs 0.038, a cloud 0.0.

Measured, with the memory built once and loaded, against the same reactive
policy at 1058.2:

    bc+planner 631.6   −426.6, t=−2.65, 3W/5L

Against 638.5 for the same model with the collision term dead. The risk term
is neutral on the game metric; it is kept because what it replaces is provably
a no-op, not because it was shown to help.

Ten duels now put the planner between 531 and 639 whatever is done to it, while
the policy it overrides ranges 563 to 1550. The planner does not add to the
policy, it caps it: it takes the wheel on most decision ticks, and five
templates scored by predicted progress off a 48×48 crop cannot do what the
policy learned from demonstration. The next thing worth trying is not a better
model of the same thing — it is letting the planner see past the crop, or
scoring plans by something other than distance.

## The planner was never the problem

Swap the learned model for the console itself — save the state, play each
candidate for real, read Mario's own coordinate and his own death, restore —
and change nothing else. The five crude templates stay, the commitment stays,
and the policy's own next moves join the candidate list, rolled out as a
sequence rather than as a first press.

Four seeds, 3000 frames, against the same reactive policy:

| arm | best_x | median | deaths | vs bc | t |
| --- | --- | --- | --- | --- | --- |
| bc | 668.5 | 652.0 | 11 | | |
| oracle h=48 | 2860.0 | 3105.5 | 2 | +2191.5 | +8.25 |
| oracle h=96 | 2497.5 | 2733.0 | 5 | +1829.0 | +4.45 |
| oracle h=144 | 2445.0 | 2681.5 | 4 | +1776.5 | +3.80 |

Four times the distance, a fifth of the deaths, three seeds of four finishing
the level. Confirmed on eight seeds that took no part in any tuning: +1804.4,
t=+4.78, median 3113.

Two things fall out of it. The shorter horizon wins, which contradicts the
argument that put `SEQ` at 48 in the first place — that reasoning was about a
learned model's ability to tell actions apart, not about planning, and with an
exact model a longer horizon only adds ways to be wrong. And the oracle picks
the policy's own plan about 72% of the time: "BC as a prior" did not have to be
built, an honest search arrives at it.

Then price the model against the objective by swapping only the model back:

| arm | best_x | deaths | vs bc | t |
| --- | --- | --- | --- | --- |
| bc | 668.5 | 11 | | |
| learned h=48 | 794.5 | 12 | +126.0 | +0.74 |
| oracle h=48 | 2860.0 | 2 | +2191.5 | +8.25 |

Identical candidates, identical scorer, identical commitment. **The learned
model captures 126 of 2192 points — 6% of what a perfect model is worth.** Where
the oracle takes the policy's plan 126–144 times out of 188, the learned model
takes it 5–18 times and calls for a jump in a quarter of all ticks instead.
It does not merely mispredict; it systematically overrules a good plan.

So the objective, the templates, the horizon and the commitment are all
adequate, and the entire remaining gap is the model. This retires the earlier
wording that "the model is not the bottleneck" — what the duels showed was that
further accuracy on (dx, dy) does not help, which is a different claim. A model
can predict its chosen quantities perfectly and still be blind for control: in
front of a pit both `run` and `jump` have plausible kinematics for thirty
frames, and the outcome is decided by the edge of the platform, which is not in
the crop.

### Pessimism is not a substitute for a model

If a weak model overrules a good policy, make it earn the wheel: override only
when a plan beats the policy's own by a margin. On the four development seeds
the curve looked convincing — 794.5 at margin 0, 937.0 at 20, 1041.0 at 60,
with deaths falling 12, 9, 7 — and beyond 120 it collapses to exactly the
policy's own number three times over, which at least proves the harness
degenerates honestly.

On eight seeds that had no part in choosing it, margin 60 scores 909.4 against
the policy's 892.2: **+17.1, t=+0.12.** The +372.5 was the margin being fitted
to the seeds it was measured on — the same mistake that once turned +176.4 into
−10.3, made again with the warning already written down. On the fresh seeds the
planner picks the policy 169 times out of 177, so the arm is the policy wearing
a planner's overhead.

The oracle number is now the yardstick. A new model is not "n% more accurate",
it is "n% of 1804", and today's is at 1%.

### Refusing to name the hero makes it worse

The 4% of frames where the tracker names the wrong object are not noise around
a good answer, they are a different answer: the error jumps from 0 px to 87.
So `pick_hero` was given the obvious guard — a minimum confidence, a margin
over the runner-up, and a requirement that the slot was actually seen this
frame rather than being one of the ghosts the tracker keeps alive for 300
missed frames.

Measured on the same 900 frames of live play:

| | before | after |
| --- | --- | --- |
| frames with a hero | 892 | 843 |
| error beyond 16 px | 3% | **7%** |
| the 0.55–0.80 band | 22 frames, 50% bad | 53 frames, 79% bad |

Backwards, and the reason is that the guard aims at the wrong half of the
distribution. `missed == 0` throws out **Mario himself** on the frames where
the tracker lost him for a step — behind scenery, or merged with an enemy —
and he was the right answer there, with high confidence and a position one
frame old. With him gone the field is left to some other object, which then
wins unopposed. A margin cannot catch that: it is not a tie, it is a lone wrong
candidate.

Reverted. The same trick does work in the branch collector and is kept there,
where it took hero visibility from 56% to 95% — but that is a different
question. Choosing a reference point once, before branching, is better done by
declining a doubtful moment; per-frame tracking has no such luxury, because
declining leaves a vacuum that something else fills.

What this actually argues for is not refusing but not losing: carrying the
hero's identity between frames instead of recomputing an argmax on every one.

## A battery of decisions, and the death that cannot be avoided

A duel costs twenty minutes and answers with a number noisier than most of what
it is asked to measure. Prediction error costs seconds and measures the wrong
thing. So the benchmark was rebuilt out of the question the planner actually
asks: play, and wherever the six candidate plans genuinely diverge, record the
frame, the hero, and what the console says each plan is worth. Scoring any
model on the saved battery then needs no emulator and takes seconds.

The first build admitted any point where some plan died — 28 of 200 — and the
model chose a dying plan at every one of them. Before writing that down it was
worth checking, and the check overturned it: at all 28 points **all six plans
die.** Mario was already doomed when the branch was taken. Nothing there is a
decision.

Tightening the rule to deaths that are actually avoidable — some plan dies,
some plan lives — finds **zero such points in 13,560 frames**. Within a
48-frame horizon, death in this game is never a choice: either it is not
visible yet, or it is already settled.

That retires a whole line of work. The collision term, the danger verdicts, the
death-rate weighting — none of them could have helped at this horizon, and
measured, none of them did. It also explains the oracle's death count, which is
2 against the policy's 11 with no death term in its score at all: safety is not
foreseen one plan ahead, it emerges from repeatedly taking the plan that gets
furthest, because dying costs progress across all the plans that follow.

On the 200 surviving points, against a chance rate of 0.167:

| model | top-1 | pairwise | mean regret |
| --- | --- | --- | --- |
| ego_world_v6 | 0.275 | 0.667 | 12.5 px |
| ego_world_v7 | 0.330 | 0.769 | 12.5 px |

Median regret is 0 px for both: at most points the choice is harmless and the
damage is concentrated in a few expensive mistakes — the same shape as the
tracker's error, where 96% of frames are exact and the rest are wrong by 87 px.
Note that v7 ranks better here while scoring worse in the duel, which is a
warning about both instruments: the battery is a proxy measured on 200 points,
the duel is the real thing measured on eight.

## Which of the three suspects it was

A learned model that ranks plans badly can be failing in three places: what the
observation contains, what the loss rewards, or what forty-eight recurrent
steps do to an error. A duel cannot separate them, so remove two of the three —
one forward pass from one observation straight to the six returns the console
measured, no trajectory and no recurrence — and then vary only the input.

4000 branch points across 41 playthroughs, split by playthrough because
neighbouring points are seconds apart in one stretch of level. Trained listwise
against `softmax(G / 20 px)`, each point weighted by how much the choice is
worth, because the returns are mostly ties with a few expensive exceptions and
squared error on the value optimises a median that is already right.

Held out, 936 points from ten playthroughs the probe never saw:

| | top-1 | pairwise | regret | p90 |
| --- | --- | --- | --- | --- |
| always bc | 0.643 | — | 7.4 | 32.0 |
| always jump now | 0.085 | — | 5.0 | 20.0 |
| ego_world_v6 (recurrent, crop) | 0.261 | 0.646 | 6.1 | 23.5 |
| probe, crop + velocity | 0.299 | 0.841 | **3.9** | 13.5 |
| probe, + overview strip | 0.246 | 0.859 | **2.8** | **7.0** |
| probe, + OAM + grounded | 0.311 | 0.864 | 3.0 | 7.0 |
| probe, privileged RAM | 0.363 | 0.853 | **2.4** | **3.5** |

The recurrent model and the first probe see **exactly the same thing** — the
48×48 crop and the hero's velocity — and the probe halves the regret and takes
pairwise ordering from 0.65 to 0.84. Dropping the rollout and scoring the loss
on the decision is worth more than anything that was ever done to the
observation. That is the answer: the largest single culprit was the chain, not
the eyes.

The observation is not innocent either, and the number says exactly how guilty.
The overview strip barely moves the mean, 3.9 to 2.8, and halves the tail, 13.5
to 7.0 — the crop is adequate on the ordinary frame and blind precisely where
it is expensive, which is the pit that starts beyond its right edge. Sprite
positions on top of the strip add nothing further. The console's own numbers
reach 2.4 with a p90 of 3.5, so between the best pixels and perfect knowledge
there is still a factor of two in the tail.

For scale, the ego model at 6.1 px is barely better than always doing what the
policy says at 7.4. Ranked by what it buys, from the current 6.1:

    direct prediction + a listwise loss   6.1 -> 3.9
    an overview strip as well             3.9 -> 2.8   (p90 13.5 -> 7.0)
    perfect knowledge of the state        2.8 -> 2.4   (p90 7.0 -> 3.5)

One more thing worth keeping: every probe has a *worse* top-1 than always
following the policy, which is right 64% of the time, and every probe has far
less regret. The policy's mistakes are rare and expensive; a probe's are
frequent and cheap. Top-1 and regret disagree here, and control cares about the
second.

## The probe drives, and the audit says what it costs

The plan-value probe needs no emulator to run: the policy's plan is a *slot* it
was trained to value, not a sequence to be simulated, so it ranks six options
from one frame and either lets the policy keep the wheel or executes a
template. That makes it the first arm here that is a controller rather than a
measuring device.

Twelve seeds, 9000 frames, progress counted across level boundaries because the
camera's x restarts at zero and a longer window measured on `xscroll` scores a
run *lower* the moment it finishes 1-1:

    bc        1392.2   median 1482.5   deaths 67
    bc+probe  1240.0   median 1162.5   deaths 74   −152.2, t=−0.99, 5W/7L

A draw, after six configurations that lost by 400 to 500. Widening the window
bought almost nothing for either arm — 1209 to 1392 for the policy across three
times the frames, and the furthest of all 24 runs is 1909, which on a
level-folded scale means **nobody finished 1-1**. Six deaths per run each. The
runs are limited by dying, not by time, and the extra frames buy more attempts
at the same wall. That also explains the oracle's four-fold margin: it is the
only arm that does not die.

Then the audit — the oracle evaluating every decision the probe makes, on the
trajectory the probe itself produced. 1626 decisions:

    regret        mean 2.55 px    median 0.00    p90 0.0
    P(regret > 16 px)   5.4%
    P(regret > 32 px)   4.8%
    P(regret > 64 px)   0.2%
    CVaR worst 5%       46.6 px
    the policy's plan was best in 85%; the probe took it 11%
    chose a plan that dies:  50 / 1626

Two hypotheses die here. **Distribution shift is not the explanation**: online
regret on the probe's own states is 2.55 px against 2.8 px measured offline on
the policy's states. The probe is exactly as good where it drives as where it
was tested. And **average regret is the wrong summary** — the median is zero, so
most decisions are free, and everything is in a 5% tail worth 46.6 px.

The number that matters is the last one. **3.1% of the probe's decisions choose
a plan that kills Mario.** And it contradicts the earlier finding that no death
is avoidable within 48 frames: that was measured on the policy's trajectory.
Once the probe drives, avoidable deaths exist, and it walks into fifty of them.

A fatal *label* is not a death, though, and the counts say so plainly. The
audit ran on four seeds, not the twelve of the duel above: 1626 decisions is
406 a run, the 50 fatal choices are 12.5 a run, and the runs actually died 5
times each. Two and a half fatal labels per death. A plan is scored over 48
frames but only its first 16 are executed before the controller replans, so
some of those futures are never reached, and several consecutive decisions
looking at the same approaching death are one event and not three. Reading the
3.1% as "this is the death rate" would be wrong, and any head trained on these
labels has to group them by the death they anticipate, or one future corpse
becomes a fistful of correlated positives.

So the next thing the probe needs is not a better estimate of distance. It
needs to predict that a plan is fatal, and refuse it — and the label it learns
should be the consequence of the decision, which is "execute the commitment,
hand back to the controller, and see", not "hold this plan for 48 frames".

### A perfect safety head is worth nothing

Before training a head to predict that a plan is fatal, borrow one: let the
probe score exactly as before, and let the console mask off every candidate it
knows will kill him. The probe then takes its best surviving plan. This is the
same substitution that settled the planner question — put the ideal component
in first and measure its ceiling.

The mask does its job. Audited, the arm makes **zero avoidable fatal choices**;
the only fatal picks left are states where all six plans die, and nine such
labels turn out to be three actual deaths.

Eight fresh seeds, 9000 frames:

    bc                   978.5   median  746.0   deaths 48
    bc+probe+veto:next  1098.8   median 1062.5   deaths 49   +120.2, t=+0.70

**Deaths: 48 against 49.** A perfect safety oracle, removing every avoidable
fatal choice, changes the death count by one in the wrong direction.

So deaths are not caused at the moment the planner sees them. By the time any
of the six plans can be evaluated, the outcome is already settled — which is
the same thing the battery said when it found no avoidable death in 13,560
frames of the policy's play, and the two together now say it about both
trajectories. A hazard head, learned or perfect, is looking too late. That
question is closed without building one.

Which leaves the oracle's own death count to explain: 2 against the policy's
11, with no death term in its score at all. It avoids death by being right
about *progress* far enough out that the paths towards death score badly before
they become inescapable. Safety here is not a constraint to be added, it is
what an accurate long-range return already implies — and the probe cannot
reproduce it because its errors, though rare, are concentrated exactly in the
states that precede trouble.

### Near-perfect decisions, and still a draw

Collecting on the probe's own trajectory — the DAgger states, the ones the
policy never visits — produced 3096 oracle-labelled decisions across eight
seeds, and the audit of them settles the remaining question before the data was
even used for training.

    regret        mean 1.20 px    median 0.00    p90 0.0
    P(regret > 16 px)   2.9%      CVaR worst 5%   23.8 px
    the policy's plan was best in 88%; the probe took it 12%
    chose a plan that dies:  115 / 3096
      of those:  0 avoidable,  115 with every plan fatal
      about 36 distinct deaths behind those 115 labels

**Not one of the 115 fatal choices had a safe alternative.** Every time the
probe picked a plan that kills, all six plans killed. The veto experiment
showed this with the mask switched on; here it is with the mask off, on a
different set of seeds, and it is the same. Death is never a choice at the
moment the planner is asked.

And the decisions are close to optimal: 1.20 px of mean regret against 7.4 for
always deferring to the policy. The probe is nearly as good as the console at
the question it is asked, on the states it actually visits, and the game is
still a draw — 1229.4 against 1126.8, t=+0.63.

So per-decision regret is the wrong accounting for a sequential problem. A
choice with zero immediate regret still puts Mario in a different place, and a
48-frame return cannot see that one of two equally-good-looking places is a
corner. The oracle wins on the same horizon and the same six plans because its
numbers are exact, and among the near-ties it takes the one that is genuinely
better — hundreds of times, compounding into a trajectory that never reaches
the wall at all.

That is the argument for a terminal value, arriving from measurement rather
than from theory: the 48-frame return has been imitated to 1.2 px and imitating
it better cannot help, because what it leaves out is everything after frame 48.

## Five ways of spoiling a perfect model, and none of them reproduce the probe

The probe ranks plans with a within-point correlation of 0.73 against the
console's own numbers — measured after centring, because a raw correlation is
swamped by the fact that at a good moment every plan does well and at a bad one
every plan does badly. Its mean regret is 2.8 px where always deferring to the
policy costs 7.4 and its own state-blind average order costs 7.2. It is not a
constant and it is not blind. And it draws.

So the question was what kind of imperfection turns a planner that nearly
finishes the level into one that draws. Take the oracle's own values and spoil
them, five ways, six seeds each, everything else identical:

| what was done to the oracle's numbers | best_x | deaths | vs bc |
| --- | --- | --- | --- |
| nothing | 2098.0 | 10 | +971.5 |
| gaussian, σ = 1.2 px — the probe's error | 2401.3 | 13 | +1274.8 |
| gaussian, σ = 4 px | 2443.7 | 13 | +1317.2 |
| gaussian, σ = 12 px | 1824.8 | 15 | +698.3 |
| σ = 1.2 px frozen to Mario's position | 2598.2 | 7 | +1471.7 |
| σ = 4 px frozen to position | 2514.3 | 9 | +1387.8 |
| ±25 px on 5% of scores, exact otherwise | 2383.7 | 10 | +1257.2 |
| ±50 px on 5% of scores, exact otherwise | 2713.8 | 9 | +1587.3 |

The policy scores 1126.5 on these seeds. **Nothing here breaks it.** Not the
size of the error, not its persistence — a mistake that repeats every time
Mario stands in the same place is no worse than one redrawn each visit — and
not its shape, with a tail of fifty pixels on one decision in twenty.

Three hypotheses died in that table. That a pixel of error crosses a decisive
boundary; that a deterministic error traps a controller which loops back to the
place it misjudges; that the damage lives in a heavy tail the mean hides. All
were plausible, all are wrong, and the probe's deficiency is not describable as
noise on correct values at any magnitude, persistence or shape.

### One real mismatch, found while looking

The controller was not executing the plan it scored. The probe values plans
48 frames long, and the arm executed templates **rebuilt at the commitment
length**, which is not the same recipe:

    jump later, as valued:     12 frames of running, then 4 of jump
    jump later, as executed:   12 frames of running, then 10 of jump — 22 long

That is the template the probe picks most often. Fixed to a straight prefix of
the plan that was scored, and measured on fresh seeds: 1083.9 against the
policy's 1140.9, t=−0.40. Unchanged, as expected for sixteen frames out of
forty-eight — but the controller now does what the number describes, which it
did not before.

## The same harness, and a win that was the seeds

Every comparison so far ran the oracle in one script and the probe in another.
Putting the probe inside `oracle_mpc` leaves exactly one thing different
between the arms — who assigns the numbers. The policy's candidate is built the
same way, the commitment is the same, the executed prefix is the same.

Eight seeds it had never seen, 9000 frames:

| arm | best_x | median | deaths | vs bc | t |
| --- | --- | --- | --- | --- | --- |
| bc | 919.8 | 786.0 | 50 | | |
| oracle-5×48 | 2351.4 | 2132.5 | 44 | +1431.6 | +4.51 |
| probe | 1267.5 | 1159.0 | 46 | **+347.8** | **+2.87** |

**The probe beats the policy**, significantly, for the first time. In its own
script the same checkpoint drew twice — −57 and −152 — so the harness was worth
about four hundred points, and the two scripts differ in how the policy's
option is executed: here it is a pre-sampled sequence held for the commitment,
there the policy re-decides live every four frames.

The capture of the oracle's gain, per seed and not pooled, with death as the
only thing that stops either arm:

    seed 700  bc 1319  probe 1313  oracle 2354   -0.01
    seed 701  bc  785  probe 1003  oracle 1900   +0.20
    seed 702  bc  702  probe 1673  oracle 3114   +0.40
    seed 703  bc  789  probe 1011  oracle 1911   +0.20
    seed 704  bc 1563  probe 2106  oracle 1408   -3.50
    seed 705  bc  702  probe 1007  oracle 3105   +0.13
    seed 706  bc  787  probe  720  oracle 3113   -0.03
    seed 707  bc  711  probe 1307  oracle 1906   +0.50

    median +0.16, bootstrap 95% CI [-0.03, +0.40]

Two things to keep honest here. The ratio is far noisier than the difference —
its denominator is a random variable, and on seed 704 the oracle scored *below*
the policy, which makes that row meaningless rather than terrible. And the
median capture of 16% has a confidence interval touching zero, so the right
summary is the difference: +347.8 at t=+2.87, with the share of the ceiling it
represents somewhere between nothing and forty per cent.

Still, after a week in which every arm lost or drew, a learned scorer inside
the planner is finally ahead of the policy it overrides.

### Except it was not the harness, and probably not a win

The claim above that the harness is worth four hundred points was made without
running the other harness on the same seeds. Doing that: `probe_duel` on seeds
700–707 gives bc 919.8 and probe 1359.8, **+440.0** — the same baseline to the
decimal and a slightly larger margin than the script that was supposed to be
better. The harness is worth nothing. The seeds were worth everything.

Executing the policy's option as a held sequence rather than a live
re-decision, tested with `--bc-live` on those seeds, is also worth nothing:
probe 1340.5 against 1267.5, the wrong way for the hypothesis and inside the
noise either way.

Every paired seed the probe has ever run, pooled:

| seeds | n | bc | probe | difference | t |
| --- | --- | --- | --- | --- | --- |
| 200–211 | 12 | 1392.2 | 1240.0 | −152.2 | −0.99 |
| 600–607 | 8 | 1140.9 | 1083.9 | −57.0 | −0.40 |
| 700–707 | 8 | 919.8 | 1359.8 | +440.0 | +2.39 |
| **all** | **28** | | | **+44.2** | **+0.43** |

Bootstrap 95% CI on the pooled difference: [−148, +241]. Fifteen wins,
thirteen losses.

So the probe is **not** established as better than the policy. One set of eight
seeds happened to be hard for the policy and kind to the probe, and on it the
difference reached t=+2.39; two other sets went the other way. The heading
above stands as written because it is what the run said, and this is what
checking it said.

The lesson is about the instrument, not the model. The policy's own score
ranges 920 to 1392 across seed sets of eight, which is larger than any effect
measured all week. Eight seeds cannot resolve differences of this size, and
every conclusion in this file that rests on eight — including several that were
reported as settled — is worth only as much as that.

## Thirty-two seeds, and what survives them

Eight seeds cannot resolve this game: the policy's own mean ran from 920 to
1392 across sets of eight, which is larger than any effect measured in a week
of work. So the standard moved to 32, and the three arms were run again on
seeds 1000–1031, none of them used for anything before.

| arm | best_x | median | deaths | vs bc | t | W/L |
| --- | --- | --- | --- | --- | --- | --- |
| bc | 1027.0 | 1006.0 | 176 | | | |
| oracle-5×48 | 2666.5 | 3110.5 | 163 | +1639.5 | +13.47 | 31/1 |
| probe | 1302.2 | 1318.0 | 200 | +275.1 | +2.64 | 24/8 |

Bootstrap 95% CI: oracle [+1397, +1865], probe **[+74, +480]**.

Two things settle here. The oracle's ceiling is not a seed effect — 31 wins in
32, t=+13.47, and it is worth about 1.6× the policy's entire score. And the
probe **is** better than the policy after all: +275.1 with a confidence
interval clear of zero, 24 wins to 8.

Which reinstates, in weaker form, the claim retracted a few hours earlier. The
retraction was right on what was then known — three sets of eight seeds
disagreeing in sign, pooling to +44 with an interval spanning zero — and the
properly powered run says the effect is real but **smaller than the +440 that
prompted the excitement**. The honest number is +275, and the honest lesson is
that the earlier +440 and the earlier −152 were the same experiment, both
undersampled.

Median capture of the oracle's gain is 0.13. A learned scorer takes an eighth
of what a perfect one is worth, and that eighth is now measured rather than
hoped for.

## Item 2: a tail on the score, and a metric that was hiding the answer

Scoring a plan by its own progress says nothing about whether the place it
lands in is a corner. So each candidate now plays only the sixteen frames that
will actually be executed, and then the policy keeps playing for another
ninety-six — the score becomes the consequence of the *decision*, not the
progress of an open-loop template.

The first run of this reported +201.7 at t=+1.17 and looked like nothing. It
was the metric. `oracle_mpc` measured progress by the camera's x, which resets
to zero at a level boundary, and 26 of 32 runs now finish 1-1 — so every good
run was being capped at about 3120 and everything after it discarded. The same
bug was fixed in `probe_duel` a day earlier and not carried across. Folded
across levels, the same 32 seeds say:

| arm | best_x | median | deaths | finished 1-1 | reached 1-3 | vs bc |
| --- | --- | --- | --- | --- | --- | --- |
| bc | 1027.0 | 1006.0 | 176 | 0/32 | 0/32 | |
| oracle-5×48 | 5166.5 | 7110.5 | 163 | 20/32 | 0/32 | +4139.5, t=+9.23 |
| oracle + tail | **6618.1** | 7122.0 | 147 | **26/32** | **4/32** | +5591.1, t=+11.46 |

Paired, the tail is worth **+1451.6** over the plain oracle, t=+1.91, CI
[−23, +2922], 23 wins to 8, sign test p=0.0053. Stated carefully: the tail
improves the result on most seeds and raises clears of 1-1 from 20/32 to 26/32,
and the size of the mean gain is estimated only loosely. The interval includes
zero because a level boundary makes the distribution lumpy — clearing 1-1 jumps
the score by four thousand — so the sign test is the more expressive
instrument here.

McNemar on the clears themselves is *not* significant: 11 seeds cleared only
with the tail, 5 only without, exact two-sided p = 0.21. Sixteen discordant
pairs are too few. The consistent result is on progress, not on completions.

Deaths per run barely separate the arms, because the better ones spend their
extra time in level geometry they have never seen. Normalised by distance
covered they separate sharply:

| arm | deaths per run | per 1000 px of progress |
| --- | --- | --- |
| bc | 5.50 | 5.36 |
| oracle-5×48 | 5.09 | 0.99 |
| oracle + tail | 4.59 | **0.69** |

The policy dies eight times as often per pixel earned as the oracle with a
tail.

Two things worth keeping separate from the headline. **The policy finished 1-1
in none of 32 runs**, so the "wall at 1900" written up yesterday was a fact
about the policy, not the game: the oracle walks through it, and with a tail it
gets into 1-3. And the earlier claim that the horizon should stay at 48 stands —
this is not a longer horizon for the *plan*, it is a continuation under a fixed
policy after the plan ends, which is what a terminal value approximates.

## The tail target is mostly the continuation's dice

Learning the tail return should have been the payoff: a perfect planner using
it gains +1451 over one that does not, so the target is worth something. Trained
on 4000 points across 41 playthroughs, ranked on the advantage over the policy's
own slot, with separate heads for dying in the tail and crossing a level:

| input | top-1 | pairwise | regret | p90 |
| --- | --- | --- | --- | --- |
| always bc | 0.428 | — | 33.3 | 110.0 |
| always jump now | 0.156 | — | 28.1 | 102.0 |
| strip | 0.257 | 0.535 | 27.9 | 99.0 |
| + OAM | 0.261 | 0.541 | 28.2 | 97.0 |
| privileged RAM | 0.296 | 0.523 | 25.8 | 96.0 |

Pairwise barely above a coin, and **the console's own memory does no better**.
On the 48-frame target the same probe with the same inputs reached pairwise
0.859 and 2.8 px against 7.4. When privileged state cannot predict a label, the
label is usually not a function of the state.

It is not. The continuation is the policy sampling at temperature 0.9, so the
same decision is followed by different play each time it is scored. Measuring
that directly — every candidate scored twice from one save state, with the
policy drawing differently:

    continuation at temperature 0.9
      spread across candidates within a point       21.6 px
      spread between two scorings of the same one   22.4 px
      centred correlation, the part a ranking uses  +0.312
      same best candidate both times                50%

    continuation at temperature 0
      spread across candidates                       5.8 px
      spread between two scorings                    0.0 px
      centred correlation                           +1.000
      same best candidate both times                100%

**The noise in the label is the same size as the signal**, and the oracle
choosing the best of six agrees with itself half the time. No learner can do
better than the reproducible part, and at a centred correlation of 0.31 there
is very little of it.

Which also explains why the oracle *plays* better with the tail while the
target is unlearnable. The oracle is not estimating an expectation — it sees
one realised future, and a future that survived 112 frames really is better
than one that died in them. The learner is asked for the mean over futures,
and no single example in the data is that mean.

The fix is to fix the continuation. `BCPolicy._sample` divided by the
temperature without a guard, so asking for a deterministic policy produced NaN
logits and a refusal from numpy rather than greedy play; `StatePolicy` had
always guarded it. With that repaired, temperature 0 gives a target that
reproduces exactly, and a spread across candidates of 5.8 px which is now all
signal.

## The learned tail probe, in the game (2026-08-21)

With a target that reproduces, the probe learns. On 962 held-out decisions
from ten runs it beats both the policy and the best constant template:

    always bc                pairwise 0.516   regret 23.0 px
    always "jump now"        pairwise 0.530   regret 19.4 px
    probe, pixels            pairwise 0.558   regret 15.6 px
    probe, sprite boxes      pairwise 0.560   regret 17.9 px
    probe, console RAM       pairwise 0.586   regret 14.4 px

The baselines in that table used to be measured over all four thousand points
while the probes were measured over the nine hundred held out — two different
sets read as one comparison. They are now on the same split.

Then the same probes, scoring inside the planner, over thirty-two paired
seeds. Progress is folded across levels; the ceiling arm scores every
candidate through the console with a 96-frame continuation.

    bc                  median   787   mean   896   clears  0/32   deaths 70
    probe, pixels       median  1321   mean  1344   clears  0/32   deaths 85
    probe, console RAM  median   786   mean   990   clears  0/32   deaths 85
    exact tail          median  3121   mean  4257   clears 12/32   deaths 42

    probe, pixels       +449 px  [+225, +675]   t = +3.77   wins 26/32
    probe, console RAM   +94 px  [ -79, +272]   t = +1.04
    exact tail         +3361 px  [+2598, +4176] t = +8.09   wins 32/32

The learned probe is worth 13% of the exact tail. The effect is real — it
beats the policy on 26 of 32 seeds — and it is nowhere near half.

**The offline metric inverts the ordering.** Console RAM is the best input by
regret and the worst by progress, with an interval across zero. Regret is
measured at the states the *policy* reached; the probe drives the game
somewhere else, and the sharper the fitted function, the worse it travels. So
offline pairwise and regret are a filter, never an acceptance test.

That also names the bottleneck, and it is not perception: a perfect
description of the console state does not help. It is the distribution of
states the labels come from.

## Doom has no boundary (2026-08-21)

Every fatal decision has no safe alternative, so the choice that kills must be
an earlier one. `doomed.py` looks one level further: play each candidate's
commitment, then ask the console whether *every* plan from the state it lands
in dies. A decision where some candidates lead to such a state and some do not
is the last moment where the choice still matters.

    lookahead   safe    boundary   lost
     48 frames  95.7%      0.0%    4.3%
     96 frames  92.4%      0.0%    7.6%
    192 frames  87.5%      0.0%   12.5%

368 decisions, eight runs, three horizons. **The boundary class is empty at
all three.** A position is either wholly recoverable or wholly lost, and
which of the six macro-templates is taken never decides it. The same thing
appears independently in the offline data: of 962 held-out decisions, 94 have
a death, and in all 94 every candidate dies.

Labelling doom is therefore not a direction — there is nothing at the
boundary to label. Whatever separates a run that dies from one that does not
happens either further back than 192 frames or at a finer grain than these
six templates can express.

## A perfect hero does not help (2026-08-21)

Item 4 was hero tracking as data association — a beam or an HMM instead of the
current greedy pick. Before writing one, measure what the tracker gets wrong,
against the console's own Mario. His screen position is at $3AD and $CE and
his velocities at $57 and $9F, located by matching every byte in the page
against the quantity it should equal: 0.94 px and 0.00 px of disagreement,
correlations of +0.94.

Over 12000 frames the tracker returns no hero at all in 23% of them, and of
the rest, after removing the constant offset between a box centre and a
sprite corner, 74.4% are within 4 px of Mario and 17% are more than 32 px
away — a different sprite entirely. The planner is therefore looking at the
right object in roughly 57% of frames.

That is a real fault. It is not a limiting one. A probe trained on the
console's hero and playing with the console's hero, over the same 32 paired
seeds:

    probe, tracked hero   +449 px  [+225, +675]   t = +3.77
    probe, console hero   +387 px  [+204, +580]   t = +3.88

    console minus tracked  -62 px  [-342, +217]   t = -0.43   better on 14/32

Perfect identity is worth nothing measurable, and if anything slightly less
than the tracker. Offline it looked marginally better — pairwise 0.591
against 0.558 — which is the same inversion the privileged-input arm showed.

So data association is not the bottleneck either, and the HMM is not worth
writing. Between this, the privileged-input arm and the five perturbation
experiments, everything on the perception side has now been given the
console's own answer and none of it moved the game.

## DAgger made it worse (2026-08-21)

The probe scores best offline on the input that plays worst, so what limits it
is not what it sees but where it is asked to look: regret is measured at the
states the policy reached, and the probe drives elsewhere. The standard answer
is DAgger — collect from the learner's own trajectory, label with the same
oracle, aggregate, retrain. `decision_battery build --driver` does exactly
that, on seeds disjoint from the evaluation set.

4000 new points over 42 runs, 508 of them containing a death against the
original 426 — the probe does take the game to worse places, as expected.
Trained on the union of old and new, 8000 points, and played on the same 32
paired seeds:

    bc                median   787   deaths 70
    probe, original   median  1321   deaths 85   +449 px [+222, +692]
    probe, DAgger     median   643   deaths 44   -110 px [-253,  +34]

Worse than the probe it came from, and worse than the policy. What it chooses
says why:

                     bc   jump later  wait  back off  jump now  run
    original        48%      29%       4%      2%       11%      7%
    after DAgger    18%      30%      21%     14%        9%      9%

It became timid. Waiting went from 4% to 21%, backing off from 2% to 14%,
deference to the policy from 48% to 18%, and deaths halved. It learned not to
die instead of learning to advance: the new data is richer in deaths, death is
terminal in the target, and so aggregation moved the objective towards
avoidance. DAgger supplied many examples of where the probe goes wrong and
none of where it should have gone instead.

The distribution-shift explanation is therefore not confirmed in its naive
form. It may still be right with a balanced target, an oracle continuation
instead of the policy's, or more than one round — but none of that is
measured, and the record should not read as though it were.

## Neither replanning more often nor refusing to answer (2026-08-21)

Two cheap things the design had never varied, on the same 32 paired seeds.

    bc                       median   787   deaths 70
    probe, commit 16         median  1321   deaths 85   +449 px
    probe, commit 8          median  1008   deaths 85   +303 px
    ensemble of 3, defer     median  1011   deaths 89   +270 px

    commit 8   vs the probe it came from: -146 px [-444, +146]  better on 15/32
    ensemble   vs the probe it came from: -179 px [-441,  +63]  better on 16/32

Both intervals cross zero and both win on half the seeds, so the honest
reading is no effect either way — and certainly not the gain being looked for.
Neither goes into the controller.

What each rules out is worth more than the arms themselves.

Halving the commitment does not help, so the probe's errors are not rare
mistakes that persist too long. Replanning twice as often re-derives the same
wrong answer twice as often. That argues against the whole family of
"correct more frequently" fixes.

Three probes trained from different seeds agree precisely where all three are
wrong together, so disagreement is a poor detector of error. The fault is not
variance across training runs — an ensemble would find that — but a bias they
share: they learned the same wrong function rather than three noisy versions
of the right one.

Which is also what DAgger's failure said. Three independent measurements point
the same way: the value function this data and this architecture can produce
is systematically the wrong one, and neither the rate of correction, nor a
vote, nor fresh states change it.

## Reproducible and wrong: the two continuations (2026-08-21)

Searching a macro-action space with the oracle made it worse — 2744 against
5119 over eight seeds, with CEM taken in only 13% of decisions. Widening the
candidate set cannot hurt a planner whose scores are correct, and the scores
are not: under a continuation at temperature 0.9 the same plan scored twice
differs by 22 px while different plans differ by 21. The best of thirty-eight
noisy draws is the luckiest one.

If that is the whole story, a continuation that reproduces should raise the
ceiling. It lowers it. The same oracle, the same 32 paired seeds, differing
only in the temperature the value is measured under:

    bc                       median   787   clears  0/32   deaths 70
    oracle, tail at T = 0.9  median  3121   clears 12/32   deaths 42   +3361 px
    oracle, tail at T = 0    median  2362   clears  7/32   deaths 48   +2458 px

    T=0 minus T=0.9:  -903 px [-1959, +211]   better on 8/32

Temperature 0 is not the same policy with the noise removed. It is a
different, greedier policy, and a deterministic policy in this game stalls: it
takes the same argmax against the same pipe forever. Its value is perfectly
reproducible and describes a continuation that never happens, because when the
planner does hand back the wheel, the policy plays at 0.9.

So the two available targets are both defective, in opposite ways: at 0.9 the
label is relevant and half noise, at 0 it is reproducible and biased. The
morning's fix traded variance for bias without noticing the trade.

This also re-prices the learned probe. It was trained on the deterministic
target, so what it imitates is a scorer worth +2458, not +3361, and its share
is 449/2458 = 18% rather than 13%.

And it names something that has not been tried: the mean over several draws of
the 0.9 continuation. N times the cost, but relevant and reproducible at once,
which neither existing target is.

## The optimiser's curse, demonstrated by reversing it (2026-08-21)

The same CEM search, the same eight seeds, the same macro-action vocabulary,
differing only in the continuation the value is measured under:

    tail at T = 0.9   templates median 5119, clears 4/8
                      + CEM     median 2744, clears 0/8
                      cem minus templates  -2041 px [-3902, -189]   4/8

    tail at T = 0     templates median 2634, clears 2/8
                      + CEM     median 7128, clears 5/8
                      cem minus templates  +2168 px [+645, +4090]   8/8

The sign reverses. With a noisy score, widening the candidate set loses 2041
px; with a reproducible one it gains 2168, on every seed rather than half.
Taking the best of many noisy estimates selects for the overestimated, and
here that is not a correction but the difference between clearing 1-1 in none
of eight runs and in five.

Both defects are visible at once in the medians:

                      templates   + CEM
    tail at T = 0.9        5119    2744
    tail at T = 0          2634    7128

Noise hurts in proportion to how widely you maximise; bias hurts the same
either way. With six candidates the noise largely averages out and the bias of
a stalling greedy continuation is what shows; with thirty-eight the noise
dominates everything.

7128 is the furthest anything in this project has reached, and it came from
enlarging the action vocabulary — which the first, mis-scored run had made
look worthless. Eight seeds, so it is remeasured at 32 before anything is
built on it.

## The ceiling was never the ceiling (2026-08-21)

Averaging the tail over four draws of the same 0.9 continuation, six templates
as always, 32 paired seeds:

    bc                            median   787   clears  0/32   deaths 70
    probe, learned                median  1321   clears  0/32   deaths 85   +449
    oracle, tail at T = 0         median  2362   clears  7/32   deaths 48  +2458
    oracle, one draw at T = 0.9   median  3121   clears 12/32   deaths 42  +3361
    oracle, four draws at T = 0.9 median  7117   clears 20/32   deaths 20  +4590

    four draws minus one:  +1229 px [+234, +2206]  t = +2.36  better on 21/32

The median more than doubles, clears go from 12 to 20 of 32, and deaths halve
— with no change to the candidate set. The only difference is that the value
is an average of four futures instead of one.

Everything measured against "the exact tail" today was measured against one
draw from a noisy distribution. The target the probe was trained to imitate is
worth +2458 or +3361 depending on which defect it carries; a properly defined
one is worth +4590. A share of the gap that read as 13% is 18% against the
target actually used, and less than 10% against the target that should have
been used.

With the CEM reversal this is one mechanism, and it accounts for the whole
day: noise in the score hurts in proportion to how widely it is maximised.
Six candidates, moderately — 3361 where 4590 was available. Thirty-eight,
catastrophically — 2744 where 7128 was. A learned probe is a noisier estimator
still, which is the company its 18% keeps.

Which means the negative results of the day — DAgger, the shorter commitment,
the ensemble's refusal — were all measured against a mis-specified ceiling on
a target that was half dice. They are not safe to keep as refutations, and
they need remeasuring against a target that is properly defined.

## On a target that is properly defined, the probe is worth nothing (2026-08-21)

The averaged tail — four draws of the 0.9 continuation — is the first target
that is both relevant and reproducible. 4000 points, 39 runs, same collector.
Offline the learned probes only draw level with a constant:

    always bc          pairwise 0.491   regret 28.1
    always "jump now"  pairwise 0.562   regret 16.7
    probe, pixels      pairwise 0.567   regret 17.0
    probe, OAM         pairwise 0.548   regret 16.5
    probe, console RAM pairwise 0.513   regret 15.2

And in the game, over the same 32 paired seeds:

    probe on the deterministic target   median 1321   +449 px  t = +3.77
    probe on the averaged target        median  785     -9 px  t = -0.10

The interval is [-170, +172]. Not a weak effect — no effect. What it chooses
changed completely:

                      bc   jump now   run   jump later  wait  back off
    deterministic    48%       11%     7%          29%    4%        2%
    averaged         16%       36%    26%          14%    5%        4%

The old probe handed the wheel back half the time; the new one almost always
imposes its own choice, usually a jump, and arrives exactly where the policy
would have.

Three explanations, and the measurements so far do not separate them:

1. The old +449 was noise exploitation. The target was half dice, the probe
   found structure in it, and the structure happened to help.
2. The old target's *bias* was accidentally useful. A deterministic
   continuation is a greedy policy that stalls, so a plan that survives it is
   one that gets clear of trouble. That may be a better heuristic than the
   true expectation.
3. The averaged target is simply harder: less spread between candidates means
   less signal per point at the same sample size.

The second is the most interesting and the most likely: a correctly defined
value can be a worse thing to learn than an incorrectly defined one, when the
error points somewhere useful.

## The control that should have been first (2026-08-21)

Take the same template every decision. No state, no learning, no branching.
32 paired seeds, the same as everything else:

    bc                            median   787   clears  0/32  deaths 70      +0
    probe, averaged target        median   785   clears  0/32  deaths 85      -9
    probe, deterministic target   median  1321   clears  0/32  deaths 85    +449
    always "jump now"             median  2600   clears 12/32  deaths 42   +3178
    oracle, one draw              median  3121   clears 12/32  deaths 42   +3361
    oracle, four draws            median  7117   clears 20/32  deaths 20   +4590

Priced against the habit rather than the policy:

    oracle, four draws   +1412 px [+241, +2468]  t = +2.41  better on 23/32
    oracle, one draw      +183 px [-905, +1263]  t = +0.32  better on 18/32
    probe, deterministic -2729 px                           better on  1/32
    probe, averaged      -3187 px                           better on  0/32

Three things follow, and none of them are comfortable.

**A single-draw oracle is indistinguishable from a fixed habit.** The +3361
that every learned model was priced against all day is 95% "commit to a jump
for sixteen frames" and 5% choosing.

**The learned probe loses to the habit on 32 seeds out of 32.** Its +449 over
the policy was less than a free heuristic gives, and every "share of the
ceiling" computed today used a denominator that was mostly free.

**The value of choosing is +1412, not +4590.** That is what the averaged
oracle has over the habit, and it is the only part any learned scorer could
ever have been competing for. The rest is the policy's inability to commit to
a macro-action, which needs no model at all.

The measurement costs four minutes and it should have been the first arm ever
run. Every fraction in the sections above is being restated against this
denominator.

## Commitment, not choice (2026-08-21)

All three forward habits, each taking one template every decision and holding
it for sixteen frames:

    bc                        median   787   clears  0/32   deaths 70      +0
    always "run"              median  2409   clears  7/32   deaths 53   +2418
    always "jump now"         median  2600   clears 12/32   deaths 42   +3178
    always "jump later"       median  2828   clears 13/32   deaths 51   +3358
    oracle, one draw          median  3121   clears 12/32   deaths 42   +3361
    oracle, four draws        median  7117   clears 20/32   deaths 20   +4590

    oracle, four draws  minus the best habit:  +1232 px [+235, +2220] t=+2.41
    oracle, one draw    minus the best habit:     +3 px [-990,  +973] t=+0.01

Three pixels. An arm that rewinds the console, scores six plans through 144
frames of future each and spends 160000 emulator frames a run plays exactly as
well as "always jump, a little late". The two numbers, +3358 and +3361, come
from independent runs.

**What works is committing, not choosing.** Every habit that moves right and
holds its plan for sixteen frames is worth two to three thousand pixels; the
policy, re-deciding every four frames, is worth none of it. It never really
jumps, because it never holds A long enough.

**One thing from the whole day survives as a real gain: averaging the tail.**
It is the only arm that beats a habit — +1232, 20 clears against 13, and 20
deaths against 51. That is what choosing well is worth, and it is 3.7 times
smaller than the number called "the ceiling" this morning.

So the problem is not to learn a plan's value; a constant needs no value and
does nearly as well. The problem is to learn the exception — *when not to
jump* — which is 20 deaths against 51, a rare-event detection rather than an
estimation.

## Two different cures, one destination (2026-08-21)

CEM at 32 seeds, continuation at temperature 0:

    templates            median 3054   clears  8/32   deaths 36
    templates + CEM      median 7122   clears 18/32   deaths 12   +1656  t=+3.31

    cem minus the best habit:      +985 px [ -41, +2033]  t = +1.86  24/32
    cem minus the averaged oracle: -247 px [-1359,  +906]  t = -0.42  21/32

The search with a clean score and the six templates with an averaged score
arrive at the same place — 7122 against 7117 by median, 18 clears against 20.
Two unrelated treatments of the same disease, landing together, which suggests
they now share whatever the next barrier is.

Against the habit, though, CEM's +985 has an interval touching zero. Only the
averaged tail separates from a free heuristic with confidence: +1232
[+235, +2220].

Of everything tried, exactly one thing reliably beats doing the same simple
action every time: averaging the value over several futures.

## Averaging saturates at four draws (2026-08-21)

    oracle, one draw     median 3121   clears 12/32   deaths 42   +3361
    oracle, four draws   median 7117   clears 20/32   deaths 20   +4590
    oracle, eight draws  median 7114   clears 18/32   deaths 27   +4215

    eight minus four:  -375 px [-1390, +689]  t = -0.68  better on 12/32

Nothing. Eight draws are not better than four. Averaging is not a knob to keep
turning but a one-off repair: the damage was done by the single draw, and four
already removes it. Every future measurement can use four and pay no more.

The day in three lines: a free habit is worth +3358 and matches a single-draw
oracle to three pixels; averaging over four futures is worth +1232 on top of
the habit and is the only thing that reliably beats it; the learned probe
beats the habit on none of 32 seeds.

## Retraction: the habit control was an oracle (2026-08-22)

The section above is wrong and is withdrawn. `--fixed` short-circuited the
score to a one-hot but did not stop the scoring loop, which appended the
console's real values to the same list; the maximum then took the oracle's
answer whenever any plan gained more than one pixel. The arms called "always
jump later" and "always jump now" chose their template 42% and 45% of the
time and the oracle's pick the rest, so what looked like a free habit beating
a planner was two oracles being compared with each other — which is also the
whole of the three-pixel coincidence between them.

With the loop actually skipped, a real constant is worth **-266 px** against
the policy, not +3178. Blindly holding one template walks into things.

Withdrawn: "always jumping is worth +3178", "a single-draw oracle is
indistinguishable from a fixed habit", "the value of choosing is +1412".
Unaffected: the averaged tail's +1229 over a single draw, its saturation at
four draws, and the CEM reversal — none of those arms use `--fixed`.

The question the control was meant to answer — how much of the oracle's gain
is commitment rather than choice — is still open and still worth answering.

## The habits, measured honestly (2026-08-22)

With the scoring loop actually skipped, 32 paired seeds:

    bc                     median  787   clears 0/32   deaths 70       +0
    always "jump later"    median  483   clears 0/32   deaths 43     -413
    always "jump now"      median  604   clears 0/32   deaths  0     -292
    always "run"           median  628   clears 0/32   deaths 96     -268
    oracle, one draw       median 3121   clears 12/32  deaths 42    +3361
    oracle, four draws     median 7117   clears 20/32  deaths 20    +4590

Every constant is worse than the policy. Commitment alone explains none of the
oracle's advantage: what the oracle has is choice, and the earlier picture —
a large gap between a planner with correct values and everything else — is
restored intact.

"Always jump now" is the instructive one: no deaths at all in 32 runs, and
604 px. It jumps in place forever, perfectly safe and perfectly stuck.

The rescue arm — keep the habit unless the console says this plan is fatal —
lands on 483, identical to the habit it protects. A death veto on top of a
constant buys nothing, which is the same answer `doomed.py` gave from the
other direction: the plans that kill are not the ones a veto can see.

## Sixteen futures, and what they say (2026-08-23)

500 decision points, sixteen continuations per candidate, saved once. Nested
N built from the same rollouts, so N is not confounded with a fresh sample.
Two disjoint panels of N; agreement, the spread of their paired differences,
and the regret of one panel's choice under the sixteen-draw mean:

      N   top-1   top-set   sd of paired diff   regret vs 16   entropy
      1   62.6%    63.3%          55.4 px           5.7 px       0.00
      2   62.8%    76.3%          36.2 px           3.8 px       0.20
      4   64.0%    76.5%          27.9 px           2.3 px       0.30
      8   69.6%    77.2%          21.3 px           1.1 px       0.34

**There is no saturation.** Regret falls monotonically, 5.7 to 1.1, and the
spread of paired differences falls as 1/sqrt(N) (55.4, 36.2, 27.9, 21.3
against the predicted 55.4, 39.2, 27.7, 19.6). The earlier claim that four
draws are enough came from an online tie between four and eight on 32 seeds,
which is noise, not a plateau. Withdrawn.

The number that explains the failed training:

    spread across candidates, the signal      14.9 px
    sd of paired differences at N=4, the noise 27.9 px

At four draws the measurement error on a *difference* is twice the difference
itself. The "correctly defined" training set was collected at exactly N=4, so
the probe that learned nothing from it was learning a label with twice as
much noise as signal.

Common random numbers — one seeded stream per draw, shared across candidates —
shrink that noise for free:

           CRN      independent
    N=2   36.2 px      44.4 px
    N=4   27.7 px      32.6 px
    N=8   19.7 px      25.3 px

An 18-22% reduction in the standard deviation of a paired difference, worth
roughly a factor 1.5 in draws.

Selection rules, each fitted on one panel of eight and scored on the other,
because scoring them against the sixteen-draw mean is circular — the mean
wins by construction:

    mean              4.7 px
    median            5.3 px
    lower quartile    5.8 px
    CVaR worst 25%    6.2 px
    mean - 1 sd       5.9 px

No risk-sensitive rule beats the mean. There is no case for learning a
quantile; the hypothesis is tested and dead.

The greedy T=0 continuation, the target that produced a probe worth +449:

    regret against the honest sixteen   5.2 px
    same best candidate                 65.6%
    picks bc 47%, the honest oracle picks bc 43%

Its choice is about as bad as a single draw. So the bias does not make the
choice better, and "the bias is accidentally useful" fails in that form. What
survives is the other half: T=0 is perfectly reproducible while an averaged
target at N=4 is not, and a biased-but-clean label may simply be learnable
where an unbiased-but-noisy one is not.

## The pre-registered chooser failed (2026-08-23)

`docs/preregistration.md` fixed the architecture, the target, the loss, the
number of draws, the single development comparison, the confirmatory block
4000-4031 and the criterion, before the training data finished collecting.

Development, 438 held-out points from the 2400-point matrix:

    soft target, weighted     regret 10.6 px   <- winner
    soft target, unweighted   regret 12.5 px
    hard target, weighted     regret 11.0 px
    hard target, unweighted   regret 11.0 px
    always "jump now"         regret 11.1 px

    student minus the constant: +0.51 px, bootstrap [-2.50, +3.41]

Both halves of the target earned their place — soft beats hard, weighting
beats not weighting, and soft *without* weighting is the worst of the four,
which says a smeared label needs the confidence weight to be usable. But the
advantage over a constant was indistinguishable from zero, and the stop rule
asks only whether the student is worse, so the confirmatory run went ahead on
a margin that was recorded in advance as thin.

Confirmatory, seeds 4000-4031, never used before:

    bc                median 1316   mean 1225   clears 0/32   deaths 77
    always jump now   median  604   mean  604   clears 0/32   deaths  0
    student           median  643   mean  794   clears 0/32   deaths 42

    student minus bc              -432 px [-626, -237]  p = 0.0005   fail
    student minus always jump now +190 px [ +94, +302]  p = 0.0001   PASS

**FAILED**: the student is beaten by the policy, significantly, and beats only
the constant. The direction is reported as failed and is not re-analysed.

What it chose is the clue worth keeping: `wait` 34%, `bc` 26%, `jump later`
14%, `back off` 14%, `jump now` 10%, `run` 3%. The teacher it was distilled
from picks `bc` 46% of the time and `wait` 5%. The student did not learn the
teacher's policy — it learned to stand still, which on this training signal is
the safest way to be wrong.

That is consistent with everything measured: at the sample sizes available the
label's noise on a *difference* is nearly twice the difference itself, so the
part of the teacher a student can actually recover is the part that survives
that noise, and "waiting is rarely catastrophic" survives it.

## Correction: the label was never mostly noise (2026-08-23)

Two sections above say that at four draws "the measurement error on a
difference is twice the difference itself", and the failed chooser was
explained by it. That comparison put two different quantities side by side and
is withdrawn.

`13.3 px` was the average within-point standard deviation of the six candidate
values. `24.2 px` was the spread between *two* panels' estimates of a pairwise
difference, which is sqrt(2) times one panel's error. Neither is the signal a
learner has to predict, which is the standard deviation of the true pairwise
difference across points and pairs.

Measured properly on the 2400-point matrix — one panel's estimate against the
sixteen-draw mean:

    signal: sd of a true pairwise difference    33.4 px

    N = 1   error 33.0 px   noise/signal 0.99
    N = 2   error 22.0 px   noise/signal 0.66
    N = 4   error 15.0 px   noise/signal 0.45
    N = 8   error  8.5 px   noise/signal 0.25

At four draws the label carries roughly twice as much signal as noise, not
half. At sixteen — what the failed student was trained on — the error is
smaller still.

So label noise does not explain the failure, and "collect four times as many
draws" is not the fix. What remains is that the student agrees with its
teacher on only 52.5% of held-out points drawn from the teacher's own
distribution, while landing in the top-set 79% of the time. It is a weak
imitator where the label is good, not a good imitator of a bad label.

## How accurate an imitator has to be (2026-08-23)

Two days of distilling an argmax never asked what accuracy buys. A perfect
scorer, forced to take a uniformly random wrong candidate a fixed fraction of
the time, 32 paired seeds, four-draw tail:

    error   median   clears   advantage over bc
     0%      7117    20/32          +4590
    20%      1909     4/32          +1666
    50%      1012     0/32           +181  [+1, +360]

Half the advantage requires an error rate below **15.7%**; a quarter requires
below 30.5%. Half of all decisions wrong leaves +181 of +4590 — a perfect
value function, used badly, is worth almost nothing.

The student agrees with its teacher on 52.5% of held-out points: an error rate
of 47.5%, and 41.8% with the console's own memory as input. Both are far past
the point where imitation stops paying. It did worse still — beaten by the
policy — because its errors are not uniform: they concentrate on `wait`.

This closes the direction by arithmetic rather than by any detail of training.
Distilling the choice over six templates cannot work unless accuracy reaches
roughly 85%, and nothing measured here comes near: pixels 52.5%, sprite lists
52.5%, console RAM 58.2%.

What the same measurements suggest instead: two templates carry 87% of the
full candidate set's value and three carry 97%. A chooser over two candidates
starts at 50% from a coin rather than 17%, and the accuracy needed to profit
is a different problem from the six-way one. That is a change of the question,
which is what a closed direction is for.

## Not covariate shift: the student is a wait-attractor (2026-08-23)

The failed student picks `wait` 34% of the time; its teacher, measured on the
teacher's own states, picks it 5%. Two causes look identical from outside — the
teacher would also wait on the states the student reaches, or it would not.
1200 decisions collected with the student driving, each labelled by the same
sixteen-draw teacher:

    candidate     teacher here   student   teacher on bc states
    bc                  48.2%     30.2%           45.8%
    run                  9.4%      3.2%            9.0%
    jump now             9.1%      7.3%           12.5%
    jump later          13.0%     13.7%           12.1%
    wait                 7.6%     37.0%            5.9%
    back off            12.7%      8.7%           14.7%

**The teacher's policy barely moves.** On the student's own states it still
says `bc` 48% and `wait` 8%, against 46% and 6% on the policy's states. The
right answer did not change; the student is wrong.

Agreement falls from 52.5% in distribution to **32.3%** on its own states, so
there is a distribution effect — but it is the student generalising worse off
its training distribution, not the target moving. DAgger addresses the second
and this is the first.

The confusion is a single attractor. From every teacher row, a third to a half
of decisions go to `wait`:

                        bc    run  jump now  jump later  wait  back off
    teacher bc         46%     2%       4%          7%   33%        7%
    teacher run        12%     5%      14%          9%   50%       10%
    teacher jump now   14%     3%      20%         22%   39%        2%
    teacher jump later 15%     9%       1%         26%   38%       10%
    teacher back off   16%     1%      12%         23%   38%       11%

And here is why the training signal never stopped it:

    oracle regret of the student's choice   7.7 px  (always bc: 8.1 px)
    regret on the decisions it got wrong   11.4 px
    spurious waits: 50.2% of all errors, costing 2.1 px each

A spurious `wait` costs two pixels inside a 144-frame horizon. It is the
cheapest possible error to make, so a loss built on that horizon barely
penalises it — and a student minimising that loss drifts to it from every
state. In the game the same behaviour costs 432 px against the policy, because
the horizon prices one wait and the run pays for all of them.

That is a property of the target, not of the student: a value measured over
144 frames cannot see what standing still repeatedly does to a run.

## A longer horizon does not price inaction (2026-08-23)

If waiting is cheap because the label only looks 144 frames ahead, looking
further should make it expensive. 200 points, four draws, the penalty for
taking `wait` instead of the best candidate, at four horizons:

    horizon   best - wait   best - bc   spread   wait/spread   wait is best
        96        23.3 px      17.9 px  14.4 px         1.62          10.5%
       192        35.4 px      38.4 px  24.3 px         1.46          13.5%
       288        68.5 px      78.5 px  53.3 px         1.29          12.5%
       480       137.2 px     153.7 px 105.1 px         1.31          12.0%

The absolute penalty grows sixfold, and so does everything else. Relative to
the spread the loss is actually fitting, waiting gets *less* distinctive, not
more: 1.62 down to 1.31. Lengthening the tail will not fix the wait attractor.
The policy recovers the lost ground; what a longer window adds is mostly
divergence between all the candidates, not a bill for standing still.

## Where the student waits (2026-08-23)

The average penalty for waiting is 23.3 px. The student's own spurious waits
cost 2.1 px. It is not waiting at random — it is waiting exactly where waiting
is nearly free. Splitting its 1200 decisions by the hero's speed:

    spurious waits among the slowest third of states   83.3%
    spurious waits among the fastest third              7.6%

    regret of a spurious wait, slow states   2.0 px  (n=330)
    regret of a spurious wait, fast states   3.3 px  (n= 30)

The rule it learned is "if Mario is not moving, wait". Per decision that is
almost correct — a stalled Mario loses little by stalling one more window. Over
a run it is fatal, because the states where he is not moving are exactly the
ones that need an action to leave, and choosing to wait there is choosing to
stay.

Hero velocity is one of the two scalars in the student's input vector, and the
image half is a 48x48 crop plus a downscaled band. Given a weak image signal
and one strong scalar, the network found the degenerate rule that scalar
supports. That is worth recording against the earlier claim that perception is
closed: it is closed for the *value* system, but a chooser built on this input
has a shortcut available, and it took it.

## DAgger converges, and the attractor moves (2026-08-24)

The earlier round used a target that was half noise. With the sixteen-draw
teacher and the learner's own 1200 states added to training, on held-out
learner-induced states:

    old student, on its own states      agreement 32.3%
    DAgger student, on those states     agreement 47.4%
    DAgger student, on its *own* states agreement 45.2%

It holds up when the distribution moves again, so this converges rather than
displacing the problem — the effect DAgger exists for, and one the noisy-target
round could not have shown. Withdrawing the earlier "DAgger made it worse" as a
statement about DAgger: it was a statement about that target.

But the collapse did not go away, it changed target:

    candidate    teacher here   old student   DAgger student
    bc                 48.6%         30.2%            62.9%
    wait                6.5%         37.0%             2.9%

`wait` fell from 37% to 2.9% and `bc` rose from 30% to 63% against a teacher
that says 49%. The confusion shows it: where the teacher says `bc` the student
agrees 79% of the time, and from every other row it sends 43-51% to `bc`
anyway.

    regret 8.0 px overall, 14.7 px where wrong; always bc is 8.5 px

So the student collapses onto whichever action is safest under the loss —
first the cheapest error, now the plurality class. This is majority-class
collapse, and it is what cross-entropy does when the signal is weak relative
to the marginal.

The consolation is that this failure mode is benign: a model that defers to
the policy plays roughly like the policy, where the previous one played 432 px
worse. It also sharpens the next design, because the only question left is the
narrow one — when to override the default — which is exactly a binary gate.

## The binary gate: a lower ceiling and a much flatter curve (2026-08-25)

Restricting the arm to {bc, jump now} and scoring both through the console,
32 paired seeds, four-draw tail:

    bc                          median  787   clears  0/32   deaths 70
    gate, perfect               median 1672   clears  2/32   deaths 75   +1110
    gate, 20% spurious jumps    median 1655   clears  0/32   deaths 88    +689
    gate, 20% missed jumps      median 1479   clears  1/32   deaths 82    +860
    gate, 20% of both           median 1552   clears  0/32   deaths 87    +675
    full oracle, six candidates median 7117   clears 20/32   deaths 20   +4590

**The offline subset analysis overstated this badly.** On the matrix,
{bc, jump now} recovered 68% of the full candidate set's per-decision value.
In the game a *perfect* gate recovers 1110 of 4590 — 24%. Per-decision regret
does not transfer, which is now the third time the same lesson has appeared.

The two errors are not symmetric, and in the useful direction:

    20% spurious jumps  -421 px [-1011, +97]   keeps 62% of the gate's gain
    20% missed jumps    -250 px [ -831, +347]  keeps 78%
    20% of both         -435 px [ -997, +62]   keeps 61%

Jumping when the console says defer costs nearly twice what deferring when it
says jump does. Defaulting to `bc` under uncertainty is therefore the right
design, and it is now measured rather than assumed.

The real argument for the gate is not its ceiling but the shape of its curve:

    six candidates: +4590 -> +1666 at 20% error   keeps 36%
    binary gate:    +1110 ->  +689 at 20% error   keeps 62%

A learner that is wrong a fifth of the time destroys two thirds of what a
six-way chooser could win and only a third of what a gate could. Against the
gate's ceiling, the accuracy a student already reaches is worth something; the
question is whether 62% of +1110 survives the gap between a corruption model
and a real classifier, which the six-way experiment says is not a small gap.

Also worth noting: the gate has *more* deaths than the policy (75 against 70,
and 88 with spurious jumps). Removing `wait` and `back off` removes the ways
of not dying, so a gate buys progress with risk.

## The gate, and a metric that had to be fixed first (2026-08-25)

A gate over {bc, jump now}: target `delta = Q(jump now) - Q(bc)`, label
`P(delta > 0)` by bootstrap over the sixteen draws, loss weighted by
`|E delta|`, 3600 points from the teacher's and the learner's states.

The first table said the weighting hurt — plain AUC 0.548 with it against
0.668 without. That comparison was rigged by construction: plain AUC counts
every decision alike, which is exactly the objective the unweighted model
minimises and exactly what the weighted model is built to ignore. Scoring each
pair by how much is at stake gives the fair table:

    input        weighting    AUC    AUC weighted by stake
    strip        yes         0.548          0.587
    strip        no          0.668          0.525
    strip, no speed  no      0.675          0.534
    console RAM  no          0.740          0.661
    console RAM  yes         0.609          0.670

Each model wins on its own objective, so "the weighting hurts" is withdrawn.
What survives the change of metric is the input:

**The console's own state carries substantially more of this signal than the
pixels do** — 0.670 against 0.587 on the stake-weighted measure, 0.740 against
0.668 on the plain one. That is the clearest evidence yet that perception is
not closed for a controller of this shape, against the earlier claim that it
was closed on the strength of the value experiments.

**Removing hero velocity changes nothing** — 0.668 to 0.675 plain, 0.525 to
0.534 weighted. The shortcut that produced the six-way student's `wait`
attractor is not what carries the binary gate, so that hypothesis does not
generalise past the task that produced it.

None of this is yet useful. The privileged gate's operating points:

    threshold  jumps   false positives  false negatives
       0.5     15.8%        10.3%            65.4%
       0.6      6.1%         3.3%            84.4%
       0.7      2.3%         0.8%            92.7%

Jumping is right in 22.6% of decisions; at the only threshold with a tolerable
false-positive rate it finds a third of them. A gate this conservative is
close to being the policy with extra steps.

## Neither data nor epochs: the models memorise (2026-08-25)

Training AUC separates "not enough data" from "not enough information", and
answers a third question as well. All three collections merged, 3597 training
points against the earlier 2807:

    input        epochs   train AUC   test AUC   test weighted by stake
    console RAM      40       0.858      0.730            0.533
    console RAM     150       0.919      0.734            0.568
    pixels          150       0.974      0.674            0.524

Four times the training and a quarter more data move the training fit and
leave the test where it was — 0.730 to 0.734. The pixel model fits its
training states almost perfectly, at 0.974, and generalises worst of the
three.

So the limit is neither information (the inputs clearly carry the label on
states the model has seen), nor capacity (it fits them), nor, in this range,
data volume. It is transfer. Each decision point is close to unique in the
representation the network is given, so it memorises instead of generalising.

That is an argument about features rather than about scale: "there is a pit
thirty pixels ahead" is one fact across many states, and a raw crop makes
every instance of it look different. The console's own state does better —
0.730 against 0.674 — because a few of its bytes are that kind of fact
already.

(803 test points, so the standard error on an AUC is about 0.02: the 0.06
between privileged and pixels is real, the 0.004 between 40 and 150 epochs is
not.)

## F1 fires, and the split was never a split (2026-08-25)

Two instruments, no new features. Four rotated folds, both inputs, every AUC
reported separately on the points where the tracker agrees with the console
about where Mario is and on the points where it does not:

    input        AUC   stake   train   hero ok   hero lost    kNN   kNN RP
    pixels     0.562   0.670   0.725     0.560       0.518  0.707    0.665
    console    0.608   0.700   0.661     0.575       0.610  0.746    0.734

    console minus pixels, all points  +0.046  (+0.070, +0.080, -0.002, +0.037)
    console minus pixels, hero ok     +0.014  (+0.062, +0.025, -0.049, +0.019)
    console minus pixels, hero lost   +0.092  (+0.005, +0.196, +0.045, +0.121)
    population: hero ok 73%, hero lost 27%

**The advantage of console memory lives in the quarter of frames where the
tracker has lost Mario.** Where it has him — the only regime a deliverable
operates in — the two inputs are within noise, and one fold is negative. The
earlier claim that "the console's own state carries substantially more of this
signal than the pixels do", and the conclusion drawn from it that perception is
not closed for a controller of this shape, are withdrawn: that was a mixture of
two populations reported as one number.

Worse, and more useful. A training-free k-NN readout on the same features and
the same split scores 0.707 where the trained network scores 0.562, and a
random projection of the same width scores 0.665. So nearest-neighbour
structure is mostly generic — and that pointed at the reason:

    distance in world x from a held-out point to the nearest training point
      exactly 0 px   74.6%
      within  2 px   95.6%
      within  4 px   99.3%
      within  8 px  100.0%

**Every run walks through the same level.** Splitting by run does not separate
the states: a held-out point almost always has a near-twin in training, at the
same place in 1-1, and copying its label works. Memorising the level is a
viable strategy that survives a by-run split, which is exactly the strategy the
train/test gap said the network was using.

So every test AUC in this project's gate and chooser experiments is optimistic,
including the ones used to reject designs. A split that measures transfer has
to hold out a region of the level by world x, or another level entirely. That
is the next change, and it comes before any feature work.

## Forty-five percent of the training set was a tie taught as a refusal (2026-08-26)

300 states, each labelled twice by an independent sixteen-draw matrix from the
*same* save state, so the label's own Monte Carlo variance is measurable
rather than assumed.

The first thing it showed was not about variance. In **40%** of decisions,
jumping and deferring produce **byte-identical** futures — the same returns in
all sixteen draws, in both repeats. 88% of those are airborne states, against
32% of the rest: the button does nothing because Mario is already in the air.

In `draw_matrix_all.npz`, the set the gate trained on, that fraction is
**45.0%**. And the gate's label is `P(delta > 0)` by bootstrap, which on an
exact tie is **0** — so nearly half the training points said *certainly do not
jump* about a decision that has no consequence either way. That is a
mechanical push towards exactly the majority-`bc` collapse the DAgger student
showed, and it is a bug in the target rather than a property of the problem.

With the ties dropped, 180 states:

    Monte Carlo sd of the label      9.1 px
    spread between candidates       19.5 px
    sd of delta across all states   50.6 px

    representation   close pairs   hidden (excess)   distant pairs   hidden/spread
    strip                51.2 px          50.3 px         50.1 px            2.58
    oam                  51.2 px          50.3 px         50.2 px            2.58
    console RAM          44.1 px          43.1 px         53.4 px            2.21

Knowing two states look alike constrains their answers barely at all: the
closest half-percent of pairs differ about as much as arbitrary pairs, and the
variation hidden by the representation is 2.2 to 2.6 times the spread a
chooser has to resolve. Console RAM is the only one that constrains anything
(44.1 against 53.4 for distant pairs).

**The caveat is large and structural.** With 180 points in a thirteen-thousand
dimensional space, the "closest 0.5% of pairs" are not close in any absolute
sense — the threshold lands at 17.1 standardised units. This bounds the
question rather than answering it: it says these representations do not
collapse states usefully at the sample size available, not that no
representation could. Answering it properly needs either many more points or a
representation low-dimensional enough for near-duplicates to exist, which is
the argument for the surface profile stated as a measurement instead of a
hope.

## Eighty thousand labels, and the answer is no (2026-08-26)

Every offline number in this project came from a few thousand decisions,
because a label cost 9360 emulator frames. Two candidates and four draws cost
848, and storing the 48x48x6 input instead of the frame costs 14 kB instead of
161, so eight parallel shards produced **80000 labelled decisions** in an
afternoon — eighteen times the whole previous corpus. The question that buys
is direct: is the crop a bad representation, or an under-fed one?

Two splits, because they ask different things. By run is new playthroughs of a
level already seen. By world x, in quantile blocks with a 320 px buffer purged
around the test region, is ground the model has not seen.

    split            2500 points      largest        n
    by run          0.537 ± 0.041   0.568 ± 0.005   35000
    by world x      0.498 ± 0.051   0.421 ± 0.007   29854

**On known ground, fourteen times the data buys +0.031 of AUC.** The variance
across seeds collapses from ±0.041 to ±0.005: the model becomes stable, and
stably mediocre. Reaching a useful 0.85 this way is not a matter of more
shards.

**On unseen ground it goes below chance and stays there.** 0.421 with a spread
of 0.007 across three seeds is not "failed to learn" — that would be 0.50 with
a wide spread. It is a rule that transfers with the wrong sign: what the model
finds is tied to the geography it was trained on, and in a different stretch
of level the same appearance calls for the opposite action. More data makes it
more confidently wrong.

A third measurement explains part of every earlier gate number. With ties
labelled as they were — probability zero, "certainly do not jump" — the same
model on the same split scores **0.630 ± 0.001** against **0.568 ± 0.005** with
ties at 0.5. A tie is almost always an airborne state, so the network was
being paid 0.062 of AUC for detecting flight, which is easy and says nothing
about the decision. Every gate AUC reported before 2026-08-26 carries that
component.

So the answer to "why not just use the screen" is measured rather than
argued: the screen is what we feed, and with eighteen times the labels the
model learns the place rather than the decision. That is the case for a
representation in which two stretches of level that call for the same action
look the same — which is what the surface profile is for, and it now has a
competitor with an honestly measured zero.

## Correction: that was one block, and the spread was across seeds (2026-08-26)

The section above reports 0.421 ± 0.007 on ground held out by world x and reads
it as a rule transferring with the wrong sign. Four audits say otherwise, and
the headline is withdrawn.

**The block was degenerate.** Fold 3's test range is `x = [593, 601]` — nine
pixels of level holding 10113 points, because runs pile up where they get
stuck. That is one location, not a region, and ±0.007 was the spread across
model initialisations on it, not across geography.

All eight blocks, same training size, one seed each:

    block   test    pool  trained    AUC   share +
        0   9984   54006    15000  0.745     0.219
        1  10000   38101    15000  0.380     0.201
        2   9839   15641    15000  0.475     0.471
        3  10113   29854    15000  0.449     0.504
        4   8961   28841    15000  0.509     0.207
        5  11103   35215    15000  0.523     0.262
        6   9999   30437    15000  0.586     0.254
        7  10001   65456    15000  0.388     0.130

    macro-average 0.507 ± 0.111, range 0.380 to 0.745, 4 of 8 above chance

**On unseen ground the model is at chance on average**, with block-to-block
variation an order of magnitude larger than the seed-to-seed variation that
was mistaken for it. It transfers well to the level's opening (0.745) and
inverts on two other stretches. "Systematically wrong" is not supported;
"unreliable, and the unreliability is geographic" is.

**The purge is audited, not assumed.** Across all eight blocks: zero training
points whose 120 px input window overlaps the test range, and zero whose
candidate branch or 96-frame continuation can reach it at 2.5 px per frame.
The 320 px buffer exceeds the 280 px of actual reach.

**Fourteen times the data was not fourteen times the diversity.**

    2500 points → 892 distinct world x, 2.8 revisits each
    80000       → 2123 distinct,       37.7 revisits each

Thirty-two times the points bought 2.4 times the places. What the extra data
buys is repeated observation of ground already seen, which is why it collapses
the seed spread and adds almost no transferable signal.

**A single bit beats the network.** `airborne`, one boolean from console
memory, scores AUC 0.583 on its own. The trained CNN on 35000 points scores
0.568. Among ground states alone the same bit scores exactly 0.500, as it
must. So the earlier claim that ties were worth 0.062 of AUC as a flight
detector is narrowed: differences of AUC do not decompose additively, but the
single-feature baseline is now measured and our model does not clear it.

Withdrawn from the section above: "raw pixels tested at scale and rejected",
"more data makes it more confidently wrong", and the additive attribution of
0.062. What survives: repeated traversals of one level do not buy transfer,
and the current pixel model does not beat a one-bit baseline.

## The cheap label is fine; the target has its own ceiling (2026-08-26)

Eighty thousand decisions were collected at four draws rather than sixteen,
because that is eleven times cheaper. On the six-candidate task four draws
agreed with sixteen on 69.1% of top-1 picks, so the binary target needed its
own audit before anything rested on it. All of it offline, from the 300 states
that carry two independent sixteen-draw labels each, with panels of four cut
from those sixteen. Exact ties are dropped — 121 of 300 — because they would
count as perfect agreement and flatter every number.

    sign of delta agrees, 4 draws against another 4 on the same state   0.709
    sign of delta agrees, 4 draws against an independent 16             0.704
    sign of delta agrees, 16 draws against an independent 16            0.754

    as a ranking of the independent sixteen-draw value
      4-draw label    AUC 0.839
      16-draw label   AUC 0.865

    sd of a 4-draw delta, two panels on one state    18.8 px
    sd of delta across states                        51.0 px

**Four draws did not trade quality for quantity.** Against an independent
sixteen it reproduces the sign 0.704 of the time where sixteen manages 0.754,
and it ranks the true value at 0.839 against 0.865. The eleven-fold saving
cost about three points of AUC.

**But sixteen draws do not agree with themselves either**, and that is the
more useful number. A label's sign reproduces 75% of the time, because the
decisions are mostly close:

    |delta| below  5 px in 38% of live decisions
    |delta| below 10 px in 47%
    |delta| below 20 px in 56%

Nearly half of all real decisions are near-ties where the sign is close to
arbitrary. So a sixteen-draw label reproduces another one at about **0.865 of AUC**.
That is the label's test-retest reliability, not a ceiling: a model estimating
the latent expectation could in principle rank a single noisy panel better
than another noisy panel does. It is the reference the numbers below are read
against, not a bound on them.

Which puts the day's results in their place:

    reference: a sixteen-draw label predicting another  0.865
    a single airborne bit                              0.583
    the trained network, 35000 points, ties fixed      0.568

There is real headroom between 0.57 and 0.87, and the difference between 0.839
and 0.865 has not had a paired bootstrap over the 179 states it rests on. The
practical conclusion is narrow and safe: four draws are good enough for bulk
collection, and the model is failing well short of the label's own
reliability without clearing a one-bit baseline.

## The binary gate's premise does not hold across levels (2026-08-26)

Before building a representation for a {bc, jump now} gate, the premise was
checked where it had never been: on other levels. Five development levels,
eight paired seeds each, three arms — the policy, an oracle restricted to
{bc, jump now}, and the full six-candidate oracle. Levels 7-1 and 8-1 are held
back and not opened.

    level      BC   oracle-2 vs BC        oracle-6 vs BC        share
    2-1     16496   +856 [ +646, +1068]   +928 [ +855, +1019]     92%
    3-1     32336   +400 [ +305,  +497]   +377 [ +260,  +511]    106%
    4-1     48876   +400 [ -161,  +885]  +1991 [+1063, +3374]     20%
    5-1     64612   +166 [ -209,  +577]  +1146 [ +748, +1574]     14%
    6-1     80337   +637 [ +122, +1221]  +1206 [ +696, +1682]     53%

    share of the six-candidate gain taken by two candidates: 57% ± 37%
    (on 1-1 it was 24%)

**Jumping is not a stable second candidate.** The fraction of the planner's
advantage that {bc, jump now} recovers ranges from 14% to 106% across five
levels, and on two of them the gate's own gain has an interval crossing zero.
A cross-level binary gate on this pair is not justified by the data, whatever
representation it is given.

**And the planner's advantage is itself a 1-1 number.** The six-candidate
oracle gains +377 to +1991 on these levels against +4590 on 1-1 — four to
twelve times less. Every ratio quoted in this project against "the ceiling"
was quoted against the most favourable level in the game.

What is stable is the shape of the oracle's choice: it defers to the policy
43-63% of the time on every level, and no single template takes more than 15%
of the rest. So the decision the planner is making is not "jump or not" — it
is a genuinely six-way choice, and the same on every level tested.

## The tile map was not found by matching, and is not being guessed (2026-08-26)

The privileged geometry input needs an exact structured description of the
ground, and the natural source is the console's own tile map. Where it lives
was searched for the same way Mario's coordinates were — by matching every
candidate window in RAM against what the pixels show — rather than taken from
memory or a wiki.

41 camera-aligned frames (only those where the scroll sits on a tile boundary,
so a column of pixels maps to exactly one column of the map), 12 rows by 15
columns, every offset from $400 to $760 read as two screens of 13x16 stored
column-major:

    baseline, predict empty everywhere    plain 0.822, balanced 0.500
    best window found, $63a               balanced 0.623

Plain agreement is useless here because 82% of cells are empty and predicting
nothing scores 0.822 — higher than any window scored. On balanced accuracy the
best candidate reaches 0.623, which is above chance but nowhere near a decoded
map.

So the search failed, and the honest reading is that the pixel mask it was
matched against is not clean enough for this: it flags clouds, bushes, enemies
and text as "solid", none of which the map stores that way. Fixing that means
either a better solidity ground truth or a different matching signal.

Recorded rather than resolved, and the privileged geometry input will not be
built on a guessed address. The alternative for the battery is geometry
derived from the pixel mask itself, which is not privileged and therefore not
an upper bound — a limitation to state rather than paper over.

## A physics reference finds the map, and turns out to be the better artefact (2026-08-26)

The pixel-based search failed because the reference was wrong. Physics gives a
clean one: Mario's body cannot overlap a solid tile, so every tile his box
covers is empty — thousands of certain labels a run — and when he is standing,
the tile under his feet is solid. Tiles with no evidence stay unknown and are
not scored.

Against that reference, sweeping every window in $400-$520, row-major against
column-major, seven row offsets and all 32 ring phases:

    level 1-1   balanced 0.991
    level 3-1   balanced 0.997
    baseline, predict empty everywhere    0.500

**A window in RAM does decode the geometry**, at 99% balanced accuracy on two
levels independently, and the family of equally-scoring solutions is periodic
in 32 bytes — so the map is row-major with 32 bytes to a row, two screens
wide, in the $400-$500 region.

What is *not* pinned is the exact address: base and ring phase are degenerate
against each other (shifting the base one byte is shifting the phase one
column), and the best pair differs between the two levels in a way one global
rule has not yet explained. So the address stays unquoted.

It does not need to be. **The contact map is itself the structured exact
geometry the battery needs** — true solidity in world tile coordinates,
derived from what the console let Mario do rather than from an address taken
on faith. It is privileged, it is exact where it has evidence, and it says so
where it does not. The detour produced a better artefact than the thing it was
chasing.

## The battery answers: nothing beats a bit and a velocity (2026-08-28)

Seven inputs, one head, BC-centred advantage regression, leave-one-level-out
over six levels, three seeds, ~14800 stratified decision points with full
save-states. Macro over held-out levels:

    input                     regret  weighted   top1   captured
    airborne + velocity        17.9     67.1    0.161   0.221 +/- 0.012
    pixels (strip)             17.8     67.0    0.166   0.224 +/- 0.007
    geometry profile           18.1     62.9    0.210   0.216 +/- 0.005
    geo + history              18.4     63.0    0.224   0.205 +/- 0.006
    geo + hist + objects       18.6     65.2    0.219   0.195 +/- 0.010
    ...+ BC probabilities      18.5     64.2    0.228   0.198 +/- 0.005
    everything + pixels        18.7     65.0    0.187   0.187 +/- 0.008

**Every input captures the same ~20% of the oracle's per-decision gain, and
it is the same 20% the two-scalar baseline captures.** The exact contact-map
geometry, the sprite positions, the action history and the policy's own
probabilities add top-1 agreement (0.16 to 0.23) but no captured value —
they help pick among near-ties, not among the decisions that matter. On
level 5-1 every input goes negative.

Perception was not the bottleneck for value estimation, and now the
richest privileged description of the *present* is not the bottleneck for
choice either. What separates the oracle from every model in this battery is
that the oracle looks at the future. The information that decides the choice
is not in the current state as we can describe it — which is the epistemic
POMDP reading: the value of a candidate depends on how the stochastic
continuation unrolls, and no static description of now predicts that
beyond ~20%.

The honest reading of the whole programme so far: distillation of this
oracle into a reactive chooser has a low information-theoretic ceiling, and
the working direction is the one the research synthesis put last month —
use the learner to allocate the planner's compute, not to replace the
planner's look at the future.

## The prior fails its offline gate: thinning beats pruning (2026-08-28)

Four procedures simulated from the stored per-draw returns, macro over six
held-out levels, three seeds:

    procedure    rollouts   regret
    full               24    0.00 px      (by construction)
    uniform-2          12    2.84 px
    prior-3            12    5.22 px
    prior-soft         15    3.65 px

    top-3 recall of the true best         0.723
    pruned candidate wins substantially   9.5% of decisions

At matched compute, **uniformly halving the draws beats letting the prior
prune candidates** — 2.84 px against 5.22 — and the soft allocation loses to
plain thinning while paying more. The prior keeps the truly best candidate in
its top three only 72% of the time, and in one decision out of ten the
candidate it discards wins by more than sixteen pixels. Rollouts are cheap to
thin because the noise averages; candidates are expensive to prune because a
dropped best is unrecoverable.

So the pre-registered offline gate for D1 fails and the online run does not
happen — which is the gate doing its job. The learner has now been priced in
all four roles this programme defined for it: value estimator (~zero over a
constant online), chooser (below a one-bit baseline), DAgger student (below
the majority class), and compute allocator (below uniform thinning). All four
failures are consistent with the battery's finding: the information that
decides is not a function of the present state under any description tried.

## The last two arms: memory loses, thinning holds (2026-08-28)

**Episodic memory loses to the policy it was meant to improve.** Steering by
the oracle's own stored decisions — five nearest within 32 px of world x,
3600 points of memory on 1-1, RAM-keyed, zero rollouts at runtime:

    bc            median 787   clears  0/32   deaths 70      +0
    knn memory    median 609   clears  0/32   deaths 45   -209 [-354, -60]

The offline story (k-NN at 0.707 beating every net) does not survive contact
with the closed loop, for the reason that killed every student: the memory's
choice perturbs the trajectory, the perturbed trajectory leaves the memory's
coverage, and where the memory is silent or stale its committed 16-frame
templates are worse than the policy's own reflexes. Fewer deaths, less
progress — the cautious signature again.

**Halving the draws holds up online.** The oracle at two draws instead of
four, same seeds:

    oracle 2 draws   median 5116   clears 16/32   deaths 34   +3966 [+3124, +4765]
    oracle 4 draws   median 7117   clears 20/32   deaths 20   +4590 (recorded)

Offline the thinning cost 2.84 px per decision; online it keeps roughly 86%
of the four-draw advantage at 55% of the branch frames (263k against 487k a
run). Not free — the compounding is real — but the compute-progress trade is
now a measured curve with three points (1, 2, 4 draws), and it is the only
lever in the whole programme that moved anything without learning.

## The adaptive budget, calibrated offline: the trigger is not the one specified (2026-08-28)

The next programme opens where the last one closed: allocate the planner's
own rollouts sequentially. The reviewer's scheme — two paired rollouts for
everyone, then third and fourth only where the expected regret of stopping
is high, threshold frozen on five levels and tested on the sixth — went
through the stored draw matrices before touching the emulator, and the
matrices rewrote its middle.

The specified statistic fails. Expected stopping regret under a Gaussian
with a *globally* calibrated sigma is anti-informative: the top quarter of
points it calls riskiest carry 19–36% of the real 2→4 saving — worse than
escalating at random. The noise is strongly heteroscedastic and one sigma
per level washes out exactly the signal the trigger needs. The winner-death
component is empty too (lift 0.000), because deaths are already priced into
the penalised returns it reads.

Two of the reviewer's other components carry everything. Winner instability
— the two draws disagreeing about the argmax — marks ~43% of decisions that
hold 70–75% of the saving. The point's own draw disagreement carries the
rest: ranking stable points by expected regret under a *per-point* sigma
(fifteen pairwise d0−d1 disagreements at the point, shrunk toward the
train-levels global with weight six) puts 46–58% of the saving into the top
quarter of escalations on every held-out level, where the global-sigma
version managed a fifth of that.

The frozen test uses the independent reference: on the 16-draw matrix of
1-1 the procedure sees only draws 0–3, the reference is the mean of draws
4–15, and tau comes from the five other levels:

    uniform-2       cost 12     regret 5.01 px
    adaptive        cost 16.5   regret 3.91      ← 94% of the 2→4 saving
    random at 16.5  cost 16.5   regret 4.58      (the convex-hull line)
    full-4          cost 24     regret 3.85

The adaptive point sits far below the line between the fixed budgets, with
every constant frozen on other levels. `scripts/experiments/adaptive.py` is
the stand; `oracle_mpc.py --draws 2 --adaptive 26.709` is the same rule
inside the planner, escalating all six candidates together so nothing is
ever pruned. The online run on 32 paired seeds decides whether the offline
94% survives compounding.

## The adaptive budget online: below the line, not above it (2026-08-28)

32 paired seeds, `--draws 2 --adaptive 26.709`, every constant frozen before
the run. The trigger behaved exactly as calibrated — 37.7% of decisions
escalated against the offline 37.2%, 344k branch frames a run against the
predicted ~335k. The progress did not follow:

    oracle 2 draws    +3966 [3124, 4765]   263k frames   16/32 clears
    adaptive          +3463 [2787, 4165]   344k          12/32
    control line @344k +4192                              (to beat)
    oracle 4 draws    +4590 [3827, 5307]   487k          20/32

The pre-declared criterion — the adaptive point above the straight line
between the fixed budgets — fails decisively: the line sits above even the
upper edge of the confidence interval. Paired seed-by-seed against the
stored oracle-2 run (bc rows byte-identical, so the pairing is exact):
adaptive − oracle-2 = −503 [−1478, +512], 11 wins to 21 losses, at 31% more
compute.

Two honest readings, both recorded. First, the offline→online gap claims
its fourth victim: 94% of the per-decision saving against an independent
reference did not survive the closed loop, joining the gate (68%→24%), the
k-NN chooser, and the input-order inversions. Per-decision px measured 144
frames ahead is still not the quantity the trajectory compounds. Second,
the criterion itself is near-unresolvable at this sample size: the line
exceeds the oracle-2 point by only 226 px while the paired CI spans ~2000,
so only a large win could ever have passed — but the observed direction is
negative regardless, and a rule that cannot show its win inside the noise
of 32 seeds is not a rule this stand can certify.

What survives: the stand (`adaptive.py`), the trigger diagnosis (global
sigma anti-informative, winner instability + per-point sigma carry the
signal), and the unchanged conclusion of the compute curve — at these
budgets the only certified allocation remains the uniform one. The
adaptive arm's implementation stays in `oracle_mpc.py` behind `--adaptive`
for whoever brings either more seeds or a per-decision CRN harness that
can pair the arms draw for draw.

## Common random numbers: the 2→4 step itself is inside the noise (2026-08-28)

The adaptive arm's negative came with a caveat — the criterion might be
unresolvable at 32 seeds. That is now measured rather than suspected. Every
continuation draw is seeded by (run seed, world x of the decision,
candidate, draw index) with the global stream swapped out and restored, so
arms sharing a seed play byte-identical games until a genuine policy
difference: the bc rows of all three arm sets are byte-identical, an
escalated decision sees exactly the draws the 4-draw oracle would see at
that state, and one seed of thirty-two stayed identical between adaptive
and oracle-2 for the whole 3000 frames.

All three arms rerun on the shared noise, 32 paired seeds:

    oracle-2   median 3124   +3827.5 [3044, 4592]   264k   15/32 clears
    adaptive   median 3126   +3625.2 [2810, 4457]   340k   14/32
    oracle-4   median 7106   +4081.8 [3274, 4885]   475k   17/32

    adaptive − oracle-2   −202  [−1438, +1068]   14 wins / 17 losses
    oracle-4 − oracle-2   +254  [ −919, +1441]   19 wins / 13 losses
    adaptive − oracle-4   −457  [−1542,  +608]   13 wins / 19 losses

The headline is the middle row. **Even always-escalating against
never-escalating — the whole 2→4 draws step — is statistically
unresolvable per-trajectory at 32 paired seeds under common random
numbers.** The published curve's +624 between the fixed budgets sits well
inside this band; the medians differ dramatically (3124 against 7106)
because the distribution is bimodal around clearing 1-1, but the means do
not. A criterion asking the adaptive point to clear an interpolation line
between two statistically indistinguishable endpoints was unwinnable and
unlosable from the start; that it was pre-registered anyway is the mistake
this entry records.

What stands after the noise floor is drawn honestly:

  * per-decision, against an independent 12-draw reference, four draws
    beat two by 1.16 px and the frozen trigger keeps 94% of that at 69%
    of the cost — measured on thousands of points, real;
  * per-trajectory, none of it is certifiable at n=32 — only effects the
    size of the planner itself (+3800 over bc) clear the floor;
  * uniform thinning keeps its title by default, not by victory: the
    adaptive rule costs 29% more and shows no gain twice.

The reusable artifact is the harness: `--crn` pins every draw to the
decision's state, which is what makes "identical until a genuine
difference" a property rather than a hope. Anyone returning to draw
allocation should measure per-decision on the stored matrices and treat
online runs at this scale as a smoke test, not a verdict.

## A longer tail buys nothing (2026-08-28)

The cheapest item on the improve-the-oracle list: value the same 48-frame
plans under a 144-frame continuation instead of 96, on the CRN harness, 32
paired seeds against the stored tail-96 run:

    tail-96    +3827.5 [3044, 4592]   264k frames   15/32 clears   32 deaths
    tail-144   +3662.1 [2841, 4530]   367k          14/32          30

    tail-144 − tail-96   −165  [−1323, +958]   15 wins / 16 losses / 1 tie

The hypothesis was that pits and enemies beyond the 96-frame horizon are
invisible to the value and a longer look would remove late deaths. Deaths
barely moved (30 against 32) and progress did not follow at all, at +39%
frames. By the standing rule — behavioural changes without measurable gain
are rolled back — the tail stays at 96. Same caveat as everything at this
scale: the CI spans ±1100, so this certifies "no large effect", not "no
effect"; but a change that costs 39% more compute carries the burden of
proof, and it showed nothing.

## Two-step search: no progress, half the deaths (2026-08-28)

Ordered pairs of the five behaviours at 48 frames each — 25 compositions
plus the policy's own 96-frame plan — against the same six-candidate
oracle, both at two draws on the CRN harness, 32 paired seeds:

    o2, 6 candidates    +3827.5 [3044, 4592]    264k    15/32   32 deaths
    two-step, 26        +3675.9 [2957, 4414]   1388k    13/32   14
    (o4, 6 candidates   +4081.8 [3251, 4889]    475k    17/32   28)

    two-step − o2   −152  [−1295, +1004]   16 wins / 16 losses

On progress the verdict is the usual one: dead even at 5.3× the compute,
so by the no-gain rule the six-candidate set keeps the wheel. The
compositions are genuinely used — the policy's plan takes only 40% of
decisions, and `jump later+jump now`, `back off+jump now` lead the rest —
they just do not buy pixels.

What they do buy is the first sub-planner-sized effect this online
harness has ever resolved: **deaths halve, 1.00 → 0.44 per run, paired
−0.56 [−1.03, −0.09]** — the interval excludes zero, fifteen seeds
improve against eight that worsen. Deaths are a far lower-variance
outcome than best_x, which is why the same 32 seeds that cannot see a
150-pixel effect can see this one. The reading: 96-frame compositions
let the planner refuse doomed commitments that a 48-frame template
cannot express its way out of, and the safety does not convert into
progress because the value being maximised is progress px — survival is
only rewarded through its contribution to x. A planner whose objective
priced death explicitly would presumably trade some of this safety back
for speed; that is a design question, not a measurement, and it is left
here as one.

## Four draws do not rescue the compositions (2026-08-29)

The overnight control for the optimizer's curse: the same 26-candidate
two-step search, four draws instead of two, so its values are as clean as
the best fixed-budget oracle's. CRN, 32 paired seeds:

    o2                +3827.5    264k    15/32   32 deaths
    o4                +4081.8    475k    17/32   28
    two-step, 2 draws +3675.9   1388k    13/32   14
    two-step, 4 draws +3466.7   2296k    12/32   18

    ts4 − ts2   −209  [−1301, +884]    15/17
    ts4 − o4    −615  [−1833, +611]    16/16

Cleaner evaluation moved nothing: the two-step arm sits where it sat,
now at 8.7× the six-candidate cost. The curse is refuted as the
explanation — the compositions genuinely do not buy progress on this
level. The safety effect survives at four draws (deaths −0.44
[−0.84, −0.03] against o2), confirming it was never an artefact of
noisy values.

That closes the improve-the-oracle list as far as flags reach: a longer
tail buys nothing, compositions buy safety the progress objective cannot
spend, and self-tail is ~54× — out of reach of this machine. The
six-candidate, uniform-draws oracle remains the reference planner, now
having defended its title against every cheap challenger.

## The death price changes nothing, and the matrices say why (2026-08-29)

The structural follow-up to the two-step safety finding: price death into
the value instead of vetoing it. `--death-price P` makes every draw count
— a dead one contributes the x it reached minus P — so the value becomes
E[progress] − P·P(death), where the majority-veto rule made a death in
the minority of draws free.

Two prices, 400 and 1000 px, CRN, 32 paired seeds each:

    dp400  − o2 veto    0.0   [0, 0]   0 wins / 0 losses / 32 ties
    dp1000 − o2 veto    0.0   [0, 0]   0 / 0 / 32

Byte-identical, all sixty-four runs. Not one of ~11,400 decisions changed
its argmax. The stored matrices explain it: across 14,809 decision points
on six levels there is **not a single candidate that dies in a minority
of its draws** — if a plan leads to death, both draws die; if it does
not, neither does. On this candidate set death is a deterministic
property of the plan, P(death) is 0 or 1, and the veto and the price are
the same rule wearing different clothes. The freebie the price was meant
to close does not exist here.

Which sharpens the two-step finding into its real shape: the halved
deaths did not come from better death accounting — the six-candidate
oracle's accounting was already exact — they came from a richer plan
space in which an escape exists that the six templates cannot express.
The lever is the candidate set, not the objective. A cheap corollary
experiment would be the six templates plus a handful of hand-built
escape compositions rather than all 25 pairs; left on the list.

## The escapes work — against the right baseline (2026-08-29)

Three hand-picked compositions — the pairs the 26-candidate search
actually used — added to the plain template set at 96 frames, with the
control nobody had run: the plain six templates at 96 frames. CRN, 32
paired seeds:

    o2, h=48, 6 cand      +3827.5    264k   15/32   32 deaths
    h=96, 6 cand          +1978.1    304k    7/32   63
    h=96, 6+3 escapes     +3540.7    474k   13/32   28
    (two-step, 26 cand    +3675.9   1388k   13/32   14)

    escapes − h96 plain   +1563  [+529, +2564]   27 wins / 5 losses
    escapes − o2 h48       −287  [−1276, +723]   15 / 17
    h96 plain − o2 h48    −1849  [−2937, −698]    5 / 26
    escapes − two-step     −135  [−1231, +945]   15 / 17

Three findings, two of them clearing the noise floor with room to spare.

**Monolithic 96-frame templates are toxic.** The control collapses:
half the progress, double the deaths of the same behaviours at 48
frames. An open-loop "run" held for 96 frames is a commitment nothing
can rescue; the 48-frame horizon was never an arbitrary choice, it was
load-bearing.

**Compositions repair almost all of it.** Adding just three rescue
pairs buys +1563 paired pixels (CI well clear of zero) and halves the
deaths against the monolithic-96 control (−1.09 [−1.69, −0.50]). And
they carry the whole composition value: against the full 25-pair search
the three hand-picked pairs are dead even at a third of the cost.

**The champion stands.** Against the 48-frame six-candidate oracle the
escapes are noise (−287) at 1.8× the cost. So the two-step story ends
re-attributed once more: its safety was never something a richer space
adds to a healthy planner — it is what compositionality gives back
after long open-loop plans take it away. At this game's tempo, six
short templates replanned every 16 frames already sit at the local
optimum, and every structural variation tried today lands at or below
it for more compute.

## The planner transfers online: five levels, five positives (2026-08-29)

The online economics had only ever been measured on 1-1. The same
oracle-2 (six candidates, 48-frame horizon, 96-frame tail, two draws,
CRN) against the same policy on every other dry land level, 32 paired
seeds each:

    level    bc median -> oracle     paired gain            w/l    deaths o/bc
    2-1        16436   ->  17416     +800  [ +666,  +923]   31/1     65/68
    3-1        32401   ->  32669     +509  [ +339,  +710]   31/1     86/63
    4-1        49188   ->  50292    +2483  [+1662, +3391]   31/1     59/62
    5-1        64516   ->  65757    +1495  [+1067, +2042]   32/0     76/87
    6-1        80340   ->  81269    +1314  [+1018, +1605]   28/4     81/69

Every confidence interval clears zero; 153 of 160 paired seeds go to the
planner. No level-specific tuning of any kind — the templates, horizon,
commit, margin and draw count are exactly the 1-1 champion's.

The magnitude varies fourfold (+509 on 3-1 to +2483 on 4-1) and so does
the character: on 3-1 and 6-1 the planner buys progress with extra
deaths (86 vs 63, 81 vs 69 — aggression through the night bridges and
cannon fields), on 5-1 it wins while dying less (76 vs 87). The same
value function expresses as caution or courage depending on what the
level prices.

This closes the transfer question the programme left open: **the
planner's advantage is not a 1-1 artefact.** Exact dynamics plus six
generic behaviours is a general lever on this game, online, at every
level tried. What remains untested is other games — the pipeline
(integration states, contact maps, the CRN harness) is built for it,
but SMB's x-scroll physics is doing unquantified work in the templates.

## A second game: the planner carries a policy that cannot walk (2026-08-29)

Contra (J), the first non-Mario game the oracle has ever run on. The
adaptation cost is worth recording because of how small it was: a
game-generic position (Contra's integration publishes a 16-bit camera
scroll and a level counter), a game-generic boot (no SMB countdown clock
— pulse START until the lives counter goes live, then six hundred
buttonless frames for the AREA intro, because START during play is
pause), and **not one change to the templates**: B+RIGHT that means "run"
in Mario means "advance firing" in Contra, A+B+RIGHT jumps in both.

32 paired seeds, oracle-2 CRN against the game's own BC policy:

    bc (bc_contra_attn)   median 0      max 0      0 deaths
    oracle-2              median 2367   max 2537   32 deaths   262k frames

    oracle − bc   +2270  [+2099, +2381]   31 wins / 0 losses / 1 tie

The policy is not weak here — it is inert: zero camera progress on every
seed, standing at the spawn shooting at nothing. And the tail the oracle
values plans under is that same inert policy, which makes the number
sharper than it looks: the value differences that drive +2270 come
almost entirely from what the 48-frame template prefix itself reaches.
Six generic behaviours plus exact dynamics walk nine screens into the
jungle on the first try, one death per run, through a game the learned
half of the system cannot take a single step in.

The pipeline claim is now tested end to end: integration in, planner
out, template set untouched. What the x-scroll physics of SMB was
suspected of doing in the templates turns out to be nothing Contra's
physics does not also accept.

## Third and fourth game, and a camera found by scanning (2026-08-29)

Contra (U) is the sibling ROM — same RAM map, same result: the policy
inert on every seed, the oracle at +2134 [+1916, +2304], 31/0/1.

Super C is the more interesting one, because stable-retro's registry
ships hundreds of NES ROMs whose RAM maps list only lives and score — no
position, nothing to value progress by. The camera was found empirically
in minutes: boot, hold advance, snapshot RAM, keep the bytes that are
flat while idle and monotone while running; the low byte wraps at 0x6B
and 0x6C ticks exactly at each wrap, verified through two wraps. With
that one address pair and the Contra checkpoint as a fixed weak tail:

    Super C   bc median 224 (max 864)   oracle median 1360 (max 2368)
              +1210  [+1046, +1380]   31/0/1

Four games now, one planner, one template set, zero per-game tuning
beyond a position address and a boot sequence:

    SMB 1-1     +3828      Contra (J)  +2270
    SMB 2-1..6-1 +509..+2483   Contra (U)  +2134
                             Super C     +1210

Every interval clears zero; across the ten environments the planner
wins 371 of 384 paired seeds. The RAM-scan method makes essentially the
whole stable-retro side-scroller library reachable at a few minutes per
game — no trained policy required, since Contra established the oracle
works over an inert tail.

## The scan method meets its boundary (2026-08-30)

The RAM-scan that found Super C's camera in minutes is now a tool
(`find_camera.py`: bytes flat at idle, monotone under advance, high byte
verified at the wraps) — and its boundary is now measured too. Ninja
Gaiden has no global monotone camera word: its levels are cut into
scenes and the camera resets at each boundary, so every high-byte
candidate jumps back mid-run. Kung Fu, probed the same way in its
left-scrolling variant, shows no stable camera pair at all under a
1,800-frame advance. The method opens games with one continuous scroll
counter — the Contra family proved that — and stops where progress
needs composing from scene counters, which is per-game archaeology
again. Recorded so the next sweep across the library starts with the
right expectations.

## The wall falls: doomed states, a damage term, and honest continues (2026-08-30)

Contra's level-1 boss stopped the planner cold, and the diagnosis was
worth the fight. The camera halts at the wall, so progress goes flat; the
kill needs ~2,250 frames of sustained diagonal fire (measured by script
with granted lives); and — the real finding — the wall zone is full of
**doomed states**: at a hand-evaluated decision every one of eight
candidates died inside its own 48-frame prefix. When every value is the
death floor, argmax falls through to the policy's plan by tie, the inert
policy stands in the bullet stream, and the next doomed state arrives.
The bullet was already in flight before the decision was made — the
mistake lives beyond the tail's horizon.

Three additions, each earned by a measurement:

  * **a damage term in the value** — four object-slot bytes found by
    diff-scan (flat without fire, monotone under hits, sum 66→0),
    worth 40 px per hit and switched on only where the camera has hit
    the wall, so the value stays continuous;
  * **the project's first game-specific templates**, recorded as such:
    fire up-right, prone under the stream, a jumping diagonal — the
    six-for-everything claim now reads six-plus-what-the-weapon-needs;
  * **honest continues**: after the last life the runner presses START
    on the continue screen the way a player would, the level restarts
    from its beginning, and the credits are counted in the row.

The result: level 1 cleared end to end — best_x 4192 (the camera 192 px
into level 2), four continues, eight deaths, the combat templates chosen
104 times where they matter. The CRN harness made the recorded video
byte-identical to the test run. `--save-final`/`--load-state` (the boss
lab: resume where a run ended, with lives granted) stay in the runner.

## Retraction: the wall did not fall — my own bonus lied to me (2026-08-30)

The previous entry claimed level 1 cleared at best_x 4192. The owner
watched the video and saw the truth in one sentence: the run never got
past the wall. 4192 is not level·4000+192 — it is 3072 + 40·28: the
damage bonus I had added to the planner's value was flowing into the
run metric too, and 28 hits of wall damage dressed themselves up as a
level transition. The video's final minute shows the wall approach,
deaths 8, wall standing.

The instrument bug is fixed the only correct way — by separating the
two quantities. `game_value()` is what the planner maximises (position
plus the wall-damage term); `game_pos()` is pure position and is all
the run metric and the CRN keys ever see. The honest state of the boss
fight: 28 of 66 hits across four credits and eight deaths. The doomed-
states diagnosis, the templates, the continues and the damage term all
stand; the victory does not, yet.

The lesson is an old one from this project's own list — a plausible
number is not a measurement — with a new twist: the misleading number
was not the game's, it was mine, added the same afternoon. Any term
added to an objective must be kept out of the metric by construction,
not by discipline.

## The wall, part two: a weapon byte, two dead hypotheses, and the real bottleneck (2026-08-30)

The capsule idea came from the owner, and the weapon byte is now real:
0xAA verified by poke-and-look (0 fires thin white, 3 fans out as
spread, real pickups read 16/19 — flag bits over a low-nibble tier, so
the value masks with 0x0F). A weapon term went into `game_value`
(spread +400 px, mid-tier half), and pickups turn out to happen
naturally — wmax 16 even in arms without the term.

Then the measurements killed two hypotheses in a row. Every honest run
that reaches the wall takes off exactly **28 hits** — a plateau, not
noise. Hypothesis one, the all-DEATH tie loop (doomed states hand the
wheel to the inert policy), was fixed on principle — dead candidates
now rank by damage dealt before dying ("die usefully", the floor still
keeps every live plan above every dead one) — and the plateau did not
move: 28, 28, 0, 0 on the same seeds. Hypothesis two, "the core needs
a straight shot from the ledge", died in a nine-variant position lab:
2-4 hits per 500 frames from any ledge. The video frames showed why —
the core sits behind an armour cross at mid-height, ground fire hits
plate, and the immortal script's kill rate was riding on respawn
invincibility, falling from the top through the sensor line.

What the data actually says: 28 hits is the budget of ONE arrival with
about one life, and credits two through four mostly never make it back
to the wall (their rows read hits 0). The wall dies when each credit
arrives at all and arrives with three lives — three good arrivals
exceed 66. The bottleneck is journey survival, and the repository
already owns the right tool for that: the Go-Explore robustification
loop, checkpoints along the route, return and reinforce. That is the
designed next block, not an evening flag.

Tree reuse between replans (the MCTS borrow) was examined and shelved
honestly: the search is flat, and carrying constant-template values
across replans is an unguaranteed heuristic that owes the offline
stand a verdict before it touches the online planner.

## The wall, part three: a held button, and every piece solved separately (2026-08-30)

The owner's muscle memory ("fire the diagonal point-blank") found in one
sentence what three of my hypotheses had missed — and the root cause
under it was a held button. In Contra, B held down fires exactly one
bullet; every fire template held B, so "fire up-right" had been a
decoration emitting one shot per 48 frames, while every lab script that
worked had been tapping. With the templates switched to 2-on/2-off
taps, the planner from the wall state does what the owner described:
walks to contact and strips the wall — **66 hits of 66, zero deaths,
one credit**, best_x 4000 on the now-pure position metric (the level
byte itself).

From a cold start the picture is a staircase of solved and unsolved:
seeds 5-7 reach the wall, take exactly 28 hits again and die out;
seeds 3, 4, 8 never pass x≈2820 — the cliff staircase before the base,
snipers above, every candidate at the death floor while climbing. The
journey costs 6-8 deaths per run, so what arrives at the wall has no
lives left to spend on the fight the planner has already proven it can
win from a healthy state.

Every component now has a measured verdict: the route to 3072 —
passable; the wall kill from a good arrival — solved; the composition —
blocked on journey survival, concentrated at one place. That is the
Go-Explore robustification loop's exact job description (checkpoint
before the cliff, find a surviving climb, reinforce), and it is the
designed next block rather than tonight's flag.

## A0: the console answers the buttons, causally (2026-09-02)

The button probe of the morning was circular — it found "forward" through
the game's position counter, the very per-game variable a universal agent
should be discovering. `controllability.py` uses nothing per-game: from
one save-state it runs a synchronous no-op branch and, beside it, one
branch per chord and mode (press-edge, hold, tap) for 32 frames, and
measures only what differs from doing nothing — screen pixels at the best
horizontal alignment within ±8 px (a nudged residual velocity scrolls the
camera a pixel and would otherwise own every brick edge), RAM bytes, and
whether a life was lost. Two confounds had to be removed by construction:
scrolling chords are excluded from the body vote, and the snapshot waits
ninety frames for momentum to die.

What it reads off SMB and Contra with no RAM map:

  * **controllability**: Contra's title screen and spawn animation answer
    to nothing; SMB after a death answers to nothing; play states do;
  * **the body**: ~143 px at Mario's spot; 500–770 px at Contra's soldier;
  * **position bytes**: 0x86 (SMB player x) among the bidirectional
    responders; Contra's 0x65/0x68 camera bytes as right-only (the camera
    never backs up) and 253 as the player's screen x;
  * **fire and its semantics**: SMB — zero projectile pixels under any B
    chord; Contra — B+UP/tap 31–33 px of new sprite off the body, B/tap 9,
    B+DOWN/tap 9–119, and **zero under hold or press** — the held-B bug of
    two days ago, read straight off the console.

The audio channel is left out for now: the savestate does not carry the
APU phase, so every branch differs from no-op in sound regardless of
input. Next: templates assembled from this scan alone, then compared
paired against the hand-written sets.

## A1 on A0, and the first game with no hand-written address (2026-09-03)

`scan_templates()` builds the candidate set from `control_<game>.json`
alone: forward is the RIGHT chord that pushes the scan's own position
bytes furthest (ties to the chord that also fires), fire is the B chord
with the most projectile pixels off the body with the scan's own tap/hold
mode, jump is forward plus A by convention. It reproduces the hand sets —
SMB 5 of 5, Contra 7 of 8 — with no per-game input. Rush'n Attack, a game
nothing in the repo had touched, scans as a tap-fire knife game with
RIGHT/hold forward and zero pixels under held B.

**The camera.** The remaining per-game variable was the position, and
Rush'n Attack refused the method of the 29th: byte 20 is an 8-bit scroll
that wraps every 256 px and *no byte anywhere* ticks at the wrap — there
is no 16-bit camera to find. Two changes to `find_camera.py`:

  * candidates are ranked by agreement with the picture's own horizontal
    shift (best alignment of 4-frame-apart playfields within ±12 px,
    correlated with the byte's unwrapped increments). Without it walking
    animation counters win: they are flat while idle and monotone under
    advance, exactly the old signature. Byte 20 scores 0.57 against 0.19
    for the runner-up; on Contra the same ranking returns 100/101, the
    RAM-map camera, at 1.0;
  * a lo-only record is written when no high byte exists, and the planner
    unwraps it symmetrically (a drop past 128 is a wrap forward, a rise
    past 128 a wrap back), so restoring a savestate across the boundary
    unwinds the count. Each worker is anchored from the decision's x0.

**The run that was not.** The first closed-loop run — position and
templates from the scans, a Contra-trained tail — printed +555 px over
the policy. The frames disagreed: all eight candidates carried the same
score, and that score drifted negative. Spawned workers re-import the
module, so `SCAN_POS`, set in main after argument parsing, was empty in
every worker, and they scored Rush'n Attack with Contra's position
formula. The planner was blind, took the policy's plan 106 times in 113
decisions, and the "gain" was an offset the second arm inherited from the
first arm's title screen (the unwrap counter persisted across runs). Both
are fixed by construction: the scan position is an initializer argument,
the counter is dropped after boot. `POS_DEBUG=1` prints the main and
worker positions at every decision, which is how the fix was checked.

The bc arm's per-seed results reproduce exactly across processes
(563, 546, 592, 587, 601, 618 for seeds 0–5) and rise almost monotonically
with the seed index up to seed 24, then reset. It is deterministic and
shared with the paired arm at the same seed, so the comparison stands, but
the regularity is unexplained and noted.

**Result, 32 paired seeds, 1800 frames, nothing hand-written:**

| arm | best_x median | IQM | mean | deaths / run |
|---|---|---|---|---|
| bc (Contra-trained tail, inert-ish) | 684 | 686 | 690 | 4.06 |
| oracle 48/96, 2 draws, scan templates + scan position | 848 | 840 | 834 | 3.09 |

Paired difference +145 px (median +112), bootstrap 95% CI [+124, +166],
32 of 32 seeds to the planner, 0.97 fewer deaths per run. 3,616 decisions:
the policy's own plan 68%, wait 22%, forward 4%, the rest under 2% each —
the same shape as SMB and Contra, where the planner's value is in the
handful of decisions where waiting or a different button avoids a death.
The absolute numbers are small because the whole 1800-frame budget is
~850 px of a game that scrolls at 0.3 px/frame; this is not a level
clear, it is the planner working in a game whose position, buttons and
fire semantics it discovered itself.

The price of this game, per the reviewer's metric: zero manually
specified variables; calibration — one A0 scan (~4 min), one camera scan
(~2 min); human minutes — none inside the loop, several hours outside it
fixing the two tool bugs the game exposed (the 8-bit scroll, the blind
worker). Both fixes are general; the next game pays only the scans.

## C2, the reroot, measured on paper before any stand (2026-09-03)

The reviewer's C2: after executing 16 frames, find the actual savestate
among the computed descendants, make it the root, keep its continuations,
compute only the missing branches. In this planner the search is not
needed — the executed 16 frames *are* the chosen candidate's first 16
frames (commit = 16, prefix = 48), so the exact state match is guaranteed
every decision, and the question is only what the match buys.

  * The chosen candidate's remaining prefix (frames 16–48) and its tail
    draws are the only exactly reusable work. Every other candidate
    diverges at frame 0 of the new root. Reusing them as a "continue"
    candidate costs zero frames and saves one candidate's worth:
    48 + draws·96 = 240 of ~1,850 frames per decision on Rush'n Attack
    (8 candidates, 2 draws), 13%; ~17% on SMB with 6.
  * But the reused tail ends 16 frames earlier than a fresh one would —
    the continue candidate looks 128 frames ahead where the others look
    144. That is a behavioural change to one candidate, and by the noise
    floor of the 29th a 13% budget shift is not resolvable online at 32
    seeds; there is no offline stand for it either, because the stored
    draw matrices hold values, not trajectories, and reuse changes the
    window the value is taken over.
  * The bc plan is built by rolling the policy 48 frames in the main
    process and then re-simulated as a prefix by a worker; shipping the
    frame-48 state would save 48 frames on 68% of decisions, ~2.5%.

Ceiling: 13–17% of branch frames, all of it in the tails' shifted
window, none of it exact. Parallelism already cut the seconds; the
work itself has no large exact-reuse component in a fixed-template MPC
with commitment. C2 is closed as "measured, below the floor" and not
implemented; the reroot idea belongs to a tree search with progressive
expansion, which this planner is not. On to B1'.

## B1': failure-triggered rollback meets the cliff, and loses to a composition (2026-09-03)

Implemented as `--rollback N`: a ring of the last N savestates on the
game's clock (every 16 frames); when every candidate is doomed, rewind one
step, re-plan with the horizon grown by the frames rewound so the death
stays in view, and commit the survivor; give up on this life after N
steps. Discarded frames count against the budget. Three versions were
measured on Contra's cliff (`cliff.state`, x≈2820, 900 frames, 8 seeds,
two lives), against the plain planner and two-step:

| arm | passes the cliff (reaches 3072) | deaths / 8 runs | branch frames / run |
|---|---|---|---|
| plain 48/96 ×2 | 1 / 8 | 12 | ~107k |
| rollback 4, survivor committed whole, plain templates | 0 / 8 | 11 | ~107k |
| rollback 4, compositions at the rescue re-plan only | 0 / 8 | 11 | ~280k |
| rollback 4, danger window: compositions on until the game clock passes the doom's lookahead, normal commits | **5 / 8** | 9 | 230–420k |
| two-step everywhere (26 candidates) | **8 / 8** | 7 | 306–347k |

What the rollback does, every seed: fires (attempts 4–7 per run in the
first two versions), finds a survivor two to four steps back, and the
survivor is a stall — the committed prefix ends, the doom is back, the
depth escalates to the cap and fails. The trap is not "a decision made
earlier than 144 frames": from 64 frames back with a 208-frame lookahead
no *single* template passes, and from those same states two-step passes
at once. The cliff is a staircase of ledges under turret fire, and the
passing runs string jumps at timings the six templates do not offer
(two-step's choices there: run+jump now 7%, run+jump later 7%, run+run
6%, jump later+run 5%, jump now+run 4%, jump now+jump now 3% — a finer
jump-timing grid, no single dominant pair).

Verdict: rollback is closed. The danger-window version works only because
it turns compositions on, and two-step turned on everywhere beats it 8/8
to 5/8 at the same cost on this stretch. The general lesson is the one
from SMB's escapes in reverse: the planner's remaining failures on the
journey are *capability* failures of the candidate set, not lookahead
failures, and the cheap fix is the right compositions, not a rewind.
Next: an escapes set for Contra distilled from two-step's choices (five
run/jump pairs, 13 candidates instead of 26), measured on the cliff and
then over the whole level.

**Escapes on the cliff (same stand, 8 seeds):** 7 of 8 reach 3072, 8
deaths against the policy's 13, ~172k branch frames per run — 1.6× the
plain set and half of two-step, which passes 8 of 8 at ~330k. The
full-level campaign (32 paired seeds, 4500 frames, escapes vs plain) is
running; the question is how many seeds arrive at the wall at all.

## The journey, whole: escapes over the full level (2026-09-03)

32 paired seeds, 4500 frames, from power-on, CRN; the plain set (8
candidates) against the plain set plus the five Contra escapes (13).

| arm | reach the wall (3072) | best_x median | IQM | mean | deaths / run | wall hits at arrival | branch frames / run |
|---|---|---|---|---|---|---|---|
| plain | 5 / 32 | 2728 | 2699 | 2705 | 1.94 | 31.2 | 586k |
| escapes | **22 / 32** | **3072** | 3041 | 2915 | 1.97 | 31.6 | 912k |

Paired best_x +209 px (median +250), bootstrap 95% CI [+99, +312]; 23
wins, 3 ties, 6 losses; deaths identical. Seeds reaching the wall: both
3, escapes only 19, plain only 2. Where the plain arm stops is four
places, not one — 2367 (×3), 2480–2495 (×9), 2720–2750 (×7), 2821–2826
(×8) — and the escapes pass all four, so the "cliff" was the last of a
family of jump-timing bottlenecks, not a single spot. Cost 1.56×.

The wall is now the budget's problem: arrivals land around frame 3500
and the 1000 frames left buy ~31 of the 66 hits. The next run is the same
escapes arm at 7000 frames, and the metric for a clear is the pure
position crossing 4000 (the level counter folds in at that stride),
which the damage term cannot reach by construction.

## Correction: the wall "hits" in the campaign tables were not hits (2026-09-03)

The full-level tables above report "wall hits at arrival ~31". That
number is `66 − (sum of the four slot bytes)`, and 66 was the sum in the
one savestate the wall work started from. A real arrival state saved
from the escapes campaign (`arrival1.state`, x=3071) has the same four
bytes at 17, 4, 16, 1 — sum 38 — before a shot is fired, and eight lab
seeds from it report exactly 28 "hits" each because 66 − 38 = 28. The
7000-frame run reports the same 28/28/44/28 on its first seeds as the
4500-frame run did: the extra budget bought nothing because nothing was
being bought. The baseline is not a property of the wall; the four bytes
are object-slot fields whose meaning depends on what occupies the slots.

What stands: reaching 3072 is pure camera position and is unaffected.
What is withdrawn: every hit count in the tables of this day, and the
sentence "the wall is now the budget's problem". What has never been
observed, in any run, is the wall dying — the level counter has never
advanced, and the Aug 30 recording shows the cannons intact to the game
over. The metric is now the *drop* of the slot sum from its value at
arrival (`wall_drop`, with `wall_slots_at_arrival` beside it) and is
labelled a slot-byte drop, not damage; the planner's damage term is
untouched (a shared offset within a decision) but its validity on real
arrivals is unverified and is the next thing to establish — by the level
counter or not at all.

**What the slot bytes are (same day, scripted).** From `contra_wall.state`
with two lives granted, walking to the wall and tapping UP+RIGHT+B: byte
1410 (0x582) falls 30 → 14 by exactly one per tap, one tap every four
frames, then drops to zero when the run dies — the HP of *one* wall part
in *that* slot. Bytes 1333–1335 do not move under this fire. In the real
arrival the slot at 1410 holds a 1, i.e. some other object. So the
"damage" the planner has been pricing is one part's HP when that part
happens to sit in a fixed slot, and a general damage signal needs the
object *type* per slot, not four fixed addresses. Open; the kill signal
remains the level counter.

**7000 frames, same escapes arm, 32 seeds:** 23 reach the wall (22 at
4500), median 3072, mean 2979, 2.25 deaths per run, 31 of 32 runs used
their continue, **0 level clears** (pure position never crosses 4000).
The extra 2500 frames change nothing at the wall, which is the expected
result once the hit counts are known to be phantom: the planner arrives,
loses its last lives to the wall, continues from the level start. The
wall is a boss problem with no verified damage signal yet; the journey
to it is solved at 22–23 of 32.

## The wall falls, seen this time (2026-09-03, lab)

With the damage term on the object tables (types at 0x530, HP at 0x580;
the wall = types 17/16/16/4, 72 HP), eight seeds from the real arrival
state `arrival1.state` (x=3071, two lives granted by the lab flag),
1500 frames, escapes:

| seed | deaths | typed wall HP arrival → min | best_x |
|---|---|---|---|
| 0 | 1 | 72 → 0 | 3072 |
| 1 | 2 | 77 → 0 | 3072 |
| 2 | 1 | 72 → 0 | 3072 |
| **3** | **0** | 72 → 0 | **4000** |
| 4 | 1 | 72 → 0 | 3072 |
| 5 | 2 | 72 → 22 | 3072 |
| 6 | 2 | 72 → 0 | 3072 |
| **7** | **0** | 72 → 0 | **4000** |

Two seeds advance the level counter — pure position 4000 — with zero
deaths, so the lab's extra lives played no part in them. Seed 3 was
re-run with the recorder and the frames checked: explosions on the upper
cannon and the sensor, the wall face bare, then the stage-clear
intermission with the hero and the Japanese "arrived at point A" text.
That is the first wall kill and the first level clear observed in this
project, and the first claim of one made after looking. "HP → 0" in the
other seeds is ambiguous by itself — the typed sum also reads 0 when the
objects despawn on a death — which is why the level counter stays the
only kill criterion. The choices at the wall: `fire up-right` 12–22 per
run, the tapped diagonal from two days ago.

Running now: the same arm end to end from power-on, 32 seeds, 7000
frames; the number that matters is seeds with position ≥ 4000.

## Contra level 1, cleared from power-on: 12 of 32 (2026-09-03)

The same escapes arm with the typed damage term, end to end from
power-on, 32 seeds, 7000 frames, CRN:

| | count |
|---|---|
| reach the wall (position ≥ 3072) | 23 / 32 |
| **clear the level (level counter, position ≥ 4000)** | **12 / 32** |
| deaths per run | 2.03 |
| best_x median / mean | 3072 / 3327 |

Seed 1 was re-run with the recorder and the frames checked: the wall
fight with both cannons and the sensor exploding, the bare wall face, the
"arrived at point A" intermission, the **AREA 2 : BASE 1** title card
with REST 1, and the first corridor of the base. The clears used 1–2
deaths each; none used a continue before the kill.

Of the eleven seeds that reached the wall and did not clear, seven
arrived with a typed HP sum above 72 (88–168): turrets from the cliff
share the cannons' object type and were still in the tables, and the
clipped term `max(0, 72 − HP)` is silent until the sum falls under 72.
The unclipped variant (`--wall-unclipped`, the same differences within a
decision, no baseline) is running paired on the same seeds.

The road here, for the record: the fire semantics from A0 (tap, not
hold), the escapes from two-step's choices, the object tables from a RAM
dump under fire, and two retractions along the way. The learned policy
alone still moves zero pixels in this game.

**Level 2, first look (2026-09-03, scripted).** The base does not scroll,
so the planner's value is flat there; the first level-2 recording was
also paused by the harness (auto-continue fired on camera = 0; now gated
on lives = 0). From a level-start state, standing still and tapping B:
the room's sensor is object type 20 with 8 HP, it dies in ~8 taps, the
wall opens; holding UP walks into room 2, whose wall is the same red
panel with a sensor. A scan for a byte stepping +1 at the room change
found nothing clean before the scripted run died in room 2. The base's
progress signal — rooms cleared — is the next per-game variable to find,
or the first case for a damage-only objective over the object table.
