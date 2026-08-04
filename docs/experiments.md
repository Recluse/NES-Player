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
