# Manual-assisted zero-shot play — protocol

Written 2026-10-08, before the pipeline that feeds it is built and before any
number from a held-out game is seen. It freezes what the test is, what the
agent is handed, what it is forbidden to see, and what counts as success.

This is the test [manuals.md](manuals.md) and the README set up and have not
run. Everything below exists so that when it does run, it is the test and not
something that grew after the fact.

## 1. The claim

An agent that has never seen a game, given the documentation that shipped in
the box and nothing else that a person would not have, plays further than the
same agent without that documentation.

Not a stronger claim: not that it beats a human, not that it clears the game,
not that it beats the privileged planner. The planner is allowed to read the
emulator and rewind it; that arm is reported for scale, in italics, as in the
honest track of the README.

## 2. Roles

| role | may read | may not read |
|---|---|---|
| policy / instincts | frames, audio, its own actions | RAM, prior internals |
| planner (non-privileged arm) | frames, audio, prior, its own scans of buttons/camera/HP | RAM position, savestates from the future |
| manual-prior builder | the PDF, its own vision of the game, the schema | running results on confirmatory seeds |
| evaluator | RAM, everything | — (it writes metrics only) |

The fence is the one in [specification.md](specification.md): RAM is ground
truth for the evaluator and never an input to play.

## 3. What the agent is handed

1. **The ROM**, through the normal emulator interface.
2. **Auto-start**: START pulses and the existing pixel-based TitleWatch, so
   the game reaches a playable state. This is not knowledge; a person also
   presses start.
3. **The manual**, as a PDF — the scan of the document in the box, the same
   one a buyer had. English release preferred (readable for recordings);
   Japanese accepted when it is the one matching the ROM, as with Contra J.
4. **The prior file**, produced by the pipeline of §5 from that PDF and from
   on-screen scans — not hand-written by the experimenter who will run the
   game.

### Allowed in the prior

* `goal`, `target.concepts`, `exit.way_on`, `controls`, `hazards`,
  `player.role`, `economy` — text facts, each with a page citation.
* `scope` — when the manual itself scopes a fact (by area, scroll mode, menu
  versus play).
* Extra button templates that are literally in the manual's control table
  (e.g. mirrored fire directions).

### Forbidden in the prior

* Object type numbers, hit points, RAM addresses, camera addresses, tile IDs.
  Those are found by the existing scans (`controllability.py`,
  `object_tables.py`, `find_camera.py`, `hero_tiles.py`, `section_scan.py`) at
  runtime, on this game, with no tuning.
* Any number that came from watching this game's confirmatory seeds.
* Hand-written `target_px` weights. Pricing is the scan's HP arithmetic, or a
  generic rule frozen in §5.4 — not a value tuned on the held-out game.

If the scan refuses to answer (as `vice_explore` already teaches it to), the
prior stays without that slot. A missing slot is honest; a guessed one is not.

## 4. Games

### 4.1 Already spent

Contra (J/U), Super C, Rush'n Attack, Super Mario Bros., Vice / Project Doom
and every game in `runs/` or `docs/experiments.md` has been watched and tuned.
**None of them is eligible for the held-out claim.** They may be used only as
*dev* games for building the pipeline.

### 4.2 Eligible for the held-out claim

A game is eligible if all of the following hold:

* a ROM is legally supplied by the user and imports into stable-retro;
* a manual PDF exists in the archives listed in [manuals.md](manuals.md);
* it is **not** in any training set, any tuning run, or any prior written so
  far;
* its score or progress is readable from the screen by the existing HUD
  reader, or a position counter is found by a scan that refuses hand-typed
  addresses (the `vice_explore` standard);
* nobody on the project plays it manually for tuning before the run.

### 4.3 Split, frozen now

| block | use | count |
|---|---|---|
| **dev** | build the pipeline, debug the schema, measure parse precision | ≥ 2 games (Contra J/US start here) |
| **held-out** | the claim, run once after freeze | **2 games**, named in §4.4, not opened by the pipeline author until the freeze date |
| **optional secret** | a third held-out, if budget allows, named by a second person | 0 or 1 |

### 4.4 Naming the held-out

Two names, written into this file by a dated note **after** the pipeline is
frozen and **before** the first held-out run:

* *Date — held-out A: …, held-out B: …, chosen by … , eligible because …*

Until that note exists, this protocol has not been executed and the claim has
not been tested. Choosing a game after seeing pipeline results on it spends
the game.

**Candidates gathered 2026-10-15** from the local No-Intro set. Each needs a
manual PDF in the archives of [manuals.md](manuals.md) before it is eligible.
None of these has been played or tuned in this repository:

| candidate | why it fits the protocol |
|---|---|
| Ninja Gaiden (USA) | platformer, horizontal progress matches the metric, famous manual |
| DuckTales (USA) | platformer, score on screen, clear control page |
| Mega Man 3 or 4 (USA) | platformer/shooter, bosses as room transitions |
| Kid Icarus (USA) | mixed vertical/horizontal scroll, tests scope-by-area |
| Blaster Master (USA) | mixed vehicle/foot, tests the scroll-mode axis of the manual |
| Faxanadu (USA) | slower RPG-ish platformer, different tempo |

**Pending choice (2026-10-15).** Recluse picks two from the table (or
substitutes). Pipeline author must not open those two until the parse freeze.
A candidate is spent the moment its confirmatory seeds are looked at.

**Dated note — 2026-10-18 (second pair, generalization test):**
* **held-out C: Kung-Fu Heroes (USA)** — chosen by Recluse, beat-em-up,
  new genre. ROM: `Kung-Fu Heroes (USA).nes`. Manual:
  archive.org `kung-fu-heroes-nes-hiresscans` (600 dpi).
* **held-out D: Pac-Man (USA)** — chosen by Recluse, maze game,
  single screen, score on screen. ROM: `Pac-Man (USA).nes`. Manual:
  archive.org `PacManNESHiResScans` (600 dpi).
* Chosen 2026-10-18 as a generalization test after Chip 'n Dale and
  Galaga were confirmed. Neither game appears in any training set,
  tuning run, or prior in this repository.

The pipeline author has not run `play`, `explore`, `train`, or any
experiment on Kung-Fu Heroes or Pac-Man and will not until the parse
freeze for this pair. Schema, scoring function and success criterion
are unchanged from the first pair.

**Dated note — 2026-10-15:**
* **held-out A: Chip 'n Dale - Rescue Rangers (USA)** — chosen by Recluse
  (chosen after DuckTales was withdrawn for having no NES manual PDF).
  Platformer, score on screen, co-op, Capcom 1990. ROM:
  `Chip 'n Dale - Rescue Rangers (USA).nes`. Manual:
  `manuals/Chip-n-Dale-Rescue-Rangers.pdf` (gamesdatabase, 20 pages).
* **held-out B: Galaga (USA)** — chosen by Recluse 2026-10-15, single-screen
  shooter, score on screen, no scroll. ROM: `Galaga - Demons of Death
  (USA).nes`. Manual: `manuals/Galaga_1985_Bandai.pdf` (gamesdatabase,
  17 pages). Famicom alternate scan: archive.org `galaga-fc-hiresscans`.
* **DuckTales (USA)** — withdrawn 2026-10-16 as held-out A: no NES manual
  PDF exists on the mirrors checked (see `manuals/INVENTORY.md`). ROM was
  present but the protocol requires a manual a person can open.

The pipeline author (this agent) has not run `play`, `explore`, `train`, or
any experiment on Chip 'n Dale or Galaga and will not until the parse freeze.
The manual PDFs must be confirmed in the archives of
[manuals.md](manuals.md) before either game is eligible to run.

**The pair is complete as of 2026-10-15.** Confirmatory seeds 5000–5031 stay
unopened. Next: freeze the parse pipeline (schema + gold + grounding) on the
dev games (Contra J, Vice, Battletoads), then open the two held-out manuals
and run the claim once.

## 5. The pipeline (what must freeze before held-out)

### 5.1 Parse

PDF pages → structured prior (schema in `assets/priors/schema/`). Every
extracted fact carries `page` and a quote or caption. A human may correct the
parse **once**, as a judge of the machine's output, and the correction is
logged. The judge does not add facts the parse did not propose, and does not
see held-out results.

**Dev metric, reported before held-out:** precision and recall of extracted
facts against a hand-made gold set on dev games, side by side with two
baselines: empty schema, and raw OCR text shoved at the same model.

### 5.2 Ground

Concept → on-screen entity, checked by a scan where possible ("the sensor
dies and the exit opens"), and confirmed once with a frame on file. The
mapping table is part of the artefacts.

**Dev metric:** fraction of gold concepts mapped to the correct entity, with
random-blob and all-HP-tiles baselines beside it.

### 5.3 Scans

Buttons, tap/hold, camera, object tables, hero tiles, section counter — the
existing scripts, run on the held-out game **after** the parse is frozen.
Scans may not use the manual's *values*; they may use its *vocabulary* only
through the grounding table.

### 5.4 Pricing

Hit points per target type come from `object_tables.py`. The conversion to
pixels of progress is either:

* the stage's own position stride over the scan's remaining-distance estimate
  (the Contra wall arithmetic), or
* a single generic rule frozen now: *one hit point of a goal object is worth
  1/16 of one screen of the progress counter*, as a floor when the stride is
  unknown.

The generic rule is deliberately coarse. It is not tuned on held-out.

## 6. Arms

All arms see the same ROM, the same auto-start, the same frame budget.

| arm | what it gets | why |
|---|---|---|
| random | nothing | floor |
| instincts | nothing (self-calibrating) | the thing that worked zero-shot on Castlevania |
| planner, no prior | scans only | isolates the manual's contribution |
| **planner + machine prior** | scans + prior from the PDF | **the claim** |
| planner + hand prior (dev only) | scans + current hand-written prior | ceiling of the manual channel on dev games |
| *privileged planner* | emulator as world model | scale, in italics, not a peer |

The hand-prior arm is **not** run on held-out games. Its existence on dev is
the measurement that the machine prior is worth something like what a person
extracts.

## 7. Seeds, budget, metrics

* **Development seeds:** 0–31, as everywhere else. For pipeline debugging.
* **Confirmatory seeds for the held-out claim:** **5000–5031**, thirty-two
  paired seeds, never used by any arm in this project before the freeze. (The
  4000–4031 block belongs to the learned-chooser preregistration and is not
  reused.)
* **Frame budget:** 3000 frames per run, commit 16, horizon 48 — the
  standard of [preregistration.md](preregistration.md). A longer budget is a
  protocol change, not a stretch.
* **Paired** across arms on the same seeds (common random numbers where the
  harness allows).

Metrics, all with median, IQM, and paired bootstrap 95% intervals, and with
environment frames and decision count reported separately:

1. `progress = level * 4000 + x` from a **scan-found** position, not a
   hand-typed address;
2. score read off the screen by `HudReader` (verified against RAM on dev);
3. clears / room transitions, counted by the section scan;
4. deaths per run.

Baseline discipline: every accuracy or rate is reported beside its baseline,
and the headline is the fraction of the *privileged planner's* gap the manual
closes — not "percent better than random".

## 8. Success criterion, stated in advance

The claim is confirmed only if **both** hold on the held-out block, pooled
across the two games and also reported per game:

* **planner + machine prior** beats **planner, no prior** on `progress` —
  paired permutation test on per-seed differences, 10000 permutations,
  two-sided, p < 0.05;
* **planner + machine prior** beats **instincts** on `progress` by the same
  test, p < 0.05.

Reported beside, not as criteria: the gap to the privileged planner, score,
deaths, and whether the machine prior beats the hand prior on dev (it should
be within its CI to argue the channel is not broken).

If only the first holds, the honest statement is "the manual helps the
planner, not shown to beat plain instincts". If neither holds, the direction
is reported as failed at the size this budget can resolve — see the
resolution table in [preregistration.md](preregistration.md) — and is not
re-analysed into a success by subsetting seeds, dropping a game, changing the
metric, or extending the budget.

## 9. What would make this invalid

* Running held-out games before the pipeline freeze date.
* Letting the pipeline author play or tune on a held-out game.
* Writing type IDs, HP, or `target_px` into a prior by hand for a held-out
  game.
* Choosing which two held-out games to report after seeing any of their
  results.
* Replacing a held-out game because "the ROM did not import" after looking at
  numbers; import failures are fixed before the freeze date or the game is
  replaced **before** any run, with a dated note.
* Changing the frame budget, the metric, or the success criterion after the
  fact. Any change requires a dated note in this file, and the run no longer
  counts as confirmatory.

## 10. Order of work

1. Freeze this file (done, 2026-10-08). Fill §4.4 when the pipeline is frozen.
2. Schema + parse on **dev** games (manual → prior).
3. Grounding table (concept → entity).
4. Scans run on held-out, values not types from the manual.
5. Dated note naming held-out A and B.
6. Run once on seeds 5000–5031. Publish, including a negative result.

Until step 5's note exists, nothing here is a result.

### Pipeline freeze — 2026-10-16

**Frozen on dev games** (Contra J, Contra US, Vice; Battletoads for HUD):
schema v2, gold sets, parse rules, grounding assembly, HUD reader, menu
polygon, measurements registry. Dev scores recorded in
`docs/measurements.jsonl`:

| dev parse | recall | vs raw OCR |
|---|---|---|
| Contra US | 1.0 | 1.0 (tie) |
| Contra J | 0.25 | — (control table is a picture) |
| Vice | 0.86 | 0.57 (parser wins) |

**Held-out pair** (§4.4): Chip 'n Dale (A) + Galaga (B). Both READY:
manual PDF + ROM present, neither parsed as a scored gold set, neither
played. A premature 3-fact extract of Galaga during inventory is quarantined
in `manuals/parsed/_quarantine/` and is not a gold set.

### Results — 2026-10-16

Ran on seeds 5000–5031, 3000 frames, commit 16, horizon 48.

**Held-out A, Chip 'n Dale - Rescue Rangers:** the claim is confirmed.
Planner + manual prior reaches median 239; planner without the prior
reaches 92. Paired difference +144 px, 19 of 20 common seeds won,
bootstrap 95% CI [109, 145] — zero is outside. The prior equals the
reactive policy alone (239), where the no-prior planner needs horizon 96
to reach 234. The manual's contribution is the crate combat: page 8's
"Press B to pick up objects; pressing B while holding them throws them
right or left" becomes composed templates (`pick and throw right`,
`pick and throw up`, `run jump throw`). Without those templates the same
planner scores 92 at the same horizon.

**Held-out B, Galaga:** the claim is confirmed at horizon 256. At the
default horizon (48) the claim failed: planner + prior 81, planner
without prior 81, policy alone 81, difference 0 px, CI [0, 2]. The
score window (fire → bullet → hit → score) is 60–120 frames, longer
than 48 or 96. With the horizon widened to 256, combat reward
(alignment + fire), and Galaga-specific templates from agent-play,
the planner reaches median 85 vs policy 81: **+3 px [2, 4], 6 of 6
paired seeds**. The constraint is the action-result window, not the
manual. Published in `docs/measurements.jsonl` as
`manual-assisted-galaga` (the h=48 negative) and
`galaga-h256-oracle-beats-bc` (the h=256 confirmation).

**What the test shows.** A manual helps when it names a *sequence of
actions* the game wants (pick up, then throw) that the scan cannot
discover, **and** the horizon reaches the action-result window. At
Chip 'n Dale both conditions hold at h=48. At Galaga the window is
three times longer and the horizon had to grow to 256. A manual alone
is not enough; a manual plus a horizon that matches the game's
action-result window is.
Action-to-result windows differ by orders of magnitude across games —
24 frames for a thrown crate, 60–120 for a Galaga bullet, hours in some
RPGs — and a fixed horizon is a property of the game, not a
hyperparameter.

## 11. Provenance

Every run writes `run.json` beside its artefacts per the global invariant:
command, git commit, dirty flag, versions, uv.lock sha, seed, ROM sha256,
core name+sha, and the arm's prior file hash. The parse artefacts (gold set,
extraction output, grounding table) are committed the same way, with the PDF
**not** committed (they are the rights holders' — see [manuals.md](manuals.md)).

## 12. Action-result windows (added 2026-10-18)

The horizon is not a hyperparameter; it is a property of the game.

| game | action | result | window | works at |
|---|---|---|---|---|
| Chip n Dale | pick up crate | throw hits enemy | 24 frames | h=48 |
| Contra | destroy sensor | stage ends | 200+ frames | damage term |
| Galaga | fire | bullet hits, score | 60–120 frames | h=256 |
| Metroid | buy item | use in boss fight | **hours** | not yet |

For windows inside the horizon, the planner sees the consequence and
chooses well. For windows longer than the horizon, it cannot: every
candidate scores the same, and the chooser has nothing to learn from.

The Galaga result (h=256 beats BC 6/6) confirms this. The Metroid class
of games needs either:

1. **Hierarchical subgoals** — decompose "reach the boss" into
   room-level steps the horizon can see;
2. **Forward models** — predict the far-future reward from the current
   state instead of simulating to it;
3. **Continuous proxy signals** — the combat reward (alignment + fire)
   that stands in for the consequence within the window.

All three are research directions, not MVP. The skill library (stage 10)
is the first piece of (1): compositions of actions that a single
template cannot express.

