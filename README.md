# NES Player

Two things live in this repository, and honesty requires introducing them
separately.

The first is a perception stack for playing NES games the way a person does:
seeing the screen, hearing the sound, and nothing else. The emulator's memory
is closed to the policy — it is read only by training and evaluation, as
ground truth for checking what the agent read off the screen.

The second is a measurement journal — and the journal's verdict, as of now, is
that the strongest player in the repository is not that stack. It is a planner
that openly cheats: it uses the emulator itself as a world model. That verdict
was reached by paired experiments, survived every attempt to overturn it, and
is the most interesting thing here.

## What plays today

`scripts/experiments/oracle_mpc.py`: at each decision the planner saves the
console state, plays out a handful of 48-frame button templates plus the
policy's own plan, continues each with the reactive policy for 96 frames,
averages 2–4 such futures, commits 16 frames of the winner, and repeats.
Progress comes from the game's own position counter; a plan that dies is not
compared on distance.

The same planner, the same handful of templates, no per-game tuning beyond a
position address and a boot sequence — and, since 3 September, not even the
address:

| environment | paired gain over the policy | seeds |
|---|---|---|
| SMB 1-1 | +3828 (and 20/32 full clears at 4 draws) | 32 |
| SMB 2-1 … 6-1 | +509 … +2483, every CI clear of zero | 32 each |
| Contra (J) | +2270; **level 1 cleared from power-on in 12/32** with five composed templates and a damage term on the game's object tables | 32 |
| Contra (U) | +2134 | 32 |
| Super C | +1210 | 32 |
| Rush'n Attack | +145 [+124, +166], 32/32 — position **and** templates found by scanning, zero hand-written addresses | 32 |
| Contra (J), the base | first room opened in **17/32** from a room state, with the manual's two notes (what to destroy, where the door is) and every other input from scans | 32 |

Every per-game input the planner uses in Contra is now found by a scan
rather than typed: which buttons do what and whether fire is tapped or held
(`controllability.py`), the templates built from that (`scan_templates`),
the camera or an 8-bit scroll (`find_camera.py`), the object tables that
hold hit points (`object_tables.py`), the player's own sprites
(`hero_tiles.py`), and the stage's section counter from the fades between
rooms (`section_scan.py`). What stays human is the manual: a small file per
game saying what the game wants destroyed and where the exit is
(`assets/priors/`), which is what a person reads before playing too.

Read the caveats before the numbers impress you: the planner is privileged
by construction — it looks at futures instead of predicting them; on Contra
the learned policy is inert (median 0 on every seed), so the gain there is
carried by the template prefixes; the journal records four retractions from
these two days, each caught by a frame or a scripted check rather than by
thought (a "hits" count that was a savestate's constant, a vertical term that
was measuring enemy fire, a "horizon problem" and an "arithmetic problem"
that were one term returning zero); and the base result is one room, not
the stage — in the next room the soldier parks in a corner again.

## What learning could not take over

The obvious next step — distil the planner into a network and stop paying for
rollouts — was tried to the end and failed with measured causes. A learner was
priced in five roles (value estimator, action chooser, binary gate, DAgger
student, compute allocator) and lost all five; a battery of seven input
representations, from two scalars to the full privileged state, captured the
same ~20% of the planner's per-decision gain; the practically useful remainder
was reachable only by simulating the future. Two structural findings survived
every control: monolithic 96-frame plans are toxic (half the progress, double
the deaths) and a few hand-written compositions buy most of it back; and at 32
paired seeds the online noise floor certifies only planner-sized effects —
finer economics is measured offline, on stored rollout matrices, under common
random numbers. The full chronology, retractions included, is
[docs/experiments.md](docs/experiments.md).

## The perception stack

The original point stands as a goal: skills that carry over to games the agent
has never seen, from pixels and sound alone.

![The live dashboard: Super Mario Bros. with the attention overlay, action
probabilities, tracked objects and the numbers the agent read off the
screen](docs/images/dashboard.jpg)

*Left: the game with Grad-CAM showing where the network is looking, boxes around
the objects the motion tracker found, and rings where a sound was predicted to
come from. The gamepad below lights up the buttons actually being pressed —
here B and RIGHT, matching the 0.68 the model gives that action. Right: the
action distribution, uncertainty over time, tracked objects, live conv features,
remembered sprites, the sounds heard, and the agent's own running commentary.*

*`hud read 300 1 1 1 391` is score, coins, world and timer, read off the picture
with no labels and no memory access. The game's own HUD in the same frame reads
000300, ×01, 1-1, TIME 391.*

## What it does

- **Finds objects by motion**, corrected for camera scroll, and works out which
  one it controls by correlating world-space velocity with button presses.
  Subtracting the scroll is the part that matters: when the camera follows the
  hero his screen velocity is nearly zero, and without the correction the agent
  cannot recognise itself at all.
- **Reads the screen.** Digits are learned with no labels whatsoever: in any
  counter the lowest digit's transitions form a ring 0→9, and zero is found
  topologically, as the glyph with a closed hole. Checked against RAM, the timer
  reads with correlation 1.000 and 95.8% exact matches.
- **Listens.** A hit, damage or a scene change is often audible before it is
  visible. Adding audio raises action-prediction accuracy by 7 points on 4 of 4
  seeds.
- **Builds a world model** around the object it controls and plans 16 steps
  ahead over behaviour templates.

## Perception stack results

| | |
|---|---|
| First contact with an unseen game (Castlevania) | **700 points** from instincts alone; a random agent and a transferred network both score **0** |
| Trained games | 7, action-prediction accuracy **0.70–0.94** |
| Few-shot on unseen games, 3 episodes | Gradius 0.707 → **0.787**, Battletoads 0.594 → **0.671** |
| Attention inside object boxes | 12.5% → **21.5%** (38.0% multi-game) against 13.8% for a uniform gaze |
| Whole model | **16.5 MB** — vision, hearing and decision together |

Negative results are published beside the positive ones. Zero-shot policy
transfer **does not work**: on an unseen game the multi-game base covers 9.7
units of distance against 126 for a random agent. Features transfer; behaviour
does not. Slot attention failed to decompose NES frames into objects, because a
tiled background is described just as well by positional blobs. The full log is
in [docs/experiments.md](docs/experiments.md).

## Where the data comes from

No demonstrations are required. An instinct policy — no training, just control
calibration and a handful of exploration rules — plays the game itself, headless
at about a thousand frames per second, and behavioural cloning learns from its
episodes. Collecting a dataset for a new game takes minutes.

![The launcher: pick a mode, a game, a checkpoint and recording
options](docs/images/launcher.png)

![Double Dragon: the agent closes with an enemy, turns to face it and strikes;
when surrounded it backs away](docs/images/beat-em-up.jpg)

## Running it

macOS on Apple Silicon and Linux x86_64 work out of the box; Windows goes
through WSL2, since stable-retro ships no native Windows build. A GPU is
optional — CUDA, Metal or CPU, selected automatically. Everything runs
through [uv](https://github.com/astral-sh/uv), Astral's Python package and
project manager — install it first (`curl -LsSf https://astral.sh/uv/install.sh | sh`
on macOS and Linux, or see its README), then:

```bash
uv sync
uv run pytest -q
```

**No ROM images are distributed with this project.** Import your own:

```bash
uv run python -m retro.import /path/to/roms
```

Trained weights are release assets, not repository contents — seven checkpoints
at 16 MB each is not what git is for:

```bash
uv run python scripts/fetch_models.py --list   # see what there is
uv run python scripts/fetch_models.py          # fetch it, about 105 MB
```

You can skip that entirely. The instinct policy needs no model at all, and
training one for a new game takes about half an hour; see
[docs/models.md](docs/models.md).

Then, either the launcher:

```bash
./start.sh
```

or directly:

```bash
# Watch a trained model play
uv run nes-player play --game SuperMarioBros-Nes-v0 --checkpoint runs/bc_smb_attn3 \
    --window --realtime --hd --auto-start

# Let the instincts explore a new game and record what they do
uv run nes-player explore --game Gradius-Nes-v0 \
    --record datasets/explore_gradius --loop --max-frames 3600

# Train on those episodes
uv run nes-player train-bc --episode datasets/explore_gradius \
    --out runs/bc_gradius --audio --attn 1.0 --epochs 3
```

## Documentation

Everything is in [`docs/`](docs/README.md): the
[specification](docs/specification.md),
[architecture](docs/architecture.md),
[command line](docs/cli.md),
[dashboard](docs/dashboard.md),
[perception](docs/perception.md),
[training](docs/training.md),
[emulation cores](docs/cores.md),
[models](docs/models.md),
[ROMs](docs/roms.md) and the
[experiment log](docs/experiments.md).

## Licence

MIT, © 2026 Ruslan Semov — see [LICENSE](LICENSE).

Two things carry their own terms and are **not** distributed here: the libretro
emulation cores (fceumm, nestopia, quicknes) are GPL-2.0 and are downloaded from
the official libretro builds onto your machine at first use, and ROM images
belong to their rights holders and are yours to supply.
