# NES Player

An agent that learns to play NES games the way a person does: it sees the screen
and hears the sound, and nothing else. The emulator's memory is closed to the
policy — it is read only by the training loop and by evaluation, as ground truth
for checking whether what the agent read off the screen was correct.

The point is not to finish one game. The point is that skills carry over to
games the agent has never seen.

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

## Measured results

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
optional — CUDA, Metal or CPU, selected automatically.

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
