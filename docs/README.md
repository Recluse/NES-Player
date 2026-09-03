# NES Player documentation

A research system: an autonomous agent learns to play NES games through a
human-like interface — pixels and sound in, gamepad out. Nothing else.

Start with [specification.md](specification.md) for what the project is trying
to prove, then pick a document below.

## Map

| Document | What is in it |
|---|---|
| [specification.md](specification.md) | Goals, principles, acceptance criteria, roadmap |
| [architecture.md](architecture.md) | Components, data flow, the threading of the live window |
| [cli.md](cli.md) | Every `nes-player` command, with flags and examples |
| [dashboard.md](dashboard.md) | The live panel: what each pane shows, hotkeys, clickable pad |
| [perception.md](perception.md) | Object tracking, object memory, sound events, screen reading |
| [training.md](training.md) | Datasets, behavioural cloning, audio, attention, self-imitation, world model |
| [models.md](models.md) | The trained checkpoints: what they score, how to fetch them, how to make your own |
| [cores.md](cores.md) | Emulation cores: which are supported, how to switch, what differs |
| [roms.md](roms.md) | Which ROMs the tests expect, by checksum. None are distributed |
| [experiments.md](experiments.md) | The experiment log, negative results included |

## Implementation status

| Stage | State |
|---|---|
| 0. Repository and infrastructure | done — uv, Python 3.14, ruff, pytest (132 tests including regression) |
| 1. Emulator harness | done — stable-retro/fceumm headless, determinism, savestates |
| 2. Dataset builder | done — FM2 replay to Zarr; desynchronising movies contribute a verified prefix |
| 3. Behavioural cloning baseline | done — plus self-imitation |
| 4. Audio | done — multimodal model, sound events, sound meaning |
| 5. Object discovery | partial — motion tracker and attention supervision; neural slots v1 was a negative result |
| 6. World model | done — ego model v4 |
| 7. Planner | done — MPC over the ego model |
| 8. Multi-game pretraining | done — transfer confirmed, including a strict held-out split |
| 9. Online adaptation | not started |
| 10. Skill library | not started |

All ten MVP acceptance criteria are met. Every number quoted anywhere in these
documents is reproducible with a script in `scripts/experiments/`.

## Trained models

Weights are published as release assets rather than committed. Fetch them with
`uv run python scripts/fetch_models.py`, or train your own in about half an hour
— see [models.md](models.md) for the full table and both routes.

| Game | Checkpoint | Accuracy | Majority baseline |
|---|---|---|---|
| Balloon Fight | `runs/bc_balloonfight_attn` | 0.944 | 0.411 |
| Contra (US) | `runs/bc_contra_attn` | 0.943 | 0.377 |
| Gradius | `runs/bc_gradius_attn` | 0.872 | 0.449 |
| Double Dragon | `runs/bc_dd_attn2` (needs `--state default`) | 0.864 | 0.318 |
| Super Mario Bros. | `runs/bc_smb_attn3` | 0.708 | 0.717 |
| Battletoads | `runs/bc_battletoads_attn` | 0.702 | 0.430 |
| Multi-game base / observer | `runs/bc_base41_attn1` | 0.944 | 0.377 |

Datasets are not published: they are hundreds of gigabytes, and
`explore --record` regenerates equivalent ones in minutes.

## Quick start

Requires [uv](https://github.com/astral-sh/uv) (Astral's Python package and
project manager).

```bash
uv sync                      # environment
uv run pytest -q             # tests; the integration ones need imported ROMs
uv run python -m retro.import /path/to/roms

./start.sh                   # graphical launcher: game, mode, checkpoint, recording
```

Or by hand — watch a trained model play:

```bash
uv run nes-player play --game SuperMarioBros-Nes-v0 --checkpoint runs/bc_smb_attn3 \
    --window --realtime --hd --auto-start --loop
```

Explore a new game with instincts while a network watches over its shoulder:

```bash
uv run nes-player explore --game ContraU-Nes-v0 --integrations integrations \
    --observer runs/bc_base41_attn1 --window --realtime --hd --loop
```
