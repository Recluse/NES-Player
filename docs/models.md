# Trained models

The weights are **not** in this repository. A single behavioural-cloning
checkpoint is 16 MB and there are seven of them, which is not what git is for.
They are published as release assets and described by
[`assets/models.json`](../assets/models.json), which carries a sha256 for each.

```bash
uv run python scripts/fetch_models.py --list      # what is available
uv run python scripts/fetch_models.py             # all of it, about 105 MB
uv run python scripts/fetch_models.py bc_smb_attn3
```

Downloads land in `runs/` and are verified against their checksum before being
unpacked. Set `NES_MODELS_URL` to fetch from a mirror.

## What is in there

| Model | Game | Accuracy | Majority baseline |
|---|---|---|---|
| `bc_balloonfight_attn` | Balloon Fight | 0.944 | 0.411 |
| `bc_contra_attn` | Contra (US) | 0.943 | 0.377 |
| `bc_gradius_attn` | Gradius | 0.872 | 0.449 |
| `bc_dd_attn2` | Double Dragon (needs `--state default`) | 0.864 | 0.318 |
| `bc_smb_attn3` | Super Mario Bros. | 0.708 | 0.717 |
| `bc_battletoads_attn` | Battletoads | 0.702 | 0.430 |
| `bc_base41_attn1` | multi-game base, also the observer | 0.944 | 0.377 |
| `ego_smb4` | ego world model, Super Mario Bros. | action advantage 1.19 | — |
| `av_smb` | sound localisation, Super Mario Bros. | top-1 0.254 | chance 0.016 |

Read the accuracy against the baseline beside it, always. Super Mario Bros.
looks like the worst model in the table and is not: its training data is a TAS,
where the optimal route holds one button for long stretches, so a constant
prediction already scores 0.717. Balloon Fight's 0.944 against 0.411 is a much
smaller achievement than it looks, and Mario's 0.708 against 0.717 is a much
larger one.

Every checkpoint is a directory with two files: `model.pt` and a `meta.json`
holding the action vocabulary, the modality, mel statistics and the full
training history.

## Using one

```bash
uv run nes-player play --game SuperMarioBros-Nes-v0 --checkpoint runs/bc_smb_attn3 \
    --window --realtime --hd --auto-start
```

`ego_smb4` and `av_smb` are picked up automatically by `play` when present: the
first draws the ghost trajectory and powers `--planner`, the second marks where
a sound came from.

## Or train your own

You do not need any of these. Nothing in this project requires a demonstration
or a pretrained model — that is rather the point. The instinct policy plays an
unfamiliar game without any training at all, and its episodes are the training
data:

```bash
# 1. Instincts play and record, headless at about 1000 frames per second
uv run nes-player explore --game Gradius-Nes-v0 \
    --record datasets/explore_gradius --loop --max-frames 3600

# 2. Clone the behaviour, with sound and attention supervision
uv run nes-player train-bc --episode datasets/explore_gradius \
    --out runs/bc_gradius --audio --attn 1.0 --epochs 3
```

On a laptop that is roughly half an hour end to end, and it is how every model
in the table above was made, except Super Mario Bros., which came from a TAS.

## Size

16.5 MB per model, of which **89% is the final fully connected layer** and only
2% is the vision stack. If these ever need to be smaller, that layer is where
the whole answer is — not the convolutions, which is where one would instinctively
look first.
