# Training

Everything is PyTorch on a laptop accelerator — Metal on Apple Silicon, CUDA on
Linux, CPU otherwise. The models are small; training takes minutes, not hours.

## Where the data comes from

1. **The instincts playing by themselves.** `explore --record` writes episodes to
   Zarr, and headless that is hundreds of episodes in minutes. Contra, Super
   Mario Bros., Battle City, Double Dragon, Ice Climber, Gradius, Balloon Fight,
   Battletoads and BT&DD were all collected this way, 23–57 episodes each — a
   complete "instincts → data → model" loop with no demonstrations of any kind.
   Typical outcome: 0.70–0.94 validation accuracy.
2. **TAS movies** replayed deterministically into Zarr episodes. Super Mario
   Bros. syncs bit for bit. Games with lag frames (Contra, BT&DD in two-player)
   desynchronise, because fceumm and FCEUX disagree about which frames lag.
3. **The verified prefix of a desynchronising movie.** If a movie diverges half
   way through, the part before the divergence is still perfectly good data —
   BT&DD contributes 4798 frames of both players that way.

## Behavioural cloning (`policy/bc.py`)

- Input: a stack of 4 RGB frames at 120×112. Colour is kept — on the NES it
  carries information, from power-up state to whether a Goomba is a Goomba.
- Action vocabulary: the distinct button bitmasks that actually occur in the
  data, rather than all 256 combinations.
- `BCNet`: conv 32/64/64 → FC 512 → head, about 4M parameters.
- `BCNetAV` (`--audio`): plus an audio encoder — a 32×52 log-mel window, roughly
  260 ms, through a small CNN to 128 dimensions — concatenated before the head.
- A sample is frames [i-4..i-1] predicting the action at frame i. Validation is
  the last 10% of the episode, or the last 10% of episodes when training on a
  directory. Never a random frame split: consecutive frames are near-duplicates,
  and a random split leaks the answer.
- `--attn` adds attention supervision: cross-entropy between the spatial softmax
  of the last conv layer's activations and a mask of the motion tracker's boxes,
  cached in `attn_mask.npy` beside the episode.

The attention loss exists because of a measurement. Left alone, the network put
12.5% of its attention mass inside object boxes while a uniform gaze would put
13.8% there — it was doing *worse than looking nowhere in particular*, keying on
background that happened to correlate with the action. With supervision it
reaches 21.5%, and 38.0% on the multi-game base, at a cost of 2–3 points of
accuracy.

At inference the policy decides at about 15 Hz. Jump duration is added outside
the network (see `JumpShaper` in `cli/play.py`): a full Super Mario Bros. jump
needs roughly 32 frames of held A, which a policy resampled every 4 frames
cannot express by itself.

## Self-imitation (`policy/improve.py`)

A cheap stand-in for reinforcement learning: run N rollouts with sampling, score
them, retrain on the best third, repeat. Eight rounds on Super Mario Bros. moved
mean progress from 335 to 472, with the final round taking no deaths.

It overwrites the checkpoint's `model.pt`. Copy the checkpoint first.

Two rewards:

- **from RAM** (default): `progress + 2·Δscore − 300·death`, read from debug RAM
  in the training loop only, never by the policy;
- **from pixels** (`--visual`): accumulated camera scroll via phase correlation,
  with jumps over 16 px discarded as scene changes. RAM is not opened at all, so
  this works on games for which no memory map exists. It correlates 0.87 with
  true progress.

## World model

**v1/v2 (`world_model/model.py`) — a negative result, kept as history.** Latent
dynamics, encoder → GRU → decoder. The model ignored actions entirely:
open-loop advantage about 1.0. The diagnosis is worth remembering — the latent
was dominated by background and scroll, and the effect of a button press, two
pixels of hero displacement, is a grain of sand in the MSE of a 256-dimensional
latent.

**v3/v4 (`world_model/ego.py`) — the one that works.** A 48×48 crop around the
controlled object, which the motion tracker supplies, plus its velocity and the
action → GRU(128) → (dx, dy). Median-smoothed trajectories, scheduled sampling.
The metric is a 16-step open-loop rollout with true actions against shuffled
ones: v3.0 gives 1.063, v3.1 gives 1.093, v4 on 43 episodes gives **1.19**.

Used for the ghost trajectory on the dashboard (`play --ghost`, inferred on the
CPU because the GPU belongs to the brain thread) and for the MPC planner
(`--planner`).

The planner's history is the useful part: on a weak world model, planning
**lost** to plain reaction. It only started winning after the fourth iteration
of the model. A planner is only as good as what it plans in.

## Reproducibility

- Seeds: `train_bc(..., seed=N)`, `train_wm(..., seed=N)`.
- Checkpoint metadata — action vocabulary, modality, mel statistics, training
  history — lands in `runs/<name>/meta.json`.
- Results are recorded in [experiments.md](experiments.md), negative ones
  included.
- ROMs are verified by hash: SHA-1 in the integrations, MD5 from the `.fm2`
  headers.
