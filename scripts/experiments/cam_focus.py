"""What share of the Grad-CAM mass falls inside the tracker's object boxes.

The chance level is the mean mask area — what a uniform gaze would score — and
reading the two together is the whole point: an untrained model scored 12.5%
against a chance level of 13.8%, meaning it was looking at scenery.

Usage:
  uv run python scripts/experiments/cam_focus.py <episode_dir> <checkpoint> [...]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from nes_player.data.reader import Episode
from nes_player.policy.bc import BCPolicy, episode_attn_masks

ep = Episode(Path(sys.argv[1]))
masks = episode_attn_masks(ep).astype(np.float32)
n = len(ep)
start = n - int(n * 0.3)

chance = []
for k, ckpt in enumerate(sys.argv[2:]):
    pol = BCPolicy(ckpt)
    scores = []
    for i in range(start, n):
        pol.push_audio(ep.frame_audio(i))
        want = (i - start) % 5 == 0 and masks[i].sum() > 0
        out = pol.act(ep.frames[i], with_cam=want)
        if not want or out[2] is None or out[2].sum() < 1e-6:
            continue
        scores.append(float((out[2] * masks[i]).sum() / out[2].sum()))
        if k == 0:
            chance.append(float(masks[i].mean()))
    print(f"{ckpt}: cam-in-box {np.mean(scores):.3f} (n={len(scores)})")
print(f"chance (uniform cam): {np.mean(chance):.3f}")
