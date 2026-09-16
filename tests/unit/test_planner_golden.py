"""The planner has to be the same planner it was yesterday.

`oracle_mpc.run` is the strongest player in this repository and the source of
every planner number in `docs/experiments.md`, and until now nothing checked it
at all. Two of the four retractions on that page were changes to the metric and
to a scoring term — exactly the kind of change this catches, because it does
not ask whether the planner is good, only whether it still does what it did.

The policy it needs is built here from a fixed generator rather than trained:
no checkpoint ships with the repository, and pinning the weights ourselves also
means a change in how torch initialises cannot move the golden. A policy of
random weights exercises the loop but not much of the policy's own behaviour,
so this is a regression test for the decision loop and not a claim about play.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/experiments"))

# What the loop did at the commit this test was written. Any of these moving is
# either a deliberate change to the planner or a bug; both want to be noticed.
GOLDEN = {
    "best_x": 95,
    "branch_frames": 8208,
    "deaths": 0,
    "chosen": {"bc": 10, "fire up-right": 1, "jump now": 1, "run": 7},
}


def _fixture(out: Path) -> Path:
    from nes_player.policy.bc import DEFAULT_OFFSETS, BCNet

    masks = [0, 1 << 0, 1 << 7, (1 << 0) | (1 << 7)]
    out.mkdir(parents=True, exist_ok=True)
    model = BCNet(len(masks), in_ch=len(DEFAULT_OFFSETS) * 3)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.from_numpy(
                rng.normal(0, 0.02, tuple(p.shape)).astype("float32")))
    torch.save(model.state_dict(), out / "model.pt")
    (out / "meta.json").write_text(json.dumps({
        "vocab_masks": masks,
        "vocab_names": ["NOOP", "A", "RIGHT", "A+RIGHT"],
        "frame_offsets": list(DEFAULT_OFFSETS),
        "modality": "video", "arch": "conv", "input_hw": [112, 120],
    }))
    return out


def _run(ckpt: Path) -> dict:
    import oracle_mpc as m

    # frames=300, horizon=48, commit=16, repeat=4, seed=0, temperature=1.0
    return m.run(str(ckpt), "ContraJ-Nes-v0", None, 300, 0, 1.0, 4, 48, 16)


@pytest.mark.skipif(not (ROOT / "integrations/ContraJ-Nes-v0").exists(),
                    reason="needs the Contra integration")
def test_the_planner_still_decides_what_it_decided(tmp_path):
    row = _run(_fixture(tmp_path / "ckpt"))
    got = {k: row[k] for k in GOLDEN}
    assert got == GOLDEN, f"the planner's decisions moved:\n{got}\n{GOLDEN}"


@pytest.mark.skipif(not (ROOT / "integrations/ContraJ-Nes-v0").exists(),
                    reason="needs the Contra integration")
def test_the_planner_repeats_itself(tmp_path):
    """Same seed twice. Without this the golden above could pass by luck."""
    ckpt = _fixture(tmp_path / "ckpt")
    assert _run(ckpt) == _run(ckpt)
