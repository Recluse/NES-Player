"""DTPolicy: Decision Transformer as a reactive policy for the MPC planner.

Same interface as BCPolicy.act(frame_rgb, temperature) -> (pressed, ranked).
Maintains (return-to-go, state, action) history for the DT context window.

    from nes_player.policy.dt_policy import DTPolicy
    policy = DTPolicy("runs/dt_galaga.pt", target_rtg=50.0)
    pressed, ranked = policy.act(frame_rgb)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from nes_player.policy.decision_transformer import DTConfig, count_params

BUTTONS = ["A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]


def bitmask_to_index(mask: int) -> int:
    bits = [i for i in range(8) if mask & (1 << i)]
    if len(bits) == 1:
        return bits[0]
    return 8


def index_to_buttons(idx: int) -> frozenset[str]:
    if idx < 8:
        return frozenset({BUTTONS[idx]})
    return frozenset({"A", "B"})  # combination token


def frame_features(frame_rgb: np.ndarray) -> list[float]:
    """Extract 6-dim state features from a frame."""
    band = frame_rgb[8:24, 195:240]
    sc = int(((band[..., 0] > 180) & (band[..., 1] > 180)
              & (band[..., 2] > 180)).sum())
    bright = float(frame_rgb.mean())
    red = float((frame_rgb[..., 0] > 150).mean())
    green = float((frame_rgb[..., 1] > 150).mean())
    blue = float((frame_rgb[..., 2] > 150).mean())
    dark = float((frame_rgb.mean(axis=-1) < 40).mean())
    return [sc, bright, red, green, blue, dark]


class StateProj(torch.nn.Module):
    def __init__(self, d_model: int = 384):
        super().__init__()
        self.proj = torch.nn.Linear(6, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DTPolicy:
    """Decision Transformer policy with BC-compatible interface."""

    def __init__(self, ckpt_path: str | Path, target_rtg: float = 50.0,
                 context_len: int = 32):
        self.target_rtg = target_rtg
        self.context_len = context_len
        self._states: list[list[float]] = []
        self._actions: list[int] = []
        self._returns: list[float] = []

        cfg = DTConfig(n_actions=9, d_model=384, n_layers=6, n_heads=6)
        self.model = cfg.build(state_encoder=StateProj())
        ckpt = Path(ckpt_path)
        if ckpt.exists():
            self.model.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=True)
            )
        self.model.eval()

    def reset(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._returns.clear()

    def act(self, frame_rgb: np.ndarray, temperature: float = 1.0,
            with_cam: bool = False):
        """Returns (pressed, ranked) matching BCPolicy interface."""
        feat = frame_features(frame_rgb)
        self._states.append(feat)
        self._returns.append(self.target_rtg)

        # Build context window
        ctx = min(self.context_len, len(self._states))
        states = self._states[-ctx:]
        returns = self._returns[-ctx:]
        actions = self._actions[-ctx:] if self._actions else [0]
        # Pad actions if needed
        while len(actions) < ctx:
            actions = [0] * (ctx - len(actions)) + actions

        r = torch.tensor([returns], dtype=torch.float32)
        s = torch.tensor([states], dtype=torch.float32)
        a = torch.tensor([actions], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(r, s, a)[0, -1, :]  # last timestep

        # Temperature sampling
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            action_idx = int(torch.multinomial(probs, 1).item())
        else:
            action_idx = int(logits.argmax())

        self._actions.append(action_idx)
        pressed = index_to_buttons(action_idx)

        # Ranked distribution
        probs = torch.softmax(logits, dim=-1).numpy()
        ranked = sorted(
            [(index_to_buttons(i), float(p)) for i, p in enumerate(probs)],
            key=lambda x: -x[1],
        )
        return pressed, ranked
