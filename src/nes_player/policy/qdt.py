"""QDT: Q-learning Decision Transformer.

Improves DT's return-to-go conditioning by relabelling RTG with a learned
value function (Yamagata et al., 2022, arXiv:2209.03993).

Steps:
1. Learn value function V(s) from agent-play data (Q-learning / MSE).
2. Relabel RTG: if V(s_t) > R_t, replace R_t with V(s_t) (Algorithm 1).
3. Train DT on relabelled data.

    from nes_player.policy.qdt import relabel_rtg, ValueFunction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ValueFunction(nn.Module):
    """Simple MLP value function V(s) -> scalar."""

    def __init__(self, state_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


def train_value_function(
    states: np.ndarray,
    returns: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> ValueFunction:
    """Train V(s) to predict return-to-go from state features.

    states: (N, T, state_dim) — window of state features
    returns: (N, T) — return-to-go at each timestep
    """
    vf = ValueFunction(state_dim=states.shape[-1]).to(device)
    opt = torch.optim.Adam(vf.parameters(), lr=lr)

    # Flatten to individual timesteps
    s = torch.tensor(states, dtype=torch.float32, device=device)
    r = torch.tensor(returns, dtype=torch.float32, device=device)

    # (N, T, state_dim) -> (N*T, state_dim)
    s_flat = s.reshape(-1, s.shape[-1])
    r_flat = r.reshape(-1)

    n = len(s_flat)
    # Subsample if too many timesteps
    if n > 10000:
        idx_sub = torch.randperm(n)[:10000]
        s_flat = s_flat[idx_sub]
        r_flat = r_flat[idx_sub]
        n = len(s_flat)

    bs = 512
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            s_batch = s_flat[idx]
            r_batch = r_flat[idx]

            pred = vf(s_batch)
            loss = nn.functional.mse_loss(pred, r_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(idx)

        if epoch % 5 == 0:
            print(f"  vf epoch {epoch}: loss={total_loss/n:.4f}")

    return vf


def relabel_rtg(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    vf: ValueFunction,
    device: str = "cpu",
) -> np.ndarray:
    """Relabel return-to-go using learned value function (Algorithm 1).

    For each trajectory, walk backwards:
        R_T = 0
        R_{t-1} = r_{t-1} + max(R_t, V(s_t))

    states: (N, T, state_dim)
    rewards: (N, T) — immediate rewards
    returns: (N, T) — original return-to-go
    """
    N, T, _ = states.shape
    relabeled = np.zeros((N, T), dtype=np.float32)

    s = torch.tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        v = vf(s.reshape(-1, s.shape[-1])).reshape(N, T).cpu().numpy()

    for i in range(N):
        R = 0.0
        for t in range(T - 1, -1, -1):
            # R_{t} = r_t + max(R_{t+1}, V(s_{t+1}))
            # But we need R_{t-1} = r_{t-1} + max(R_t, V(s_t))
            # Start from end: R_T = 0
            if t == T - 1:
                relabeled[i, t] = 0.0
            else:
                relabeled[i, t] = rewards[i, t] + max(relabeled[i, t + 1], v[i, t + 1])
            # Actually Algorithm 1: R_{tau-1} = r_{tau-1} + max(R_tau, V(s_tau))
            # So we compute R_{t-1} from R_t and V(s_t)

    # Recompute properly following Algorithm 1
    for i in range(N):
        R = 0.0  # R_T = 0
        relabeled[i, T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            # R_t = r_t + max(R_{t+1}, V(s_{t+1}))
            R_next = relabeled[i, t + 1]
            V_next = v[i, t + 1]
            relabeled[i, t] = rewards[i, t] + max(R_next, V_next)

    return relabeled


def consistency_relabel(
    rewards: np.ndarray,
    rtg: np.ndarray,
    context_len: int = 32,
) -> np.ndarray:
    """Ensure RTG consistency within DT context windows (Algorithm 2).

    For each window, copy last RTG and work backwards:
        R_hat_t = r_t + R_hat_{t+1}
    """
    N, T = rewards.shape
    consistent = np.zeros_like(rtg)

    for i in range(N):
        for t in range(T):
            # Window ending at t
            start = max(0, t - context_len + 1)
            # Start from R_hat_t = rtg[i, t]
            R_hat = rtg[i, t]
            consistent[i, t] = R_hat
            for tau in range(t - 1, start - 1, -1):
                R_hat = rewards[i, tau] + R_hat
                consistent[i, tau] = R_hat

    return consistent
