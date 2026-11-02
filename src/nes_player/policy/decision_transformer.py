"""Decision Transformer: reinforcement learning as sequence modeling.

Instead of BC's `frame → action`, DT conditions on `(return-to-go, frame,
action-history)` and predicts the next action. The return-to-go is the
dial: set it high and the model plays aggressively; set it low and it
plays safe. The horizon stops mattering because the target is given, not
simulated.

Architecture (Malady 2026-10-20: 10–15M parameters, not GPT-2's 124M):

    4 transformer layers, d_model=256, 4 heads, ~12–15M params
    conv backbone from BC (frame → 256-d embedding)
    action embedding: 8 buttons → 256-d
    return-to-go: scalar → 256-d

Tokens per timestep: [R_t, s_t, a_t]. Position = timestep embedding.
Predict a_t from (R_t, s_t, a_{<t}) with teacher forcing.

Not a replacement for the planner: the planner sets the goal (return-to-go),
DT generates the actions. The lexicon and grounding stay where they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if torch is not None:
    class TimestepEmbedding(nn.Module):
        """Sinusoidal timestep embedding, added to every token of that step."""

        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            self.d_model = d_model
            self.max_len = max_len
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe)

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return self.pe[t]  # (B, d_model)


    class DecisionTransformer(nn.Module):
        """(return-to-go, state, action) → action.

        The state encoder is pluggable: pass a module that maps frames to
        d_model embeddings (the BC conv backbone works). If none is given,
        states are expected as already-embedded vectors.
        """

        def __init__(
            self,
            n_actions: int = 8,
            d_model: int = 256,
            n_layers: int = 4,
            n_heads: int = 4,
            max_timesteps: int = 512,
            state_encoder: nn.Module | None = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_actions = n_actions
            self.d_model = d_model
            self.max_timesteps = max_timesteps

            # projections: each of (R, s, a) becomes d_model
            self.ret_proj = nn.Linear(1, d_model)
            self.state_encoder = state_encoder or nn.Identity()
            self.state_proj = nn.Linear(d_model, d_model)
            self.action_emb = nn.Embedding(n_actions, d_model)

            self.timestep_emb = TimestepEmbedding(d_model, max_timesteps)

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.action_head = nn.Linear(d_model, n_actions)

        def forward(
            self,
            returns: torch.Tensor,      # (B, T) return-to-go
            states: torch.Tensor,       # (B, T, ...) frames or embeddings
            actions: torch.Tensor,      # (B, T) action indices
            timesteps: torch.Tensor | None = None,  # (B, T)
        ) -> torch.Tensor:
            B, T = returns.shape
            if timesteps is None:
                timesteps = torch.arange(T, device=returns.device).unsqueeze(0).expand(B, -1)

            # embed each modality
            r = self.ret_proj(returns.unsqueeze(-1))          # (B, T, d)
            if states.dim() > 2:
                s = self.state_encoder(states)                # (B, T, d)
            else:
                s = states
            s = self.state_proj(s)
            a = self.action_emb(actions)                       # (B, T, d)

            t_emb = self.timestep_emb(timesteps)               # (B, T, d)
            r = r + t_emb
            s = s + t_emb
            a = a + t_emb

            # interleave tokens: [R, s, a] per timestep → (B, 3T, d)
            tokens = torch.stack([r, s, a], dim=2).reshape(B, T * 3, self.d_model)

            # causal mask over the 3T sequence
            mask = torch.triu(
                torch.full((T * 3, T * 3), float("-inf"), device=returns.device),
                diagonal=1,
            )
            out = self.transformer(tokens, mask=mask)

            # read out the action positions (every 3rd token starting at 2)
            a_out = out[:, 2::3, :]                          # (B, T, d)
            return self.action_head(a_out)                   # (B, T, n_actions)

        def act(self, returns, states, actions, timesteps=None):
            """Predict the next action for the last timestep."""
            logits = self.forward(returns, states, actions, timesteps)
            return logits[:, -1, :].argmax(dim=-1)


@dataclass
class DTConfig:
    n_actions: int = 8
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    max_timesteps: int = 512
    dropout: float = 0.1

    def build(self, state_encoder=None):
        return DecisionTransformer(
            n_actions=self.n_actions,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            max_timesteps=self.max_timesteps,
            state_encoder=state_encoder,
            dropout=self.dropout,
        )


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
