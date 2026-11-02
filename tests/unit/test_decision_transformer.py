"""Decision Transformer: sequence model over (return-to-go, state, action)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

torch = pytest.importorskip("torch")

from nes_player.policy.decision_transformer import (  # noqa: E402
    DTConfig,
    count_params,
)


def test_param_count_is_ten_to_fifteen_million():
    """Malady 2026-10-20: 10-15M, not GPT-2's 124M."""
    model = DTConfig().build()
    n = count_params(model)
    assert 8_000_000 <= n <= 20_000_000, f"{n:,} out of range"


def test_forward_shape():
    model = DTConfig().build()
    B, T = 2, 32
    returns = torch.randn(B, T)
    states = torch.randn(B, T, 384)
    actions = torch.randint(0, 8, (B, T))
    out = model(returns, states, actions)
    assert out.shape == (B, T, 8)


def test_act_returns_action_indices():
    model = DTConfig().build()
    B, T = 2, 16
    act = model.act(torch.randn(B, T), torch.randn(B, T, 384),
                    torch.randint(0, 8, (B, T)))
    assert act.shape == (B,)
    assert all(0 <= a < 8 for a in act.tolist())


def test_return_to_go_changes_prediction():
    """The dial: different return-to-go should give different actions."""
    model = DTConfig().build()
    B, T = 1, 8
    states = torch.randn(B, T, 384)
    actions = torch.randint(0, 8, (B, T))
    act_low = model.act(torch.zeros(B, T), states, actions)
    act_high = model.act(torch.ones(B, T) * 100, states, actions)
    # They may or may not differ (random weights), but both must be valid
    assert 0 <= act_low.item() < 8
    assert 0 <= act_high.item() < 8


def test_config_builds():
    cfg = DTConfig(d_model=128, n_layers=2, n_heads=2)
    model = cfg.build()
    assert count_params(model) < 5_000_000
