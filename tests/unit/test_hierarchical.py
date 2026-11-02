"""Tests for hierarchical subgoal chains."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nes_player.policy.hierarchical import (  # noqa: E402
    SubgoalChain,
    make_manual_chain,
    make_uniform_chain,
)


def test_chain_advances_on_reaching_threshold():
    chain = SubgoalChain(thresholds=[100, 500, 1000])
    assert chain.current_target == 100
    assert not chain.update(50)
    assert chain.current_target == 100
    assert chain.update(120)
    assert chain.current_target == 500
    assert chain.update(600)
    assert chain.current_target == 1000
    assert chain.update(1500)
    assert chain.done


def test_chain_progress():
    chain = make_uniform_chain(800, n_subgoals=4)
    assert chain.current_target == 200
    chain.update(300)
    assert chain.progress == 0.25
    chain.update(500)
    assert chain.progress == 0.5


def test_shaped_value_within_horizon():
    chain = SubgoalChain(thresholds=[100, 500])
    # Target is 100, position 0, horizon 200: target within horizon
    v = chain.shaped_value(position=0, horizon_px=200)
    assert v > 0  # positive reward for reachable target
    # Target is 100, position 50, horizon 200: closer
    v2 = chain.shaped_value(position=50, horizon_px=200)
    assert v2 > v  # closer is better


def test_shaped_value_beyond_horizon():
    chain = SubgoalChain(thresholds=[1000])
    v = chain.shaped_value(position=0, horizon_px=100)
    assert v < 0  # unreachable target: negative distance shaping


def test_manual_chain():
    chain = make_manual_chain([256, 512, 1024, 2048])
    assert chain.current_target == 256
    assert len(chain.remaining()) == 4


def test_reset():
    chain = SubgoalChain(thresholds=[100, 200])
    chain.update(150)
    chain.reset()
    assert chain.current_target == 100
    assert chain.progress == 0
