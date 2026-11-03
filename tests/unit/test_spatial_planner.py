"""Tests for spatial path planning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nes_player.policy.spatial_planner import SpatialPlanner  # noqa: E402


def make_test_frame(walls: list[tuple[int, int]], targets: list[tuple[int, int]],
                    agent: tuple[int, int], size: int = 28, cell: int = 8):
    """Create a synthetic frame with walls, targets, and an agent."""
    frame = np.zeros((size * cell, size * cell, 3), dtype=np.uint8)
    frame[:] = 128  # mid-gray = path

    for gy, gx in walls:
        y0, y1 = gy * cell, (gy + 1) * cell
        x0, x1 = gx * cell, (gx + 1) * cell
        frame[y0:y1, x0:x1] = 0  # dark = wall

    for gy, gx in targets:
        y0, y1 = gy * cell, (gy + 1) * cell
        x0, x1 = gx * cell, (gx + 1) * cell
        frame[y0:y1, x0:x1] = 255  # bright = target

    agy, agx = agent
    y0, y1 = agy * cell, (agy + 1) * cell
    x0, x1 = agx * cell, (agx + 1) * cell
    frame[y0:y1, x0:x1] = 200  # bright-ish = agent

    return frame


def test_build_map():
    planner = SpatialPlanner(cell_size=8)
    # Simple corridor: walls on sides, path in middle
    walls = [(y, 0) for y in range(28)] + [(y, 27) for y in range(28)]
    targets = [(14, 14)]
    frame = make_test_frame(walls, targets, (14, 5))
    wall_map, target_map, _ = planner.build_map(frame)
    assert wall_map[14, 0]  # wall
    assert not wall_map[14, 14]  # path
    assert target_map[14, 14]  # target


def test_bfs_path_straight():
    planner = SpatialPlanner(cell_size=8)
    # Open map
    wall_map = np.zeros((28, 28), dtype=bool)
    path = planner.bfs_path(wall_map, (14, 5), [(14, 15)])
    assert len(path) == 11  # 10 steps + start
    assert path[0] == (14, 5)
    assert path[-1] == (14, 15)


def test_bfs_path_around_wall():
    planner = SpatialPlanner(cell_size=8)
    wall_map = np.zeros((5, 5), dtype=bool)
    # Wall blocking the direct path
    wall_map[2, 1] = True
    wall_map[2, 2] = True
    wall_map[2, 3] = True
    path = planner.bfs_path(wall_map, (2, 0), [(2, 4)])
    assert len(path) > 5  # longer than direct
    assert path[0] == (2, 0)
    assert path[-1] == (2, 4)
    # Verify path avoids walls
    for y, x in path:
        assert not wall_map[y, x]


def test_path_to_buttons():
    planner = SpatialPlanner(cell_size=8)
    path = [(0, 0), (0, 1), (0, 2), (1, 2)]
    actions = planner.path_to_buttons(path, cell_frames=2)
    assert len(actions) == 6  # 3 steps * 2 frames each
    assert frozenset({"RIGHT"}) in actions[:2]
    assert frozenset({"DOWN"}) in actions[-2:]


def test_plan_simple():
    planner = SpatialPlanner(cell_size=8, wall_threshold=40, target_threshold=180)
    walls = [(y, 0) for y in range(28)] + [(y, 27) for y in range(28)]
    targets = [(14, 14)]
    frame = make_test_frame(walls, targets, (14, 5))
    actions, info = planner.plan(frame)
    assert info["path_length"] > 0
    assert len(info["targets"]) > 0
    assert info["agent_pos"] is not None


def test_no_targets():
    planner = SpatialPlanner(cell_size=8)
    wall_map = np.zeros((5, 5), dtype=bool)
    target_map = np.zeros((5, 5), dtype=bool)
    targets = planner.find_targets(target_map)
    assert targets == []
    path = planner.bfs_path(wall_map, (2, 2), targets)
    assert path == []
