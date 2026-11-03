"""Spatial path planning for maze/route games (Pac-Man class).

The manual for these games describes rules ("eat dots", "avoid ghosts"),
not button sequences. Templates don't help: the problem is spatial
(where to go), not temporal (what to press).

This module builds a grid map from the screen and plans routes with BFS:
  1. Map: walls (dark) vs paths (bright) from frame pixels.
  2. Position: agent sprite location on the grid.
  3. Targets: dots/food items visible on screen.
  4. BFS: shortest path from agent to nearest target, avoiding walls.
  5. Output: a sequence of direction buttons (LEFT/RIGHT/UP/DOWN).

    from nes_player.policy.spatial_planner import SpatialPlanner
    planner = SpatialPlanner(cell_size=8)
    plan = planner.plan(frame_rgb)  # list of button sets
"""

from __future__ import annotations

from collections import deque

import numpy as np

# NES button bitmasks (for returning actions compatible with the policy)
BUTTONS = {"A": 1, "B": 2, "SELECT": 4, "START": 8,
           "UP": 16, "DOWN": 32, "LEFT": 64, "RIGHT": 128}


class SpatialPlanner:
    """Grid-based path planner for maze games."""

    def __init__(self, cell_size: int = 8, wall_threshold: float = 40.0,
                 target_threshold: float = 180.0):
        self.cell_size = cell_size
        self.wall_threshold = wall_threshold
        self.target_threshold = target_threshold

    def build_map(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build grid maps from frame pixels.

        Returns (wall_map, target_map, agent_map):
          wall_map:   True where walls (dark pixels)
          target_map: True where targets (bright dots/items)
          agent_map:  True where agent sprite (bright, not wall, not target)
        """
        h, w = frame_rgb.shape[:2]
        gh, gw = h // self.cell_size, w // self.cell_size
        wall_map = np.zeros((gh, gw), dtype=bool)
        target_map = np.zeros((gh, gw), dtype=bool)

        for gy in range(gh):
            for gx in range(gw):
                cell = frame_rgb[
                    gy * self.cell_size:(gy + 1) * self.cell_size,
                    gx * self.cell_size:(gx + 1) * self.cell_size,
                ]
                brightness = float(cell.mean())
                # Bright pixel fraction (targets are small bright dots)
                bright_frac = float((cell.mean(axis=-1) > self.target_threshold).mean())
                dark_frac = float((cell.mean(axis=-1) < self.wall_threshold).mean())

                if dark_frac > 0.6:
                    wall_map[gy, gx] = True
                elif bright_frac > 0.1:
                    target_map[gy, gx] = True

        # Agent: bright cells that are not walls and not targets
        # (simplified: use the center region of the screen)
        agent_map = ~(wall_map | target_map)
        return wall_map, target_map, agent_map

    def find_agent(self, wall_map: np.ndarray, target_map: np.ndarray,
                   prev_pos: tuple[int, int] | None = None) -> tuple[int, int]:
        """Find the agent's position on the grid.

        Uses the brightest non-wall cell near the center (or prev_pos).
        """
        gh, gw = wall_map.shape
        candidates = np.argwhere(~wall_map)

        if prev_pos is not None:
            # Find the closest non-wall cell to the previous position
            distances = np.sum((candidates - np.array(prev_pos)) ** 2, axis=1)
            idx = np.argmin(distances)
            return tuple(candidates[idx])

        # Default: center of the playable area
        center = np.array([gh // 2, gw // 2])
        distances = np.sum((candidates - center) ** 2, axis=1)
        idx = np.argmin(distances)
        return tuple(candidates[idx])

    def find_targets(self, target_map: np.ndarray,
                     max_targets: int = 5) -> list[tuple[int, int]]:
        """Find target positions (bright dots/food items)."""
        targets = np.argwhere(target_map)
        if len(targets) == 0:
            return []
        # Limit to nearest targets
        center = np.array(target_map.shape) // 2
        distances = np.sum((targets - center) ** 2, axis=1)
        sorted_idx = np.argsort(distances)
        return [tuple(targets[i]) for i in sorted_idx[:max_targets]]

    def bfs_path(self, wall_map: np.ndarray,
                 start: tuple[int, int],
                 goals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """BFS from start to the nearest goal, avoiding walls.

        Returns the path as a list of (gy, gx) positions.
        """
        if not goals:
            return []
        goal_set = set(goals)
        gh, gw = wall_map.shape

        queue = deque([(start, [start])])
        visited = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            (y, x), path = queue.popleft()
            if (y, x) in goal_set:
                return path

            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < gh and 0 <= nx < gw:
                    if not wall_map[ny, nx] and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append(((ny, nx), path + [(ny, nx)]))

        return []  # no path found

    def path_to_buttons(self, path: list[tuple[int, int]],
                        cell_frames: int = 4) -> list[frozenset[str]]:
        """Convert a grid path to a sequence of button presses.

        Each grid step becomes `cell_frames` frames of holding a direction.
        """
        if len(path) < 2:
            return []

        actions = []
        for i in range(1, len(path)):
            y0, x0 = path[i - 1]
            y1, x1 = path[i]
            dy, dx = y1 - y0, x1 - x0

            if dy < 0:
                btn = "UP"
            elif dy > 0:
                btn = "DOWN"
            elif dx < 0:
                btn = "LEFT"
            elif dx > 0:
                btn = "RIGHT"
            else:
                continue

            for _ in range(cell_frames):
                actions.append(frozenset({btn}))

        return actions

    def plan(self, frame_rgb: np.ndarray,
             prev_pos: tuple[int, int] | None = None,
             cell_frames: int = 4) -> tuple[list[frozenset[str]], dict]:
        """Plan a route from the agent to the nearest target.

        Returns (actions, info) where info has the map, path, and targets.
        """
        wall_map, target_map, _ = self.build_map(frame_rgb)
        agent_pos = self.find_agent(wall_map, target_map, prev_pos)
        targets = self.find_targets(target_map)
        path = self.bfs_path(wall_map, agent_pos, targets)
        actions = self.path_to_buttons(path, cell_frames)

        info = {
            "agent_pos": agent_pos,
            "targets": targets[:5],
            "path_length": len(path),
            "wall_map": wall_map,
            "target_map": target_map,
        }
        return actions, info
