"""BFS A* expert + multi-modal optimal action mask (CNN_pathway clone).

Verbatim copy of Equivariant_pathway/expert.py — the expert is purely
about the grid + goal + fire layout, so it is policy-agnostic. Cloned
here so CNN_pathway has zero cross-deps on Equivariant_pathway.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np

from envs.maze_env import (
    ACTION_DELTAS,
    TILE_FIRE,
    TILE_GOAL,
    TILE_WALL,
)

NUM_ACTIONS = 4
INF = 10**9


def build_grid(grid_size, fire_positions, goal_pos, wall_template=None):
    if wall_template is not None:
        grid = np.array(wall_template, dtype=np.int32).copy()
    else:
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    for r, c in fire_positions:
        grid[int(r), int(c)] = TILE_FIRE
    gr, gc = goal_pos
    grid[int(gr), int(gc)] = TILE_GOAL
    return grid


def compute_distance_map(grid, goal_pos):
    H, W = grid.shape
    dist = np.full((H, W), INF, dtype=np.int64)
    gr, gc = int(goal_pos[0]), int(goal_pos[1])
    if not (0 <= gr < H and 0 <= gc < W):
        return dist
    dist[gr, gc] = 0
    q = deque([(gr, gc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr, nc] == TILE_WALL or grid[nr, nc] == TILE_FIRE:
                continue
            if dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def optimal_action_mask(dist_map, agent_pos, grid, num_actions=NUM_ACTIONS):
    H, W = grid.shape
    mask = np.zeros(num_actions, dtype=np.float32)
    ar, ac = int(agent_pos[0]), int(agent_pos[1])
    if dist_map[ar, ac] >= INF:
        return mask
    here = dist_map[ar, ac]
    if here == 0:
        return mask
    for a, (dr, dc) in ACTION_DELTAS.items():
        if a >= num_actions:
            continue
        nr, nc = ar + dr, ac + dc
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if grid[nr, nc] == TILE_WALL or grid[nr, nc] == TILE_FIRE:
            continue
        if dist_map[nr, nc] == here - 1:
            mask[a] = 1.0
    return mask


class AStarExpert:
    def __init__(self, grid, goal_pos, num_actions=NUM_ACTIONS):
        self.grid = grid
        self.goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        self.num_actions = num_actions
        self.dist_map = compute_distance_map(grid, self.goal_pos)

    def optimal_actions(self, agent_pos):
        return optimal_action_mask(self.dist_map, agent_pos, self.grid, self.num_actions)

    def sample_action(self, agent_pos, rng):
        mask = self.optimal_actions(agent_pos)
        if mask.sum() == 0:
            return None
        idx = np.flatnonzero(mask)
        return int(rng.choice(idx))

    def shortest_path_length(self, start):
        d = int(self.dist_map[int(start[0]), int(start[1])])
        return d if d < INF else -1
