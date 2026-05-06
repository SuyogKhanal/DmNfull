"""Semantic-channel grid encoder for the equivariant policy.

Maps a maze grid + agent + goal + fire positions into a (5, H, W) tensor.
Channels are exactly the ones the dmnpol EquivariantUNetPolicy expects:
    0: agent (one-hot at agent cell)
    1: goal  (one-hot at goal cell)
    2: fire  (binary mask over fire cells)
    3: wall  (binary mask over wall cells)
    4: free  (1 - occupied; redundant but ablatable)

Resolution-independent: H and W follow the input grid, so the same encoder
handles 5x5, 6x6, etc. without changes. The agent channel doubles as the
gather mask the equivariant model uses to pick the per-cell logit.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from envs.maze_env import TILE_FIRE, TILE_FREE, TILE_GOAL, TILE_WALL

NUM_CHANNELS = 5
CH_AGENT = 0
CH_GOAL = 1
CH_FIRE = 2
CH_WALL = 3
CH_FREE = 4


def encode_state(
    grid: np.ndarray,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    fire_positions: Sequence[Sequence[int]],
    num_classes: int = NUM_CHANNELS,
) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int32)
    H, W = grid.shape
    out = np.zeros((num_classes, H, W), dtype=np.float32)

    ar, ac = int(agent_pos[0]), int(agent_pos[1])
    gr, gc = int(goal_pos[0]), int(goal_pos[1])
    if 0 <= ar < H and 0 <= ac < W:
        out[CH_AGENT, ar, ac] = 1.0
    if 0 <= gr < H and 0 <= gc < W:
        out[CH_GOAL, gr, gc] = 1.0

    fire_mask = np.zeros((H, W), dtype=np.float32)
    for r, c in fire_positions:
        fire_mask[int(r), int(c)] = 1.0
    fire_mask = np.maximum(fire_mask, (grid == TILE_FIRE).astype(np.float32))
    out[CH_FIRE] = fire_mask

    out[CH_WALL] = (grid == TILE_WALL).astype(np.float32)

    occupied = np.maximum.reduce([out[CH_AGENT], out[CH_GOAL], out[CH_FIRE], out[CH_WALL]])
    out[CH_FREE] = 1.0 - occupied
    return out


def encode_from_env(env) -> np.ndarray:
    """Convenience wrapper: encode the current state of a MazeNavEnv instance."""
    return encode_state(
        grid=env.grid,
        agent_pos=tuple(env.agent_pos),
        goal_pos=tuple(env.goal_pos),
        fire_positions=[tuple(p) for p in env.fire_positions],
    )
