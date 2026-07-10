"""Bird's-eye-view RGB renderer for the RGBCNNPolicy stack.

Replicates ``MazeNavEnv._get_bird_eye_view`` exactly (same colours,
same cell layout, same image size) but as a pure-numpy function so
training / dataset code can render without spinning up the env.

Output is (3, H_px, W_px) float32 in [0, 1]. Default cell_px is 16,
matching the env's render of 80x80 for a 5x5 grid.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from envs.maze_env import (
    AGENT_COLOR,
    TILE_COLORS,
    TILE_FIRE,
    TILE_FREE,
    TILE_GOAL,
    TILE_WALL,
)

DEFAULT_CELL_PX = 16
DEFAULT_IMG_SIZE = 80


def _color_for(grid: np.ndarray, agent_pos: Tuple[int, int],
               r: int, c: int) -> Tuple[int, int, int]:
    if (r, c) == tuple(agent_pos):
        return AGENT_COLOR
    tile = int(grid[r, c])
    if tile == TILE_GOAL:
        return (50, 200, 80)
    if tile == TILE_FIRE:
        return (220, 60, 20)
    return TILE_COLORS.get(tile, (100, 100, 100))


def render_rgb(
    grid: np.ndarray,
    agent_pos: Tuple[int, int],
    cell_px: int = DEFAULT_CELL_PX,
    img_size: int = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """Return an (img_size, img_size, 3) uint8 array. cell_px is computed
    as img_size // grid_size if the caller did not pin one explicitly.
    """
    grid = np.asarray(grid, dtype=np.int32)
    H, W = grid.shape
    cell_px = int(img_size // H) if cell_px is None else int(cell_px)
    img = np.zeros((H * cell_px, W * cell_px, 3), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            y0, x0 = r * cell_px, c * cell_px
            y1, x1 = y0 + cell_px, x0 + cell_px
            img[y0:y1, x0:x1] = _color_for(grid, agent_pos, r, c)
    return img


def encode_state(
    grid: np.ndarray,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    fire_positions: Sequence[Sequence[int]],
    cell_px: int = DEFAULT_CELL_PX,
) -> np.ndarray:
    """Public entry-point used by the dataset and rollout code. Returns
    a (3, H_px, W_px) float32 tensor in [0, 1] so it plugs straight into
    ``torch.from_numpy(x).float()``.

    The signature matches Equivariant_pathway/encoder.encode_state on
    purpose — every caller (dataset, rollout, collect_demos) can import
    this module instead of the equivariant one without changing arg
    names.
    """
    g = np.asarray(grid, dtype=np.int32).copy()
    # Make sure goal/fire actually appear on the grid we render; the
    # equivariant pipeline keeps them out-of-band, but the RGB renderer
    # picks them up from the grid tiles.
    fires = [tuple(int(x) for x in p) for p in (fire_positions or [])]
    for r, c in fires:
        g[r, c] = TILE_FIRE
    gr, gc = int(goal_pos[0]), int(goal_pos[1])
    g[gr, gc] = TILE_GOAL
    img = render_rgb(g, agent_pos, cell_px=cell_px)
    img = img.astype(np.float32) / 255.0
    # (H, W, 3) -> (3, H, W)
    return np.transpose(img, (2, 0, 1))
