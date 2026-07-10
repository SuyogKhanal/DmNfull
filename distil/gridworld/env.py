"""GridWorld env helpers for the DISTIL loop: build/configure a maze from a sampled
layout, read the geometric scene tuple, BFS shortest path (for expert demos +
takeover), and a standalone bird-eye rasterizer for the VLM frames (start / t_flag /
end) at arbitrary agent cells.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .expert import INF, build_grid, compute_distance_map
from .maze_env import (ACTION_DELTAS, AGENT_COLOR, RENDER_IMG_SIZE, TILE_COLORS,
                       TILE_FIRE, TILE_GOAL, MazeNavEnv)


def make_maze(layout: Optional[Dict] = None) -> MazeNavEnv:
    """A MazeNavEnv forced to a sampled layout (no randomization, no render window)."""
    env = MazeNavEnv(render_mode=None)
    if layout is not None:
        env.configure_layout(layout["grid"], layout["start"], layout["goal"], layout["fires"])
    return env


def reset_to(env: MazeNavEnv, layout: Dict):
    env.configure_layout(layout["grid"], layout["start"], layout["goal"], layout["fires"])
    return env.reset()


def scene_of(env: MazeNavEnv) -> Tuple:
    """(grid_list, agent_cell, goal_cell, fires) — the input the classifier consumes."""
    return (env.grid.tolist(), tuple(env.agent_pos), tuple(env.goal_pos),
            [tuple(f) for f in env.fire_positions])


def bfs_path(grid, start, goal) -> List[Tuple[int, int]]:
    """Greedy shortest path (descend the BFS distance map). [] if unreachable."""
    grid = np.asarray(grid, dtype=np.int32)
    dmap = compute_distance_map(grid, tuple(goal))
    cur = tuple(int(v) for v in start)
    if dmap[cur] >= INF:
        return []
    path = [cur]
    guard = 0
    while cur != tuple(goal) and guard < grid.size * 4:
        guard += 1
        nxt = None
        for dr, dc in ACTION_DELTAS.values():
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and dmap[nr, nc] == dmap[cur] - 1:
                nxt = (nr, nc); break
        if nxt is None:
            break
        cur = nxt; path.append(cur)
    return path


def bird_eye(grid, agent, goal, fires, size: int = RENDER_IMG_SIZE) -> np.ndarray:
    """Standalone (H,W,3) uint8 bird-eye raster with the agent at `agent` (no env needed)."""
    grid = np.asarray(grid, dtype=np.int32).copy()
    gs = grid.shape[0]
    for r, c in fires:
        grid[int(r), int(c)] = TILE_FIRE
    grid[int(goal[0]), int(goal[1])] = TILE_GOAL
    cell_px = size // gs
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(gs):
        for c in range(gs):
            if (r, c) == tuple(agent):
                color = AGENT_COLOR
            elif grid[r, c] == TILE_GOAL:
                color = (50, 200, 80)
            elif grid[r, c] == TILE_FIRE:
                color = (220, 60, 20)
            else:
                color = TILE_COLORS.get(int(grid[r, c]), (100, 100, 100))
            img[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px] = color
    return img


def render_frames(layout: Dict, agent_cells: Dict[str, Tuple], out_dir: str, ep_id: int) -> Dict:
    """Save start/high_loss/end bird-eye PNGs for the VLM. Returns {role: path}."""
    from PIL import Image
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for role, cell in agent_cells.items():
        img = bird_eye(layout["grid"], cell, layout["goal"], layout["fires"])
        p = str(Path(out_dir) / f"ep{ep_id}_{role}.png")
        Image.fromarray(img).save(p)
        paths[role] = p
    return paths
