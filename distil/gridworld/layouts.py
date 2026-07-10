"""Deterministic solvable-layout sampler for GridWorld DISTIL.

The paper's GridWorld is the general randomized 5x5 task (3 obstacles/fires, varied
start+goal), NOT the single fixed 'multimodal' maze. Screening pools, the shared
bootstrap, and the frozen held-out set are all draws from this sampler. Every layout
is reproducible from its integer seed (so a SELECT re-roll reconstructs the exact
scene), BFS-solvable, with Manhattan(start,goal) >= min_manhattan and exactly
`n_fire` fire cells disjoint from start/goal (mirrors the repo's layout_sampler
constraints).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from .maze_env import ACTION_DELTAS, TILE_FIRE, TILE_FREE

GRID_SIZE = 5


def _bfs_reachable(grid: np.ndarray, start, goal) -> bool:
    """4-connected BFS over free cells (fires impassable); True if goal reachable."""
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    q = deque([tuple(start)])
    seen[start[0], start[1]] = True
    while q:
        r, c = q.popleft()
        if (r, c) == tuple(goal):
            return True
        for dr, dc in ACTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and grid[nr, nc] != TILE_FIRE:
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def sample_layout(seed: int, grid_size: int = GRID_SIZE, n_fire: int = 3,
                  min_manhattan: int = 4) -> Dict:
    """Return {grid, start, goal, fires} — BFS-solvable, reproducible from `seed`."""
    rng = np.random.default_rng(int(seed))
    for _ in range(200):
        cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        si = rng.integers(len(cells)); start = cells[si]
        gi = rng.integers(len(cells)); goal = cells[gi]
        if start == goal or abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < min_manhattan:
            continue
        free = [p for p in cells if p != start and p != goal]
        idx = rng.choice(len(free), size=min(n_fire, len(free)), replace=False)
        fires = [free[i] for i in idx]
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        for r, c in fires:
            grid[r, c] = TILE_FIRE
        if _bfs_reachable(grid, start, goal):
            return {"grid": grid.tolist(), "start": list(start), "goal": list(goal),
                    "fires": [list(f) for f in fires], "seed": int(seed)}
    # fallback: a guaranteed-solvable corner-to-corner layout with no fires
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    return {"grid": grid.tolist(), "start": [0, 0], "goal": [grid_size - 1, grid_size - 1],
            "fires": [], "seed": int(seed)}


def sample_layouts(seed_base: int, n: int, **kw) -> List[Dict]:
    return [sample_layout(seed_base + i, **kw) for i in range(n)]
