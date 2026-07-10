"""GridWorld SELECT / BRIDGE demo-collection primitives.

SELECT: re-instantiate a failed layout, take over at the t_flag agent cell, and the
BFS expert demonstrates the optimal path from there to the goal (== the robot side's
replay-to-divergence + expert takeover). BRIDGE: prescribe ONE new middle-ground start
cell between 2-3 cited failures (a corridor placement), validated BFS-solvable before
any expert effort (the infeasibility gate); the BFS expert then solves it.

A demo = {grid, start, goal, fires, cells} (the expert's path); training recomputes the
BFS optimal-action mask at each cell, so only the visited path cells are needed. Budget
counts only SUCCESSFUL demos (a non-solvable BRIDGE returns None -> re-prescribe).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .env import bfs_path
from .maze_env import TILE_FIRE
from .expert import build_grid, compute_distance_map, INF


def _demo(grid, start, goal, fires, cells) -> Dict:
    return {"grid": grid, "start": list(start), "goal": list(goal),
            "fires": [list(f) for f in fires], "cells": [list(c) for c in cells]}


def collect_select_gw(desc) -> Tuple[Optional[Dict], bool, Dict]:
    """Expert takeover from the t_flag agent cell to the goal (BFS optimal path)."""
    grid, goal, fires = desc.layout["grid"], desc.layout["goal"], desc.layout["fires"]
    cell = desc.agent_cell
    path = bfs_path(grid, cell, goal)
    if len(path) < 2 or tuple(path[-1]) != tuple(goal):
        return None, False, {"len": len(path)}
    return _demo(grid, cell, goal, fires, path), True, {"len": len(path)}


def _nearest_free(grid_arr, cell, goal, fires) -> Optional[Tuple[int, int]]:
    gs = grid_arr.shape[0]
    r = int(np.clip(cell[0], 0, gs - 1)); c = int(np.clip(cell[1], 0, gs - 1))
    fireset = {tuple(f) for f in fires}
    for rad in range(gs):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs and (nr, nc) != tuple(goal) \
                        and (nr, nc) not in fireset and grid_arr[nr, nc] != TILE_FIRE:
                    return (nr, nc)
    return None


def collect_bridge_gw(cited: List) -> Tuple[Optional[Dict], bool, Dict]:
    """Prescribe a mid-cluster start (corridor placement) on the representative's
    layout; BFS-solvability gate; expert solves it. None if infeasible (re-prescribe)."""
    if not cited:
        return None, False, {"reason": "no_cited"}
    rep = cited[0]
    grid, goal, fires = rep.layout["grid"], rep.layout["goal"], rep.layout["fires"]
    grid_arr = build_grid(len(grid), fires, goal, wall_template=np.asarray(grid, dtype=np.int32))
    mean_cell = np.round(np.mean([c.agent_cell for c in cited], axis=0)).astype(int)
    start = _nearest_free(grid_arr, mean_cell, goal, fires)
    if start is None:
        return None, False, {"reason": "no_free_cell"}
    dmap = compute_distance_map(grid_arr, tuple(goal))
    if dmap[start] >= INF or int(dmap[start]) < 2:      # unsolvable or trivially adjacent
        return None, False, {"reason": "infeasible", "start": list(start)}
    path = bfs_path(grid, start, goal)
    if len(path) < 2 or tuple(path[-1]) != tuple(goal):
        return None, False, {"reason": "no_path"}
    return _demo(grid, start, goal, fires, path), True, {"len": len(path), "start": list(start)}
