"""Training dataset for the CNN+MLP maze policy.

Walks the layouts declared in `training_layouts.yaml`, samples random
(start, goal, fire) configurations on each layout's wall template, plans a
fire-free shortest path with BFS, and records every (image, state, action)
transition as a training sample. The held-out test layouts are never used.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.maze_layouts import MAZE_LAYOUTS
from envs.maze_env import (
    ACTION_DELTAS,
    MazeNavEnv,
    TILE_FIRE,
    TILE_FREE,
    TILE_GOAL,
    TILE_WALL,
)


_HOST_MAZE = "multimodal"


def _bfs_plan(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[int]:
    H, W = grid.shape
    if start == goal:
        return []
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        (r, c), actions = queue.popleft()
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            tile = grid[nr, nc]
            if tile in (TILE_WALL, TILE_FIRE):
                continue
            visited.add((nr, nc))
            new_actions = actions + [action]
            if (nr, nc) == goal:
                return new_actions
            queue.append(((nr, nc), new_actions))
    return []


def _free_cells(grid_template: np.ndarray) -> List[Tuple[int, int]]:
    return [
        (r, c)
        for r in range(grid_template.shape[0])
        for c in range(grid_template.shape[1])
        if grid_template[r, c] == TILE_FREE
    ]


def _install_layout(layout_grid: List[List[int]]) -> None:
    """Temporarily overwrite MAZE_LAYOUTS[multimodal] with the supplied grid so
    MazeNavEnv can render it. Returns nothing; use _restore_layout to undo."""
    MAZE_LAYOUTS[_HOST_MAZE]["grid"] = layout_grid


def _build_env_for_layout(
    layout_grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    fires: List[Tuple[int, int]],
    seed: int,
) -> MazeNavEnv:
    _install_layout(layout_grid)
    MAZE_LAYOUTS[_HOST_MAZE]["start"] = list(start)
    MAZE_LAYOUTS[_HOST_MAZE]["goal"] = list(goal)
    env = MazeNavEnv(
        maze_name=_HOST_MAZE,
        render_mode="rgb_array",
        randomize_start=False,
        randomize_goal=False,
        randomize_fire=False,
        num_fire_tiles=len(fires),
        seed=seed,
        fire_positions=list(fires),
    )
    env.reset()
    env.grid[env.grid == TILE_GOAL] = TILE_FREE
    env.goal_pos = goal
    env.grid[goal[0], goal[1]] = TILE_GOAL
    env.agent_pos = start
    env.start_pos = start
    env.visited = np.zeros_like(env.grid, dtype=bool)
    env.visited[start] = True
    env.trajectory = [start]
    return env


@dataclass
class _Sample:
    image: np.ndarray   # (H, W, 3) uint8
    state: np.ndarray   # (state_dim,) float32
    action: int


def _sample_layout_demo(
    layout_grid: List[List[int]],
    num_fire_tiles: int,
    rng: np.random.Generator,
    seed: int,
    max_attempts: int = 25,
) -> List[_Sample]:
    grid_template = np.array(layout_grid, dtype=np.int32)
    free = _free_cells(grid_template)
    if len(free) < 2 + num_fire_tiles:
        return []

    for _ in range(max_attempts):
        idxs = rng.choice(len(free), size=2 + num_fire_tiles, replace=False)
        chosen = [free[int(i)] for i in idxs]
        start, goal, *fires = chosen
        plan = _bfs_plan(
            _grid_with_fires(grid_template, fires), tuple(start), tuple(goal),
        )
        if plan:
            break
    else:
        return []

    env = _build_env_for_layout(layout_grid, tuple(start), tuple(goal), fires, seed)
    samples: List[_Sample] = []
    obs = env._get_obs()
    for action in plan:
        samples.append(
            _Sample(
                image=obs["image"].copy(),
                state=obs["state"].astype(np.float32).copy(),
                action=int(action),
            )
        )
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    env.close()
    return samples


def _grid_with_fires(template: np.ndarray, fires: List[Tuple[int, int]]) -> np.ndarray:
    g = template.copy()
    for r, c in fires:
        g[r, c] = TILE_FIRE
    return g


def load_training_layouts(yaml_path: str) -> Dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f) or {}


class CNNMLPDemoDataset(Dataset):
    """Collects demos in-memory at construction time."""

    def __init__(
        self,
        layouts_yaml: str,
        seed: int = 0,
        augment: bool = True,
    ):
        cfg = load_training_layouts(layouts_yaml)
        self.layouts_cfg = cfg
        layouts = cfg.get("training_layouts") or []
        if not layouts:
            raise ValueError(f"No training_layouts found in {layouts_yaml}")

        rng = np.random.default_rng(seed)
        self.augment = augment
        self.samples: List[_Sample] = []

        per_layout_counts: List[Tuple[str, int]] = []
        for layout in layouts:
            name = layout.get("name", "<unnamed>")
            n_demos = int(layout.get("n_demos", 50))
            num_fires = int(layout.get("num_fire_tiles", 2))
            grid = layout["grid"]

            count_before = len(self.samples)
            for d in range(n_demos):
                demo_seed = int(rng.integers(0, 2**31 - 1))
                demo_samples = _sample_layout_demo(
                    layout_grid=grid,
                    num_fire_tiles=num_fires,
                    rng=rng,
                    seed=demo_seed,
                )
                self.samples.extend(demo_samples)
            per_layout_counts.append((name, len(self.samples) - count_before))

        print(f"[cnn-dataset] Collected {len(self.samples)} (img, state, action) samples")
        for name, n in per_layout_counts:
            print(f"[cnn-dataset]   layout={name:<22s} samples={n}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = s.image.astype(np.float32) / 255.0
        if self.augment:
            brightness = np.random.uniform(0.85, 1.15)
            contrast = np.random.uniform(0.90, 1.10)
            img = np.clip(img * brightness * contrast, 0.0, 1.0)
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(s.state).float(),
            torch.tensor(s.action, dtype=torch.long),
        )
