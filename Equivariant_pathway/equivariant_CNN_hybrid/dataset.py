"""Demo dataset for the EquivariantCNNHybridPolicy.

Each sample yields BOTH a 3-channel RGB tensor (B, 3, H_px, W_px)
*and* a 5-channel semantic grid tensor (B, 5, H, W) plus the multi-label
optimal-action mask. Re-uses the equivariant pathway's BFS/A* expert
for the mask and the equivariant encoder for the grid; uses
CNN_pathway.encoder_rgb for the RGB render so the visual input matches
exactly what RGBCNNPolicy was trained on.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.encoder import NUM_CHANNELS, encode_state as encode_grid
from Equivariant_pathway.expert import (
    NUM_ACTIONS, build_grid, compute_distance_map, optimal_action_mask,
)
from CNN_pathway.encoder_rgb import encode_state as encode_rgb, DEFAULT_CELL_PX


def _infer_grid_size(demo: Dict) -> int:
    traj = demo.get("trajectory", []) or []
    fires = demo.get("fire_positions", []) or []
    goal = demo.get("goal_pos", [0, 0]) or [0, 0]
    start = demo.get("start_pos", [0, 0]) or [0, 0]
    max_idx = 0
    for r, c in list(traj) + list(fires) + [list(goal), list(start)]:
        max_idx = max(max_idx, int(r), int(c))
    return max_idx + 1


class HybridDemoDataset(Dataset):
    def __init__(
        self,
        demo_dir: str,
        num_actions: int = NUM_ACTIONS,
        include_failed: bool = False,
        glob_pattern: str = "*.json",
        max_demos: Optional[int] = None,
        recursive: bool = True,
        cell_px: int = DEFAULT_CELL_PX,
    ):
        self.demo_dir = demo_dir
        self.num_actions = num_actions
        self.cell_px = int(cell_px)
        self.samples: List[Dict[str, Any]] = []
        self._dist_cache: Dict[Tuple, np.ndarray] = {}
        self._grid_cache: Dict[Tuple, np.ndarray] = {}
        pattern = os.path.join(demo_dir, "**", glob_pattern) if recursive else os.path.join(demo_dir, glob_pattern)
        files = sorted(glob.glob(pattern, recursive=recursive))
        if max_demos is not None and max_demos > 0:
            files = files[:max_demos]
        self.num_demo_files = len(files)
        self.demo_paths = list(files)
        for path in files:
            try:
                with open(path, "r") as f:
                    demo = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not include_failed and not demo.get("success", True):
                continue
            self._index_demo(demo)
        if not self.samples:
            raise ValueError(f"No usable demos under {demo_dir}")
        print(f"[hybrid-dataset] {self.num_demo_files} demos -> {len(self.samples)} samples")

    def _index_demo(self, demo: Dict) -> None:
        gs = _infer_grid_size(demo)
        fires = demo.get("fire_positions", []) or []
        goal = demo.get("goal_pos") or demo.get("trajectory", [[0, 0]])[-1]
        key = (gs, tuple(map(tuple, fires)), tuple(goal))
        if key not in self._grid_cache:
            grid = build_grid(gs, fires, goal, wall_template=None)
            self._grid_cache[key] = grid
            self._dist_cache[key] = compute_distance_map(grid, tuple(goal))
        traj = demo.get("trajectory", []) or []
        for t in range(len(traj) - 1):
            self.samples.append({
                "key": key, "agent_pos": tuple(traj[t]),
                "goal_pos": tuple(goal),
                "fires": tuple(map(tuple, fires)),
                "grid_size": gs, "layout_id": demo.get("layout_id"),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        grid = self._grid_cache[s["key"]]
        dist = self._dist_cache[s["key"]]
        x_grid = encode_grid(grid=grid, agent_pos=s["agent_pos"],
                             goal_pos=s["goal_pos"],
                             fire_positions=s["fires"],
                             num_classes=NUM_CHANNELS)
        x_rgb = encode_rgb(grid=grid, agent_pos=s["agent_pos"],
                           goal_pos=s["goal_pos"],
                           fire_positions=s["fires"],
                           cell_px=self.cell_px)
        mask = optimal_action_mask(dist, s["agent_pos"], grid,
                                   num_actions=self.num_actions)
        return {
            "rgb":       torch.from_numpy(np.ascontiguousarray(x_rgb)).float(),
            "grid":      torch.from_numpy(np.ascontiguousarray(x_grid)).float(),
            "mask":      torch.from_numpy(np.ascontiguousarray(mask)).float(),
            "agent_pos": torch.tensor(s["agent_pos"], dtype=torch.long),
            "grid_size": int(x_grid.shape[-1]),
        }


def collate_fixed_size(batch: List[Dict]) -> Dict[str, Any]:
    sizes = [b["grid_size"] for b in batch]
    if len(set(sizes)) == 1:
        return {
            "rgb":       torch.stack([b["rgb"]       for b in batch], dim=0),
            "grid":      torch.stack([b["grid"]      for b in batch], dim=0),
            "mask":      torch.stack([b["mask"]      for b in batch], dim=0),
            "agent_pos": torch.stack([b["agent_pos"] for b in batch], dim=0),
            "grid_size": sizes[0],
        }
    groups: Dict[int, List[Dict]] = {}
    for b in batch:
        groups.setdefault(b["grid_size"], []).append(b)
    return {
        "groups": [
            {
                "rgb":       torch.stack([b["rgb"]       for b in g], dim=0),
                "grid":      torch.stack([b["grid"]      for b in g], dim=0),
                "mask":      torch.stack([b["mask"]      for b in g], dim=0),
                "agent_pos": torch.stack([b["agent_pos"] for b in g], dim=0),
                "grid_size": gs,
            }
            for gs, g in groups.items()
        ]
    }
