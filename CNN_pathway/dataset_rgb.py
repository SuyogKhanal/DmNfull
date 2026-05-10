"""Demo dataset for the RGBCNNPolicy training loop.

Walks every JSON demo under ``demo_dir`` and yields one
``(rgb_input, optimal_mask)`` sample per recorded transition. The
optimal-action mask is recomputed via BFS at __getitem__ time so
multi-modal targets (two equally good actions in the same state)
remain jointly supervised.

The demo schema is the SAME schema both Equivariant_pathway and
CNN_pathway demos use — trajectory + start_pos + goal_pos +
fire_positions + actions. We re-derive the per-step RGB observation
from those fields via CNN_pathway/encoder_rgb.py so we don't have to
trust the saved 'images' field (which can be missing on demos
collected by the equivariant collector).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from CNN_pathway.encoder_rgb import encode_state, DEFAULT_CELL_PX
from CNN_pathway.expert import (
    NUM_ACTIONS,
    build_grid,
    compute_distance_map,
    optimal_action_mask,
)


def _infer_grid_size(demo: Dict) -> int:
    traj = demo.get("trajectory", []) or []
    fires = demo.get("fire_positions", []) or []
    goal = demo.get("goal_pos", [0, 0]) or [0, 0]
    start = demo.get("start_pos", [0, 0]) or [0, 0]
    max_idx = 0
    for r, c in list(traj) + list(fires) + [list(goal), list(start)]:
        max_idx = max(max_idx, int(r), int(c))
    return max_idx + 1


class RGBDemoDataset(Dataset):
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
                print(f"[rgb-dataset] skipping unreadable demo {path}")
                continue
            if not include_failed and not demo.get("success", True):
                continue
            self._index_demo(demo)
        if not self.samples:
            raise ValueError(
                f"No usable demos under {demo_dir} (pattern={glob_pattern})."
            )
        print(f"[rgb-dataset] Loaded {self.num_demo_files} demos -> "
              f"{len(self.samples)} (rgb, mask) samples")

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
            agent = tuple(traj[t])
            self.samples.append({
                "key":         key,
                "agent_pos":   agent,
                "goal_pos":    tuple(goal),
                "fires":       tuple(map(tuple, fires)),
                "grid_size":   gs,
                "layout_id":   demo.get("layout_id"),
                "maze_name":   demo.get("maze_name"),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        grid = self._grid_cache[s["key"]]
        dist = self._dist_cache[s["key"]]
        x = encode_state(
            grid=grid,
            agent_pos=s["agent_pos"],
            goal_pos=s["goal_pos"],
            fire_positions=s["fires"],
            cell_px=self.cell_px,
        )
        mask = optimal_action_mask(dist, s["agent_pos"], grid,
                                   num_actions=self.num_actions)
        return {
            "input":     torch.from_numpy(np.ascontiguousarray(x)).float(),
            "mask":      torch.from_numpy(np.ascontiguousarray(mask)).float(),
            "agent_pos": torch.tensor(s["agent_pos"], dtype=torch.long),
            "img_size":  int(x.shape[-1]),
        }


def collate_fixed_size(batch: List[Dict]) -> Dict[str, Any]:
    img_sizes = [b["img_size"] for b in batch]
    if len(set(img_sizes)) == 1:
        return {
            "input":     torch.stack([b["input"]     for b in batch], dim=0),
            "mask":      torch.stack([b["mask"]      for b in batch], dim=0),
            "agent_pos": torch.stack([b["agent_pos"] for b in batch], dim=0),
            "img_size":  img_sizes[0],
        }
    groups: Dict[int, List[Dict]] = {}
    for b in batch:
        groups.setdefault(b["img_size"], []).append(b)
    return {
        "groups": [
            {
                "input":     torch.stack([b["input"]     for b in g], dim=0),
                "mask":      torch.stack([b["mask"]      for b in g], dim=0),
                "agent_pos": torch.stack([b["agent_pos"] for b in g], dim=0),
                "img_size":  sz,
            }
            for sz, g in groups.items()
        ]
    }
