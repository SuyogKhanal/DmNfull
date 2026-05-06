"""Equivariant policy training dataset.

Walks every JSON demo in `demo_dir` and yields one (input, optimal_mask)
sample per recorded transition. Two key design choices, mirroring dmnpol:

  1. Optimal-action mask is recomputed via BFS at __getitem__ time, NOT
     read from the recorded action. So when the human/BFS demonstrator
     happened to pick one of two equally good actions, both actions
     remain positives in the mask. This preserves the multi-modality the
     equivariant trainer needs to actually learn that "go right" and "go
     down" are jointly optimal in the open square.

  2. p4m augmentation (8-fold: 4 rotations x 2 reflections) is OFF by
     default — the model is built-in equivariant so spatial augmentation
     is redundant. The flag is exposed so non-equivariant ablations can
     reuse the same dataset.

Demo schema (matches both play_maze.save_demo and the BFS expert):
    trajectory     : list of [r, c] cells, length T+1
    goal_pos       : [r, c]
    fire_positions : list of [r, c]
    start_pos      : [r, c] (optional; first trajectory entry used otherwise)
    success        : bool (optional; demos that didn't reach the goal are
                     skipped by default)
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from Equivariant_pathway.encoder import NUM_CHANNELS, encode_state
from Equivariant_pathway.expert import (
    NUM_ACTIONS,
    build_grid,
    compute_distance_map,
    optimal_action_mask,
)


def _infer_grid_size(demo: Dict) -> int:
    """Best-effort grid-size inference from any of the spatial fields."""
    traj = demo.get("trajectory", []) or []
    fires = demo.get("fire_positions", []) or []
    goal = demo.get("goal_pos", [0, 0]) or [0, 0]
    start = demo.get("start_pos", [0, 0]) or [0, 0]
    max_idx = 0
    for r, c in list(traj) + list(fires) + [list(goal), list(start)]:
        max_idx = max(max_idx, int(r), int(c))
    return max_idx + 1


class EquivariantDemoDataset(Dataset):
    """In-memory (input, optimal_mask) dataset built from JSON demos."""

    def __init__(
        self,
        demo_dir: str,
        num_actions: int = NUM_ACTIONS,
        include_failed: bool = False,
        glob_pattern: str = "*.json",
        max_demos: Optional[int] = None,
        recursive: bool = True,
    ):
        self.demo_dir = demo_dir
        self.num_actions = num_actions

        self.samples: List[Dict[str, Any]] = []
        self._dist_cache: Dict[Tuple, np.ndarray] = {}
        self._grid_cache: Dict[Tuple, np.ndarray] = {}

        if recursive:
            pattern = os.path.join(demo_dir, "**", glob_pattern)
            files = sorted(glob.glob(pattern, recursive=True))
        else:
            pattern = os.path.join(demo_dir, glob_pattern)
            files = sorted(glob.glob(pattern))

        if max_demos is not None and max_demos > 0:
            files = files[:max_demos]
        self.num_demo_files = len(files)
        self.demo_paths: List[str] = list(files)

        for path in files:
            try:
                with open(path, "r") as f:
                    demo = json.load(f)
            except (json.JSONDecodeError, OSError):
                print(f"[eq-dataset] skipping unreadable demo {path}")
                continue
            if not include_failed and not demo.get("success", True):
                # play_maze marks success=true when the goal was reached;
                # BFS expert demos always end at the goal, so this just
                # filters demos truncated by fire-hits.
                continue
            self._index_demo(demo)

        if not self.samples:
            raise ValueError(
                f"No usable demos under {demo_dir} (pattern={glob_pattern}, "
                f"recursive={recursive}). Collect demos with the BFS expert "
                f"or play_maze before training."
            )

        per_layout: Dict[str, int] = {}
        for s in self.samples:
            per_layout[s.get("layout_id") or "unknown"] = (
                per_layout.get(s.get("layout_id") or "unknown", 0) + 1
            )
        print(f"[eq-dataset] Loaded {self.num_demo_files} demos -> "
              f"{len(self.samples)} (input, mask) samples")
        for lid, n in sorted(per_layout.items()):
            print(f"[eq-dataset]   layout_id={lid:<48s} samples={n}")

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
            num_classes=NUM_CHANNELS,
        )
        mask = optimal_action_mask(dist, s["agent_pos"], grid,
                                   num_actions=self.num_actions)
        return {
            "input":     torch.from_numpy(np.ascontiguousarray(x)).float(),
            "mask":      torch.from_numpy(np.ascontiguousarray(mask)).float(),
            "agent_pos": torch.tensor(s["agent_pos"], dtype=torch.long),
            "grid_size": int(x.shape[-1]),
        }


def collate_fixed_size(batch: List[Dict]) -> Dict[str, Any]:
    """Stacks a batch when every sample shares the same grid size (the
    common case for 5x5 training). Mixed-size batches fall back to a
    list-of-groups dict; the trainer then iterates over groups.
    """
    grid_sizes = [b["grid_size"] for b in batch]
    if len(set(grid_sizes)) == 1:
        return {
            "input":     torch.stack([b["input"]     for b in batch], dim=0),
            "mask":      torch.stack([b["mask"]      for b in batch], dim=0),
            "agent_pos": torch.stack([b["agent_pos"] for b in batch], dim=0),
            "grid_size": grid_sizes[0],
        }
    groups: Dict[int, List[Dict]] = {}
    for b in batch:
        groups.setdefault(b["grid_size"], []).append(b)
    return {
        "groups": [
            {
                "input":     torch.stack([b["input"]     for b in g], dim=0),
                "mask":      torch.stack([b["mask"]      for b in g], dim=0),
                "agent_pos": torch.stack([b["agent_pos"] for b in g], dim=0),
                "grid_size": gs,
            }
            for gs, g in groups.items()
        ]
    }
