"""Expand training_layouts.yaml into a flat play_maze-compatible JSON.

For every base layout in the YAML we:
  * sample n_demos concrete (start, goal, 3-fire) configurations
  * validate each with BFS so it's actually solvable
  * write each as one entry in `layouts: [...]`, including the wall `grid` and
    a `layout_name` tag that CNN_pathway/train.py uses to group demos.

Then the user runs:

    python scripts/play_maze.py \
        --layouts-from CNN_pathway/training_layouts_play.json \
        --demo_dir CNN_pathway/demos

play_maze auto-advances after each save, so all configurations are recorded
in one sitting. The `grid` field per layout is honoured by play_maze (it
installs the wall template into MAZE_LAYOUTS["multimodal"] before each env).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.maze_env import ACTION_DELTAS, TILE_FIRE, TILE_FREE, TILE_WALL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "training_layouts.yaml"))
    p.add_argument("--out", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "training_layouts_play.json"))
    p.add_argument("--demo_dir", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "demos"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _bfs_reachable(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    H, W = grid.shape
    if start == goal:
        return False
    queue = deque([start])
    visited = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            if grid[nr, nc] in (TILE_WALL, TILE_FIRE):
                continue
            if (nr, nc) == goal:
                return True
            visited.add((nr, nc))
            queue.append((nr, nc))
    return False


def _sample_config(
    template: np.ndarray,
    num_fires: int,
    rng: np.random.Generator,
    min_manhattan: int,
    max_attempts: int = 200,
) -> Dict:
    free = [(r, c) for r in range(template.shape[0])
                   for c in range(template.shape[1])
                   if template[r, c] == TILE_FREE]
    if len(free) < 2 + num_fires:
        raise ValueError("not enough free cells for the requested config")

    for _ in range(max_attempts):
        idxs = rng.choice(len(free), size=2 + num_fires, replace=False)
        chosen = [free[int(i)] for i in idxs]
        start, goal, *fires = chosen
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < min_manhattan:
            continue
        grid = template.copy()
        for r, c in fires:
            grid[r, c] = TILE_FIRE
        if _bfs_reachable(grid, tuple(start), tuple(goal)):
            return {
                "start_pos": [int(start[0]), int(start[1])],
                "goal_pos":  [int(goal[0]),  int(goal[1])],
                "fire_positions": [[int(r), int(c)] for r, c in fires],
            }
    raise RuntimeError(
        f"could not sample a solvable config in {max_attempts} attempts "
        f"(template too constrained or min_manhattan too large)."
    )


def main():
    args = parse_args()
    with open(args.layouts, "r") as f:
        spec = yaml.safe_load(f) or {}

    base_layouts = spec.get("training_layouts") or []
    if not base_layouts:
        raise SystemExit(f"no training_layouts in {args.layouts}")
    num_fires = int(spec.get("num_fire_tiles", 3))
    min_manhattan = int(spec.get("min_manhattan_dist", 4))

    rng = np.random.default_rng(args.seed)
    expanded: List[Dict] = []
    for layout in base_layouts:
        name = layout["name"]
        template = np.array(layout["grid"], dtype=np.int32)
        n_demos = int(layout.get("n_demos", 5))
        for k in range(n_demos):
            cfg = _sample_config(template, num_fires, rng, min_manhattan)
            cfg["layout_name"] = name
            cfg["layout_index"] = k
            cfg["grid"] = layout["grid"]
            cfg["rationale"] = (
                f"Training layout '{name}' demo {k+1}/{n_demos}."
            )
            expanded.append(cfg)

    out = {
        "demo_dir": args.demo_dir,
        "source":   args.layouts,
        "n_layouts_total": len(expanded),
        "layouts":  expanded,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[expand] {len(base_layouts)} base layouts → "
          f"{len(expanded)} concrete configs (3 fires each)")
    print(f"[expand] wrote: {out_path}")
    print(f"[expand] play_maze:")
    print(f"    python scripts/play_maze.py --layouts-from {out_path} "
          f"--demo_dir {args.demo_dir}")


if __name__ == "__main__":
    main()
