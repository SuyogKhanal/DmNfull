"""Random 5x5 layout sampler for the held-out test set.

Produces 50 unique solvable random layouts and writes them to
heldout_test_layouts.yaml. The sampler:
  - draws random (start, goal) pairs with Manhattan distance >= 4 (so the
    rollout has to do real work; trivial corner-adjacent goals don't
    discriminate models)
  - draws random fire positions (default 3 fires, never on start/goal)
  - excludes any layout whose (start, goal, fires) signature collides
    with the hand-designed test_layouts.yaml so the held-out set is
    actually held out
  - rejects any sampled layout where BFS finds no fire-free path from
    start to goal (would be unsolvable for the expert too, so meaningless
    to evaluate against)

The grid itself is always all-free 5x5 (no walls) — the only obstacles
are the fires. Matches the rest of the DmNfull pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bfs_solvable(start: Tuple[int, int], goal: Tuple[int, int],
                  fires: Set[Tuple[int, int]], gs: int) -> bool:
    if start == goal:
        return False
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < gs and 0 <= nc < gs):
                continue
            if (nr, nc) in fires:
                continue
            if (nr, nc) in seen:
                continue
            if (nr, nc) == goal:
                return True
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def _signature(start, goal, fires) -> Tuple:
    return (
        tuple(start),
        tuple(goal),
        tuple(sorted(tuple(f) for f in fires)),
    )


def _load_blocked_signatures(yaml_paths: List[str]) -> Set[Tuple]:
    sigs: Set[Tuple] = set()
    for path in yaml_paths:
        if not Path(path).exists():
            continue
        with open(path, "r") as f:
            spec = yaml.safe_load(f) or {}
        for key in ("test_layouts", "training_layouts", "heldout_test_layouts", "layouts"):
            for L in spec.get(key, []) or []:
                sigs.add(_signature(L["start_pos"], L["goal_pos"], L["fire_positions"]))
    return sigs


def sample_layouts(
    n: int = 50,
    grid_size: int = 5,
    num_fires: int = 3,
    min_manhattan: int = 4,
    seed: int = 0,
    blocked_signatures: Optional[Set[Tuple]] = None,
    max_attempts: int = 20000,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    blocked = blocked_signatures or set()
    layouts: List[Dict] = []
    seen: Set[Tuple] = set()
    attempts = 0
    cell_count = grid_size * grid_size

    while len(layouts) < n and attempts < max_attempts:
        attempts += 1
        cells = rng.permutation(cell_count)
        start = (int(cells[0] // grid_size), int(cells[0] % grid_size))
        goal  = (int(cells[1] // grid_size), int(cells[1] % grid_size))
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < min_manhattan:
            continue
        fire_pool = [
            (int(c // grid_size), int(c % grid_size))
            for c in cells[2:]
            if (int(c // grid_size), int(c % grid_size)) not in (start, goal)
        ]
        if len(fire_pool) < num_fires:
            continue
        fires = set(fire_pool[:num_fires])
        sig = _signature(start, goal, fires)
        if sig in seen or sig in blocked:
            continue
        if not _bfs_solvable(start, goal, fires, grid_size):
            continue
        layouts.append({
            "name": f"heldout_{len(layouts)+1:03d}",
            "description": "Random held-out 5x5 layout (auto-generated).",
            "grid": [[0] * grid_size for _ in range(grid_size)],
            "start_pos": list(start),
            "goal_pos":  list(goal),
            "fire_positions": [list(f) for f in sorted(fires)],
        })
        seen.add(sig)

    if len(layouts) < n:
        raise RuntimeError(
            f"Only sampled {len(layouts)}/{n} unique solvable layouts after "
            f"{attempts} attempts. Lower min_manhattan or num_fires?"
        )
    return layouts


def write_yaml(layouts: List[Dict], out_path: Path,
               img_size: int = 80, grid_size: int = 5, cell_px: int = 16):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "img_size":             img_size,
        "grid_size":            grid_size,
        "cell_px":              cell_px,
        "heldout_test_layouts": layouts,
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[layout-sampler] wrote {len(layouts)} layouts -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "heldout_test_layouts.yaml"))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--grid_size", type=int, default=5)
    p.add_argument("--num_fires", type=int, default=3)
    p.add_argument("--min_manhattan", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block_yaml", type=str, action="append", default=[
        str(REPO_ROOT / "Equivariant_pathway" / "test_layouts.yaml"),
        str(REPO_ROOT / "Equivariant_pathway" / "training_layouts.yaml"),
    ], help="Layout YAML(s) whose (start,goal,fires) signatures must NOT be "
            "duplicated by the held-out sample. Repeat the flag to add more.")
    return p.parse_args()


def main():
    args = parse_args()
    blocked = _load_blocked_signatures(args.block_yaml)
    print(f"[layout-sampler] blocked {len(blocked)} known signatures from "
          f"{len(args.block_yaml)} yaml(s)")
    layouts = sample_layouts(
        n=args.n, grid_size=args.grid_size, num_fires=args.num_fires,
        min_manhattan=args.min_manhattan, seed=args.seed,
        blocked_signatures=blocked,
    )
    write_yaml(Path(args.out), layouts=layouts, grid_size=args.grid_size)


if __name__ == "__main__":
    main()
