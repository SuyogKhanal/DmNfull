"""Random 5x5 layout sampler (CNN_pathway clone of Equivariant version).

Pure Python / numpy / yaml — no policy dependencies. Cloned here so the
CNN_pathway sweep is fully self-contained.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bfs_solvable(start, goal, fires, gs):
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


def _signature(start, goal, fires):
    return (tuple(start), tuple(goal),
            tuple(sorted(tuple(f) for f in fires)))


def _load_blocked_signatures(yaml_paths):
    sigs = set()
    for path in yaml_paths:
        if not Path(path).exists():
            continue
        with open(path, "r") as f:
            spec = yaml.safe_load(f) or {}
        for key in ("test_layouts", "training_layouts", "heldout_test_layouts", "layouts"):
            for L in spec.get(key, []) or []:
                sigs.add(_signature(L["start_pos"], L["goal_pos"], L["fire_positions"]))
    return sigs


def sample_layouts(n=50, grid_size=5, num_fires=3, min_manhattan=4,
                   seed=0, blocked_signatures=None, max_attempts=20000):
    rng = np.random.default_rng(seed)
    blocked = blocked_signatures or set()
    layouts: List[Dict] = []
    seen: Set = set()
    attempts = 0
    cell_count = grid_size * grid_size
    while len(layouts) < n and attempts < max_attempts:
        attempts += 1
        cells = rng.permutation(cell_count)
        start = (int(cells[0] // grid_size), int(cells[0] % grid_size))
        goal  = (int(cells[1] // grid_size), int(cells[1] % grid_size))
        if abs(start[0]-goal[0]) + abs(start[1]-goal[1]) < min_manhattan:
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
            "grid": [[0]*grid_size for _ in range(grid_size)],
            "start_pos": list(start),
            "goal_pos":  list(goal),
            "fire_positions": [list(f) for f in sorted(fires)],
        })
        seen.add(sig)
    if len(layouts) < n:
        raise RuntimeError(
            f"Only sampled {len(layouts)}/{n} unique solvable layouts after "
            f"{attempts} attempts."
        )
    return layouts


def write_yaml(layouts, out_path: Path,
               img_size=80, grid_size=5, cell_px=16):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "img_size": img_size, "grid_size": grid_size, "cell_px": cell_px,
        "heldout_test_layouts": layouts,
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[layout-sampler] wrote {len(layouts)} layouts -> {out_path}")
