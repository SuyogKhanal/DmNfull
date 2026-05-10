"""Persist every layout the CNN_pathway sweep touches as JSON + PNG.

Clone of Equivariant_pathway/layout_tracker.py with no behavioural
change — the renderer is policy-agnostic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.maze_env import TILE_FREE, TILE_WALL, TILE_FIRE, TILE_GOAL  # noqa: F401

CELL_PX = 32
PALETTE = {
    "free":  (240, 240, 240),
    "wall":  (40,  40,  40),
    "fire":  (220, 60,  20),
    "goal":  (50,  200, 80),
    "agent": (30,  100, 220),
    "grid":  (180, 180, 180),
}


def render_layout(layout: Dict, cell_px: int = CELL_PX) -> Image.Image:
    grid = np.asarray(layout.get("grid", [[0]*5 for _ in range(5)]), dtype=np.int32)
    H, W = grid.shape
    img = Image.new("RGB", (W*cell_px, H*cell_px), PALETTE["free"])
    draw = ImageDraw.Draw(img)
    fire_set = {tuple(f) for f in layout.get("fire_positions", []) or []}
    goal = tuple(layout.get("goal_pos", [-1, -1]))
    start = tuple(layout.get("start_pos", [-1, -1]))
    for r in range(H):
        for c in range(W):
            x0, y0 = c*cell_px, r*cell_px
            x1, y1 = x0+cell_px, y0+cell_px
            if (r, c) in fire_set:
                color = PALETTE["fire"]
            elif (r, c) == goal:
                color = PALETTE["goal"]
            elif grid[r, c] == TILE_WALL:
                color = PALETTE["wall"]
            else:
                color = PALETTE["free"]
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=PALETTE["grid"])
    if 0 <= start[0] < H and 0 <= start[1] < W:
        sx0 = start[1]*cell_px + cell_px//4
        sy0 = start[0]*cell_px + cell_px//4
        sx1 = sx0 + cell_px//2
        sy1 = sy0 + cell_px//2
        draw.ellipse((sx0, sy0, sx1, sy1), fill=PALETTE["agent"])
    return img


def save_layouts(layouts: List[Dict], base_dir: Path, label: str,
                 extra_meta: Optional[Dict] = None):
    base_dir.mkdir(parents=True, exist_ok=True)
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cleaned = []
    for L in layouts:
        c = {
            "name":            L.get("name") or "unnamed",
            "description":     L.get("description", ""),
            "grid":            L.get("grid", []),
            "start_pos":       list(L.get("start_pos", [])),
            "goal_pos":        list(L.get("goal_pos", [])),
            "fire_positions": [list(p) for p in L.get("fire_positions", []) or []],
        }
        cleaned.append(c)
        try:
            img = render_layout(c)
            img.save(images_dir / f"{c['name']}.png")
        except Exception as e:
            print(f"[layout-tracker] render failed for {c.get('name')}: {e}")
    payload = {"label": label, "n_layouts": len(cleaned), "layouts": cleaned}
    if extra_meta:
        payload.update(extra_meta)
    with open(base_dir / "layouts.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_failures_from_full_output(full_output_path: Path, base_dir: Path,
                                   label: str):
    """Read a rollout's full_output.json and render every failed episode's
    layout (from dynamic_config) as a PNG + JSON record."""
    if not full_output_path.exists():
        return
    full = json.load(open(full_output_path))
    fails = []
    for e in full.get("phase_a", {}).get("all_rollouts", []) or []:
        if e.get("success"):
            continue
        dyn = e.get("dynamic_config", {}) or {}
        fails.append({
            "name": f"{label}_ep{e.get('episode_id','?')}",
            "description": "Failed episode layout (auto-saved by CNN_pathway sweep).",
            "grid": [[0]*5 for _ in range(5)],
            "start_pos": list(dyn.get("start_pos", [])),
            "goal_pos":  list(dyn.get("goal_pos", [])),
            "fire_positions": [list(p) for p in dyn.get("fire_positions", []) or []],
        })
    save_layouts(fails, base_dir=base_dir, label=label)
