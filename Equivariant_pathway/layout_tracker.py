"""Persist every layout the cycle touches as JSON + a rendered PNG.

The full active-loop cycle accumulates several layout sets per profile:
  - the 5 hand-designed training layouts (once)
  - the 4 hand-designed test layouts (once)
  - the 50 random held-out layouts (once, regenerated per --seed)
  - per round, the layouts the LLM prescribed (recommended_layouts.json)
  - per round, the test/heldout layouts the agent failed (so we can
    diff which generalisation gaps closed across rounds)

This module saves the spec as JSON AND renders each layout into a small
PNG (one cell-per-pixel block, agent / goal / fire colour-coded the
same way pygame would draw them) so a folder browser is enough to
sanity-check coverage.

Output layout
-------------
<base_dir>/
  layouts.json              # all layouts in one file (re-readable spec)
  images/
    <layout_name>.png       # one rendered tile per layout
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

# Reuse the env tile constants so the rendered colours line up with what
# pygame would draw at runtime (sanity-check images then look the same).
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
    grid = np.asarray(layout.get("grid", [[0] * 5 for _ in range(5)]), dtype=np.int32)
    H, W = grid.shape
    img = Image.new("RGB", (W * cell_px, H * cell_px), PALETTE["free"])
    draw = ImageDraw.Draw(img)

    fire_set = {tuple(f) for f in layout.get("fire_positions", []) or []}
    goal = tuple(layout.get("goal_pos", [-1, -1]))
    start = tuple(layout.get("start_pos", [-1, -1]))

    for r in range(H):
        for c in range(W):
            x0, y0 = c * cell_px, r * cell_px
            x1, y1 = x0 + cell_px, y0 + cell_px
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
        sx0 = start[1] * cell_px + cell_px // 4
        sy0 = start[0] * cell_px + cell_px // 4
        sx1 = sx0 + cell_px // 2
        sy1 = sy0 + cell_px // 2
        draw.ellipse((sx0, sy0, sx1, sy1), fill=PALETTE["agent"])
    return img


def save_layouts(
    layouts: List[Dict],
    base_dir: Path,
    label: str,
    extra_meta: Optional[Dict] = None,
):
    """Persist layouts as JSON + per-layout PNGs under <base_dir>.

    `label` is forwarded into the JSON metadata so a downstream consumer
    can tell train/test/heldout/round_<N>_recommended/round_<N>_failures
    artefacts apart at a glance.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cleaned: List[Dict] = []
    for L in layouts:
        c = {
            "name":            L.get("name") or "unnamed",
            "description":     L.get("description", ""),
            "grid":            L.get("grid", []),
            "start_pos":       list(L.get("start_pos", [])),
            "goal_pos":        list(L.get("goal_pos", [])),
            "fire_positions": [list(p) for p in L.get("fire_positions", []) or []],
        }
        for opt in ("n_repetitions", "rationale", "corridor",
                    "parent_demo_id", "layout_index", "guidance"):
            if opt in L:
                c[opt] = L[opt]
        cleaned.append(c)
        png_path = images_dir / f"{c['name']}.png"
        try:
            render_layout(c).save(str(png_path))
        except Exception as e:
            print(f"[layout-tracker] WARN: render failed for {c['name']}: {e}")

    payload = {
        "label":      label,
        "n_layouts":  len(cleaned),
        "layouts":    cleaned,
        **(extra_meta or {}),
    }
    with open(base_dir / "layouts.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[layout-tracker] [{label}] wrote {len(cleaned)} layouts -> {base_dir}")


def save_layouts_from_yaml(yaml_path: Path, base_dir: Path, label: str):
    with open(yaml_path, "r") as f:
        spec = yaml.safe_load(f) or {}
    layouts = (
        spec.get("training_layouts")
        or spec.get("test_layouts")
        or spec.get("heldout_test_layouts")
        or spec.get("layouts")
        or []
    )
    save_layouts(layouts, base_dir, label, extra_meta={"source_yaml": str(yaml_path)})


def save_failures_from_full_output(full_output_path: Path, base_dir: Path,
                                   label: str):
    """Extract every failed episode's (start, goal, fires) from a rollout's
    full_output.json and persist them as a layout set so we have a visual
    record of which generalisation gaps the agent had this round."""
    with open(full_output_path, "r") as f:
        data = json.load(f)
    rollouts = (data.get("phase_a", {}) or {}).get("all_rollouts", []) or []
    failures = []
    for r in rollouts:
        if r.get("success", False):
            continue
        dyn = r.get("dynamic_config", {}) or {}
        failures.append({
            "name": f"{r.get('maze_name', 'episode')}_ep{r.get('episode_id', '?')}",
            "description": f"Failure at episode {r.get('episode_id')}",
            "grid": [[0] * 5 for _ in range(5)],
            "start_pos":      list(dyn.get("start_pos", [])),
            "goal_pos":       list(dyn.get("goal_pos", [])),
            "fire_positions": [list(p) for p in dyn.get("fire_positions", []) or []],
        })
    save_layouts(failures, base_dir, label,
                 extra_meta={"source_full_output": str(full_output_path)})
