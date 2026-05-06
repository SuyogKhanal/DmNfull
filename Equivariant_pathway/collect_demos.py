"""BFS-expert demo collector for the Equivariant_pathway.

Two modes:
  1. From a layouts YAML (the default): record a fixed number of demos per
     layout (n_repetitions field on each layout entry, with a CLI override).
     This is how the initial 10-demo training set is collected:
         python -m Equivariant_pathway.collect_demos \
             --layouts Equivariant_pathway/training_layouts.yaml \
             --demo_dir Equivariant_pathway/demos \
             --num_demos 10

  2. From a recommended_layouts.json (the active-loop format): used by
     P4/P5/P6 rounds to autonomously collect the demos the LLM
     prescribed. Same I/O contract as scripts/rule_based_collector.py
     in the diffusion pathway, but writes EquivariantUNetPolicy-friendly
     demos (the schema is identical — both pathways consume the same
     trajectory + start/goal/fire fields).

Demo schema written here is the union of what BOTH datasets understand:
  - CNN_pathway/dataset.py reads observations / images / actions
  - Equivariant_pathway/dataset.py reads trajectory / goal_pos / fires
So the same files train either pathway with no conversion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from configs.maze_layouts import MAZE_LAYOUTS
from envs.maze_env import (
    ACTION_DELTAS, ACTION_NAMES, MazeNavEnv,
    TILE_FREE, TILE_FIRE, TILE_GOAL,
)
from Equivariant_pathway.expert import AStarExpert, build_grid

_HOST_MAZE = "multimodal"


def _build_forced_env(layout: Dict, seed: int) -> MazeNavEnv:
    """Same forced-layout construction CNN_pathway/rollout_test.py uses."""
    MAZE_LAYOUTS[_HOST_MAZE]["grid"]  = layout.get("grid",
        [[0]*5 for _ in range(5)])
    MAZE_LAYOUTS[_HOST_MAZE]["start"] = list(layout["start_pos"])
    MAZE_LAYOUTS[_HOST_MAZE]["goal"]  = list(layout["goal_pos"])
    fires = [tuple(f) for f in layout["fire_positions"]]
    env = MazeNavEnv(
        maze_name=_HOST_MAZE,
        render_mode="rgb_array",
        randomize_start=False,
        randomize_goal=False,
        randomize_fire=False,
        num_fire_tiles=len(fires),
        seed=seed,
        fire_positions=fires,
    )
    env.reset()
    forced_goal  = tuple(layout["goal_pos"])
    forced_start = tuple(layout["start_pos"])
    env.grid[env.grid == TILE_GOAL] = TILE_FREE
    env.goal_pos = forced_goal
    env.grid[forced_goal[0], forced_goal[1]] = TILE_GOAL
    env.agent_pos = forced_start
    env.start_pos = forced_start
    env.visited = np.zeros_like(env.grid, dtype=bool)
    env.visited[forced_start] = True
    env.trajectory = [forced_start]
    return env


def _record_one(env: MazeNavEnv, expert: AStarExpert,
                rng: np.random.Generator, layout_id: str,
                demo_dir: Path, suffix: str = "") -> Optional[Path]:
    obs = env._get_obs()
    obs_seq = [{"state": obs["state"], "image": obs["image"]}]
    actions: List[int] = []
    rewards: List[float] = []

    while True:
        agent = tuple(env.agent_pos)
        a = expert.sample_action(agent, rng)
        if a is None:
            break
        obs, reward, terminated, truncated, _ = env.step(a)
        obs_seq.append({"state": obs["state"], "image": obs["image"]})
        actions.append(int(a))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    if not actions:
        return None

    payload = {
        "maze_name":      _HOST_MAZE,
        "timestamp":      int(time.time() * 1000) + int(rng.integers(0, 1_000_000)),
        "layout_id":      layout_id,
        "source":         "equivariant_pathway_bfs",
        "start_pos":      [int(x) for x in env.start_pos],
        "goal_pos":       [int(x) for x in env.goal_pos],
        "fire_positions": [[int(x) for x in p] for p in env.fire_positions],
        "trajectory":     [[int(p[0]), int(p[1])] for p in env.get_trajectory()],
        "observations":   [o["state"].tolist() for o in obs_seq],
        "images":         [o["image"].tolist() for o in obs_seq],
        "actions":        actions,
        "rewards":        rewards,
        "total_reward":   float(sum(rewards)),
        "success":        bool(rewards[-1] > 5.0) if rewards else False,
    }
    demo_dir.mkdir(parents=True, exist_ok=True)
    fname = demo_dir / f"demo_{_HOST_MAZE}_{layout_id}{suffix}_{payload['timestamp']}.json"
    with open(fname, "w") as f:
        json.dump(payload, f)
    return fname


def collect_from_yaml(
    layouts_yaml: Path,
    demo_dir: Path,
    num_demos: Optional[int],
    seed: int,
) -> List[Path]:
    """Collect from training_layouts.yaml-style file. Loops layouts in
    YAML order. Each layout's `n_repetitions` field controls how many
    demos to record on that layout; `--num_demos` truncates the global
    total once enough have been written (default = sum of n_repetitions).
    """
    with open(layouts_yaml, "r") as f:
        spec = yaml.safe_load(f) or {}
    layouts = (
        spec.get("training_layouts")
        or spec.get("test_layouts")
        or spec.get("heldout_test_layouts")
        or spec.get("layouts")
        or []
    )
    rng = np.random.default_rng(seed)
    saved: List[Path] = []
    print(f"[eq-collect] {len(layouts)} layouts in {layouts_yaml}")

    for li, layout in enumerate(layouts):
        reps = max(1, int(layout.get("n_repetitions", 1) or 1))
        layout_id = layout.get("name") or f"layout{li+1}"
        for rep in range(reps):
            if num_demos is not None and len(saved) >= num_demos:
                break
            env = _build_forced_env(layout, seed=seed + li * 100 + rep)
            grid = build_grid(
                grid_size=env.grid.shape[0],
                fire_positions=env.fire_positions,
                goal_pos=env.goal_pos,
            )
            expert = AStarExpert(grid, env.goal_pos)
            if expert.shortest_path_length(env.agent_pos) <= 0:
                print(f"[eq-collect] {layout_id} rep{rep+1}: unsolvable; skipping")
                env.close()
                continue
            path = _record_one(env, expert, rng, layout_id, demo_dir,
                               suffix=f"_rep{rep+1}")
            env.close()
            if path is not None:
                saved.append(path)
                print(f"[eq-collect] saved {path.name}")
        if num_demos is not None and len(saved) >= num_demos:
            break

    print(f"[eq-collect] DONE — wrote {len(saved)} demos -> {demo_dir}")
    return saved


def collect_from_recommended(
    layouts_path: Path,
    demo_dir: Path,
    seed: int,
) -> List[Path]:
    """Collect from a recommended_layouts.json (active-loop format)."""
    with open(layouts_path, "r") as f:
        spec = json.load(f)
    layouts = spec.get("layouts") if isinstance(spec, dict) else spec
    if not isinstance(layouts, list) or not layouts:
        raise SystemExit(f"No 'layouts' list in {layouts_path}")
    rng = np.random.default_rng(seed)
    saved: List[Path] = []
    print(f"[eq-collect] {len(layouts)} prescribed layouts")

    for idx, layout in enumerate(layouts):
        layout_id = (
            f"demo{layout.get('parent_demo_id','?')}"
            f"_layout{layout.get('layout_index','?')}"
            f"_rep{layout.get('repetition','1')}"
        ) if "parent_demo_id" in layout else f"prescribed_{idx+1}"
        try:
            env = _build_forced_env(layout, seed=seed + idx)
            grid = build_grid(
                grid_size=env.grid.shape[0],
                fire_positions=env.fire_positions,
                goal_pos=env.goal_pos,
            )
            expert = AStarExpert(grid, env.goal_pos)
            if expert.shortest_path_length(env.agent_pos) <= 0:
                print(f"[eq-collect] {layout_id}: unsolvable; skipping")
                env.close()
                continue
            path = _record_one(env, expert, rng, layout_id, demo_dir)
            env.close()
            if path is not None:
                saved.append(path)
        except Exception as e:
            print(f"[eq-collect] {layout_id}: failed ({e!r}); skipping")
            continue

    summary = {
        "layouts_from": str(layouts_path),
        "demo_dir":     str(demo_dir),
        "n_layouts":    len(layouts),
        "n_saved":      len(saved),
        "saved":        [str(p) for p in saved],
    }
    summary_path = demo_dir.parent / "bfs_collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[eq-collect] DONE  saved={len(saved)}  summary={summary_path}")
    return saved


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", type=str, default=None,
                   help="A training/test/heldout layouts YAML.")
    p.add_argument("--layouts_from", type=str, default=None,
                   help="A recommended_layouts.json from the active loop.")
    p.add_argument("--demo_dir", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "demos"))
    p.add_argument("--num_demos", type=int, default=None,
                   help="Cap total demos collected from the YAML (round 1 uses 10).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if not (args.layouts or args.layouts_from):
        raise SystemExit("Provide either --layouts or --layouts_from.")
    demo_dir = Path(args.demo_dir)
    if args.layouts:
        collect_from_yaml(Path(args.layouts), demo_dir, args.num_demos, args.seed)
    if args.layouts_from:
        collect_from_recommended(Path(args.layouts_from), demo_dir, args.seed)


if __name__ == "__main__":
    main()
