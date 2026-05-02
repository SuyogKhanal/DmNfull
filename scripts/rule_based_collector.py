"""Rule-based BFS demo collector — the headless fallback for play_maze.

When the human expert isn't available (e.g. inside a slurm job that has to run
unattended), this script takes the same recommended_layouts.json the LLM
produced and walks every layout autonomously: build the env on the forced
layout -> BFS-plan a fire-free shortest path from start to goal -> execute the
plan -> save the demo in the EXACT format play_maze writes, into the EXACT
demo_dir the active loop expects. The active loop / run_round can then retrain
on the aggregated dataset and continue without ever blocking on a human.

Usage
-----
  python scripts/rule_based_collector.py \
      --layouts-from results/active_loop/p3/loop_<id>/round_<N>/recommended_layouts.json

  # Override the demo_dir embedded in the JSON:
  python scripts/rule_based_collector.py --layouts-from <path> --demo_dir demos/active_loop/p3/loop_<id>/round_<N>

  # Skip layouts whose start/goal/fire spec is unsolvable by BFS:
  python scripts/rule_based_collector.py --layouts-from <path> --skip-unsolvable
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.maze_env import MazeNavEnv, ACTION_DELTAS, ACTION_NAMES, TILE_FREE, TILE_FIRE, TILE_GOAL


MAZE_NAME = "multimodal"


def bfs_plan(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[int]:
    """Shortest fire-free path from `start` to `goal`.

    Returns the action sequence (0=UP,1=DOWN,2=LEFT,3=RIGHT) or [] if no path
    exists. Walls and fires are both treated as impassable; the goal cell is
    the destination.
    """
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
            if tile in (1, TILE_FIRE):  # 1 == TILE_WALL
                continue
            visited.add((nr, nc))
            new_actions = actions + [action]
            if (nr, nc) == goal:
                return new_actions
            queue.append(((nr, nc), new_actions))
    return []


def _build_forced_env(layout: Dict, seed: int) -> MazeNavEnv:
    """Construct a MazeNavEnv on the forced layout with all randomisation off,
    then post-edit the agent / goal cells so the env matches the spec exactly.
    Mirrors the same path play_maze.create_env uses for forced layouts."""
    fires = [tuple(f) for f in layout["fire_positions"]]
    env = MazeNavEnv(
        maze_name=MAZE_NAME,
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
    if forced_goal in env.fire_positions:
        raise ValueError(f"forced goal {forced_goal} overlaps a fire cell")
    if forced_start in env.fire_positions:
        raise ValueError(f"forced start {forced_start} overlaps a fire cell")
    if forced_start == forced_goal:
        raise ValueError(f"forced start equals forced goal: {forced_start}")

    env.grid[env.grid == TILE_GOAL] = TILE_FREE
    env.goal_pos = forced_goal
    env.grid[forced_goal[0], forced_goal[1]] = TILE_GOAL

    env.agent_pos = forced_start
    env.start_pos = forced_start
    env.visited = np.zeros_like(env.grid, dtype=bool)
    env.visited[env.agent_pos] = True
    env.trajectory = [env.agent_pos]
    return env


def _save_demo(
    env: MazeNavEnv,
    obs_seq: List[Dict],
    action_seq: List[int],
    reward_seq: List[float],
    demo_dir: Path,
    layout_id: str,
) -> Path:
    """Write the demo JSON in the same shape play_maze.save_demo produces, so
    downstream training treats rule-based and human demos identically."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    config = env.get_dynamic_config()
    filename = demo_dir / f"demo_{MAZE_NAME}_{layout_id}_bfs_{timestamp}.json"
    # config["start_pos"] reports env.agent_pos, which after stepping the BFS
    # plan equals the goal cell. The episode's true spawn is env.start_pos
    # (set in _build_forced_env to the forced start).
    payload = {
        "maze_name":      MAZE_NAME,
        "timestamp":      timestamp,
        "layout_id":      layout_id,
        "source":         "rule_based_bfs",
        "start_pos":      [int(x) for x in env.start_pos],
        "goal_pos":       [int(x) for x in config["goal_pos"]],
        "fire_positions": [[int(x) for x in p] for p in config["fire_positions"]],
        "trajectory":     [[int(x) for x in p] for p in env.get_trajectory()],
        "observations":   [o["state"].tolist() for o in obs_seq],
        "images":         [o["image"].tolist() for o in obs_seq],
        "actions":        [int(a) for a in action_seq],
        "rewards":        [float(r) for r in reward_seq],
        "total_reward":   float(sum(reward_seq)),
        "success":        bool(reward_seq[-1] > 5.0) if reward_seq else False,
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    return filename


def _layout_id_for(layout: Dict, idx: int, total: int) -> str:
    if layout.get("parent_demo_id") is not None and layout.get("layout_index") is not None:
        return (
            f"demo{layout['parent_demo_id']}"
            f"_layout{layout['layout_index']}"
            f"_rep{layout.get('repetition','1')}"
        )
    return f"layout{idx+1}of{total}"


def collect_for_layout(layout: Dict, demo_dir: Path, idx: int, total: int, seed: int = 0) -> Optional[Path]:
    """Build env, plan, execute, save. Returns the saved demo path or None when
    BFS finds no path on this layout (caller decides how to handle)."""
    print(f"\n[bfs] layout {idx+1}/{total} | start={layout['start_pos']} goal={layout['goal_pos']} fires={layout['fire_positions']}")
    env = _build_forced_env(layout, seed=seed)
    initial_obs = env._get_obs()
    obs_seq: List[Dict] = [{"state": initial_obs["state"], "image": initial_obs["image"]}]
    action_seq: List[int] = []
    reward_seq: List[float] = []

    plan = bfs_plan(env.grid, tuple(env.agent_pos), tuple(env.goal_pos))
    if not plan:
        print(f"[bfs] WARNING: no fire-free path on this layout; skipping.")
        env.close()
        return None
    print(f"[bfs] BFS plan length = {len(plan)} actions: {[ACTION_NAMES[a] for a in plan]}")

    for action in plan:
        obs, reward, terminated, truncated, info = env.step(action)
        obs_seq.append(obs)
        action_seq.append(action)
        reward_seq.append(reward)
        if terminated or truncated:
            break

    success = bool(reward_seq[-1] > 5.0) if reward_seq else False
    layout_id = _layout_id_for(layout, idx, total)
    saved = _save_demo(env, obs_seq, action_seq, reward_seq, demo_dir, layout_id)
    env.close()
    print(
        f"[bfs] saved {saved.name} | steps={len(action_seq)} "
        f"reward={sum(reward_seq):+.2f} success={success}"
    )
    return saved


def parse_args():
    p = argparse.ArgumentParser(description="Rule-based BFS demo collector for prescribed layouts.")
    p.add_argument("--layouts-from", type=str, required=True, dest="layouts_from",
                   help="Path to a recommended_layouts.json produced by run_round / active_loop.")
    p.add_argument("--demo_dir", type=str, default=None,
                   help="Where to save demos. Defaults to the demo_dir embedded in the JSON.")
    p.add_argument("--skip-unsolvable", action="store_true", dest="skip_unsolvable",
                   help="Don't error on layouts where BFS finds no path; just skip them.")
    p.add_argument("--seed-base", type=int, default=0, dest="seed_base",
                   help="Per-layout env seed = seed_base + layout_index. Demos are deterministic given the layout.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.layouts_from, "r") as f:
        spec = json.load(f)
    layouts = spec.get("layouts") if isinstance(spec, dict) else spec
    if not isinstance(layouts, list) or not layouts:
        raise SystemExit(f"no 'layouts' list in {args.layouts_from}")

    demo_dir_str = args.demo_dir or spec.get("demo_dir")
    if not demo_dir_str:
        raise SystemExit("demo_dir not provided and not present in the JSON")
    demo_dir = Path(demo_dir_str)
    demo_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bfs] {len(layouts)} layouts queued")
    print(f"[bfs] demo_dir : {demo_dir}")

    saved_paths: List[Path] = []
    skipped: List[Dict] = []
    for idx, layout in enumerate(layouts):
        try:
            saved = collect_for_layout(layout, demo_dir, idx, len(layouts), seed=args.seed_base + idx)
            if saved is None:
                skipped.append({"index": idx, "reason": "no_bfs_path", "layout": layout})
            else:
                saved_paths.append(saved)
        except Exception as e:
            if args.skip_unsolvable:
                print(f"[bfs] WARNING: layout {idx} raised {e!r}; skipping.")
                skipped.append({"index": idx, "reason": str(e), "layout": layout})
            else:
                raise

    summary = {
        "layouts_from": args.layouts_from,
        "demo_dir":     str(demo_dir),
        "n_layouts":    len(layouts),
        "n_saved":      len(saved_paths),
        "n_skipped":    len(skipped),
        "saved":        [str(p) for p in saved_paths],
        "skipped":      skipped,
    }
    summary_path = demo_dir.parent / "bfs_collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[bfs] DONE. saved={len(saved_paths)} skipped={len(skipped)}")
    print(f"[bfs] summary: {summary_path}")


if __name__ == "__main__":
    main()
