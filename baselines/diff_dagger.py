import json
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.rollout import run_rollouts
from envs.maze_env import MazeNavEnv, TILE_FIRE, TILE_WALL, ACTION_NAMES


_ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _bfs_solve(grid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[int]]:
    rows, cols = grid.shape
    if start == goal:
        return []
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        pos, actions = queue.popleft()
        for a, (dr, dc) in _ACTION_DELTAS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            tile = int(grid[nr, nc])
            if tile == TILE_WALL or tile == TILE_FIRE:
                continue
            np_pos = (nr, nc)
            if np_pos in visited:
                continue
            new_actions = actions + [a]
            if np_pos == goal:
                return new_actions
            visited.add(np_pos)
            queue.append((np_pos, new_actions))
    return None


def _build_intervention_for_episode(episode: Dict) -> Optional[Dict]:
    key_frames = episode.get("key_frames", [])
    hl_step = None
    for kf in key_frames:
        if kf["role"] == "highest_loss_frame":
            hl_step = int(kf["step_idx"])
            break
    if hl_step is None:
        return None
    if hl_step <= 0 or hl_step >= len(episode["steps"]) - 1:
        return None

    hl_info = episode["steps"][hl_step].get("info", {})
    hl_pos  = tuple(hl_info.get("agent_pos", (0, 0)))

    env = MazeNavEnv(
        maze_name=episode["maze_name"],
        render_mode="rgb_array",
        randomize_start=True,
        randomize_goal=True,
        randomize_fire=True,
        num_fire_tiles=len(episode.get("dynamic_config", {}).get("fire_positions", [])) or 3,
        seed=episode["seed"],
    )
    env.reset()
    grid = env.grid
    goal = tuple(env.goal_pos)
    env.close()

    actions = _bfs_solve(grid, hl_pos, goal)
    if actions is None:
        return None

    return {
        "episode_id":        episode["episode_id"],
        "seed":              episode["seed"],
        "intervention_step": hl_step,
        "intervention_pos":  list(hl_pos),
        "goal_pos":          list(goal),
        "expert_actions":    [int(a) for a in actions],
        "expert_action_names":[ACTION_NAMES.get(int(a), "?") for a in actions],
        "demo_source":       "bfs_diff_dagger",
    }


def run_diff_dagger(config: Dict, out_dir: Optional[Path] = None) -> Dict:
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(config.get("tracking", {}).get("output_dir", "results/runs")).parent / "baselines" / f"diff_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_result = run_rollouts(config, out_dir)
    failures = [e for e in rollout_result["all_episodes"] if not e["success"]]

    demos: List[Dict] = []
    for ed in failures:
        try:
            d = _build_intervention_for_episode(ed)
            if d is not None:
                demos.append(d)
        except Exception as e:
            demos.append({"episode_id": ed["episode_id"], "error": str(e)})

    summary = {
        "baseline":              "diff_dagger",
        "timestamp":             datetime.now().isoformat(),
        "n_episodes":            rollout_result["n_episodes"],
        "n_successes":           len(rollout_result["success_episode_ids"]),
        "n_failures":            len(failures),
        "success_rate":          (len(rollout_result["success_episode_ids"]) / max(rollout_result["n_episodes"], 1)),
        "total_demonstrations_needed": len(demos),
        "prescribed_demos":      demos,
        "config":                config,
    }
    with open(out_dir / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[DiffDAgger] Done. {len(demos)} interventions prescribed → {out_dir}")
    return summary