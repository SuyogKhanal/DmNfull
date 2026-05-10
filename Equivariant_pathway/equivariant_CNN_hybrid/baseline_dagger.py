"""Baseline DAgger for the EquivariantCNNHybridPolicy.

Same loop semantics as Equivariant_pathway/baseline_dagger.py but the
rollout step encodes BOTH the RGB and grid inputs, calls
``model(rgb, grid)``, and the retrain subprocess invokes
``Equivariant_pathway.equivariant_CNN_hybrid.train``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from configs.maze_layouts import MAZE_LAYOUTS
from envs.maze_env import (ACTION_DELTAS, ACTION_NAMES, MazeNavEnv,
                           TILE_FREE, TILE_FIRE, TILE_GOAL)
from Equivariant_pathway.encoder import encode_from_env as encode_grid_from_env
from Equivariant_pathway.expert import AStarExpert, build_grid
from CNN_pathway.encoder_rgb import encode_state as encode_rgb
from Equivariant_pathway.equivariant_CNN_hybrid.model import EquivariantCNNHybridPolicy

_HOST_MAZE = "multimodal"
TRAIN_EVERY_N_DEMOS = 4


def _build_env(layout, seed):
    MAZE_LAYOUTS[_HOST_MAZE]["grid"]  = layout.get("grid", [[0]*5 for _ in range(5)])
    MAZE_LAYOUTS[_HOST_MAZE]["start"] = list(layout["start_pos"])
    MAZE_LAYOUTS[_HOST_MAZE]["goal"]  = list(layout["goal_pos"])
    fires = [tuple(f) for f in layout["fire_positions"]]
    env = MazeNavEnv(
        maze_name=_HOST_MAZE, render_mode="rgb_array",
        randomize_start=False, randomize_goal=False, randomize_fire=False,
        num_fire_tiles=len(fires), seed=seed, fire_positions=fires,
    )
    env.reset()
    forced_goal = tuple(layout["goal_pos"])
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


def _bfs_path(grid, start, goal):
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
            if tile in (1, TILE_FIRE):
                continue
            visited.add((nr, nc))
            new_actions = actions + [action]
            if (nr, nc) == goal:
                return new_actions
            queue.append(((nr, nc), new_actions))
    return []


def _save_corrective_demo(env, obs_seq, actions, rewards, demo_dir, layout_name, suffix):
    demo_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    payload = {
        "maze_name": _HOST_MAZE, "timestamp": timestamp,
        "layout_id": f"{layout_name}_{suffix}",
        "source": "hybrid_baseline_dagger_correction",
        "start_pos": [int(x) for x in env.start_pos],
        "goal_pos":  [int(x) for x in env.goal_pos],
        "fire_positions": [[int(x) for x in p] for p in env.fire_positions],
        "trajectory": [[int(p[0]), int(p[1])] for p in env.get_trajectory()],
        "observations": [o["state"].tolist() for o in obs_seq],
        "images": [o["image"].tolist() for o in obs_seq],
        "actions": [int(a) for a in actions],
        "rewards": [float(r) for r in rewards],
        "total_reward": float(sum(rewards)),
        "success": bool(rewards[-1] > 5.0) if rewards else False,
    }
    fname = demo_dir / f"demo_{_HOST_MAZE}_{layout_name}_{suffix}_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(payload, f)
    return fname


def _load_model(checkpoint, device):
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model = EquivariantCNNHybridPolicy(
        rgb_in_channels=3, grid_in_channels=int(ckpt.get("in_channels", 5)),
        num_actions=int(ckpt.get("num_actions", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def _find_deviation_step(layout, actions, trajectory):
    fires = [tuple(f) for f in layout["fire_positions"]]
    goal = tuple(layout["goal_pos"])
    gs = len(layout.get("grid", [[0]*5]*5))
    grid = build_grid(gs, fire_positions=fires, goal_pos=goal)
    expert = AStarExpert(grid, goal)
    for t, a in enumerate(actions):
        if t >= len(trajectory):
            break
        pos = trajectory[t]
        opt = expert.optimal_actions(pos)
        if opt.sum() == 0:
            return t
        if opt[a] < 0.5:
            return t
    return -1


def _count_corrective_demos(demo_dir):
    if not demo_dir.exists():
        return 0
    return sum(1 for p in demo_dir.rglob("*.json") if "_correct_" in p.name)


def _encode_rgb_from_env(env, cell_px=16):
    return encode_rgb(
        grid=env.grid, agent_pos=tuple(env.agent_pos),
        goal_pos=tuple(env.goal_pos),
        fire_positions=[tuple(p) for p in env.fire_positions],
        cell_px=cell_px,
    )


def run_dagger_correction_pass(
    layouts_yaml, checkpoint, demo_dir, out_dir,
    seed=0, max_steps=60, train_every_n=TRAIN_EVERY_N_DEMOS,
    train_demo_paths=None, epochs=50, skip_train=False,
    n_episodes=50, train_from_scratch=False, cell_px=16,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(layouts_yaml, "r") as f:
        spec = yaml.safe_load(f) or {}
    layouts = (spec.get("test_layouts") or spec.get("training_layouts")
               or spec.get("heldout_test_layouts") or spec.get("layouts") or [])
    if not layouts:
        raise SystemExit(f"No layouts in {layouts_yaml}")
    print(f"[hybrid-dagger] {len(layouts)} layouts; running {n_episodes} episodes")

    model = _load_model(checkpoint, device)
    cum_at_start = _count_corrective_demos(demo_dir)
    cum = cum_at_start
    next_train_at = ((cum // train_every_n) + 1) * train_every_n

    n_train_calls = 0
    saved_demos: List[Path] = []
    episode_log: List[Dict] = []
    success_count = 0
    n_corrected_in_pass = 0

    for ep_idx in range(n_episodes):
        layout = layouts[ep_idx % len(layouts)]
        ep_seed = seed + ep_idx
        env = _build_env(layout, seed=ep_seed)
        obs_seq = [{"state": env._get_obs()["state"], "image": env._get_obs()["image"]}]
        actions: List[int] = []
        rewards: List[float] = []
        terminated = truncated = False
        si = 0
        while not (terminated or truncated) and si < max_steps:
            x_grid = encode_grid_from_env(env)
            x_rgb  = _encode_rgb_from_env(env, cell_px=cell_px)
            rgb_t  = torch.from_numpy(x_rgb).float().unsqueeze(0).to(device)
            grid_t = torch.from_numpy(x_grid).float().unsqueeze(0).to(device)
            with torch.no_grad():
                a = int(model(rgb_t, grid_t).argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(a)
            obs_seq.append({"state": obs["state"], "image": obs["image"]})
            actions.append(a)
            rewards.append(float(reward))
            si += 1
        success = bool(rewards[-1] > 5.0) if rewards else False
        if success:
            success_count += 1
        traj_positions = [tuple(p) for p in env.get_trajectory()]
        env.close()

        corrected_here = False
        deviation_step: Optional[int] = None
        if not success and actions:
            deviation_step = _find_deviation_step(layout, actions, traj_positions)
            if deviation_step >= 0:
                env2 = _build_env(layout, seed=ep_seed)
                obs2 = env2._get_obs()
                obs_seq2 = [{"state": obs2["state"], "image": obs2["image"]}]
                actions2: List[int] = []
                rewards2: List[float] = []
                for a in actions[:deviation_step]:
                    obs, r, _, _, _ = env2.step(a)
                    obs_seq2.append({"state": obs["state"], "image": obs["image"]})
                    actions2.append(int(a))
                    rewards2.append(float(r))
                plan = _bfs_path(env2.grid.copy(),
                                 tuple(env2.agent_pos),
                                 tuple(env2.goal_pos))
                if plan:
                    for a in plan:
                        obs, r, term, trunc, _ = env2.step(a)
                        obs_seq2.append({"state": obs["state"], "image": obs["image"]})
                        actions2.append(int(a))
                        rewards2.append(float(r))
                        if term or trunc:
                            break
                    out_path = _save_corrective_demo(
                        env2, obs_seq2, actions2, rewards2,
                        demo_dir=demo_dir,
                        layout_name=layout.get("name", f"ep{ep_idx}"),
                        suffix=f"correct_ep{ep_idx}_dev{deviation_step}",
                    )
                    saved_demos.append(out_path)
                    cum += 1
                    n_corrected_in_pass += 1
                    corrected_here = True
                env2.close()

        episode_log.append({
            "episode_id": ep_idx, "layout_name": layout.get("name"),
            "policy_success": success, "corrected": corrected_here,
            "deviation_step": deviation_step, "policy_steps": si,
        })

        if (not skip_train) and cum >= next_train_at:
            print(f"[hybrid-dagger] cumulative={cum} >= {next_train_at}; retraining")
            cmd = [
                sys.executable, "-u", "-m",
                "Equivariant_pathway.equivariant_CNN_hybrid.train",
                "--checkpoint_dir", str(checkpoint.parent),
                "--epochs", str(epochs),
            ]
            if not train_from_scratch:
                cmd.append("--resume")
            if train_demo_paths:
                cmd += ["--demo_paths", train_demo_paths]
            else:
                cmd += ["--demo_dir", str(demo_dir)]
            rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
            if rc != 0:
                print(f"[hybrid-dagger] WARNING: retrain rc={rc}")
            n_train_calls += 1
            next_train_at += train_every_n
            best_path = checkpoint.parent / "best_hybrid_policy.pth"
            reload_path = best_path if best_path.exists() else checkpoint
            model = _load_model(reload_path, device)

    summary = {
        "layouts_yaml": str(layouts_yaml), "n_layouts_in_pool": len(layouts),
        "n_episodes": n_episodes, "n_successes": success_count,
        "n_failures": n_episodes - success_count,
        "success_rate": success_count / n_episodes if n_episodes else 0.0,
        "n_corrected_in_pass": n_corrected_in_pass,
        "cum_corrections_before": cum_at_start,
        "cum_corrections_after": cum,
        "n_demos_added": len(saved_demos), "n_train_calls": n_train_calls,
        "demo_paths": [str(p) for p in saved_demos],
        "episode_log": episode_log, "checkpoint": str(checkpoint),
        "demo_dir": str(demo_dir),
    }
    with open(out_dir / "dagger_pass_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[hybrid-dagger] PASS DONE sr={summary['success_rate']:.2f} corrected={n_corrected_in_pass}")
    return summary
