"""Rollout helper for the EquivariantCNNHybridPolicy.

Same artefact contract as the equivariant rollout — full_output.json +
per-episode frames + episode_data.json so the shared P4 LLM analysis
pipeline can consume the run with no custom adapter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from configs.maze_layouts import MAZE_LAYOUTS
from envs.maze_env import ACTION_NAMES, MazeNavEnv, TILE_FREE, TILE_GOAL
from Equivariant_pathway.encoder import encode_from_env as encode_grid_from_env
from CNN_pathway.encoder_rgb import encode_state as encode_rgb
from Equivariant_pathway.equivariant_CNN_hybrid.model import EquivariantCNNHybridPolicy

_HOST_MAZE = "multimodal"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "equivariant_CNN_hybrid" / "checkpoints" / "best_hybrid_policy.pth"))
    p.add_argument("--layouts", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif_duration_ms", type=int, default=250)
    p.add_argument("--n_episodes", type=int, default=None)
    p.add_argument("--cell_px", type=int, default=16)
    return p.parse_args()


def _build_env(layout, seed):
    MAZE_LAYOUTS[_HOST_MAZE]["grid"]  = layout["grid"]
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


def _encode_rgb_from_env(env, cell_px=16):
    return encode_rgb(
        grid=env.grid,
        agent_pos=tuple(env.agent_pos),
        goal_pos=tuple(env.goal_pos),
        fire_positions=[tuple(p) for p in env.fire_positions],
        cell_px=cell_px,
    )


def _serialise_info(info):
    out = {}
    for k, v in info.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _serialise_info(v)
        else:
            out[k] = v
    return out


def _save_gif(steps, episode_dir: Path, duration_ms: int) -> str:
    frames = []
    for s in steps:
        rgb = s.get("rgb")
        if rgb is None:
            continue
        frames.append(Image.fromarray(np.asarray(rgb)))
    if not frames:
        return ""
    episode_dir.mkdir(parents=True, exist_ok=True)
    gif_path = episode_dir / "trajectory.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=False)
    return str(gif_path)


def _save_key_frames(steps, key_frames, episode_dir: Path):
    paths = {}
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    role_to_file = {"start_frame": "start.png", "highest_loss_frame": "high_loss.png",
                    "end_frame": "end.png"}
    for kf in key_frames:
        rgb = steps[kf["step_idx"]].get("rgb")
        if rgb is None:
            continue
        out = frames_dir / role_to_file[kf["role"]]
        Image.fromarray(np.asarray(rgb)).save(str(out))
        paths[kf["role"]] = str(out)
    return paths


def _strip_rgb(episode):
    return {**episode,
            "steps": [{k: v for k, v in s.items() if k != "rgb"} for s in episode["steps"]]}


def _pick_high_loss_step(steps):
    n = len(steps)
    if n <= 2:
        return max(0, n // 2)

    def _pos(i):
        info = steps[i].get("info") or {}
        ap = info.get("agent_pos")
        if ap is None: return None
        try: return (int(ap[0]), int(ap[1]))
        except (TypeError, ValueError, IndexError): return None
    start_pos = _pos(0); end_pos = _pos(n - 1)
    best_idx, best_key = -1, None
    for i in range(1, n - 1):
        pos = _pos(i)
        differs = (pos is not None) and (pos != start_pos) and (pos != end_pos)
        rwd = float(steps[i].get("reward") or 0.0)
        key = (1 if differs else 0, -rwd)
        if best_key is None or key > best_key:
            best_key = key; best_idx = i
    return best_idx if best_idx >= 0 else (n - 1) // 2


def _run_episode(model, layout, episode_id, seed, max_steps, device, cell_px):
    env = _build_env(layout, seed=seed)
    ascii_grid = env.get_grid_image_description()
    dyn_cfg = env.get_dynamic_config()
    obs = env._get_obs()
    rgb0 = env.render()
    steps: List[Dict] = [{
        "step_idx": 0, "obs": obs["state"].tolist(), "reward": 0.0,
        "action": None, "action_name": "RESET",
        "info": _serialise_info(env._get_info()), "rgb": rgb0,
    }]
    si = 0
    terminated = truncated = False
    while not (terminated or truncated) and si < max_steps:
        x_grid = encode_grid_from_env(env)
        x_rgb  = _encode_rgb_from_env(env, cell_px=cell_px)
        rgb_t  = torch.from_numpy(x_rgb).float().unsqueeze(0).to(device)
        grid_t = torch.from_numpy(x_grid).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(rgb_t, grid_t)
        action = int(logits.argmax(dim=-1).item())
        obs, reward, terminated, truncated, info = env.step(action)
        si += 1
        steps.append({
            "step_idx": si, "obs": obs["state"].tolist(),
            "reward": float(reward), "action": action,
            "action_name": ACTION_NAMES.get(action, "?"),
            "info": _serialise_info(info), "rgb": env.render(),
        })
    success = bool(steps[-1]["info"].get("success", False))
    total_reward = sum(s["reward"] for s in steps)
    sl = _pick_high_loss_step(steps)
    key_frames = [
        {"role": "start_frame", "step_idx": 0},
        {"role": "highest_loss_frame", "step_idx": sl},
        {"role": "end_frame", "step_idx": len(steps) - 1},
    ]
    env.close()
    return {
        "episode_id": episode_id, "maze_name": layout["name"], "seed": seed,
        "total_steps": si, "total_reward": total_reward, "success": success,
        "ascii_grid": ascii_grid, "dynamic_config": dyn_cfg,
        "steps": steps, "key_frames": key_frames,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.layouts, "r") as f:
        spec = yaml.safe_load(f) or {}
    layouts = (spec.get("test_layouts") or spec.get("training_layouts")
               or spec.get("heldout_test_layouts") or spec.get("layouts") or [])
    if not layouts:
        raise SystemExit(f"No layouts list in {args.layouts}")
    if args.n_episodes is None or args.n_episodes <= 0:
        episodes_schedule = list(layouts)
    else:
        episodes_schedule = [layouts[i % len(layouts)] for i in range(args.n_episodes)]
    print(f"[hybrid-rollout] {len(layouts)} unique; running {len(episodes_schedule)} eps")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EquivariantCNNHybridPolicy(
        rgb_in_channels=3, grid_in_channels=int(ckpt.get("in_channels", 5)),
        num_actions=int(ckpt.get("num_actions", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"[hybrid-rollout] loaded {args.checkpoint}")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "results" / "hybrid_pathway" / f"run_{ts}"
    else:
        run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    all_safe, success_ids, failure_ids = [], [], []
    for ep_idx, layout in enumerate(episodes_schedule):
        ed = _run_episode(model, layout, ep_idx, seed=args.seed + ep_idx,
                          max_steps=args.max_steps, device=device, cell_px=args.cell_px)
        episode_dir = episodes_dir / f"episode_{ep_idx}"
        ed["frame_paths"] = _save_key_frames(ed["steps"], ed["key_frames"], episode_dir)
        gif_path = _save_gif(ed["steps"], episode_dir, args.gif_duration_ms)
        if gif_path:
            ed["frame_paths"]["trajectory_gif"] = gif_path
        episode_dir.mkdir(parents=True, exist_ok=True)
        with open(episode_dir / "episode_data.json", "w") as f:
            json.dump(_strip_rgb(ed), f, indent=2, default=str)
        if ed["success"]:
            success_ids.append(ep_idx)
        else:
            failure_ids.append(ep_idx)
        print(f"[hybrid-rollout] ep {ep_idx} ({layout['name']}): "
              f"steps={ed['total_steps']} success={ed['success']}")
        all_safe.append({
            "episode_id": ed["episode_id"], "maze_name": ed["maze_name"],
            "seed": ed["seed"], "total_steps": ed["total_steps"],
            "total_reward": ed["total_reward"], "success": ed["success"],
            "ascii_grid": ed["ascii_grid"], "dynamic_config": ed["dynamic_config"],
            "key_frames": ed["key_frames"], "frame_paths": ed.get("frame_paths", {}),
        })

    n_episodes_run = len(episodes_schedule)
    full_output = {
        "metadata": {
            "run_id": run_dir.name, "timestamp": datetime.now().isoformat(),
            "n_episodes": n_episodes_run, "n_unique_layouts": len(layouts),
            "seed_base": args.seed, "n_successes": len(success_ids),
            "n_failures": len(failure_ids), "phase_a_only": True,
            "model_type": "equivariant_cnn_hybrid",
            "checkpoint": args.checkpoint, "layouts_yaml": args.layouts,
        },
        "config": {
            "maze": {"name": _HOST_MAZE, "randomize_start": False,
                     "randomize_goal": False, "randomize_fire": False},
            "rollout": {"n_episodes": n_episodes_run, "seed": args.seed,
                        "checkpoint_path": args.checkpoint},
        },
        "phase_a": {
            "all_rollouts": all_safe,
            "success_episode_ids": success_ids,
            "failure_episode_ids": failure_ids,
        },
    }
    with open(run_dir / "full_output.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    cfg_used = {
        "maze": {"name": _HOST_MAZE},
        "rollout": {"n_episodes": n_episodes_run, "seed": args.seed,
                    "checkpoint_path": args.checkpoint},
        "tracking": {"output_dir": str(run_dir.parent)},
    }
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg_used, f, sort_keys=False)
    print(f"[hybrid-rollout] DONE successes={len(success_ids)}/{len(layouts)}")


if __name__ == "__main__":
    main()
