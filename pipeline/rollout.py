import os
import sys
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.maze_env import MazeNavEnv, ACTION_NAMES
from model.diffusion_policy import MazeDiffusionPolicy


_DEFAULT_META = {
    "obs_dim":        14,
    "action_dim":     4,
    "obs_horizon":    4,
    "pred_horizon":   3,
    "num_diff_steps": 200,
    "dim":            64,
    "dim_mults":      [1, 2, 4],
    "grid_size":      5,
    "cell_px":        16,
    "img_size":       80,
    "use_vision":     True,
}


def _load_meta(checkpoint_path: str) -> Dict:
    ckpt_dir = os.path.dirname(checkpoint_path) or "checkpoints"
    meta_path = os.path.join(ckpt_dir, "best_model_meta.json")
    meta = dict(_DEFAULT_META)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta.update(json.load(f))
    return meta


def _build_policy(meta: Dict, device: torch.device) -> MazeDiffusionPolicy:
    policy = MazeDiffusionPolicy(
        obs_dim=int(meta.get("obs_dim", 14)),
        action_dim=int(meta.get("action_dim", 4)),
        obs_horizon=int(meta.get("obs_horizon", 4)),
        pred_horizon=int(meta.get("pred_horizon", 3)),
        num_diffusion_steps=int(meta.get("num_diff_steps", 200)),
        device=str(device),
        use_vision=bool(meta.get("use_vision", True)),
        dim=int(meta.get("dim", 64)),
        dim_mults=tuple(meta.get("dim_mults", [1, 2, 4])),
        grid_size=int(meta.get("grid_size", 5)),
        cell_px=int(meta.get("cell_px", 16)),
        img_size=int(meta.get("img_size", 80)),
    )
    return policy


def _load_checkpoint(policy: MazeDiffusionPolicy, primary: str, fallback: str, device: torch.device) -> str:
    path = primary if os.path.exists(primary) else fallback
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {primary} or {fallback}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "ema_policy" in ckpt:
        policy.model.load_state_dict(ckpt["ema_policy"], strict=True)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        policy.model.load_state_dict(ckpt["model_state_dict"], strict=True)
    elif isinstance(ckpt, dict) and "policy" in ckpt:
        policy.model.load_state_dict(ckpt["policy"], strict=True)
    else:
        policy.model.load_state_dict(ckpt, strict=True)
    policy.model.eval()
    return path


def _serialise_info(info: Dict) -> Dict:
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


def _pick_high_loss_step(rewards: List[float]) -> int:
    if len(rewards) <= 2:
        return max(0, len(rewards) // 2)
    s0, se = 0, len(rewards) - 1
    sl = int(np.argmin(rewards))
    if sl == s0 or sl == se:
        for c in np.argsort(rewards):
            ci = int(c)
            if ci != s0 and ci != se:
                sl = ci
                break
    return sl


def _save_key_frames(steps: List[Dict], key_frames: List[Dict], episode_dir: Path) -> Dict[str, str]:
    frame_paths = {}
    episode_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    role_to_file = {
        "start_frame":        "start.png",
        "highest_loss_frame": "high_loss.png",
        "end_frame":          "end.png",
    }
    for kf in key_frames:
        role = kf["role"]
        idx = kf["step_idx"]
        rgb = steps[idx]["rgb"]
        if rgb is None:
            continue
        filename = role_to_file.get(role, f"{role}.png")
        p = frames_dir / filename
        Image.fromarray(np.asarray(rgb)).save(str(p))
        frame_paths[role] = str(p)
    return frame_paths


def _run_single_episode(
    policy: MazeDiffusionPolicy,
    maze_cfg: Dict,
    seed: int,
    episode_id: int,
    obs_horizon: int,
    use_vision: bool,
) -> Dict:
    env = MazeNavEnv(
        maze_name=maze_cfg["name"],
        render_mode="rgb_array",
        randomize_start=bool(maze_cfg.get("randomize_start", True)),
        randomize_goal=bool(maze_cfg.get("randomize_goal", True)),
        randomize_fire=bool(maze_cfg.get("randomize_fire", True)),
        num_fire_tiles=int(maze_cfg.get("num_fire_tiles", 3)),
        seed=seed,
    )
    obs, info = env.reset()
    ascii_grid = env.get_grid_image_description()
    dyn_cfg = env.get_dynamic_config()

    steps: List[Dict] = []
    rgb0 = env.render()
    steps.append({
        "step_idx":   0,
        "obs":        obs["state"].tolist(),
        "reward":     0.0,
        "action":     None,
        "action_name":"RESET",
        "info":       _serialise_info(info),
        "rgb":        rgb0,
    })

    state_deque = deque([obs["state"]] * obs_horizon, maxlen=obs_horizon)
    image_deque = deque([obs["image"]] * obs_horizon, maxlen=obs_horizon)

    terminated = False
    truncated = False
    si = 0
    while not (terminated or truncated):
        img_stack = np.stack(list(image_deque), axis=0).astype(np.float32) / 255.0
        img_stack = np.transpose(img_stack, (0, 3, 1, 2))
        obs_dict = {
            "state": np.stack(list(state_deque), axis=0).astype(np.float32),
            "image": img_stack,
        }
        action = policy.get_action(obs_dict) if use_vision else policy.get_action({"state": obs_dict["state"]})

        obs, reward, terminated, truncated, info = env.step(int(action))
        state_deque.append(obs["state"])
        image_deque.append(obs["image"])
        si += 1
        steps.append({
            "step_idx":   si,
            "obs":        obs["state"].tolist(),
            "reward":     float(reward),
            "action":     int(action),
            "action_name":ACTION_NAMES.get(int(action), "?"),
            "info":       _serialise_info(info),
            "rgb":        env.render(),
        })

    success = bool(steps[-1]["info"].get("success", False))
    total_reward = sum(s["reward"] for s in steps)

    rewards = [s["reward"] for s in steps]
    sl = _pick_high_loss_step(rewards)
    key_frames = [
        {"role": "start_frame",        "step_idx": 0},
        {"role": "highest_loss_frame", "step_idx": sl},
        {"role": "end_frame",          "step_idx": len(steps) - 1},
    ]

    env.close()

    return {
        "episode_id":      episode_id,
        "maze_name":       maze_cfg["name"],
        "seed":            seed,
        "total_steps":     si,
        "total_reward":    total_reward,
        "success":         success,
        "ascii_grid":      ascii_grid,
        "dynamic_config":  dyn_cfg,
        "steps":           steps,
        "key_frames":      key_frames,
    }


def _strip_rgb_for_json(episode: Dict) -> Dict:
    safe_steps = []
    for s in episode["steps"]:
        safe_steps.append({k: v for k, v in s.items() if k != "rgb"})
    return {
        "episode_id":     episode["episode_id"],
        "maze_name":      episode["maze_name"],
        "seed":           episode["seed"],
        "total_steps":    episode["total_steps"],
        "total_reward":   episode["total_reward"],
        "success":        episode["success"],
        "ascii_grid":     episode["ascii_grid"],
        "dynamic_config": episode["dynamic_config"],
        "steps":          safe_steps,
        "key_frames":     episode["key_frames"],
        "frame_paths":    episode.get("frame_paths", {}),
    }


def run_rollouts(config: Dict, run_dir: Path) -> Dict:
    """Phase A entry point. Runs all episodes and saves per-episode artefacts into run_dir."""
    maze_cfg   = config["maze"]
    rollout_cfg= config["rollout"]
    track_cfg  = config.get("tracking", {})

    n_episodes = int(rollout_cfg.get("n_episodes", 10))
    base_seed  = int(rollout_cfg.get("seed", 42))
    ckpt_primary  = str(rollout_cfg.get("checkpoint_path", "checkpoints/best_model_ema.pth"))
    ckpt_fallback = str(rollout_cfg.get("fallback_checkpoint_path", "checkpoints/best_model.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = _load_meta(ckpt_primary)
    policy = _build_policy(meta, device)
    loaded_path = _load_checkpoint(policy, ckpt_primary, ckpt_fallback, device)
    print(f"[Rollout] Checkpoint loaded: {loaded_path}")
    print(f"[Rollout] Meta: obs_horizon={meta.get('obs_horizon')} pred_horizon={meta.get('pred_horizon')} use_vision={meta.get('use_vision')}")

    obs_horizon = int(meta.get("obs_horizon", 4))
    use_vision  = bool(meta.get("use_vision", True))

    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    all_episodes: List[Dict] = []
    for ep_idx in range(n_episodes):
        seed = base_seed 
        print(f"\n[Rollout] Episode {ep_idx+1}/{n_episodes}  seed={seed}")
        ed = _run_single_episode(
            policy=policy,
            maze_cfg=maze_cfg,
            seed=seed,
            episode_id=ep_idx,
            obs_horizon=obs_horizon,
            use_vision=use_vision,
        )
        print(f"[Rollout]  → steps={ed['total_steps']} reward={ed['total_reward']:.2f} success={ed['success']}")

        episode_dir = episodes_dir / f"episode_{ep_idx}"
        if bool(track_cfg.get("save_frames", True)):
            ed["frame_paths"] = _save_key_frames(ed["steps"], ed["key_frames"], episode_dir)
        else:
            ed["frame_paths"] = {}

        if bool(track_cfg.get("save_per_episode_json", True)):
            episode_dir.mkdir(parents=True, exist_ok=True)
            with open(episode_dir / "episode_data.json", "w") as f:
                json.dump(_strip_rgb_for_json(ed), f, indent=2, default=str)

        all_episodes.append(ed)

    successes = [e for e in all_episodes if e["success"]]
    failures  = [e for e in all_episodes if not e["success"]]
    print(f"\n[Rollout] Summary: {len(successes)}/{len(all_episodes)} succeeded, {len(failures)} failed")

    return {
        "meta":         meta,
        "n_episodes":   n_episodes,
        "seed_base":    base_seed,
        "all_episodes": all_episodes,
        "success_episode_ids": [e["episode_id"] for e in successes],
        "failure_episode_ids": [e["episode_id"] for e in failures],
    }