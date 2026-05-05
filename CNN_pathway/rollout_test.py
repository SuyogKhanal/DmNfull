"""Roll the trained CNN+MLP policy out on the held-out test layouts.

Each test layout is rolled out exactly once. The resulting trajectories are
written into a run directory in the same shape `pipeline.pipeline_runner`
expects (full_output.json + per-episode frames + episode_data.json), so the
existing P4 LLM analysis pipeline can consume them directly without any
custom adapter — see `analyze_p4.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from configs.maze_layouts import MAZE_LAYOUTS
from envs.maze_env import ACTION_NAMES, MazeNavEnv, TILE_FREE, TILE_GOAL
from CNN_pathway.model import CNNMLPPolicy

_HOST_MAZE = "multimodal"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "checkpoints" / "best_cnn_mlp.pth"))
    p.add_argument("--layouts", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "test_layouts.yaml"),
                   help="YAML with a top-level `training_layouts` or `test_layouts` "
                        "list. Either is accepted, so the same script can evaluate the "
                        "policy on training-distribution layouts as well as the held-out "
                        "test layouts.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output run directory. Defaults to "
                        "results/cnn_pathway/run_<timestamp>.")
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _build_env(layout: Dict, seed: int) -> MazeNavEnv:
    MAZE_LAYOUTS[_HOST_MAZE]["grid"] = layout["grid"]
    MAZE_LAYOUTS[_HOST_MAZE]["start"] = list(layout["start_pos"])
    MAZE_LAYOUTS[_HOST_MAZE]["goal"] = list(layout["goal_pos"])
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


def _save_key_frames(steps: List[Dict], key_frames: List[Dict], episode_dir: Path) -> Dict[str, str]:
    frame_paths = {}
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    role_to_file = {
        "start_frame": "start.png",
        "highest_loss_frame": "high_loss.png",
        "end_frame": "end.png",
    }
    saved_arrays: Dict[str, np.ndarray] = {}
    for kf in key_frames:
        rgb = steps[kf["step_idx"]]["rgb"]
        if rgb is None:
            continue
        p = frames_dir / role_to_file[kf["role"]]
        arr = np.asarray(rgb)
        Image.fromarray(arr).save(str(p))
        frame_paths[kf["role"]] = str(p)
        saved_arrays[kf["role"]] = arr
    # Diagnostic: when the agent never moves the rendered frames are
    # visually identical at every step. Log it so the VLM downstream
    # isn't misled into thinking three different states were captured.
    pairs = [
        ("start_frame", "highest_loss_frame"),
        ("highest_loss_frame", "end_frame"),
        ("start_frame", "end_frame"),
    ]
    for a, b in pairs:
        if a in saved_arrays and b in saved_arrays:
            if saved_arrays[a].shape == saved_arrays[b].shape and \
               np.array_equal(saved_arrays[a], saved_arrays[b]):
                print(f"[cnn-rollout] WARNING: {a} == {b} (agent did not produce a "
                      f"distinguishable frame for this episode).")
    return frame_paths


def _strip_rgb(episode: Dict) -> Dict:
    safe_steps = [{k: v for k, v in s.items() if k != "rgb"} for s in episode["steps"]]
    out = {**episode, "steps": safe_steps}
    return out


def _pick_high_loss_step(steps: List[Dict]) -> int:
    """Pick a representative 'highest-loss' step for VLM analysis.

    Preference order:
      1. An intermediate step whose agent_pos differs from BOTH the start and
         end positions — gives a visually distinct frame from start/end. Tie
         broken by lowest reward (so wall-hits / fire-encounters win).
      2. Otherwise the lowest-reward intermediate step (still meaningful — it
         pinpoints the worst transition even if the rendered frame happens to
         match start/end).
      3. For very short episodes (<= 2 steps), the middle index.

    The previous implementation looked at rewards alone, which produced
    visually identical start / high-loss / end frames whenever the agent
    failed by repeatedly bumping into a wall and never moving (e.g. the
    bottom-right-start test layout when the policy hadn't generalised).
    """
    n = len(steps)
    if n <= 2:
        return max(0, n // 2)

    def _pos(i: int):
        info = steps[i].get("info") or {}
        ap = info.get("agent_pos")
        if ap is None:
            return None
        try:
            return (int(ap[0]), int(ap[1]))
        except (TypeError, ValueError, IndexError):
            return None

    start_pos = _pos(0)
    end_pos   = _pos(n - 1)

    best_idx: int = -1
    best_key = None  # (differs_from_endpoints, -reward)
    for i in range(1, n - 1):
        pos = _pos(i)
        differs = (pos is not None) and (pos != start_pos) and (pos != end_pos)
        rwd = float(steps[i].get("reward") or 0.0)
        key = (1 if differs else 0, -rwd)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = i

    if best_idx < 0:
        return (n - 1) // 2
    return best_idx


def _run_episode(model: CNNMLPPolicy, layout: Dict, episode_id: int,
                 seed: int, max_steps: int, device: torch.device) -> Dict:
    env = _build_env(layout, seed=seed)
    ascii_grid = env.get_grid_image_description()
    dyn_cfg = env.get_dynamic_config()

    obs = env._get_obs()
    rgb0 = env.render()
    steps: List[Dict] = [{
        "step_idx": 0,
        "obs": obs["state"].tolist(),
        "reward": 0.0,
        "action": None,
        "action_name": "RESET",
        "info": _serialise_info(env._get_info()),
        "rgb": rgb0,
    }]

    si = 0
    terminated = False
    truncated = False
    while not (terminated or truncated) and si < max_steps:
        img_t = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device) / 255.0
        state_t = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_t, state_t)
        action = int(logits.argmax(dim=-1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        si += 1
        steps.append({
            "step_idx": si,
            "obs": obs["state"].tolist(),
            "reward": float(reward),
            "action": action,
            "action_name": ACTION_NAMES.get(action, "?"),
            "info": _serialise_info(info),
            "rgb": env.render(),
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
        "episode_id": episode_id,
        "maze_name": layout["name"],
        "seed": seed,
        "total_steps": si,
        "total_reward": total_reward,
        "success": success,
        "ascii_grid": ascii_grid,
        "dynamic_config": dyn_cfg,
        "steps": steps,
        "key_frames": key_frames,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.layouts, "r") as f:
        spec = yaml.safe_load(f) or {}
    layouts = (
        spec.get("test_layouts")
        or spec.get("training_layouts")
        or spec.get("layouts")
        or []
    )
    if not layouts:
        raise SystemExit(
            f"no `test_layouts` / `training_layouts` / `layouts` list in {args.layouts}"
        )
    layout_kind = (
        "test_layouts" if "test_layouts" in spec
        else "training_layouts" if "training_layouts" in spec
        else "layouts"
    )
    print(f"[cnn-rollout] {layout_kind}: {len(layouts)} layouts from {args.layouts}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = CNNMLPPolicy(
        img_size=int(ckpt.get("img_size", 80)),
        grid_size=int(ckpt.get("grid_size", 5)),
        cell_px=int(ckpt.get("cell_px", 16)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"[cnn-rollout] loaded checkpoint {args.checkpoint} (val_loss={ckpt.get('val_loss','?')})")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "results" / "cnn_pathway" / f"run_{ts}"
    else:
        run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    all_episodes_safe = []
    success_ids: List[int] = []
    failure_ids: List[int] = []

    for ep_idx, layout in enumerate(layouts):
        ed = _run_episode(model, layout, ep_idx, seed=args.seed + ep_idx,
                          max_steps=args.max_steps, device=device)
        episode_dir = episodes_dir / f"episode_{ep_idx}"
        ed["frame_paths"] = _save_key_frames(ed["steps"], ed["key_frames"], episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        with open(episode_dir / "episode_data.json", "w") as f:
            json.dump(_strip_rgb(ed), f, indent=2, default=str)
        if ed["success"]:
            success_ids.append(ep_idx)
        else:
            failure_ids.append(ep_idx)
        print(f"[cnn-rollout] ep {ep_idx} ({layout['name']}): "
              f"steps={ed['total_steps']} reward={ed['total_reward']:.2f} "
              f"success={ed['success']}")
        all_episodes_safe.append({
            "episode_id": ed["episode_id"],
            "maze_name": ed["maze_name"],
            "seed": ed["seed"],
            "total_steps": ed["total_steps"],
            "total_reward": ed["total_reward"],
            "success": ed["success"],
            "ascii_grid": ed["ascii_grid"],
            "dynamic_config": ed["dynamic_config"],
            "key_frames": ed["key_frames"],
            "frame_paths": ed.get("frame_paths", {}),
        })

    full_output = {
        "metadata": {
            "run_id": run_dir.name,
            "timestamp": datetime.now().isoformat(),
            "n_episodes": len(layouts),
            "seed_base": args.seed,
            "n_successes": len(success_ids),
            "n_failures": len(failure_ids),
            "phase_a_only": True,
            "model_type": "cnn_mlp",
            "checkpoint": args.checkpoint,
        },
        "config": {
            "maze": {"name": _HOST_MAZE, "randomize_start": False,
                     "randomize_goal": False, "randomize_fire": False},
            "rollout": {"n_episodes": len(layouts), "seed": args.seed,
                        "checkpoint_path": args.checkpoint},
        },
        "phase_a": {
            "all_rollouts": all_episodes_safe,
            "success_episode_ids": success_ids,
            "failure_episode_ids": failure_ids,
        },
    }
    with open(run_dir / "full_output.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)

    cfg_used = {
        "maze": {"name": _HOST_MAZE},
        "rollout": {"n_episodes": len(layouts), "seed": args.seed,
                    "checkpoint_path": args.checkpoint},
        "tracking": {"output_dir": str(run_dir.parent)},
    }
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg_used, f, sort_keys=False)

    print(f"[cnn-rollout] DONE  successes={len(success_ids)}/{len(layouts)}  "
          f"failures={failure_ids}")
    print(f"[cnn-rollout] artefacts → {run_dir}")
    print(f"[cnn-rollout] next: python -m CNN_pathway.analyze_p4 --rollout_dir \"{run_dir}\"")


if __name__ == "__main__":
    main()
