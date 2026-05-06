"""Roll out the trained equivariant policy on a layout YAML.

Mirrors CNN_pathway/rollout_test.py: each layout is rolled out exactly once,
trajectories are written into a run dir in the SAME shape pipeline.pipeline_runner
expects (full_output.json + per-episode frames + episode_data.json), so
analyze_p4/p5/p6.py can consume the run with no custom adapter.

The differences vs CNN_pathway are:
  - Model is EquivariantUNetPolicy fed with a (5, H, W) semantic encoding,
    not the (80,80,3) RGB image + 14-d state vector.
  - We additionally save per-step Q-maps (per-cell action logits) under
    each episode so downstream tooling can visualise the equivariance
    pattern. This is cheap and matches dmnpol/evaluation/rollout.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
from Equivariant_pathway.encoder import encode_from_env
from Equivariant_pathway.model import EquivariantUNetPolicy

_HOST_MAZE = "multimodal"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "checkpoints" / "best_eq_policy.pth"))
    p.add_argument("--layouts", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "test_layouts.yaml"),
                   help="YAML with `test_layouts` / `training_layouts` / `layouts`.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Run directory. Default: results/equivariant_pathway/run_<timestamp>.")
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif_duration_ms", type=int, default=250)
    p.add_argument("--save_q_maps", action="store_true",
                   help="Save per-step Q-maps (cell -> action logits) under each "
                        "episode dir. Useful for visualising equivariance; off "
                        "by default to keep run dirs small.")
    return p.parse_args()


def _build_env(layout: Dict, seed: int) -> MazeNavEnv:
    """Same forced-layout construction CNN_pathway/rollout_test.py uses."""
    MAZE_LAYOUTS[_HOST_MAZE]["grid"]  = layout["grid"]
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


def _save_gif(steps: List[Dict], episode_dir: Path, duration_ms: int) -> str:
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


def _save_key_frames(steps: List[Dict], key_frames: List[Dict],
                     episode_dir: Path) -> Dict[str, str]:
    paths = {}
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    role_to_file = {
        "start_frame":        "start.png",
        "highest_loss_frame": "high_loss.png",
        "end_frame":          "end.png",
    }
    for kf in key_frames:
        rgb = steps[kf["step_idx"]].get("rgb")
        if rgb is None:
            continue
        out = frames_dir / role_to_file[kf["role"]]
        Image.fromarray(np.asarray(rgb)).save(str(out))
        paths[kf["role"]] = str(out)
    return paths


def _strip_rgb_and_qmaps(episode: Dict) -> Dict:
    """Drop the heaviest fields (rgb arrays + per-step Q-maps) from the
    episode dict before writing episode_data.json — they're already saved
    as PNG / NPZ on disk."""
    safe = []
    for s in episode["steps"]:
        clean = {k: v for k, v in s.items() if k not in ("rgb", "q_map")}
        safe.append(clean)
    return {**episode, "steps": safe}


def _pick_high_loss_step(steps: List[Dict]) -> int:
    """Same heuristic CNN_pathway uses: prefer an intermediate step whose
    pose differs from start / end and has the lowest reward; fall back to
    the middle index for very short episodes."""
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
    best_idx, best_key = -1, None
    for i in range(1, n - 1):
        pos = _pos(i)
        differs = (pos is not None) and (pos != start_pos) and (pos != end_pos)
        rwd = float(steps[i].get("reward") or 0.0)
        key = (1 if differs else 0, -rwd)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = i
    return best_idx if best_idx >= 0 else (n - 1) // 2


def _run_episode(model: EquivariantUNetPolicy, layout: Dict, episode_id: int,
                 seed: int, max_steps: int, device: torch.device,
                 save_q_maps: bool) -> Dict:
    env = _build_env(layout, seed=seed)
    ascii_grid = env.get_grid_image_description()
    dyn_cfg    = env.get_dynamic_config()

    obs = env._get_obs()
    rgb0 = env.render()
    init_x = encode_from_env(env)
    init_q = None
    if save_q_maps:
        with torch.no_grad():
            init_q = model.per_cell_logits(
                torch.from_numpy(init_x).float().unsqueeze(0).to(device)
            ).squeeze(0).cpu().numpy()

    steps: List[Dict] = [{
        "step_idx":    0,
        "obs":         obs["state"].tolist(),
        "reward":      0.0,
        "action":      None,
        "action_name": "RESET",
        "info":        _serialise_info(env._get_info()),
        "rgb":         rgb0,
        "q_map":       init_q,
    }]

    si = 0
    terminated = False
    truncated  = False
    while not (terminated or truncated) and si < max_steps:
        x = encode_from_env(env)
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x_t)
            q_map = model.per_cell_logits(x_t).squeeze(0).cpu().numpy() if save_q_maps else None
        action = int(logits.argmax(dim=-1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        si += 1
        steps.append({
            "step_idx":    si,
            "obs":         obs["state"].tolist(),
            "reward":      float(reward),
            "action":      action,
            "action_name": ACTION_NAMES.get(action, "?"),
            "info":        _serialise_info(info),
            "rgb":         env.render(),
            "q_map":       q_map,
        })

    success = bool(steps[-1]["info"].get("success", False))
    total_reward = sum(s["reward"] for s in steps)
    sl = _pick_high_loss_step(steps)
    key_frames = [
        {"role": "start_frame",        "step_idx": 0},
        {"role": "highest_loss_frame", "step_idx": sl},
        {"role": "end_frame",          "step_idx": len(steps) - 1},
    ]
    env.close()
    return {
        "episode_id":      episode_id,
        "maze_name":       layout["name"],
        "seed":            seed,
        "total_steps":     si,
        "total_reward":    total_reward,
        "success":         success,
        "ascii_grid":      ascii_grid,
        "dynamic_config":  dyn_cfg,
        "steps":           steps,
        "key_frames":      key_frames,
    }


def _save_q_maps(steps: List[Dict], episode_dir: Path):
    qs = []
    for s in steps:
        q = s.get("q_map")
        if q is not None:
            qs.append(np.asarray(q))
    if not qs:
        return
    out = episode_dir / "q_maps.npz"
    np.savez_compressed(out, q=np.stack(qs, axis=0))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.layouts, "r") as f:
        spec = yaml.safe_load(f) or {}
    layouts = (
        spec.get("test_layouts")
        or spec.get("training_layouts")
        or spec.get("heldout_test_layouts")
        or spec.get("layouts")
        or []
    )
    if not layouts:
        raise SystemExit(f"No layouts list in {args.layouts}")
    print(f"[eq-rollout] {len(layouts)} layouts from {args.layouts}")

    # weights_only=False is correct here: our checkpoints contain non-tensor
    # state (the args dict + the channels tuple). Setting it explicitly also
    # silences the FutureWarning torch will emit on the next major version.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    channels = tuple(ckpt.get("channels", (16, 32, 64)))
    model = EquivariantUNetPolicy(
        in_channels=int(ckpt.get("in_channels", 5)),
        channels=channels,
        num_actions=int(ckpt.get("num_actions", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"[eq-rollout] loaded {args.checkpoint} (val_loss={ckpt.get('val_loss','?')})")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "results" / "equivariant_pathway" / f"run_{ts}"
    else:
        run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    all_safe = []
    success_ids: List[int] = []
    failure_ids: List[int] = []

    for ep_idx, layout in enumerate(layouts):
        ed = _run_episode(model, layout, ep_idx, seed=args.seed + ep_idx,
                          max_steps=args.max_steps, device=device,
                          save_q_maps=args.save_q_maps)
        episode_dir = episodes_dir / f"episode_{ep_idx}"
        ed["frame_paths"] = _save_key_frames(ed["steps"], ed["key_frames"], episode_dir)
        gif_path = _save_gif(ed["steps"], episode_dir, args.gif_duration_ms)
        if gif_path:
            ed["frame_paths"]["trajectory_gif"] = gif_path
        if args.save_q_maps:
            _save_q_maps(ed["steps"], episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        with open(episode_dir / "episode_data.json", "w") as f:
            json.dump(_strip_rgb_and_qmaps(ed), f, indent=2, default=str)
        if ed["success"]:
            success_ids.append(ep_idx)
        else:
            failure_ids.append(ep_idx)
        print(f"[eq-rollout] ep {ep_idx} ({layout['name']}): "
              f"steps={ed['total_steps']} reward={ed['total_reward']:.2f} "
              f"success={ed['success']}")
        all_safe.append({
            "episode_id":     ed["episode_id"],
            "maze_name":      ed["maze_name"],
            "seed":           ed["seed"],
            "total_steps":    ed["total_steps"],
            "total_reward":   ed["total_reward"],
            "success":        ed["success"],
            "ascii_grid":     ed["ascii_grid"],
            "dynamic_config": ed["dynamic_config"],
            "key_frames":     ed["key_frames"],
            "frame_paths":    ed.get("frame_paths", {}),
        })

    full_output = {
        "metadata": {
            "run_id":       run_dir.name,
            "timestamp":    datetime.now().isoformat(),
            "n_episodes":   len(layouts),
            "seed_base":    args.seed,
            "n_successes":  len(success_ids),
            "n_failures":   len(failure_ids),
            "phase_a_only": True,
            "model_type":   "equivariant_unet",
            "checkpoint":   args.checkpoint,
            "layouts_yaml": args.layouts,
        },
        "config": {
            "maze":    {"name": _HOST_MAZE,
                        "randomize_start": False,
                        "randomize_goal":  False,
                        "randomize_fire":  False},
            "rollout": {"n_episodes": len(layouts),
                        "seed": args.seed,
                        "checkpoint_path": args.checkpoint},
        },
        "phase_a": {
            "all_rollouts":         all_safe,
            "success_episode_ids":  success_ids,
            "failure_episode_ids":  failure_ids,
        },
    }
    with open(run_dir / "full_output.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)

    cfg_used = {
        "maze":     {"name": _HOST_MAZE},
        "rollout":  {"n_episodes": len(layouts),
                     "seed": args.seed,
                     "checkpoint_path": args.checkpoint},
        "tracking": {"output_dir": str(run_dir.parent)},
    }
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg_used, f, sort_keys=False)

    print(f"[eq-rollout] DONE  successes={len(success_ids)}/{len(layouts)}  failures={failure_ids}")
    print(f"[eq-rollout] artefacts -> {run_dir}")
    print(f"[eq-rollout] next: python -m Equivariant_pathway.analyze_p4 --rollout_dir \"{run_dir}\"")


if __name__ == "__main__":
    main()
