"""Baseline DAgger for the equivariant policy — no profile reasoning loop.

How this differs from the P4/P5/P6 active loop
==============================================
P4/P5/P6 follow this cycle:
  rollout 50 episodes -> LLM analyses every failure -> LLM prescribes
  layouts to record demos on -> BFS expert records them -> retrain.

Baseline DAgger doesn't ask any LLM. Instead, during the 50-episode test
rollout (run on test_layouts.yaml), every time the agent is about to
fail (hits fire OR runs out of steps), the BFS expert immediately
"completes" the episode in-place: starting from one step BEFORE the
failure step, BFS plans a fire-free path to the goal and walks it. The
recovered trajectory (states + actions) is appended as a corrective
demo to baseline_dagger/demos/. After every 4 corrective demos written,
the equivariant policy is retrained on (initial 10 demos UNION
accumulated corrective demos), and the loop continues.

The cycle terminates when held-out random success rate >= TARGET_SR
(default 0.90). There is no fixed round cap — by design.

Why "one step before the failure" rather than the fire-hit step itself?
The failure step terminates the episode in-state, so the agent never
got to make a recovery decision there. Backing up by one step gives
the demonstrator a real navigation choice from a still-alive state,
which is what the policy actually needs to learn from.

Invocation
----------
The expected caller is run_full_cycle.py; running this module directly
performs ONE rollout-and-correct + retrain pass on whatever the current
checkpoint is. run_full_cycle wraps it in the until-90% loop.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from configs.maze_layouts import MAZE_LAYOUTS
from envs.maze_env import (
    ACTION_DELTAS, ACTION_NAMES, MazeNavEnv,
    TILE_FREE, TILE_FIRE, TILE_GOAL,
)
from Equivariant_pathway.encoder import encode_from_env
from Equivariant_pathway.expert import AStarExpert, build_grid
from Equivariant_pathway.model import EquivariantUNetPolicy

_HOST_MAZE = "multimodal"
TRAIN_EVERY_N_DEMOS = 4


def _build_env_from_layout(layout: Dict, seed: int) -> MazeNavEnv:
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


def _bfs_path(grid: np.ndarray, start: Tuple[int, int],
              goal: Tuple[int, int]) -> List[int]:
    """BFS shortest fire-free action sequence (returns [] if no path)."""
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


def _save_corrective_demo(
    env: MazeNavEnv,
    obs_seq: List[Dict],
    actions: List[int],
    rewards: List[float],
    demo_dir: Path,
    layout_name: str,
    suffix: str,
) -> Path:
    """Write the corrective demo with the same schema CNN_pathway demos use,
    so both pathways' datasets can ingest it."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    payload = {
        "maze_name":      _HOST_MAZE,
        "timestamp":      timestamp,
        "layout_id":      f"{layout_name}_{suffix}",
        "source":         "baseline_dagger_correction",
        "start_pos":      [int(x) for x in env.start_pos],
        "goal_pos":       [int(x) for x in env.goal_pos],
        "fire_positions": [[int(x) for x in p] for p in env.fire_positions],
        "trajectory":     [[int(p[0]), int(p[1])] for p in env.get_trajectory()],
        "observations":   [o["state"].tolist() for o in obs_seq],
        "images":         [o["image"].tolist() for o in obs_seq],
        "actions":        [int(a) for a in actions],
        "rewards":        [float(r) for r in rewards],
        "total_reward":   float(sum(rewards)),
        "success":        bool(rewards[-1] > 5.0) if rewards else False,
    }
    fname = demo_dir / f"demo_{_HOST_MAZE}_{layout_name}_{suffix}_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(payload, f)
    return fname


def _load_model(checkpoint: Path, device: torch.device) -> EquivariantUNetPolicy:
    # weights_only=False — checkpoints contain non-tensor state (args, channels tuple).
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model = EquivariantUNetPolicy(
        in_channels=int(ckpt.get("in_channels", 5)),
        channels=tuple(ckpt.get("channels", (16, 32, 64))),
        num_actions=int(ckpt.get("num_actions", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def run_dagger_correction_pass(
    layouts_yaml: Path,
    checkpoint: Path,
    demo_dir: Path,
    out_dir: Path,
    seed: int = 0,
    max_steps: int = 60,
    train_every_n: int = TRAIN_EVERY_N_DEMOS,
    train_demo_paths: Optional[str] = None,
    epochs: int = 50,
    skip_train: bool = False,
) -> Dict:
    """One full rollout-and-correct pass.

    Returns a dict with: n_episodes, n_failures, n_corrected, n_demos_added,
    n_train_calls, success_rate, demo_paths (list[str]).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(layouts_yaml, "r") as f:
        spec = yaml.safe_load(f) or {}
    layouts = (
        spec.get("test_layouts")
        or spec.get("training_layouts")
        or spec.get("heldout_test_layouts")
        or spec.get("layouts")
        or []
    )
    if not layouts:
        raise SystemExit(f"No layouts in {layouts_yaml}")
    print(f"[dagger] {len(layouts)} layouts from {layouts_yaml}")
    print(f"[dagger] checkpoint={checkpoint}")
    print(f"[dagger] demo_dir  ={demo_dir}")
    print(f"[dagger] out_dir   ={out_dir}")
    print(f"[dagger] train every {train_every_n} corrective demos")

    model = _load_model(checkpoint, device)

    n_corrected_total = 0
    n_demos_pending_train = 0
    n_train_calls = 0
    saved_demos: List[Path] = []
    episode_log: List[Dict] = []
    success_count = 0

    for ep_idx, layout in enumerate(layouts):
        env = _build_env_from_layout(layout, seed=seed + ep_idx)
        obs_seq = [{"state": env._get_obs()["state"], "image": env._get_obs()["image"]}]
        actions: List[int] = []
        rewards: List[float] = []
        terminated = truncated = False
        si = 0

        # Run the policy until it fails or finishes the episode.
        while not (terminated or truncated) and si < max_steps:
            x = encode_from_env(env)
            x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
            with torch.no_grad():
                a = int(model(x_t).argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(a)
            obs_seq.append({"state": obs["state"], "image": obs["image"]})
            actions.append(a)
            rewards.append(float(reward))
            si += 1

        success = bool(rewards[-1] > 5.0) if rewards else False
        if success:
            success_count += 1

        # Detect "about to fail" — fire hit OR step-budget exhausted.
        # When success, no correction is needed.
        about_to_fail = not success
        corrected_here = False
        if about_to_fail:
            # Roll back ONE step so the demonstrator picks a recovery action
            # from a still-alive state (the fire-hit step is terminal).
            if len(env.get_trajectory()) >= 2:
                # Reset the env to the pre-failure pose.
                pre_fail_pos = tuple(env.get_trajectory()[-2])
                # Rebuild env on the same layout, then walk the policy's
                # actions one shy of the failure step so visited / step_count
                # match what the policy actually saw.
                env2 = _build_env_from_layout(layout, seed=seed + ep_idx)
                obs_seq2 = [{"state": env2._get_obs()["state"],
                             "image": env2._get_obs()["image"]}]
                actions2: List[int] = []
                rewards2: List[float] = []
                # Replay all but the last action.
                for a in actions[:-1]:
                    obs, r, term, trunc, _ = env2.step(a)
                    obs_seq2.append({"state": obs["state"], "image": obs["image"]})
                    actions2.append(int(a))
                    rewards2.append(float(r))
                # Now plan a BFS path from current pos to goal and execute.
                grid_now = env2.grid.copy()
                plan = _bfs_path(grid_now, tuple(env2.agent_pos), tuple(env2.goal_pos))
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
                        suffix=f"correct_ep{ep_idx}",
                    )
                    saved_demos.append(out_path)
                    n_corrected_total += 1
                    n_demos_pending_train += 1
                    corrected_here = True
                    print(f"[dagger] ep {ep_idx} ({layout.get('name')}): "
                          f"failed -> recovered with BFS plan len={len(plan)} "
                          f"-> saved {out_path.name}")
                else:
                    print(f"[dagger] ep {ep_idx} ({layout.get('name')}): "
                          f"failed AND BFS found no recovery path; skipping.")
                env2.close()
            else:
                print(f"[dagger] ep {ep_idx} ({layout.get('name')}): "
                      f"failed at step 0; cannot back up. skipping.")
        env.close()

        episode_log.append({
            "episode_id":     ep_idx,
            "layout_name":    layout.get("name"),
            "policy_success": success,
            "corrected":      corrected_here,
            "policy_steps":   si,
        })

        # Retrain after every N corrective demos.
        if (not skip_train) and n_demos_pending_train >= train_every_n:
            print(f"[dagger] retraining after {n_demos_pending_train} corrective demos "
                  f"(total corrected so far: {n_corrected_total})")
            cmd = [
                sys.executable, "-u", str(REPO_ROOT / "Equivariant_pathway" / "train.py"),
                "--checkpoint_dir", str(checkpoint.parent),
                "--epochs", str(epochs),
                "--resume",
            ]
            if train_demo_paths:
                cmd += ["--demo_paths", train_demo_paths]
            else:
                cmd += ["--demo_dir", str(demo_dir.parent / "demos")]
            print(f"[dagger] $ {' '.join(cmd)}")
            rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
            if rc != 0:
                print(f"[dagger] WARNING: retrain exited rc={rc}")
            n_train_calls += 1
            n_demos_pending_train = 0
            # Reload policy weights into the eval model so subsequent rollouts
            # in this same pass use the freshly-trained policy.
            best_path = checkpoint.parent / "best_eq_policy.pth"
            reload_path = best_path if best_path.exists() else checkpoint
            model = _load_model(reload_path, device)

    summary = {
        "layouts_yaml":     str(layouts_yaml),
        "n_episodes":       len(layouts),
        "n_successes":      success_count,
        "n_failures":       len(layouts) - success_count,
        "success_rate":     success_count / len(layouts) if layouts else 0.0,
        "n_corrected":      n_corrected_total,
        "n_demos_added":    len(saved_demos),
        "n_train_calls":    n_train_calls,
        "demo_paths":       [str(p) for p in saved_demos],
        "episode_log":      episode_log,
        "checkpoint":       str(checkpoint),
        "demo_dir":         str(demo_dir),
    }
    with open(out_dir / "dagger_pass_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[dagger] PASS DONE  policy_sr={summary['success_rate']:.2f}  "
          f"corrected={n_corrected_total}  trains_in_pass={n_train_calls}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "test_layouts.yaml"))
    p.add_argument("--checkpoint", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "checkpoints" / "best_eq_policy.pth"))
    p.add_argument("--demo_dir", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "profiles" / "baseline_dagger" / "demos"))
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--train_every_n", type=int, default=TRAIN_EVERY_N_DEMOS)
    p.add_argument("--demo_paths", type=str, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--skip_train", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.demo_dir).parent / "passes" / datetime.now().strftime("pass_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dagger_correction_pass(
        layouts_yaml=Path(args.layouts),
        checkpoint=Path(args.checkpoint),
        demo_dir=Path(args.demo_dir),
        out_dir=out_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        train_every_n=args.train_every_n,
        train_demo_paths=args.demo_paths,
        epochs=args.epochs,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
