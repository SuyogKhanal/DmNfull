"""Baseline DAgger for the equivariant policy — no profile reasoning loop.

How this differs from the P4/P5/P6 active loop
==============================================
P4/P5/P6 follow this cycle:
  rollout on test_layouts.yaml -> LLM analyses every failure -> LLM
  prescribes layouts to record demos on -> BFS expert records them
  -> retrain. One round = one such cycle; eval on heldout at end.

Baseline DAgger doesn't ask any LLM. It implements the classical
DAgger correction loop:

  1. Train the model on the current demo set (the initial 10, plus
     every corrective demo collected so far).
  2. Run 50 rollouts (round-robin through `layouts_yaml`, rotating
     seeds) using the freshly-trained policy.
  3. For each FAILED episode (fire-collision OR step-budget timeout):
       - Identify the *deviation point*: the first step where the
         policy chose an action that is NOT in the BFS-optimal set
         for that state. For fire collisions this is typically the
         step that walked into the fire; for timeouts it can be much
         earlier (the first wrong-direction step that doomed the run).
       - Replay the policy's actions up to (but not including) the
         deviation step in a fresh env so we capture the policy
         prefix verbatim.
       - From the pre-deviation state, BFS-plan a fire-free path to
         the goal and execute it — the expert "completes" the
         trajectory.
       - Save the FULL resulting trajectory (start_pos -> ...
         policy actions ... -> deviation_pos -> ... expert actions
         ... -> goal) as a corrective demo.
  4. After every 4 corrective demos accumulated *globally* (i.e.
     across all DAgger passes; counted by listing files in the demo
     dir), retrain the policy on (initial demos UNION every
     corrective demo) and reload the model so subsequent rollouts in
     the same pass use the freshly-trained weights. This is the key
     correctness fix vs. the previous version, which used a per-pass
     counter and therefore never retrained when a pass collected
     fewer than 4 corrections (the common case on a small layouts
     yaml).
  5. After the 50 rollouts, return the pass summary; the caller
     (run_full_cycle.py) evaluates on the held-out 50-layout set and
     decides whether to terminate (heldout SR >= 0.90) or run another
     pass.

Invocation
----------
The expected caller is run_full_cycle.py; running this module
directly performs ONE 50-episode rollout-and-correct pass on whatever
the current checkpoint is. The pass-level retraining (every 4
corrections) is internal to run_dagger_correction_pass.
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


def _find_deviation_step(
    layout: Dict,
    actions: List[int],
    trajectory: List[Tuple[int, int]],
) -> int:
    """Index of the first action that is NOT in the BFS-optimal set.

    `trajectory[t]` is the agent's position BEFORE `actions[t]` is taken.
    For fire collisions this typically pinpoints the step that walked
    into the fire. For timeouts it can be far earlier — e.g. the policy
    walked LEFT at step 0 when only RIGHT and DOWN were optimal. Either
    way, this is the position from which the expert should take over.

    Returns -1 if every recorded action was optimal (which shouldn't
    happen on a deterministic env if the episode failed; defensive).
    """
    fires = [tuple(f) for f in layout["fire_positions"]]
    goal = tuple(layout["goal_pos"])
    gs = len(layout.get("grid", [[0] * 5] * 5))
    grid = build_grid(grid_size=gs, fire_positions=fires, goal_pos=goal)
    expert = AStarExpert(grid, goal)
    for t, a in enumerate(actions):
        if t >= len(trajectory):
            break
        pos = trajectory[t]
        opt = expert.optimal_actions(pos)
        if opt.sum() == 0:
            # already at goal or unreachable; treat as deviation here
            return t
        if opt[a] < 0.5:
            return t
    return -1


def _count_corrective_demos(demo_dir: Path) -> int:
    """Return how many corrective demos have been collected so far.

    The cumulative count is what drives `train every 4 corrections`. We
    use the demo dir's file count rather than an in-memory counter so
    the state survives across passes (each pass starts from whatever
    cumulative count the previous pass left behind).
    """
    if not demo_dir.exists():
        return 0
    return sum(1 for p in demo_dir.rglob("*.json") if "_correct_" in p.name)


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
    n_episodes: int = 50,
) -> Dict:
    """One full DAgger rollout-and-correct pass over `n_episodes` rollouts.

    Layouts are drawn round-robin from `layouts_yaml` (with rotating
    seeds so successive episodes on the same layout still have a unique
    timestamp / signature when saved). Failures are corrected by
    re-running the policy in a fresh env up to the deviation step,
    then BFS-finishing from there. Retraining fires every
    `train_every_n` corrective demos counted *globally* (across all
    passes — see _count_corrective_demos).

    Returns a dict with: n_episodes, n_failures, n_corrected_in_pass,
    cum_corrections_before, cum_corrections_after, n_demos_added,
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
    print(f"[dagger] {len(layouts)} layouts in pool from {layouts_yaml}")
    print(f"[dagger] running {n_episodes} episodes (round-robin)")
    print(f"[dagger] checkpoint={checkpoint}")
    print(f"[dagger] demo_dir  ={demo_dir}")
    print(f"[dagger] out_dir   ={out_dir}")
    print(f"[dagger] train every {train_every_n} corrective demos (CUMULATIVE)")

    model = _load_model(checkpoint, device)

    # Cumulative counter — counts every corrective demo ever collected
    # by this method, NOT just the ones from this pass. The next train
    # is triggered when this counter crosses the next multiple of
    # train_every_n.
    cum_corrections_at_start = _count_corrective_demos(demo_dir)
    cum_corrections = cum_corrections_at_start
    next_train_at = ((cum_corrections // train_every_n) + 1) * train_every_n
    print(f"[dagger] cumulative corrections so far: {cum_corrections} "
          f"(next train at {next_train_at})")

    n_train_calls = 0
    saved_demos: List[Path] = []
    episode_log: List[Dict] = []
    success_count = 0
    n_corrected_in_pass = 0

    for ep_idx in range(n_episodes):
        layout = layouts[ep_idx % len(layouts)]
        ep_seed = seed + ep_idx

        env = _build_env_from_layout(layout, seed=ep_seed)
        obs_seq = [{"state": env._get_obs()["state"],
                    "image": env._get_obs()["image"]}]
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

        # Pre-failure trajectory (positions BEFORE each action). Length
        # is len(actions) + 1; index t is the agent's pose before
        # actions[t]. Last entry is the post-final-action pose, which
        # we don't index into during deviation detection.
        traj_positions = [tuple(p) for p in env.get_trajectory()]
        env.close()

        corrected_here = False
        deviation_step: Optional[int] = None
        if not success and actions:
            # Find where the policy first stepped off the BFS-optimal
            # set. For fire-hits this is typically the fire step; for
            # timeouts it can be much earlier.
            deviation_step = _find_deviation_step(layout, actions, traj_positions)
            if deviation_step < 0:
                # Defensive: episode failed but every action looked
                # optimal. Skip rather than fabricate a "correction".
                print(f"[dagger] ep {ep_idx} ({layout.get('name')}): "
                      f"failed but no deviation detected; skipping correction.")
            else:
                # Replay actions[:deviation_step] in a fresh env to put
                # the agent at the pre-deviation pose with the correct
                # visited mask + step count.
                env2 = _build_env_from_layout(layout, seed=ep_seed)
                obs2 = env2._get_obs()
                obs_seq2 = [{"state": obs2["state"], "image": obs2["image"]}]
                actions2: List[int] = []
                rewards2: List[float] = []
                for a in actions[:deviation_step]:
                    obs, r, _, _, _ = env2.step(a)
                    obs_seq2.append({"state": obs["state"], "image": obs["image"]})
                    actions2.append(int(a))
                    rewards2.append(float(r))
                # Now expert takes over from env2.agent_pos to goal.
                plan = _bfs_path(env2.grid.copy(),
                                 tuple(env2.agent_pos),
                                 tuple(env2.goal_pos))
                if plan:
                    for a in plan:
                        obs, r, term, trunc, _ = env2.step(a)
                        obs_seq2.append({"state": obs["state"],
                                         "image": obs["image"]})
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
                    cum_corrections += 1
                    n_corrected_in_pass += 1
                    corrected_here = True
                    print(f"[dagger] ep {ep_idx} ({layout.get('name')}): "
                          f"failed at deviation_step={deviation_step}; "
                          f"expert took over (plan len={len(plan)}); "
                          f"saved {out_path.name}  "
                          f"[cum corrections={cum_corrections}]")
                else:
                    print(f"[dagger] ep {ep_idx} ({layout.get('name')}): "
                          f"deviation at step {deviation_step} but BFS "
                          f"found no fire-free path; skipping.")
                env2.close()

        episode_log.append({
            "episode_id":      ep_idx,
            "layout_name":     layout.get("name"),
            "policy_success":  success,
            "corrected":       corrected_here,
            "deviation_step":  deviation_step,
            "policy_steps":    si,
        })

        # Cumulative retrain trigger: once we cross the next multiple
        # of train_every_n corrective demos (across all passes),
        # retrain on the full demo set.
        if (not skip_train) and cum_corrections >= next_train_at:
            print(f"[dagger] cumulative={cum_corrections} >= {next_train_at}; "
                  f"retraining (initial demos UNION every correction so far)")
            cmd = [
                sys.executable, "-u", "-m", "Equivariant_pathway.train",
                "--checkpoint_dir", str(checkpoint.parent),
                "--epochs", str(epochs),
                "--resume",
            ]
            if train_demo_paths:
                cmd += ["--demo_paths", train_demo_paths]
            else:
                cmd += ["--demo_dir", str(demo_dir)]
            print(f"[dagger] $ {' '.join(cmd)}")
            rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
            if rc != 0:
                print(f"[dagger] WARNING: retrain exited rc={rc}")
            n_train_calls += 1
            next_train_at += train_every_n
            # Reload weights so subsequent rollouts in this pass use
            # the freshly-trained policy.
            best_path = checkpoint.parent / "best_eq_policy.pth"
            reload_path = best_path if best_path.exists() else checkpoint
            model = _load_model(reload_path, device)

    summary = {
        "layouts_yaml":           str(layouts_yaml),
        "n_layouts_in_pool":      len(layouts),
        "n_episodes":             n_episodes,
        "n_successes":            success_count,
        "n_failures":             n_episodes - success_count,
        "success_rate":           success_count / n_episodes if n_episodes else 0.0,
        "n_corrected_in_pass":    n_corrected_in_pass,
        "cum_corrections_before": cum_corrections_at_start,
        "cum_corrections_after":  cum_corrections,
        "n_demos_added":          len(saved_demos),
        "n_train_calls":          n_train_calls,
        "demo_paths":             [str(p) for p in saved_demos],
        "episode_log":            episode_log,
        "checkpoint":             str(checkpoint),
        "demo_dir":               str(demo_dir),
    }
    with open(out_dir / "dagger_pass_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[dagger] PASS DONE  policy_sr={summary['success_rate']:.2f}  "
          f"corrected_this_pass={n_corrected_in_pass}  "
          f"trains_in_pass={n_train_calls}  "
          f"cum_corrections={cum_corrections}")
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
    p.add_argument("--n_episodes", type=int, default=50,
                   help="Number of rollouts per pass. Layouts are drawn "
                        "round-robin from --layouts with rotating seeds.")
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
        n_episodes=args.n_episodes,
    )


if __name__ == "__main__":
    main()
