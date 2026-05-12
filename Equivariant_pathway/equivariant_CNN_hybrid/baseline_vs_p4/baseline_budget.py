"""Budget-constrained baseline DAgger for the hybrid pathway.

Differences from ``equivariant_CNN_hybrid.baseline_dagger``:

* Every round rolls out the FULL correction pool, then ranks the
  failures and only saves corrective demos for the top K_remaining
  episodes, where K_remaining = total_budget - n_saved_so_far.

* Ranking key per failure episode (descending priority):
    1. ``policy_loss``  — mean BCE between policy logits and the BFS
       expert mask over every step in the rollout. Higher = the
       policy made worse decisions on this layout.
    2. ``policy_steps`` — number of policy steps taken before the
       episode terminated (counts off-policy meandering, matters
       when losses tie).
    3. random          — uniform tiebreaker seeded by the round
       index so a re-run produces the same order.

* Stops after the round in which the budget is exhausted, or after a
  round that yields zero new demos (the latter mirrors P4's stopping
  rule). The held-out eval is run BEFORE the budget check so the
  final point on the curve always reflects the freshly retrained
  model.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import torch.nn.functional as F

from configs.maze_layouts import MAZE_LAYOUTS
from envs.maze_env import (ACTION_DELTAS, MazeNavEnv,
                           TILE_FIRE, TILE_FREE, TILE_GOAL)
from CNN_pathway.encoder_rgb import encode_state as encode_rgb
from Equivariant_pathway.encoder import encode_from_env as encode_grid_from_env
from Equivariant_pathway.equivariant_CNN_hybrid.model import EquivariantCNNHybridPolicy
from Equivariant_pathway.expert import AStarExpert, build_grid, optimal_action_mask

_HOST_MAZE = "multimodal"


def _info(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [bo-budget] {msg}", flush=True)


def _section(title: str, char: str = "="):
    print(f"\n{char*80}\n{title}\n{char*80}", flush=True)


def _build_env(layout: Dict, seed: int) -> MazeNavEnv:
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


def _encode_rgb_from_env(env: MazeNavEnv, cell_px: int = 16):
    return encode_rgb(
        grid=env.grid, agent_pos=tuple(env.agent_pos),
        goal_pos=tuple(env.goal_pos),
        fire_positions=[tuple(p) for p in env.fire_positions],
        cell_px=cell_px,
    )


def _load_model(checkpoint: Path, device: torch.device) -> EquivariantCNNHybridPolicy:
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model = EquivariantCNNHybridPolicy(
        rgb_in_channels=3, grid_in_channels=int(ckpt.get("in_channels", 5)),
        num_actions=int(ckpt.get("num_actions", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def _bce_with_balanced_pos_weight(logits: torch.Tensor,
                                  mask: torch.Tensor) -> torch.Tensor:
    """Match the trainer's per-step loss so the ranking is consistent
    with what the network is actually optimising."""
    num_optimal = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    num_total = torch.full_like(num_optimal, float(mask.shape[-1]))
    pos_weight = ((num_total - num_optimal) / num_optimal).expand_as(mask)
    loss = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    weights = torch.where(mask > 0.5, pos_weight, torch.ones_like(pos_weight))
    return (loss * weights).mean()


def _bfs_path(grid: np.ndarray, start: Tuple[int, int],
              goal: Tuple[int, int]) -> List[int]:
    from collections import deque
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


def _find_deviation_step(layout: Dict, actions: List[int],
                         trajectory: List[Tuple[int, int]]) -> int:
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


def _rollout_with_loss(model: EquivariantCNNHybridPolicy, layout: Dict,
                       seed: int, max_steps: int, device: torch.device,
                       cell_px: int = 16) -> Dict:
    """Run the policy on a layout, recording every step plus the
    per-step BCE-with-expert-mask loss so failures can be ranked.
    """
    env = _build_env(layout, seed=seed)
    fires = [tuple(f) for f in layout["fire_positions"]]
    goal = tuple(layout["goal_pos"])
    gs = len(layout.get("grid", [[0]*5]*5))
    grid_static = build_grid(gs, fire_positions=fires, goal_pos=goal)
    from Equivariant_pathway.expert import compute_distance_map
    dist_map = compute_distance_map(grid_static, goal)

    obs_seq = [{"state": env._get_obs()["state"], "image": env._get_obs()["image"]}]
    actions: List[int] = []
    rewards: List[float] = []
    per_step_loss: List[float] = []
    terminated = truncated = False
    si = 0
    while not (terminated or truncated) and si < max_steps:
        x_grid = encode_grid_from_env(env)
        x_rgb  = _encode_rgb_from_env(env, cell_px=cell_px)
        rgb_t  = torch.from_numpy(x_rgb).float().unsqueeze(0).to(device)
        grid_t = torch.from_numpy(x_grid).float().unsqueeze(0).to(device)
        agent_pos = tuple(env.agent_pos)
        opt_mask = optimal_action_mask(dist_map, agent_pos, grid_static,
                                       num_actions=4)
        with torch.no_grad():
            logits = model(rgb_t, grid_t)
            mask_t = torch.from_numpy(opt_mask).float().unsqueeze(0).to(device)
            step_loss = float(_bce_with_balanced_pos_weight(logits, mask_t).item())
            a = int(logits.argmax(dim=-1).item())
        per_step_loss.append(step_loss)
        obs, reward, terminated, truncated, _ = env.step(a)
        obs_seq.append({"state": obs["state"], "image": obs["image"]})
        actions.append(a)
        rewards.append(float(reward))
        si += 1
    success = bool(rewards[-1] > 5.0) if rewards else False
    traj_positions = [tuple(p) for p in env.get_trajectory()]
    env.close()
    return {
        "success": success,
        "n_steps": si,
        "policy_loss": float(np.mean(per_step_loss)) if per_step_loss else 0.0,
        "policy_loss_sum": float(np.sum(per_step_loss)) if per_step_loss else 0.0,
        "actions": actions,
        "rewards": rewards,
        "trajectory": traj_positions,
    }


def _save_corrective_demo(layout: Dict, deviation_step: int, ep_seed: int,
                          actions: List[int], demo_dir: Path,
                          layout_name: str, suffix: str) -> Optional[Path]:
    """Replay the first ``deviation_step`` policy actions, then BFS to
    the goal, and persist the full trajectory as a corrective demo."""
    env2 = _build_env(layout, seed=ep_seed)
    obs2 = env2._get_obs()
    obs_seq2 = [{"state": obs2["state"], "image": obs2["image"]}]
    actions2: List[int] = []
    rewards2: List[float] = []
    for a in actions[:max(0, deviation_step)]:
        obs, r, _, _, _ = env2.step(a)
        obs_seq2.append({"state": obs["state"], "image": obs["image"]})
        actions2.append(int(a))
        rewards2.append(float(r))
    plan = _bfs_path(env2.grid.copy(), tuple(env2.agent_pos), tuple(env2.goal_pos))
    if not plan:
        env2.close()
        return None
    for a in plan:
        obs, r, term, trunc, _ = env2.step(a)
        obs_seq2.append({"state": obs["state"], "image": obs["image"]})
        actions2.append(int(a))
        rewards2.append(float(r))
        if term or trunc:
            break
    demo_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    payload = {
        "maze_name": _HOST_MAZE, "timestamp": timestamp,
        "layout_id": f"{layout_name}_{suffix}",
        "source": "hybrid_baseline_budget_correction",
        "start_pos": [int(x) for x in env2.start_pos],
        "goal_pos":  [int(x) for x in env2.goal_pos],
        "fire_positions": [[int(x) for x in p] for p in env2.fire_positions],
        "trajectory": [[int(p[0]), int(p[1])] for p in env2.get_trajectory()],
        "observations": [o["state"].tolist() for o in obs_seq2],
        "images": [o["image"].tolist() for o in obs_seq2],
        "actions": [int(a) for a in actions2],
        "rewards": [float(r) for r in rewards2],
        "total_reward": float(sum(rewards2)),
        "success": bool(rewards2[-1] > 5.0) if rewards2 else False,
    }
    fname = demo_dir / f"demo_{_HOST_MAZE}_{layout_name}_{suffix}_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(payload, f)
    env2.close()
    return fname


def _count_demos(demo_dir: Path) -> int:
    return sum(1 for _ in demo_dir.rglob("*.json")) if demo_dir.exists() else 0


def _retrain(demo_dir: Path, ckpt_dir: Path, seed: int, epochs: int,
             train_from_scratch: bool) -> int:
    cmd = [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.train",
        "--demo_dir", str(demo_dir),
        "--checkpoint_dir", str(ckpt_dir),
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    if not train_from_scratch:
        cmd.append("--resume")
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _eval_heldout(ckpt_dir: Path, heldout_yaml: Path,
                  results_dir: Path, tag: str, seed: int,
                  max_steps: int) -> Dict:
    out_dir = results_dir / f"eval_{tag}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.rollout_test",
        "--checkpoint", str(ckpt_dir / "best_hybrid_policy.pth"),
        "--layouts", str(heldout_yaml),
        "--out_dir", str(out_dir),
        "--seed", str(seed),
        "--max_steps", str(max_steps),
    ]
    _info("$ " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        raise RuntimeError(f"heldout eval failed (rc={rc})")
    full = json.load(open(out_dir / "full_output.json"))
    md = full.get("metadata", {}) or {}
    n = int(md.get("n_episodes", 0) or 0)
    ns = int(md.get("n_successes", 0) or 0)
    return {"n_episodes": n, "n_successes": ns,
            "success_rate": ns / n if n else 0.0}


def _load_correction_layouts(yaml_path: Path) -> List[Dict]:
    with open(yaml_path, "r") as f:
        spec = yaml.safe_load(f) or {}
    return (spec.get("heldout_test_layouts")
            or spec.get("test_layouts")
            or spec.get("training_layouts")
            or spec.get("layouts") or [])


def run_baseline_budget(
    run_index: int,
    bo_root: Path,
    shared_demo_dir: Path,
    shared_ckpt_dir: Path,
    correction_yaml: Path,
    heldout_yaml: Path,
    budget: int,
    target_sr: float,
    round_epochs: int,
    max_rounds: int,
    max_steps: int,
    seed: int,
    train_from_scratch: bool,
) -> Dict:
    """Run the budget-constrained baseline DAgger loop and persist a
    standard learning_curve.json. ``bo_root`` is the isolated root for
    this method+run; ``shared_demo_dir`` / ``shared_ckpt_dir`` hold the
    initial 20 BFS demos + initial trained model that p4 will also use.
    """
    demo_dir    = bo_root / "demos"
    ckpt_dir    = bo_root / "checkpoints"
    results_dir = bo_root / "results"
    for d in (demo_dir, ckpt_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Seed initial demos and checkpoint from the shared bootstrap.
    for src in sorted(shared_demo_dir.glob("*.json")):
        dst = demo_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    init_best = shared_ckpt_dir / "best_hybrid_policy.pth"
    init_last = shared_ckpt_dir / "last_hybrid_policy.pth"
    if init_best.exists():
        shutil.copy2(init_best, ckpt_dir / "best_hybrid_policy.pth")
    if init_last.exists():
        shutil.copy2(init_last, ckpt_dir / "last_hybrid_policy.pth")

    # Round 0 — heldout SR for the initial model.
    layouts = _load_correction_layouts(correction_yaml)
    if not layouts:
        raise SystemExit(f"no correction layouts in {correction_yaml}")
    initial = _eval_heldout(ckpt_dir, heldout_yaml, results_dir,
                            tag="round_000", seed=seed, max_steps=max_steps)
    history: List[Dict] = [{
        "round": 0,
        "cum_demos": _count_demos(demo_dir),
        "extra_demos": 0,
        "budget_remaining": budget,
        "heldout_sr": initial["success_rate"],
        "heldout_n_successes": initial["n_successes"],
        "heldout_n_episodes": initial["n_episodes"],
        "correction_sr": None,
        "n_failures_in_pool": None,
        "n_picked": 0,
        "n_new_demos": 0,
    }]
    _persist_curve(results_dir, history, budget, target_sr, demo_dir, ckpt_dir,
                   correction_yaml, heldout_yaml, run_index)

    if initial["success_rate"] >= target_sr:
        return {"history": history, "stopped_reason": "initial>=target"}

    extras_saved = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for rnd in range(1, max_rounds + 1):
        remaining = budget - extras_saved
        if remaining <= 0:
            _info(f"round {rnd}: budget exhausted ({extras_saved}/{budget}); stopping.")
            break

        _section(f"BASELINE-BUDGET ROUND {rnd}  (remaining budget = {remaining})", char="-")

        # ----------------------------------------------------------------- #
        # Pass 1: roll out the entire correction pool, score every episode  #
        # ----------------------------------------------------------------- #
        model = _load_model(ckpt_dir / "best_hybrid_policy.pth", device)
        round_records: List[Dict] = []
        n_success = 0
        for ep_idx, layout in enumerate(layouts):
            ep_seed = seed + rnd * 1000 + ep_idx
            rec = _rollout_with_loss(model, layout, ep_seed, max_steps, device)
            rec.update({"ep_idx": ep_idx, "layout": layout, "ep_seed": ep_seed})
            round_records.append(rec)
            if rec["success"]:
                n_success += 1

        failures = [r for r in round_records if not r["success"]]
        correction_sr = n_success / len(round_records) if round_records else 0.0
        _info(
            f"round {rnd}: pool_size={len(round_records)} "
            f"successes={n_success} failures={len(failures)} "
            f"correction_sr={correction_sr:.3f}"
        )

        # ----------------------------------------------------------------- #
        # Pass 2: rank failures by (loss desc, steps desc, deterministic    #
        # random tiebreak), keep top ``remaining`` and only those.          #
        # ----------------------------------------------------------------- #
        rng = random.Random(0xBADF00D + 1009 * (run_index + 1) + rnd)
        decorated = []
        for rec in failures:
            jitter = rng.random()
            decorated.append((
                -rec["policy_loss"],     # higher loss first
                -rec["n_steps"],         # more steps first on ties
                jitter,                  # deterministic random tiebreak
                rec,
            ))
        decorated.sort()
        n_take = min(remaining, len(decorated))
        picked = [d[3] for d in decorated[:n_take]]
        # Mirror the ranking decisions in the curve metadata so reruns
        # can audit exactly which layouts were selected.
        round_dir = results_dir / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        with open(round_dir / "ranking.json", "w") as f:
            json.dump([
                {
                    "rank": i + 1,
                    "ep_idx": d[3]["ep_idx"],
                    "policy_loss": d[3]["policy_loss"],
                    "policy_loss_sum": d[3]["policy_loss_sum"],
                    "n_steps": d[3]["n_steps"],
                    "jitter": d[2],
                    "layout_name": d[3]["layout"].get("name"),
                    "picked": (i < n_take),
                } for i, d in enumerate(decorated)
            ], f, indent=2)

        # ----------------------------------------------------------------- #
        # Pass 3: emit corrective demos for the picked failures only.       #
        # ----------------------------------------------------------------- #
        round_demo_dir = demo_dir / f"round_{rnd:03d}"
        round_demo_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for rec in picked:
            dev = _find_deviation_step(rec["layout"], rec["actions"], rec["trajectory"])
            if dev < 0:
                # If we can't pinpoint the deviation, save a from-start
                # BFS demo (still on the failing layout) so the slot
                # isn't wasted.
                dev = 0
            out = _save_corrective_demo(
                rec["layout"], dev, rec["ep_seed"], rec["actions"],
                demo_dir=round_demo_dir,
                layout_name=rec["layout"].get("name", f"ep{rec['ep_idx']}"),
                suffix=f"correct_r{rnd}_ep{rec['ep_idx']}_dev{dev}",
            )
            if out is not None:
                saved += 1
            else:
                _info(f"round {rnd}: skip ep{rec['ep_idx']} (no BFS path from deviation)")
        extras_saved += saved
        _info(f"round {rnd}: picked={len(picked)} saved={saved} "
              f"cum_extras={extras_saved}/{budget}")

        # ----------------------------------------------------------------- #
        # Pass 4: retrain once on the union of demos, eval on heldout.     #
        # ----------------------------------------------------------------- #
        if saved > 0:
            rc = _retrain(demo_dir, ckpt_dir, seed=seed, epochs=round_epochs,
                          train_from_scratch=train_from_scratch)
            if rc != 0:
                _info(f"round {rnd}: retrain rc={rc} (continuing)")
        post = _eval_heldout(ckpt_dir, heldout_yaml, results_dir,
                             tag=f"round_{rnd:03d}", seed=seed + rnd,
                             max_steps=max_steps)
        history.append({
            "round": rnd,
            "cum_demos": _count_demos(demo_dir),
            "extra_demos": extras_saved,
            "budget_remaining": budget - extras_saved,
            "heldout_sr": post["success_rate"],
            "heldout_n_successes": post["n_successes"],
            "heldout_n_episodes": post["n_episodes"],
            "correction_sr": correction_sr,
            "n_failures_in_pool": len(failures),
            "n_picked": len(picked),
            "n_new_demos": saved,
        })
        _persist_curve(results_dir, history, budget, target_sr, demo_dir, ckpt_dir,
                       correction_yaml, heldout_yaml, run_index)
        _info(f"round {rnd}: heldout_sr={post['success_rate']:.3f}")

        if post["success_rate"] >= target_sr:
            return {"history": history, "stopped_reason": "target_hit"}
        if saved == 0:
            return {"history": history, "stopped_reason": "no_new_demos"}
        if extras_saved >= budget:
            return {"history": history, "stopped_reason": "budget_exhausted"}

    return {"history": history, "stopped_reason": "max_rounds"}


def _persist_curve(results_dir: Path, history: List[Dict],
                   budget: int, target_sr: float,
                   demo_dir: Path, ckpt_dir: Path,
                   correction_yaml: Path, heldout_yaml: Path,
                   run_index: int):
    out = {
        "method": "baseline_only_budget_hybrid",
        "run_index": int(run_index),
        "budget": int(budget),
        "target_sr": float(target_sr),
        "demo_dir": str(demo_dir),
        "checkpoint_dir": str(ckpt_dir),
        "correction_yaml": str(correction_yaml),
        "heldout_yaml": str(heldout_yaml),
        "history": history,
    }
    with open(results_dir / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_index", type=int, required=True)
    p.add_argument("--bo_root", type=str, required=True)
    p.add_argument("--shared_demo_dir", type=str, required=True)
    p.add_argument("--shared_ckpt_dir", type=str, required=True)
    p.add_argument("--correction_yaml", type=str, required=True)
    p.add_argument("--heldout_yaml", type=str, required=True)
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--target_sr", type=float, default=0.90)
    p.add_argument("--round_epochs", type=int, default=100)
    p.add_argument("--max_rounds", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_from_scratch", type=lambda s: str(s).lower() in ("1","true","yes","on"),
                   default=True)
    args = p.parse_args()
    result = run_baseline_budget(
        run_index=args.run_index,
        bo_root=Path(args.bo_root).resolve(),
        shared_demo_dir=Path(args.shared_demo_dir).resolve(),
        shared_ckpt_dir=Path(args.shared_ckpt_dir).resolve(),
        correction_yaml=Path(args.correction_yaml).resolve(),
        heldout_yaml=Path(args.heldout_yaml).resolve(),
        budget=args.budget,
        target_sr=args.target_sr,
        round_epochs=args.round_epochs,
        max_rounds=args.max_rounds,
        max_steps=args.max_steps,
        seed=args.seed,
        train_from_scratch=args.train_from_scratch,
    )
    print(f"[bo-budget] DONE stopped_reason={result.get('stopped_reason')}")
