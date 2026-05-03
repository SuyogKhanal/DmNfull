"""Per-profile per-episode policy evaluation for the McNemar comparison.

Loads each profile's fine-tuned diffusion checkpoint and rolls out 100 episodes
on the same `(seed_base + ep_idx)` schedule, so episode_id `i` is the same
start/goal/fire layout in every profile. The resulting per_episode_success_policy.json
files are paired-binary inputs to mcnemar_analysis.py.

Per-episode inference noise is locked by reseeding torch / cuda / numpy /
random to the episode_seed before every action selection sequence — so a
given (checkpoint, episode_seed) pair is reproducible run-to-run, and any
SR difference between profiles is attributable to the checkpoints, not RNG
drift.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from envs.maze_env import MazeNavEnv
from model.diffusion_policy import MazeDiffusionPolicy
from pipeline.rollout import _build_policy, _load_meta, _load_checkpoint
from pipeline.pipeline_runner import load_config

PROFILES = ["p4_vlm_reasoning_kag_cross_plain_llm",
            "p5_vlm_reasoning_kag_rag_cross_plain_llm",
            "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm"]
PROFILE_KEYS = {"p4": PROFILES[0], "p5": PROFILES[1], "p6": PROFILES[2]}


def parse_args():
    p = argparse.ArgumentParser(description="Run paired per-episode policy eval for McNemar comparison.")
    p.add_argument("--config",      type=str, default="configs/experiment_config.yaml")
    p.add_argument("--n-episodes",  type=int, default=100, dest="n_episodes")
    p.add_argument("--seed-base",   type=int, default=0,   dest="seed_base")
    p.add_argument("--out-root",    type=str, default=None, dest="out_root",
                   help="Output directory. Default: results/mcnemar/<timestamp>/")
    p.add_argument("--p4-ckpt",     type=str, required=True, dest="p4_ckpt")
    p.add_argument("--p5-ckpt",     type=str, required=True, dest="p5_ckpt")
    p.add_argument("--p6-ckpt",     type=str, required=True, dest="p6_ckpt")
    p.add_argument("--max-steps",   type=int, default=None, dest="max_steps",
                   help="Hard episode cap (defaults to env's built-in truncation).")
    return p.parse_args()


def _seed_inference(seed: int):
    """Lock all RNG sources used by scheduler.denoise() for this episode.

    DDPMScheduler.denoise() uses torch.randn / torch.randn_like, so torch's
    CPU and CUDA generators must both be reseeded. We also seed numpy/random
    for any auxiliary call paths.
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    random.seed(int(seed))


def _build_env(maze_cfg: Dict, seed: int) -> MazeNavEnv:
    return MazeNavEnv(
        maze_name=maze_cfg["name"],
        render_mode="rgb_array",
        randomize_start=bool(maze_cfg.get("randomize_start", True)),
        randomize_goal=bool(maze_cfg.get("randomize_goal", True)),
        randomize_fire=bool(maze_cfg.get("randomize_fire", True)),
        num_fire_tiles=int(maze_cfg.get("num_fire_tiles", 3)),
        seed=seed,
    )


def _rollout_episode(
    policy: MazeDiffusionPolicy,
    maze_cfg: Dict,
    episode_seed: int,
    obs_horizon: int,
    use_vision: bool,
    max_steps: Optional[int],
) -> Dict:
    """One env episode. Inference RNG is reseeded once per episode so that
    repeating the same (checkpoint, episode_seed) pair gives the same outcome.
    """
    _seed_inference(episode_seed)
    env = _build_env(maze_cfg, seed=episode_seed)
    obs, info = env.reset()
    state_q = deque([obs["state"]] * obs_horizon, maxlen=obs_horizon)
    image_q = deque([obs["image"]] * obs_horizon, maxlen=obs_horizon)

    terminated = truncated = False
    steps = 0
    total_reward = 0.0
    while not (terminated or truncated):
        img = np.stack(list(image_q), 0).astype(np.float32) / 255.0
        img = np.transpose(img, (0, 3, 1, 2))
        obs_dict = {"state": np.stack(list(state_q), 0).astype(np.float32),
                    "image": img}
        action = policy.get_action(obs_dict) if use_vision else policy.get_action({"state": obs_dict["state"]})
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        state_q.append(obs["state"]); image_q.append(obs["image"])
        steps += 1
        if max_steps is not None and steps >= max_steps:
            truncated = True

    success = bool(info.get("success", False))
    env.close()
    return {"episode_id": None, "seed": episode_seed,
            "success": int(success), "total_steps": steps,
            "total_reward": total_reward}


def main():
    args = parse_args()
    master_cfg = load_config(args.config, ablation_path=None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (REPO_ROOT / "results" / "mcnemar" / f"run_{timestamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp":   datetime.now().isoformat(),
        "config":      args.config,
        "seed_base":   args.seed_base,
        "n_episodes":  args.n_episodes,
        "profiles":    PROFILES,
        "p4_ckpt":     args.p4_ckpt,
        "p5_ckpt":     args.p5_ckpt,
        "p6_ckpt":     args.p6_ckpt,
        "max_steps":   args.max_steps,
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maze_cfg = master_cfg["maze"]
    fallback_ckpt = str(master_cfg.get("rollout", {}).get("fallback_checkpoint_path",
                                                         "checkpoints/best_model.pth"))
    ckpt_for = {"p4": args.p4_ckpt, "p5": args.p5_ckpt, "p6": args.p6_ckpt}

    t0 = time.time()
    for short_key, profile in PROFILE_KEYS.items():
        ckpt_path = ckpt_for[short_key]
        meta = _load_meta(ckpt_path)
        policy = _build_policy(meta, device)
        loaded = _load_checkpoint(policy, ckpt_path, fallback_ckpt, device)
        obs_horizon = int(meta.get("obs_horizon", 4))
        use_vision  = bool(meta.get("use_vision", True))

        prof_dir = out_root / profile
        prof_dir.mkdir(parents=True, exist_ok=True)

        episodes: List[Dict] = []
        n_succ = 0
        pbar = tqdm(range(args.n_episodes), desc=f"policy/{short_key}", unit="ep", dynamic_ncols=True)
        for ep_idx in pbar:
            ep_seed = args.seed_base + ep_idx
            r = _rollout_episode(policy, maze_cfg, ep_seed, obs_horizon, use_vision, args.max_steps)
            r["episode_id"] = ep_idx
            episodes.append(r)
            n_succ += r["success"]
            pbar.set_postfix(sr=f"{n_succ/(ep_idx+1):.2f}")

        out_path = prof_dir / "per_episode_success_policy.json"
        with open(out_path, "w") as f:
            json.dump({
                "profile":     profile,
                "mode":        "policy",
                "checkpoint":  loaded,
                "seed_base":   args.seed_base,
                "n_episodes":  args.n_episodes,
                "n_successes": n_succ,
                "success_rate": n_succ / max(1, args.n_episodes),
                "episodes":    episodes,
            }, f, indent=2)
        print(f"[policy/{short_key}] wrote {out_path} | SR={n_succ}/{args.n_episodes}")

    print(f"\n[mcnemar_eval] DONE in {time.time()-t0:.1f}s")
    print(f"[mcnemar_eval] outputs under: {out_root}")
    print(f"[mcnemar_eval] next: python scripts/mcnemar_analysis.py --results-dir {out_root}")


if __name__ == "__main__":
    main()
