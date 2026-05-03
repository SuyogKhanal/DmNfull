"""Per-profile per-episode evaluation for the McNemar comparison.

Two evaluation modes, both producing a `per_episode_success_<mode>.json` per
profile so `mcnemar_analysis.py` can build paired 2x2 contingency tables for
P4-vs-P5 and P5-vs-P6.

  policy     Interpretation 2 — load each profile's fine-tuned diffusion
             checkpoint and roll out 100 episodes. Per-episode inference noise
             is fixed by torch.manual_seed(episode_seed) before action
             selection, so any per-episode SR difference between profiles is
             attributable to the checkpoints, not RNG drift.

  llm_actor  Interpretation 1 — every profile uses the SAME baseline checkpoint
             but the profile's LLM pipeline acts as a per-step shielding
             override on top of the diffusion policy. Profile flags
             (use_kag/use_rag/use_tkf/use_reasoning/use_plain_llm) control the
             per-step prompt, so any per-episode SR difference is attributable
             to LLM reasoning quality alone.

  both       Run both modes sequentially.

The same env-seed schedule (seed_base + episode_idx, episodes 0..N-1) is used
across every profile so the McNemar pairing is real (same start/goal/fire
layout per episode_id across profiles).
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

import yaml
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from envs.maze_env import MazeNavEnv, ACTION_NAMES
from model.diffusion_policy import MazeDiffusionPolicy
from pipeline.rollout import _build_policy, _load_meta, _load_checkpoint
from pipeline.pipeline_runner import load_config, _deep_merge as deep_merge

PROFILES = ["p4_vlm_reasoning_kag_cross_plain_llm",
            "p5_vlm_reasoning_kag_rag_cross_plain_llm",
            "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm"]
PROFILE_KEYS = {"p4": PROFILES[0], "p5": PROFILES[1], "p6": PROFILES[2]}


def parse_args():
    p = argparse.ArgumentParser(description="Run paired per-episode eval for McNemar comparison.")
    p.add_argument("--mode",        type=str, choices=["policy", "llm_actor", "both"], required=True)
    p.add_argument("--config",      type=str, default="configs/experiment_config.yaml")
    p.add_argument("--n-episodes",  type=int, default=100, dest="n_episodes")
    p.add_argument("--seed-base",   type=int, default=0,   dest="seed_base")
    p.add_argument("--out-root",    type=str, default=None, dest="out_root",
                   help="Output directory. Default: results/mcnemar/<timestamp>/")
    # policy mode: per-profile checkpoints
    p.add_argument("--p4-ckpt",     type=str, default=None, dest="p4_ckpt")
    p.add_argument("--p5-ckpt",     type=str, default=None, dest="p5_ckpt")
    p.add_argument("--p6-ckpt",     type=str, default=None, dest="p6_ckpt")
    # llm_actor mode: shared baseline checkpoint
    p.add_argument("--baseline-ckpt", type=str, default=None, dest="baseline_ckpt",
                   help="Shared baseline checkpoint for llm_actor mode. "
                        "Defaults to rollout.checkpoint_path from the master config.")
    p.add_argument("--rag-bank-dir",  type=str, default=None, dest="rag_bank_dir",
                   help="Optional pre-populated RAG bank dir for llm_actor mode (per profile, "
                        "expects subdir named like the profile). If absent, RAG retrieval is empty.")
    p.add_argument("--max-steps",     type=int, default=None, dest="max_steps",
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


def _profile_yaml(profile_name: str) -> Dict:
    p = REPO_ROOT / "configs" / "ablation_profiles" / f"{profile_name}.yaml"
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def _profile_flags(master_cfg: Dict, profile_name: str) -> Dict:
    merged = deep_merge(master_cfg, _profile_yaml(profile_name))
    return merged.get("pipeline", {}) or {}


# ---------------------------------------------------------------------------
# policy mode
# ---------------------------------------------------------------------------

def _rollout_policy_episode(
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


def run_mode_policy(args, master_cfg: Dict, out_root: Path):
    ckpt_for = {"p4": args.p4_ckpt, "p5": args.p5_ckpt, "p6": args.p6_ckpt}
    missing = [k for k, v in ckpt_for.items() if not v]
    if missing:
        raise SystemExit(
            f"[mode=policy] missing per-profile checkpoint(s): {missing}. "
            f"Pass --p4-ckpt/--p5-ckpt/--p6-ckpt."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maze_cfg = master_cfg["maze"]
    fallback_ckpt = str(master_cfg.get("rollout", {}).get("fallback_checkpoint_path",
                                                         "checkpoints/best_model.pth"))

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
            r = _rollout_policy_episode(policy, maze_cfg, ep_seed, obs_horizon, use_vision, args.max_steps)
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


# ---------------------------------------------------------------------------
# llm_actor mode
# ---------------------------------------------------------------------------

def _rollout_llm_actor_episode(
    policy: MazeDiffusionPolicy,
    actor,
    maze_cfg: Dict,
    episode_seed: int,
    obs_horizon: int,
    use_vision: bool,
    max_steps: Optional[int],
) -> Dict:
    """One episode where every step prefers the LLM's action; falls back to
    diffusion policy argmax when the LLM abstains/errors/picks an illegal move.
    """
    _seed_inference(episode_seed)
    env = _build_env(maze_cfg, seed=episode_seed)
    obs, info = env.reset()
    state_q = deque([obs["state"]] * obs_horizon, maxlen=obs_horizon)
    image_q = deque([obs["image"]] * obs_horizon, maxlen=obs_horizon)

    history = []
    terminated = truncated = False
    steps = 0
    total_reward = 0.0
    fallback_count = 0
    llm_count = 0
    while not (terminated or truncated):
        llm_action = actor.suggest_action(env, history)
        if llm_action is not None:
            action = llm_action
            llm_count += 1
        else:
            img = np.stack(list(image_q), 0).astype(np.float32) / 255.0
            img = np.transpose(img, (0, 3, 1, 2))
            obs_dict = {"state": np.stack(list(state_q), 0).astype(np.float32),
                        "image": img}
            action = (policy.get_action(obs_dict) if use_vision
                      else policy.get_action({"state": obs_dict["state"]}))
            fallback_count += 1

        history.append((tuple(env.agent_pos), int(action)))
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
            "total_reward": total_reward,
            "llm_actions": llm_count, "fallbacks": fallback_count}


def run_mode_llm_actor(args, master_cfg: Dict, out_root: Path):
    from scripts.llm_actor import LLMActor
    from pipeline.rag_bank import RAGBank

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maze_cfg = master_cfg["maze"]
    rollout_cfg = master_cfg.get("rollout", {})
    baseline = args.baseline_ckpt or str(rollout_cfg.get("checkpoint_path", "checkpoints/best_model_ema.pth"))
    fallback = str(rollout_cfg.get("fallback_checkpoint_path", "checkpoints/best_model.pth"))

    meta = _load_meta(baseline)
    policy = _build_policy(meta, device)
    loaded = _load_checkpoint(policy, baseline, fallback, device)
    obs_horizon = int(meta.get("obs_horizon", 4))
    use_vision  = bool(meta.get("use_vision", True))
    print(f"[llm_actor] shared baseline checkpoint: {loaded}")

    kag_path = master_cfg.get("kag", {}).get("document_path", "knowledge/kag_maze_knowledge.json")
    tkf_index_dir = master_cfg.get("tkf", {}).get("index_path", "results/demo_knowledge_base")

    for short_key, profile in PROFILE_KEYS.items():
        flags = _profile_flags(master_cfg, profile)

        rag_bank = None
        if flags.get("use_rag"):
            rag_path = (Path(args.rag_bank_dir) / profile) if args.rag_bank_dir \
                else Path(master_cfg.get("rag", {}).get("bank_path", "results/rag_bank"))
            try:
                rag_bank = RAGBank(
                    bank_path=str(rag_path),
                    top_k=int(master_cfg.get("rag", {}).get("top_k", 3)),
                    sim_threshold=float(master_cfg.get("rag", {}).get("sim_threshold", 0.3)),
                    clip_model=master_cfg.get("rag", {}).get("clip_model", "openai/clip-vit-large-patch14"),
                    owner_run_id=profile,
                )
            except Exception as e:
                print(f"[llm_actor/{short_key}] RAG init failed: {e!r}; continuing without RAG.")

        actor = LLMActor(
            profile_flags=flags,
            llm_cfg=master_cfg.get("llm", {}),
            kag_doc_path=kag_path if flags.get("use_kag") else None,
            rag_bank=rag_bank,
            tkf_index_dir=tkf_index_dir if flags.get("use_tkf") else None,
        )

        prof_dir = out_root / profile
        prof_dir.mkdir(parents=True, exist_ok=True)

        episodes: List[Dict] = []
        n_succ = 0
        pbar = tqdm(range(args.n_episodes), desc=f"llm_actor/{short_key}", unit="ep", dynamic_ncols=True)
        for ep_idx in pbar:
            ep_seed = args.seed_base + ep_idx
            r = _rollout_llm_actor_episode(policy, actor, maze_cfg, ep_seed, obs_horizon, use_vision, args.max_steps)
            r["episode_id"] = ep_idx
            episodes.append(r)
            n_succ += r["success"]
            pbar.set_postfix(sr=f"{n_succ/(ep_idx+1):.2f}", fb=actor.fallback_count)

        out_path = prof_dir / "per_episode_success_llm_actor.json"
        with open(out_path, "w") as f:
            json.dump({
                "profile":          profile,
                "mode":             "llm_actor",
                "baseline_checkpoint": loaded,
                "profile_flags":    flags,
                "seed_base":        args.seed_base,
                "n_episodes":       args.n_episodes,
                "n_successes":      n_succ,
                "success_rate":     n_succ / max(1, args.n_episodes),
                "total_llm_calls":  actor.call_count,
                "total_fallbacks":  actor.fallback_count,
                "episodes":         episodes,
            }, f, indent=2)
        print(f"[llm_actor/{short_key}] wrote {out_path} | SR={n_succ}/{args.n_episodes} "
              f"| fallbacks={actor.fallback_count}/{actor.call_count}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    master_cfg = load_config(args.config, ablation_path=None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (REPO_ROOT / "results" / "mcnemar" / f"run_{timestamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp":   datetime.now().isoformat(),
        "mode":        args.mode,
        "config":      args.config,
        "seed_base":   args.seed_base,
        "n_episodes":  args.n_episodes,
        "profiles":    PROFILES,
        "p4_ckpt":     args.p4_ckpt,
        "p5_ckpt":     args.p5_ckpt,
        "p6_ckpt":     args.p6_ckpt,
        "baseline_ckpt": args.baseline_ckpt,
        "rag_bank_dir": args.rag_bank_dir,
        "max_steps":   args.max_steps,
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    t0 = time.time()
    if args.mode in ("policy", "both"):
        run_mode_policy(args, master_cfg, out_root)
    if args.mode in ("llm_actor", "both"):
        run_mode_llm_actor(args, master_cfg, out_root)

    print(f"\n[mcnemar_eval] DONE in {time.time()-t0:.1f}s")
    print(f"[mcnemar_eval] outputs under: {out_root}")
    print(f"[mcnemar_eval] next: python scripts/mcnemar_analysis.py --results-dir {out_root} --mode {args.mode}")


if __name__ == "__main__":
    main()
