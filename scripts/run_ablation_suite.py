import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import yaml

from pipeline.rollout import run_rollouts
from pipeline.pipeline_runner import (
    load_config,
    rerun_pipeline_only,
    _deep_merge as deep_merge,
)
from pipeline.response_cache import ResponseCache


ALL_PROFILES = [
    "full_system",
    "no_vlm",
    "no_kag",
    "no_rag",
    "no_reasoning",
    "no_tkf",
    "no_aggregator",
]


def parse_args():
    p = argparse.ArgumentParser(description="Run a full ablation suite over a single shared rollout.")
    p.add_argument("--config",            type=str, default="configs/experiment_config.yaml")
    p.add_argument("--n-episodes",        type=int, default=None, dest="n_episodes")
    p.add_argument("--seed",              type=int, default=None)
    p.add_argument("--checkpoint",        type=str, default=None)
    p.add_argument("--profiles",          type=str, default=",".join(ALL_PROFILES))
    p.add_argument("--force-rerun-cache", action="store_true", dest="force_rerun_cache")
    p.add_argument("--launch-dashboard",  action="store_true", dest="launch_dashboard")
    return p.parse_args()


def _load_master_config(path: str, n_episodes, seed, checkpoint) -> Dict:
    cfg = load_config(path, ablation_path=None)
    if n_episodes is not None:
        cfg.setdefault("rollout", {})["n_episodes"] = int(n_episodes)
    if seed is not None:
        cfg.setdefault("rollout", {})["seed"] = int(seed)
    if checkpoint is not None:
        cfg.setdefault("rollout", {})["checkpoint_path"] = str(checkpoint)
    return cfg


def _parse_profiles(arg: str) -> List[str]:
    items = [s.strip() for s in (arg or "").split(",") if s.strip()]
    out = []
    for it in items:
        if it not in ALL_PROFILES:
            print(f"[Suite] WARNING: unknown profile '{it}' — ignoring.")
            continue
        if it not in out:
            out.append(it)
    if not out:
        out = list(ALL_PROFILES)
    return out


def _ensure_full_system_first(profiles: List[str]) -> (List[str], bool):
    if "full_system" in profiles:
        ordered = ["full_system"] + [p for p in profiles if p != "full_system"]
        return ordered, False
    return ["full_system"] + profiles, True


def _load_profile_yaml(profile_name: str) -> Dict:
    p = REPO_ROOT / "configs" / "ablation_profiles" / f"{profile_name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Ablation profile not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def _profile_config(master_cfg: Dict, profile_name: str) -> Dict:
    profile = _load_profile_yaml(profile_name)
    return deep_merge(master_cfg, profile)


def _save_rollout_only(rollout_result: Dict, target_dir: Path, master_cfg: Dict):
    target_dir.mkdir(parents=True, exist_ok=True)
    failure_ids = rollout_result["failure_episode_ids"]
    success_ids = rollout_result["success_episode_ids"]
    full_a_only = {
        "metadata": {
            "run_id":      target_dir.name,
            "timestamp":   datetime.now().isoformat(),
            "n_episodes":  rollout_result["n_episodes"],
            "seed_base":   rollout_result["seed_base"],
            "n_successes": len(success_ids),
            "n_failures":  len(failure_ids),
            "pipeline_flags": master_cfg.get("pipeline", {}),
            "phase_a_only": True,
        },
        "config": master_cfg,
        "phase_a": {
            "all_rollouts": [
                {
                    "episode_id":    e["episode_id"],
                    "maze_name":     e["maze_name"],
                    "seed":          e["seed"],
                    "total_steps":   e["total_steps"],
                    "total_reward":  e["total_reward"],
                    "success":       e["success"],
                    "ascii_grid":    e["ascii_grid"],
                    "dynamic_config":e["dynamic_config"],
                    "key_frames":    e["key_frames"],
                    "frame_paths":   e.get("frame_paths", {}),
                }
                for e in rollout_result["all_episodes"]
            ],
            "success_episode_ids": success_ids,
            "failure_episode_ids": failure_ids,
        },
    }
    with open(target_dir / "full_output.json", "w") as f:
        json.dump(full_a_only, f, indent=2, default=str)
    with open(target_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(master_cfg, f, sort_keys=False)


def _summarise_profile(profile: str, profile_dir: Path, full_output: Dict, cache: ResponseCache, prev_hits: int, prev_miss: int) -> Dict:
    metadata = full_output.get("metadata", {})
    parsed   = (full_output.get("phase_c", {}) or {}).get("parsed_prescription", {}) or {}
    n_ep     = int(metadata.get("n_episodes", 0) or 0)
    n_succ   = int(metadata.get("n_successes", 0) or 0)
    n_fail   = int(metadata.get("n_failures", n_ep - n_succ) or 0)
    sr       = (n_succ / n_ep) if n_ep > 0 else 0.0
    total_demos = int(parsed.get("total_demonstrations_needed", 0) or 0) if isinstance(parsed, dict) else 0

    delta_hits = cache.hit_count  - prev_hits
    delta_miss = cache.miss_count - prev_miss
    return {
        "profile":           profile,
        "run_dir":           str(profile_dir),
        "n_episodes":        n_ep,
        "n_successes":       n_succ,
        "n_failures":        n_fail,
        "success_rate":      sr,
        "total_demos_prescribed": total_demos,
        "llm_calls_made":    delta_miss,
        "cache_hits":        delta_hits,
    }


def _print_table(rows: List[Dict]):
    header = (
        f"{'Profile':<16} | {'SR':>5} | {'Failed':>6} | {'Demos':>6} | {'LLM calls':>9} | {'Cache hits':>10}"
    )
    sep = "-" * len(header)
    print("\n" + sep); print(header); print(sep)
    for r in rows:
        print(
            f"{r['profile']:<16} | "
            f"{r['success_rate']:>5.2f} | "
            f"{r['n_failures']:>6d} | "
            f"{r['total_demos_prescribed']:>6d} | "
            f"{r['llm_calls_made']:>9d} | "
            f"{r['cache_hits']:>10d}"
        )
    print(sep)


def _try_launch_dashboard(suite_root: Path):
    try:
        from dashboard import app as dashboard_app
    except Exception as e:
        print(f"[Suite] Dashboard import failed: {e}")
        return
    if hasattr(dashboard_app, "launch_dashboard"):
        try:
            dashboard_app.launch_dashboard(str(suite_root))
            return
        except Exception as e:
            print(f"[Suite] launch_dashboard() raised: {e} — falling back to subprocess.")
    cmd = [sys.executable, str(REPO_ROOT / "dashboard" / "app.py")]
    print(f"[Suite] Launching dashboard subprocess: {' '.join(cmd)}")
    subprocess.Popen(cmd, cwd=str(REPO_ROOT))


def main():
    args = parse_args()

    master_cfg = _load_master_config(args.config, args.n_episodes, args.seed, args.checkpoint)
    profiles_in = _parse_profiles(args.profiles)
    profiles_ordered, prepended_full = _ensure_full_system_first(profiles_in)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    suite_root = REPO_ROOT / "results" / "ablations" / run_id
    suite_root.mkdir(parents=True, exist_ok=True)

    cache_root = REPO_ROOT / "results" / "cache" / run_id
    cache = ResponseCache(cache_root, force_rerun=args.force_rerun_cache)

    manifest = {
        "run_id":           run_id,
        "timestamp":        datetime.now().isoformat(),
        "profiles_requested": profiles_in,
        "profiles_executed":  profiles_ordered,
        "full_system_prepended_silently": prepended_full,
        "cli_overrides": {
            "config":            args.config,
            "n_episodes":        args.n_episodes,
            "seed":              args.seed,
            "checkpoint":        args.checkpoint,
            "force_rerun_cache": bool(args.force_rerun_cache),
            "launch_dashboard":  bool(args.launch_dashboard),
        },
        "master_config":      master_cfg,
        "cache_root":         str(cache_root),
        "suite_root":         str(suite_root),
    }
    with open(suite_root / "suite_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'='*72}\n[Suite] Phase A — single shared rollout\n{'='*72}")
    rollout_dir = suite_root / "rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rollout_result = run_rollouts(master_cfg, rollout_dir)
    _save_rollout_only(rollout_result, rollout_dir, master_cfg)

    summaries: List[Dict] = []
    prev_hits = cache.hit_count
    prev_miss = cache.miss_count

    for profile in profiles_ordered:
        print(f"\n{'='*72}\n[Suite] Profile: {profile}\n{'='*72}")
        prof_cfg = _profile_config(master_cfg, profile)
        prof_dir = suite_root / profile
        prof_dir.mkdir(parents=True, exist_ok=True)

        try:
            full_output = rerun_pipeline_only(
                saved_run_dir=str(rollout_dir),
                overrides={"pipeline": prof_cfg.get("pipeline", {})},
                out_run_dir=str(prof_dir),
                master_config_path=args.config,
                cache=cache,
            )
        except TypeError:
            full_output = rerun_pipeline_only(
                saved_run_dir=str(rollout_dir),
                overrides={"pipeline": prof_cfg.get("pipeline", {})},
                out_run_dir=str(prof_dir),
                master_config_path=args.config,
            )

        summary = _summarise_profile(profile, prof_dir, full_output, cache, prev_hits, prev_miss)
        prev_hits = cache.hit_count
        prev_miss = cache.miss_count
        summaries.append(summary)

    suite_summary_path = suite_root / f"suite_summary_{timestamp}.json"
    with open(suite_summary_path, "w") as f:
        json.dump({
            "run_id":          run_id,
            "timestamp":       datetime.now().isoformat(),
            "profiles":        summaries,
            "total_cache_hits":   cache.hit_count,
            "total_cache_misses": cache.miss_count,
        }, f, indent=2, default=str)

    _print_table(summaries)
    print(f"\n[Suite] Suite summary written to: {suite_summary_path}")
    print(f"[Suite] All artefacts under:       {suite_root}")

    if args.launch_dashboard:
        _try_launch_dashboard(suite_root)


if __name__ == "__main__":
    main()