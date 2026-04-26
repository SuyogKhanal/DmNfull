import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from pipeline.pipeline_runner import load_config
from baselines.naive_dagger import run_naive_dagger
from baselines.diff_dagger import run_diff_dagger


def parse_args():
    p = argparse.ArgumentParser(description="Run a baseline (naive DAgger or diff-DAgger).")
    p.add_argument("--baseline",   type=str, choices=["naive", "diff"], required=True)
    p.add_argument("--config",     type=str, default="configs/experiment_config.yaml")
    p.add_argument("--ablation",   type=str, default="configs/ablation_profiles/baseline_naive.yaml")
    p.add_argument("--n_episodes", type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.ablation)

    if args.n_episodes is not None:
        cfg.setdefault("rollout", {})["n_episodes"] = args.n_episodes
    if args.seed is not None:
        cfg.setdefault("rollout", {})["seed"] = args.seed
    if args.checkpoint is not None:
        cfg.setdefault("rollout", {})["checkpoint_path"] = args.checkpoint

    if args.baseline == "naive":
        run_naive_dagger(cfg)
    else:
        run_diff_dagger(cfg)


if __name__ == "__main__":
    main()