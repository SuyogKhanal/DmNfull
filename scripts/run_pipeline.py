import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from pipeline.pipeline_runner import load_config, run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Run the full LLM-guided pipeline (Phase A + B + C).")
    p.add_argument("--config",         type=str, default="configs/experiment_config.yaml")
    p.add_argument("--ablation",       type=str, default=None,
                   help="Path to an ablation profile YAML under configs/ablation_profiles/ (e.g. configs/ablation_profiles/no_rag.yaml).")
    p.add_argument("--tag",            type=str, default=None,
                   help="Optional tag appended to the run directory name.")
    p.add_argument("--n_episodes",     type=int, default=None)
    p.add_argument("--seed",           type=int, default=None)
    p.add_argument("--checkpoint",     type=str, default=None)
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

    tag = args.tag
    if tag is None and args.ablation:
        tag = Path(args.ablation).stem

    run_pipeline(cfg, tag=tag)


if __name__ == "__main__":
    main()