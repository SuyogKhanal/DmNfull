"""Run the P6 LLM analysis on an Equivariant_pathway rollout.

P6 = P5 + TKF.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root must be on sys.path so the shared helper resolves when this
# script is launched directly (subprocess invocation by run_full_cycle.py).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway._analysis_common import run_profile_analysis, MASTER_CONFIG  # noqa: E402

PROFILE_YAML = "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm.yaml"
LABEL = "p6"
OUT_SUBDIR = "p6_analysis"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--master_config", type=str, default=str(MASTER_CONFIG))
    p.add_argument("--demo_dir", type=str, default=None,
                   help="Per-profile demo dir — TKF reads from here to compute "
                        "demo coverage. Required for P6 to do anything different "
                        "from P5.")
    p.add_argument("--rag_bank", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    extra = {}
    if args.demo_dir:
        extra.setdefault("tkf", {})["demo_dir"] = args.demo_dir
    if args.rag_bank:
        extra.setdefault("rag", {})["bank_path"] = args.rag_bank
    run_profile_analysis(
        profile_yaml_name=PROFILE_YAML,
        rollout_dir=args.rollout_dir,
        out_subdir_name=OUT_SUBDIR,
        out_dir_override=args.out_dir,
        master_config_path=args.master_config,
        label=LABEL,
        extra_overrides=extra or None,
    )


if __name__ == "__main__":
    main()
