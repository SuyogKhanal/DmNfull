"""Run the P5 LLM analysis on an Equivariant_pathway rollout.

P5 = P4 + RAG.
"""
from __future__ import annotations

import argparse

from Equivariant_pathway._analysis_common import run_profile_analysis, MASTER_CONFIG

PROFILE_YAML = "p5_vlm_reasoning_kag_rag_cross_plain_llm.yaml"
LABEL = "p5"
OUT_SUBDIR = "p5_analysis"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--master_config", type=str, default=str(MASTER_CONFIG))
    p.add_argument("--demo_dir", type=str, default=None,
                   help="Per-profile demo dir (forwarded to TKF when present).")
    p.add_argument("--rag_bank", type=str, default=None,
                   help="Per-profile RAG bank dir; required so different profile "
                        "runs don't pollute each other's retrieval bank.")
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
