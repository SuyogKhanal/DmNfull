"""Run the existing P5 LLM analysis pipeline on a CNN+MLP test rollout.

P5 = P4 + RAG (retrieval-augmented context across past episodes).
The shared logic lives in CNN_pathway/_analysis_common.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from CNN_pathway._analysis_common import run_profile_analysis, MASTER_CONFIG

PROFILE_YAML = "p5_vlm_reasoning_kag_rag_cross_plain_llm.yaml"
LABEL = "p5"
OUT_SUBDIR = "p5_analysis"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", type=str, required=True,
                   help="Run directory produced by CNN_pathway/rollout_test.py")
    p.add_argument("--out_dir", type=str, default=None,
                   help=f"Where to write the {LABEL.upper()} analysis. Defaults to "
                        f"<rollout_dir>/{OUT_SUBDIR}.")
    p.add_argument("--master_config", type=str, default=str(MASTER_CONFIG))
    return p.parse_args()


def main():
    args = parse_args()
    run_profile_analysis(
        profile_yaml_name=PROFILE_YAML,
        rollout_dir=args.rollout_dir,
        out_subdir_name=OUT_SUBDIR,
        out_dir_override=args.out_dir,
        master_config_path=args.master_config,
        label=LABEL,
    )


if __name__ == "__main__":
    main()
