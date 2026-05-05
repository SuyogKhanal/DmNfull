"""Run the existing P6 LLM analysis pipeline on a CNN+MLP test rollout.

P6 = P5 + TKF (Trusted Knowledge Filter / demo-knowledge cross-check).
TKF reads the recorded demo JSONs to build a CLIP-indexed knowledge base, so
this wrapper points it at CNN_pathway/demos by default. Override with
`--demo_dir` if you keep the play_maze demos elsewhere.

The shared logic lives in CNN_pathway/_analysis_common.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from CNN_pathway._analysis_common import (
    MASTER_CONFIG,
    REPO_ROOT,
    run_profile_analysis,
)

PROFILE_YAML = "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm.yaml"
LABEL = "p6"
OUT_SUBDIR = "p6_analysis"
DEFAULT_DEMO_DIR = REPO_ROOT / "CNN_pathway" / "demos"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", type=str, required=True,
                   help="Run directory produced by CNN_pathway/rollout_test.py")
    p.add_argument("--out_dir", type=str, default=None,
                   help=f"Where to write the {LABEL.upper()} analysis. Defaults to "
                        f"<rollout_dir>/{OUT_SUBDIR}.")
    p.add_argument("--demo_dir", type=str, default=str(DEFAULT_DEMO_DIR),
                   help="TKF demo directory (where play_maze saved your demos). "
                        "Overrides `tkf.demo_dir` in the master config so TKF "
                        "doesn't error out on the empty default `demos/` folder.")
    p.add_argument("--tkf_index_path", type=str, default=None,
                   help="Optional override for `tkf.index_path` (where the CLIP "
                        "knowledge-base index is cached).")
    p.add_argument("--master_config", type=str, default=str(MASTER_CONFIG))
    return p.parse_args()


def main():
    args = parse_args()

    tkf_overrides = {"demo_dir": args.demo_dir}
    if args.tkf_index_path:
        tkf_overrides["index_path"] = args.tkf_index_path
    extra = {"tkf": tkf_overrides}

    run_profile_analysis(
        profile_yaml_name=PROFILE_YAML,
        rollout_dir=args.rollout_dir,
        out_subdir_name=OUT_SUBDIR,
        out_dir_override=args.out_dir,
        master_config_path=args.master_config,
        label=LABEL,
        extra_overrides=extra,
    )


if __name__ == "__main__":
    main()
