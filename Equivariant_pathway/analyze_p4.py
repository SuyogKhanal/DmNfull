"""Run the P4 LLM analysis on an Equivariant_pathway rollout.

P4 = VLM + Reasoning + KAG + Cross-Episode Reasoning + Plain LLM Aggregator.
Shared logic in Equivariant_pathway/_analysis_common.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# When this module is launched as a standalone script (e.g. by
# run_full_cycle.py via subprocess), Python only adds Equivariant_pathway/
# to sys.path, NOT the repo root. The shared analysis helper lives at the
# repo level (`Equivariant_pathway._analysis_common`), so import would
# fail with ModuleNotFoundError. Prepend REPO_ROOT before the import.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway._analysis_common import run_profile_analysis, MASTER_CONFIG  # noqa: E402

PROFILE_YAML = "p4_vlm_reasoning_kag_cross_plain_llm.yaml"
LABEL = "p4"
OUT_SUBDIR = "p4_analysis"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", type=str, required=True,
                   help="Run directory produced by Equivariant_pathway/rollout_test.py")
    p.add_argument("--out_dir", type=str, default=None,
                   help=f"Where to write the {LABEL.upper()} analysis. "
                        f"Defaults to <rollout_dir>/{OUT_SUBDIR}.")
    p.add_argument("--master_config", type=str, default=str(MASTER_CONFIG))
    p.add_argument("--demo_dir", type=str, default=None,
                   help="Per-profile demo directory used by TKF (NB: P4 has TKF off, "
                        "but the override is still accepted so the wrapper is uniform "
                        "with analyze_p5/p6).")
    return p.parse_args()


def main():
    args = parse_args()
    extra = {}
    if args.demo_dir:
        extra = {"tkf": {"demo_dir": args.demo_dir}}
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
