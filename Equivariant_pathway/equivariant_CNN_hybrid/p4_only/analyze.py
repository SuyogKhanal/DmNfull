"""P4 analyzer wrapper for the hybrid p4_only pipeline.

Re-uses Equivariant_pathway/_analysis_common.run_profile_analysis (the
hybrid lives inside Equivariant_pathway/ so this is not a cross-dep
violation) and ships the same scaling-with-floor prompt addenda used
by the equivariant + CNN p4_only stacks so the comparison is
prompt-level identical across all three methods.
"""
from __future__ import annotations

from typing import Dict, Optional

from Equivariant_pathway._analysis_common import run_profile_analysis

PROFILE_YAML = "p4_vlm_reasoning_kag_cross_plain_llm.yaml"
LABEL = "p4_only_hybrid"
OUT_SUBDIR = "p4_analysis"


REASONING_ADDENDUM = (
    "SAMPLE-EFFICIENCY DIRECTIVE — pick the smallest n_demos that closes the failure mode,\n"
    "but never zero when a failure is present.\n"
    "HARD FLOOR: this episode IS a failure. n_demos for this episode must be >= 1."
)

AGGREGATOR_ADDENDUM = (
    "HOLISTIC SAMPLE-EFFICIENCY DIRECTIVE — minimise total layouts but NEVER zero, and\n"
    "scale recommendations with the diversity of the failure pool.\n"
    "1. Cluster the failures into the smallest set of distinct modes.\n"
    "2. Recommend the SMALLEST set of layouts per cluster that fixes the missing behaviour.\n"
    "3. HARD FLOOR: if n_failure_episodes >= 1, the response MUST contain at least one\n"
    "   cluster, one demonstration_prescription, one recommended_layout, and\n"
    "   total_demonstrations_needed >= 1.\n"
    "4. SCALE WITH POOL DIVERSITY (soft guideline, total recommended_layouts):\n"
    "      n_failure_episodes  1-3   -> aim for 1-2\n"
    "      n_failure_episodes  4-7   -> aim for 2-4\n"
    "      n_failure_episodes  8-15  -> aim for 3-7\n"
    "      n_failure_episodes  16+   -> aim for 5-10\n"
    "   Always keep total < n_failure_episodes (otherwise no compression).\n"
)


def run(rollout_dir: str, out_dir: Optional[str] = None,
        demo_dir: Optional[str] = None,
        master_config: Optional[str] = None) -> Dict:
    extra: Dict = {
        "llm": {
            "prompt_addendum_reasoning":  REASONING_ADDENDUM,
            "prompt_addendum_aggregator": AGGREGATOR_ADDENDUM,
        },
    }
    if demo_dir:
        extra["tkf"] = {"demo_dir": demo_dir}
    return run_profile_analysis(
        profile_yaml_name=PROFILE_YAML,
        rollout_dir=rollout_dir,
        out_subdir_name=OUT_SUBDIR,
        out_dir_override=out_dir,
        master_config_path=master_config,
        label=LABEL,
        extra_overrides=extra,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--demo_dir", default=None)
    p.add_argument("--master_config", default=None)
    args = p.parse_args()
    run(args.rollout_dir, args.out_dir, args.demo_dir, args.master_config)
