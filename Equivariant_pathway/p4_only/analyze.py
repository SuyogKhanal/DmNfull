"""P4 analyzer wrapper for the p4_only pipeline.

Calls the same `run_profile_analysis` entry point that
Equivariant_pathway/analyze_p4.py uses (so the underlying pipeline
remains the shared P4 = VLM + Reasoning + KAG + Cross-Episode +
Plain-LLM stack), but injects two run-specific prompt addenda via
`extra_overrides` -> `llm.prompt_addendum_*`. The addenda strengthen
the existing "pick the smallest layout count" guidance with explicit
holistic / sample-efficiency framing so P4 can clearly outperform
baseline DAgger on demos-per-success rate.

The addendum hooks in pipeline/reasoning.py + pipeline/aggregator.py
are gated: when llm_cfg["prompt_addendum_*"] is empty (every other
caller — analyze_p4.py, analyze_p5.py, run_full_cycle.py, etc.) the
prompts are byte-identical to the legacy versions.
"""
from __future__ import annotations

from typing import Dict, Optional

from Equivariant_pathway._analysis_common import run_profile_analysis

PROFILE_YAML = "p4_vlm_reasoning_kag_cross_plain_llm.yaml"
LABEL = "p4_only"
OUT_SUBDIR = "p4_analysis"

# ---------------------------------------------------------------------------
# Prompt addenda. Kept here (not in a YAML) so the directive lives next to
# the pipeline that uses it and a reviewer can read code + prompt in one go.
# ---------------------------------------------------------------------------
REASONING_ADDENDUM = (
    "SAMPLE-EFFICIENCY DIRECTIVE — pick the smallest n_demos that closes the failure mode.\n"
    "The orchestrator pays one BFS-expert demonstration per layout that the downstream\n"
    "prescription LLM recommends. A baseline-DAgger controller can already converge by\n"
    "recording ONE corrective demo per failed episode; the entire reason this P4 pipeline\n"
    "exists is to identify the MINIMUM, most informative demos that fix the failure mode\n"
    "in fewer expert queries. When the analysis discusses corridor + path proposals in\n"
    "section 5, frame n_demos accordingly: prefer 1-2 unless the failure is genuinely\n"
    "multi-modal (two distinct corridors, two distinct fire-blocking patterns), and never\n"
    "suggest 6+ demos without explicit evidence of multiple non-overlapping failure modes\n"
    "in this episode. Treat extra demos as a cost, not a safety margin."
)

AGGREGATOR_ADDENDUM = (
    "HOLISTIC SAMPLE-EFFICIENCY DIRECTIVE — minimise total layouts across all failures.\n"
    "Treat the failure summaries above as a SET to be covered, not a list to be itemised.\n"
    "1. Read every failure together. Identify the smallest grouping (one or more clusters)\n"
    "   that explains them. If a single corridor / fire-blocking pattern accounts for the\n"
    "   majority of failures, prefer one cluster; do not split for cosmetic balance.\n"
    "2. For each cluster, recommend the SMALLEST set of concrete layouts that, taken\n"
    "   together, force the policy to learn the missing behaviour. If one well-chosen\n"
    "   layout teaches the corridor, recommend exactly one — even if six episodes failed.\n"
    "3. You are competing against a baseline-DAgger controller that pays one expert query\n"
    "   per failed episode. Your value is sample efficiency: total recommended_layouts\n"
    "   summed across clusters should normally be MUCH smaller than n_failure_episodes.\n"
    "4. Do NOT pad recommended_layouts to match cluster count or to look thorough. The\n"
    "   orchestrator runs until heldout success rate >= 0.90; if a single round under-\n"
    "   prescribes, the next round's failures will justify the additional layout. Over-\n"
    "   prescribing is a permanent cost, under-prescribing is at most one extra round.\n"
    "5. n_repetitions on each layout should mirror this discipline: 1-2 unless the layout\n"
    "   sits at the intersection of several failure modes."
)


def run(
    rollout_dir: str,
    out_dir: Optional[str] = None,
    demo_dir: Optional[str] = None,
    master_config: Optional[str] = None,
) -> Dict:
    """Run the P4 analysis pipeline with the p4_only prompt addenda."""
    extra_overrides: Dict = {
        "llm": {
            "prompt_addendum_reasoning":  REASONING_ADDENDUM,
            "prompt_addendum_aggregator": AGGREGATOR_ADDENDUM,
        },
    }
    if demo_dir:
        extra_overrides["tkf"] = {"demo_dir": demo_dir}
    return run_profile_analysis(
        profile_yaml_name=PROFILE_YAML,
        rollout_dir=rollout_dir,
        out_subdir_name=OUT_SUBDIR,
        out_dir_override=out_dir,
        master_config_path=master_config,
        label=LABEL,
        extra_overrides=extra_overrides,
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
