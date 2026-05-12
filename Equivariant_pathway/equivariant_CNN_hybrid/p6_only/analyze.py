"""P6 analyzer wrapper for the hybrid p6_only pipeline.

P6 = P5 + TKF (Training Knowledge Fetcher). Two override knobs:

  --rag_bank   per-round RAG bank dir, isolated from P5 (different
               filesystem path) and from any other pool size.
  --demo_dir   the dir TKF scans to decide which failure modes are
               already covered. Set to P6's OWN demo dir (which
               contains ONLY the 20 initial BFS demos + per-round
               LLM-prescribed BFS demos — no baseline_dagger
               corrections, since the bootstrap source-filter rejects
               them; no other method's demos either).
"""
from __future__ import annotations

from typing import Dict, Optional

from Equivariant_pathway._analysis_common import run_profile_analysis

PROFILE_YAML = "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm.yaml"
LABEL = "p6_only_hybrid"
OUT_SUBDIR = "p6_analysis"


REASONING_ADDENDUM = (
    "SAMPLE-EFFICIENCY DIRECTIVE — pick the smallest n_demos that closes the failure mode,\n"
    "but never zero when a failure is present.\n"
    "HARD FLOOR: this episode IS a failure. n_demos for this episode must be >= 1.\n"
    "RAG NOTE: when 'Retrieved similar past failures' contains a matching case, treat it\n"
    "as evidence the previous prescription did NOT fix the corridor — propose a demo\n"
    "that DIFFERS from the matched one (different start, goal, fires, or path length)\n"
    "so the new demo adds coverage rather than duplicating what the policy already has.\n"
    "TKF NOTE: the TKF block tells you whether the failure mode already has a demo in\n"
    "the bank. If TKF reports FOUND, the policy has been TRAINED on such a demo and is\n"
    "STILL failing — recommend a demo that DIFFERS from the matched one (different\n"
    "start, goal, fires, or path length) so the new demo adds genuine coverage. Never\n"
    "recommend an identical demo, but never suppress the recommendation entirely just\n"
    "because TKF=FOUND."
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
    "   Always keep total <= n_failure_episodes (otherwise no compression).\n"
    "5. RAG NOTE: when the per-episode reasoning cites RAG-retrieved similar past\n"
    "   failures, treat those retrievals as evidence that previous prescriptions did\n"
    "   NOT fully fix the corridor / failure-mode — the policy is still failing on it\n"
    "   even after the matched demo was trained on. Prescribe at least one layout\n"
    "   that explicitly DIFFERS from the retrieved cases (different start, goal, or\n"
    "   fire layout on the SAME corridor) so the new demo adds coverage rather than\n"
    "   duplicating what the policy already has. Do NOT use RAG matches as a reason\n"
    "   to collapse otherwise-distinct failure clusters into one.\n"
    "6. TKF NOTE: TKF=FOUND on an episode means a similar demo already exists in the\n"
    "   bank AND THE POLICY IS STILL FAILING ON THAT MODE. Treat FOUND as evidence\n"
    "   the existing demo is insufficient (insufficient coverage, wrong path length,\n"
    "   or distractor-fires too easy). Prescribe a new layout that DIVERGES from the\n"
    "   matched demo along at least one axis (start / goal / fire-pattern / path\n"
    "   length). Do NOT zero out the cluster. You may still prefer FEWER NEW LAYOUTS\n"
    "   when TKF=FOUND across the whole cluster, but the minimum per cluster is 1 —\n"
    "   not 0.\n"
)


def run(
    rollout_dir: str,
    out_dir: Optional[str] = None,
    demo_dir: Optional[str] = None,
    rag_bank: Optional[str] = None,
    master_config: Optional[str] = None,
) -> Dict:
    extra: Dict = {
        "llm": {
            "prompt_addendum_reasoning":  REASONING_ADDENDUM,
            "prompt_addendum_aggregator": AGGREGATOR_ADDENDUM,
        },
    }
    if demo_dir:
        extra["tkf"] = {"demo_dir": demo_dir}
    if rag_bank:
        extra["rag"] = {"bank_path": rag_bank}
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
    p.add_argument("--rag_bank", default=None)
    p.add_argument("--master_config", default=None)
    args = p.parse_args()
    run(args.rollout_dir, args.out_dir, args.demo_dir,
        args.rag_bank, args.master_config)
