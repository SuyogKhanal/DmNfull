"""P5 analyzer wrapper for the hybrid p5_only pipeline.

P5 = P4 + RAG. The pipeline.py caller passes a per-round ``rag_bank``
path so RAG retrievals from earlier rounds in this method's run feed
into later rounds — but never cross-contaminate other methods (P4/P6)
or other pool sizes (a different sweep would supply a different
P5_ONLY_ROOT).

Re-uses the shared Equivariant_pathway/_analysis_common helper (legal:
the hybrid lives inside Equivariant_pathway/) and ships the same
scaling-with-floor base prompt addenda as P4 plus a short P5 add-on
about leveraging the RAG-retrieved similar past failures during
clustering.
"""
from __future__ import annotations

from typing import Dict, Optional

from Equivariant_pathway._analysis_common import run_profile_analysis

PROFILE_YAML = "p5_vlm_reasoning_kag_rag_cross_plain_llm.yaml"
LABEL = "p5_only_hybrid"
OUT_SUBDIR = "p5_analysis"


REASONING_ADDENDUM = (
    "SAMPLE-EFFICIENCY DIRECTIVE — pick the smallest n_demos that closes the failure mode,\n"
    "but never zero when a failure is present.\n"
    "HARD FLOOR: this episode IS a failure. n_demos for this episode must be >= 1.\n"
    "RAG NOTE: when the 'Retrieved similar past failures' block above contains a case\n"
    "that matches this episode's failure mode, lean on the prior fix it implies rather\n"
    "than fabricating a brand-new one."
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
    "5. RAG NOTE: when the per-episode reasoning above cites RAG-retrieved similar past\n"
    "   failures, prefer to merge those failures into a single cluster rather than\n"
    "   creating cosmetically-distinct clusters. The retrieved cases are evidence the\n"
    "   failure mode is recurring; one well-chosen layout for the cluster is usually\n"
    "   the right call.\n"
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
