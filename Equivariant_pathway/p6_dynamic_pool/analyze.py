"""P6 analyzer wrapper for the p6_dynamic_pool pipeline.

Same mechanics as Equivariant_pathway/p6_only/analyze.py — runs the full
P6 stack (VLM + Reasoning + KAG + RAG + TKF + Cross-Episode + Plain-LLM
Aggregator) and injects three prompt addenda from p6_dynamic_pool/prompts/
into the merged llm config:

  llm.prompt_addendum_reasoning      <- prompts/reasoning.yml      (per-ep)
  llm.prompt_addendum_cross_episode  <- prompts/cross_episode.yml  (cross-ep)
  llm.prompt_addendum_aggregator     <- prompts/aggregator.yml     (final JSON)

The prompts are tuned for dynamic-pool-expansion framing: every prescribed
layout is being added to the active correction pool, so the LLM is told to
think of its output as POOL EXTENSIONS that the policy will be evaluated on
next round (not just additional demos for retraining).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

from Equivariant_pathway._analysis_common import run_profile_analysis

PROFILE_YAML = "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm.yaml"
LABEL = "p6_dynamic_pool"
OUT_SUBDIR = "p6_analysis"

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_addendum(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.yml"
    if not path.exists():
        return ""
    with open(path, "r") as f:
        spec = yaml.safe_load(f) or {}
    return str(spec.get("addendum", "") or "").strip()


def run(
    rollout_dir: str,
    out_dir: Optional[str] = None,
    demo_dir: Optional[str] = None,
    rag_bank: Optional[str] = None,
    master_config: Optional[str] = None,
) -> Dict:
    extra_overrides: Dict = {
        "llm": {
            "prompt_addendum_reasoning":     _load_addendum("reasoning"),
            "prompt_addendum_cross_episode": _load_addendum("cross_episode"),
            "prompt_addendum_aggregator":    _load_addendum("aggregator"),
        },
    }
    if demo_dir:
        extra_overrides["tkf"] = {"demo_dir": demo_dir}
    if rag_bank:
        extra_overrides["rag"] = {"bank_path": rag_bank}
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
    p.add_argument("--rag_bank", default=None)
    p.add_argument("--master_config", default=None)
    args = p.parse_args()
    run(args.rollout_dir, args.out_dir, args.demo_dir, args.rag_bank, args.master_config)
