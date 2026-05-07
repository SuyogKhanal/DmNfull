"""P6 analyzer wrapper for the p6_only pipeline.

Calls the same `run_profile_analysis` entry point as
Equivariant_pathway/analyze_p6.py, so the underlying pipeline stack stays
the canonical P6 = VLM + Reasoning + KAG + RAG + TKF + Cross-Episode +
Plain-LLM Aggregator. The only difference is three prompt addenda loaded
from p6_only/prompts/*.yml and injected via `extra_overrides` into the
merged llm config:

  llm.prompt_addendum_reasoning      <- prompts/reasoning.yml      (per-ep)
  llm.prompt_addendum_cross_episode  <- prompts/cross_episode.yml  (cross-ep)
  llm.prompt_addendum_aggregator     <- prompts/aggregator.yml     (final JSON)

Edit any of those YAMLs to tune p6_only's prompting strategy without
touching code. Empty addendum = legacy prompt (so leaving a YAML's
`addendum:` field blank disables that hook for p6_only).

This is gated end-to-end: pipeline/reasoning.py and pipeline/aggregator.py
read these keys from llm_cfg and append when present, fall back to legacy
behaviour when absent. analyze_p4.py / analyze_p5.py / analyze_p6.py do
NOT set them, so the legacy multi-method run_full_cycle.py is unaffected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

from Equivariant_pathway._analysis_common import run_profile_analysis

PROFILE_YAML = "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm.yaml"
LABEL = "p6_only"
OUT_SUBDIR = "p6_analysis"

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_addendum(name: str) -> str:
    """Load prompts/<name>.yml and return its `addendum:` field. Missing
    file -> empty string -> legacy prompt is used."""
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
    """Run the P6 analysis pipeline with the p6_only prompt addenda."""
    extra_overrides: Dict = {
        "llm": {
            "prompt_addendum_reasoning":     _load_addendum("reasoning"),
            "prompt_addendum_cross_episode": _load_addendum("cross_episode"),
            "prompt_addendum_aggregator":    _load_addendum("aggregator"),
        },
    }
    # P6 uses TKF on the per-method demo bank; route it to p6_only/demos/
    # so the TKF coverage check operates on the bank that is actually
    # being trained on.
    if demo_dir:
        extra_overrides["tkf"] = {"demo_dir": demo_dir}
    # P6 also uses RAG. The active loop wants per-round isolation so
    # round N's RAG bank doesn't leak into round N+1.
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
