"""Thin wrapper around upstream's LLM profile-analysis entry point.

We call ``Equivariant_pathway._analysis_common.run_profile_analysis``
with our own prompt addendums (built in :mod:`p4.prompts`) so the LLM
gets self-awareness fields + mode directive on top of the existing
budget addendum.

This keeps us out of the pipeline internals (Phase A/B/C, KAG/RAG,
response cache, OAI retry) while still letting us control what the
model sees per round.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway._analysis_common import run_profile_analysis  # noqa: E402


# Profile is shared with upstream P4 — we reuse it verbatim so KAG/RAG
# context and model selection match. Our addendums layer on top.
PROFILE_YAML = "p4_vlm_reasoning_kag_cross_plain_llm.yaml"
LABEL_BATCH = "p4_batch_seqbatch"
LABEL_SEQ = "p4_seq_seqbatch"
OUT_SUBDIR = "p4_analysis"


def run_analysis(
    rollout_dir: Path,
    out_dir: Path,
    demo_dir: Path,
    label: str,
    reasoning_addendum: str,
    aggregator_addendum: str,
) -> Dict:
    """Invoke the upstream P4 analysis pipeline with our addendums.

    Returns the analyzer's result dict; the prescribed layouts land at
    ``out_dir / "recommended_layouts.json"`` (or under the prescription
    report — :func:`flatten_recommendations` normalizes that path).
    """
    extra: Dict = {
        "llm": {
            "prompt_addendum_reasoning": reasoning_addendum,
            "prompt_addendum_aggregator": aggregator_addendum,
        },
        "tkf": {"demo_dir": str(demo_dir)},
    }
    # Cluster-portable model override: when the Qwen sbatch sets
    # LLM_MODEL_NAME / VLM_MODEL_NAME these flow into the upstream
    # pipeline's `llm.model` / `llm.vlm_model` config via _deep_merge,
    # so reasoning/aggregator/plain calls and vlm_analyser request
    # the Qwen served-model-names. Unset (OpenAI cluster) -> upstream
    # experiment_config.yaml defaults are used (no behavior change).
    env_llm = os.environ.get("LLM_MODEL_NAME")
    env_vlm = os.environ.get("VLM_MODEL_NAME")
    if env_llm:
        extra["llm"]["model"] = env_llm
    if env_vlm:
        extra["llm"]["vlm_model"] = env_vlm
    # Honor the upstream env knob for sequential Phase-B fan-out so the
    # OpenAI rate-limit story is identical.
    pbw = os.environ.get("P4_PHASE_B_MAX_WORKERS")
    if pbw:
        try:
            n = int(pbw)
            if n > 0:
                extra["pipeline"] = {"phase_b_max_workers": n}
        except (TypeError, ValueError):
            pass
    return run_profile_analysis(
        profile_yaml_name=PROFILE_YAML,
        rollout_dir=str(rollout_dir),
        out_subdir_name=OUT_SUBDIR,
        out_dir_override=str(out_dir),
        master_config_path=None,
        label=label,
        extra_overrides=extra,
    )


def flatten_recommendations(analysis_dir: Path, label: str) -> Optional[Path]:
    """Resolve the path to a flat ``recommended_layouts.json``.

    Upstream writes either a pre-flattened ``recommended_layouts.json``
    or a structured ``{label}_prescription_report.json`` that we need to
    flatten ourselves. This is the same logic upstream's
    ``p4_budget._flatten_recommended_layouts`` uses — kept here as a
    local copy so we don't take a dependency on upstream's specific
    label string.
    """
    rec_path = analysis_dir / "recommended_layouts.json"
    if rec_path.exists():
        return rec_path
    report_path = analysis_dir / f"{label}_prescription_report.json"
    if not report_path.exists():
        return None
    with open(report_path, "r") as f:
        report = json.load(f)
    flat = []
    for pres in report.get("prescription", {}).get("demonstration_prescriptions", []) or []:
        cluster = pres.get("cluster", "?")
        for li, lay in enumerate(pres.get("recommended_layouts", []) or []):
            flat.append({
                "parent_demo_id": cluster,
                "layout_index": li,
                "repetition": 1,
                "n_repetitions": int(lay.get("n_repetitions", 1) or 1),
                "start_pos": list(lay.get("start_pos", [])),
                "goal_pos": list(lay.get("goal_pos", [])),
                "fire_positions": [list(p) for p in (lay.get("fire_positions") or [])],
                "rationale": lay.get("rationale", ""),
                # `steps` (the corridor) is what we use for corridor
                # blocking; surface it explicitly even when upstream
                # didn't put it in the structured report.
                "steps": lay.get("steps", ""),
            })
    with open(rec_path, "w") as f:
        json.dump({"layouts": flat, "n_layouts": len(flat)}, f, indent=2)
    return rec_path


def cap_layouts(rec_path: Path, remaining: int, mode_max: Optional[int] = None) -> Dict:
    """Hard-cap a flat ``recommended_layouts.json`` to ``min(remaining,
    mode_max)`` entries. Returns an audit dict the round can persist.
    """
    payload = json.load(open(rec_path))
    layouts = payload.get("layouts", []) or []
    proposed = len(layouts)
    cap = remaining if mode_max is None else min(int(mode_max), remaining)
    if cap <= 0:
        kept = []
    else:
        kept = layouts[:cap]
    payload["layouts"] = kept
    payload["n_layouts"] = len(kept)
    payload["n_layouts_proposed"] = proposed
    payload["n_layouts_capped_by_budget"] = max(0, proposed - len(kept))
    with open(rec_path, "w") as f:
        json.dump(payload, f, indent=2)
    return {
        "proposed": proposed,
        "kept": len(kept),
        "capped": max(0, proposed - len(kept)),
        "cap": cap,
    }


def aggregator_reasoning_text(analysis_dir: Path, label: str) -> str:
    """Best-effort extraction of the aggregator's reasoning blob for
    ``prescription_overlap.json``. Falls back to empty when the report
    layout differs from what we expect.
    """
    report = analysis_dir / f"{label}_prescription_report.json"
    if not report.exists():
        return ""
    try:
        with open(report, "r") as f:
            data = json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return ""
    # Several possible keys depending on upstream version. We try a few.
    for k in ("cross_episode_reasoning", "aggregator_reasoning", "reasoning"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v
    pres = data.get("prescription") or {}
    for k in ("cross_episode_reasoning", "reasoning_text"):
        v = pres.get(k) if isinstance(pres, dict) else None
        if isinstance(v, str) and v.strip():
            return v
    return ""
