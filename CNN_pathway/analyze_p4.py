"""Run the existing P4 LLM analysis pipeline on the CNN+MLP test rollout.

Pipeline reuse:
  - The rollout dir produced by `rollout_test.py` is shaped exactly like the
    one the ablation suite produces for Phase A, so we hand it straight to
    `pipeline.pipeline_runner.rerun_pipeline_only` with the P4 ablation
    profile merged on top of the master experiment config. No new LLM code
    is written here — the existing VLM analyser, KAG loader, reasoning
    chain and aggregator do all the work.

  - After the pipeline finishes we extract the structured prescription
    (demonstration_prescriptions / total_demonstrations_needed) and write a
    side-by-side report of {test layout failure} vs {P4 prescription} so we
    can answer the question: did P4 prescribe exactly the demonstrations
    these failures need?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from pipeline.pipeline_runner import rerun_pipeline_only

P4_PROFILE = REPO_ROOT / "configs" / "ablation_profiles" / "p4_vlm_reasoning_kag_cross_plain_llm.yaml"
MASTER_CONFIG = REPO_ROOT / "configs" / "experiment_config.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", type=str, required=True,
                   help="Run directory produced by CNN_pathway/rollout_test.py")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to write the P4 analysis. Defaults to "
                        "<rollout_dir>/p4_analysis.")
    p.add_argument("--master_config", type=str, default=str(MASTER_CONFIG))
    return p.parse_args()


def _load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _build_p4_overrides() -> Dict:
    overrides = _load_yaml(str(P4_PROFILE))
    return overrides


def _summarise_prescription(structured: Dict) -> List[Dict]:
    out: List[Dict] = []
    for pres in (structured or {}).get("demonstration_prescriptions", []) or []:
        out.append({
            "cluster": pres.get("cluster_id") or pres.get("cluster"),
            "n_demos_needed": pres.get("n_demonstrations_needed")
                              or pres.get("n_demos") or pres.get("num_demos"),
            "recommended_layouts": pres.get("recommended_layouts", []) or [],
            "rationale": pres.get("rationale") or pres.get("reason") or "",
        })
    return out


def _expected_layout_for_failure(episode: Dict) -> Dict:
    dyn = episode.get("dynamic_config", {}) or {}
    return {
        "episode_id": episode.get("episode_id"),
        "maze_name":  episode.get("maze_name"),
        "start_pos":  dyn.get("start_pos"),
        "goal_pos":   dyn.get("goal_pos"),
        "fire_positions": dyn.get("fire_positions"),
        "success":    bool(episode.get("success", False)),
        "total_steps": episode.get("total_steps"),
    }


def main():
    args = parse_args()
    rollout_dir = Path(args.rollout_dir).resolve()
    if not rollout_dir.exists():
        raise SystemExit(f"rollout_dir does not exist: {rollout_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else rollout_dir / "p4_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = _build_p4_overrides()
    print(f"[p4] rollout_dir : {rollout_dir}")
    print(f"[p4] out_dir     : {out_dir}")
    print(f"[p4] overrides   : {overrides}")

    full_output = rerun_pipeline_only(
        saved_run_dir=str(rollout_dir),
        overrides=overrides,
        out_run_dir=str(out_dir),
        master_config_path=args.master_config,
    )

    structured = (full_output.get("phase_c", {}) or {}).get("parsed_prescription", {}) or {}
    prescriptions = _summarise_prescription(structured)
    failures = [
        _expected_layout_for_failure(e)
        for e in (full_output.get("phase_a", {}) or {}).get("all_rollouts", [])
        if not e.get("success", False)
    ]

    report = {
        "rollout_dir": str(rollout_dir),
        "p4_run_dir":  str(out_dir),
        "n_failures":  len(failures),
        "test_layout_failures": failures,
        "p4_prescription": {
            "total_demonstrations_needed": structured.get("total_demonstrations_needed"),
            "demonstration_prescriptions": prescriptions,
            "failure_clusters":          structured.get("failure_clusters", []),
        },
    }
    report_path = out_dir / "p4_prescription_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("[p4] SUMMARY")
    print("=" * 72)
    print(f"failures observed    : {len(failures)}")
    for f in failures:
        print(f"  ep{f['episode_id']} ({f['maze_name']}): "
              f"start={f['start_pos']} goal={f['goal_pos']} fires={f['fire_positions']} "
              f"steps={f['total_steps']}")
    total = structured.get("total_demonstrations_needed", "?")
    print(f"\nP4 prescribed total_demos_needed = {total}")
    for p in prescriptions:
        print(f"  cluster={p['cluster']} n_demos={p['n_demos_needed']} "
              f"layouts={len(p['recommended_layouts'])}")
    print(f"\n[p4] report → {report_path}")


if __name__ == "__main__":
    main()
