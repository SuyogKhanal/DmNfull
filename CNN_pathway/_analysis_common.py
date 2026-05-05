"""Shared helper for the analyze_pN.py wrappers.

Each profile (P4, P5, P6) has its own thin wrapper script under CNN_pathway/
that calls `run_profile_analysis` here. The wrapper supplies the profile
YAML name and the output sub-directory; everything else (config merge,
pipeline rerun, prescription summarisation, on-disk report) is handled in
this module so the three wrappers cannot drift apart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from pipeline.pipeline_runner import rerun_pipeline_only

PROFILES_DIR = REPO_ROOT / "configs" / "ablation_profiles"
MASTER_CONFIG = REPO_ROOT / "configs" / "experiment_config.yaml"


def _load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _summarise_prescription(structured: Dict) -> List[Dict]:
    out: List[Dict] = []
    for pres in (structured or {}).get("demonstration_prescriptions", []) or []:
        out.append({
            "cluster": pres.get("cluster_id") or pres.get("cluster")
                       or pres.get("cluster_label"),
            "n_demos_needed": pres.get("n_demonstrations_needed")
                              or pres.get("n_demos") or pres.get("num_demos")
                              or len(pres.get("recommended_layouts", []) or []) or None,
            "recommended_layouts": pres.get("recommended_layouts", []) or [],
            "rationale": pres.get("rationale") or pres.get("reason") or "",
        })
    return out


def _expected_layout_for_failure(episode: Dict) -> Dict:
    dyn = episode.get("dynamic_config", {}) or {}
    def _coerce(p):
        if p is None:
            return None
        try:
            return [int(x) for x in p]
        except (TypeError, ValueError):
            return p
    fires = dyn.get("fire_positions") or []
    fires_clean = []
    for fp in fires:
        try:
            fires_clean.append([int(x) for x in fp])
        except (TypeError, ValueError):
            fires_clean.append(fp)
    return {
        "episode_id":     episode.get("episode_id"),
        "maze_name":      episode.get("maze_name"),
        "start_pos":      _coerce(dyn.get("start_pos")),
        "goal_pos":       _coerce(dyn.get("goal_pos")),
        "fire_positions": fires_clean,
        "success":        bool(episode.get("success", False)),
        "total_steps":    episode.get("total_steps"),
    }


def _deep_merge(base: Dict, override: Dict) -> Dict:
    import copy
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_profile_analysis(
    profile_yaml_name: str,
    rollout_dir: str,
    out_subdir_name: str,
    out_dir_override: Optional[str] = None,
    master_config_path: Optional[str] = None,
    label: str = "p?",
    extra_overrides: Optional[Dict] = None,
) -> Dict:
    """Run a single profile's analysis on an existing rollout directory.

    profile_yaml_name : name of the yaml under configs/ablation_profiles/.
    rollout_dir       : path produced by CNN_pathway/rollout_test.py.
    out_subdir_name   : default sub-folder under rollout_dir to write into.
    out_dir_override  : explicit out dir, takes precedence over the default.
    master_config_path: path to the master experiment config.
    label             : short tag (e.g. 'p4') used in print/report headers.
    extra_overrides   : extra config overrides merged on top of the profile
                        (e.g. {'tkf': {'demo_dir': 'CNN_pathway/demos'}}).
    """
    rollout = Path(rollout_dir).resolve()
    if not rollout.exists():
        raise SystemExit(f"rollout_dir does not exist: {rollout}")

    out_dir = Path(out_dir_override) if out_dir_override else rollout / out_subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_path = PROFILES_DIR / profile_yaml_name
    if not profile_path.exists():
        raise SystemExit(f"profile not found: {profile_path}")
    overrides = _load_yaml(profile_path)
    if extra_overrides:
        overrides = _deep_merge(overrides, extra_overrides)
    master_path = master_config_path or str(MASTER_CONFIG)

    print(f"[{label}] rollout_dir : {rollout}")
    print(f"[{label}] out_dir     : {out_dir}")
    print(f"[{label}] profile     : {profile_path.name}")
    print(f"[{label}] overrides   : {overrides}")

    full_output = rerun_pipeline_only(
        saved_run_dir=str(rollout),
        overrides=overrides,
        out_run_dir=str(out_dir),
        master_config_path=master_path,
    )

    structured = (full_output.get("phase_c", {}) or {}).get("parsed_prescription", {}) or {}
    prescriptions = _summarise_prescription(structured)
    failures = [
        _expected_layout_for_failure(e)
        for e in (full_output.get("phase_a", {}) or {}).get("all_rollouts", [])
        if not e.get("success", False)
    ]

    report = {
        "profile":     profile_path.name,
        "label":       label,
        "rollout_dir": str(rollout),
        "run_dir":     str(out_dir),
        "n_failures":  len(failures),
        "test_layout_failures": failures,
        "prescription": {
            "total_demonstrations_needed": structured.get("total_demonstrations_needed"),
            "demonstration_prescriptions": prescriptions,
            "failure_clusters":            structured.get("failure_clusters", []),
        },
    }
    report_path = out_dir / f"{label}_prescription_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"[{label}] SUMMARY")
    print("=" * 72)
    print(f"failures observed    : {len(failures)}")
    for fr in failures:
        print(f"  ep{fr['episode_id']} ({fr['maze_name']}): "
              f"start={fr['start_pos']} goal={fr['goal_pos']} fires={fr['fire_positions']} "
              f"steps={fr['total_steps']}")
    total = structured.get("total_demonstrations_needed", "?")
    print(f"\n{label} prescribed total_demos_needed = {total}")
    for pr in prescriptions:
        print(f"  cluster={pr['cluster']} n_demos={pr['n_demos_needed']} "
              f"layouts={len(pr['recommended_layouts'])}")
    print(f"\n[{label}] report → {report_path}")
    return report
