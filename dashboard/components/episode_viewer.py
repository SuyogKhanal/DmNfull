import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_full_output(run_dir: str) -> Dict:
    p = Path(run_dir) / "full_output.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r") as f:
        return json.load(f)


def list_failure_episodes(full_output: Dict) -> List[Dict]:
    failure_ids = set(full_output.get("phase_a", {}).get("failure_episode_ids", []))
    out = []
    for e in full_output.get("phase_a", {}).get("all_rollouts", []):
        if e.get("episode_id") in failure_ids:
            dyn = e.get("dynamic_config", {})
            out.append({
                "episode_id":     e["episode_id"],
                "seed":           e["seed"],
                "total_steps":    e.get("total_steps"),
                "total_reward":   e.get("total_reward"),
                "label":          (
                    f"Episode {e['episode_id']} "
                    f"(seed={e['seed']}, steps={e.get('total_steps','?')}, "
                    f"reward={e.get('total_reward',0.0):.2f}) "
                    f"start={dyn.get('start_pos','?')} goal={dyn.get('goal_pos','?')}"
                ),
                "frame_paths":    e.get("frame_paths", {}),
                "key_frames":     e.get("key_frames", {}),
                "dynamic_config": dyn,
                "ascii_grid":     e.get("ascii_grid", ""),
            })
    return out


def dropdown_choices(failure_episodes: List[Dict]) -> List[Tuple[str, int]]:
    return [(ep["label"], ep["episode_id"]) for ep in failure_episodes]


def _resolve_frame(run_dir: Path, episode_id: int, frame_key: str, frame_paths: Dict) -> Optional[str]:
    """
    Try multiple locations to find a frame image.

    Priority 1 - absolute paths from full_output.json frame_paths (recorded at run time).
    Priority 2 - standard relative path under the profile run_dir.
    Priority 3 - sibling rollout/ dir (ablation suite stores frames there).
    """
    filename_map = {
        "start_frame":        ["start.png", "frame_0000.png"],
        "highest_loss_frame": ["high_loss.png", "highest_loss.png", "loss_peak.png"],
        "end_frame":          ["end.png", "frame_last.png"],
    }

    # Priority 1: absolute paths from saved JSON
    if frame_key in frame_paths:
        p = Path(frame_paths[frame_key])
        if p.exists():
            return str(p)

    # Priority 2: relative to profile run_dir
    fdir = run_dir / "episodes" / f"episode_{episode_id}" / "frames"
    for fname in filename_map.get(frame_key, []):
        p = fdir / fname
        if p.exists():
            return str(p)

    # Priority 3: sibling rollout/ dir
    rollout_sibling = run_dir.parent / "rollout"
    if rollout_sibling.exists():
        rfdir = rollout_sibling / "episodes" / f"episode_{episode_id}" / "frames"
        for fname in filename_map.get(frame_key, []):
            p = rfdir / fname
            if p.exists():
                return str(p)

    return None


def get_episode_frames(run_dir: str, episode_id: int) -> Dict[str, Optional[str]]:
    run = Path(run_dir)

    # Load frame_paths from full_output.json for absolute path resolution
    frame_paths: Dict = {}
    full_out_path = run / "full_output.json"
    if full_out_path.exists():
        try:
            with open(full_out_path, "r") as f:
                fo = json.load(f)
            for ep in fo.get("phase_a", {}).get("all_rollouts", []):
                if ep.get("episode_id") == episode_id:
                    frame_paths = ep.get("frame_paths", {})
                    break
        except Exception:
            pass

    return {
        key: _resolve_frame(run, episode_id, key, frame_paths)
        for key in ["start_frame", "highest_loss_frame", "end_frame"]
    }


def get_episode_component_outputs(run_dir: str, episode_id: int) -> Dict[str, str]:
    """
    Read per-episode component outputs.  Primary: text files saved by the
    pipeline under <run_dir>/episodes/episode_<id>/.  Fallback: read directly
    from full_output.json phase_b when files are missing.
    """
    run = Path(run_dir)
    edir = run / "episodes" / f"episode_{episode_id}"

    fields = {
        "vlm_report":         edir / "vlm_report.txt",
        "kag_context":        edir / "kag_context.txt",
        "rag_retrieved":      edir / "rag_retrieved.txt",
        "reasoning":          edir / "reasoning.txt",
        "tkf_result":         edir / "tkf_result.json",
        "final_prescription": edir / "final_prescription.txt",
    }

    out = {}
    for key, p in fields.items():
        if not p.exists():
            out[key] = _read_from_phase_b(run, episode_id, key)
            continue
        try:
            if p.suffix == ".json":
                with open(p, "r") as f:
                    data = json.load(f)
                out[key] = json.dumps(data, indent=2, default=str)
            else:
                out[key] = p.read_text(encoding="utf-8")
        except Exception as e:
            out[key] = f"(error loading {p.name}: {e})"
    return out


def _read_from_phase_b(run: Path, episode_id: int, field: str) -> str:
    """Extract component output from full_output.json phase_b as fallback."""
    full_out_path = run / "full_output.json"
    if not full_out_path.exists():
        return "(not available)"
    try:
        with open(full_out_path, "r") as f:
            fo = json.load(f)
        for ep in fo.get("phase_b", {}).get("per_episode", []):
            if ep.get("episode_id") == episode_id:
                key_map = {
                    "vlm_report":         "vlm_report",
                    "kag_context":        "kag_context",
                    "rag_retrieved":      "rag_context",
                    "reasoning":          "reasoning_combined",
                    "tkf_result":         "tkf_result",
                    "final_prescription": "prescription",
                }
                val = ep.get(key_map.get(field, field), "(not available)")
                if isinstance(val, dict):
                    return json.dumps(val, indent=2, default=str)
                return str(val) if val else "(not available)"
    except Exception as e:
        return f"(error reading phase_b: {e})"
    return "(not available)"


def get_episode_metadata(failure_episodes: List[Dict], episode_id: int) -> Dict:
    for ep in failure_episodes:
        if ep["episode_id"] == episode_id:
            return ep
    return {}
