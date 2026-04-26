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
                "label":          f"Episode {e['episode_id']} (seed={e['seed']}, steps={e.get('total_steps','?')}, reward={e.get('total_reward',0.0):.2f}) start={dyn.get('start_pos','?')} goal={dyn.get('goal_pos','?')}",
                "frame_paths":    e.get("frame_paths", {}),
                "dynamic_config": dyn,
                "ascii_grid":     e.get("ascii_grid", ""),
            })
    return out


def dropdown_choices(failure_episodes: List[Dict]) -> List[Tuple[str, int]]:
    return [(ep["label"], ep["episode_id"]) for ep in failure_episodes]


def get_episode_frames(run_dir: str, episode_id: int) -> Dict[str, Optional[str]]:
    run = Path(run_dir)
    fdir = run / "episodes" / f"episode_{episode_id}" / "frames"
    mapping = {
        "start_frame":        fdir / "start.png",
        "highest_loss_frame": fdir / "high_loss.png",
        "end_frame":          fdir / "end.png",
    }
    return {k: (str(v) if v.exists() else None) for k, v in mapping.items()}


def get_episode_component_outputs(run_dir: str, episode_id: int) -> Dict[str, str]:
    run = Path(run_dir)
    edir = run / "episodes" / f"episode_{episode_id}"
    fields = {
        "vlm_report":          edir / "vlm_report.txt",
        "kag_context":         edir / "kag_context.txt",
        "rag_retrieved":       edir / "rag_retrieved.txt",
        "reasoning":           edir / "reasoning.txt",
        "tkf_result":          edir / "tkf_result.json",
        "final_prescription":  edir / "final_prescription.txt",
    }
    out = {}
    for key, p in fields.items():
        if not p.exists():
            out[key] = "(not available)"
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


def get_episode_metadata(failure_episodes: List[Dict], episode_id: int) -> Dict:
    for ep in failure_episodes:
        if ep["episode_id"] == episode_id:
            return ep
    return {}