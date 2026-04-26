import difflib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def _read_prescription(run_dir: str, episode_id: int) -> str:
    p = Path(run_dir) / "episodes" / f"episode_{episode_id}" / "final_prescription.txt"
    if not p.exists():
        return "(no prescription saved)"
    return p.read_text(encoding="utf-8")


def _read_cross_prescription(run_dir: str) -> str:
    p = Path(run_dir) / "full_output.json"
    if not p.exists():
        return "(no full_output.json)"
    with open(p, "r") as f:
        data = json.load(f)
    parsed = data.get("phase_c", {}).get("parsed_prescription", {})
    if isinstance(parsed, dict) and parsed:
        return json.dumps(parsed, indent=2, default=str)
    return data.get("phase_c", {}).get("cross_episode_reasoning", "(empty)")


def prescription_pair(original_run_dir: str, rerun_run_dir: Optional[str], episode_id: Optional[int]) -> Tuple[str, str]:
    if episode_id is not None:
        orig = _read_prescription(original_run_dir, episode_id)
        new  = _read_prescription(rerun_run_dir, episode_id) if rerun_run_dir else "(no re-run)"
    else:
        orig = _read_cross_prescription(original_run_dir)
        new  = _read_cross_prescription(rerun_run_dir) if rerun_run_dir else "(no re-run)"
    return orig, new


def unified_diff(original: str, new: str, fromfile: str = "original", tofile: str = "rerun") -> str:
    a = (original or "").splitlines(keepends=False)
    b = (new or "").splitlines(keepends=False)
    diff = list(difflib.unified_diff(a, b, fromfile=fromfile, tofile=tofile, lineterm=""))
    if not diff:
        return "(no differences)"
    return "\n".join(diff)


def html_side_by_side(original: str, new: str, label_left: str = "Original", label_right: str = "Rerun") -> str:
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<div style='display:flex; gap:12px; font-family:monospace; font-size:13px;'>"
        f"<div style='flex:1; border:1px solid #ccc; padding:8px; white-space:pre-wrap;'>"
        f"<b>{esc(label_left)}</b><hr>{esc(original)}</div>"
        f"<div style='flex:1; border:1px solid #ccc; padding:8px; white-space:pre-wrap;'>"
        f"<b>{esc(label_right)}</b><hr>{esc(new)}</div>"
        "</div>"
    )