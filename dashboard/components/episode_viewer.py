"""
dashboard/components/episode_viewer.py
=========================================================================
Read failure-episode artefacts (frames, component outputs, metadata) from a
saved pipeline run directory.

This module is *defensive on purpose*: in the ablation-suite layout, frames
are stored only in the shared rollout sibling
(`results/ablations/<run>/rollout/episodes/episode_<id>/frames/`) and per-
profile dirs may or may not have saved `.txt` reports.  Every helper here
falls back through multiple resolution paths and never raises on missing
data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Top-level full_output.json access
# ---------------------------------------------------------------------------

def load_full_output(run_dir: str) -> Dict:
    p = Path(run_dir) / "full_output.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r") as f:
        return json.load(f)


def list_failure_episodes(full_output: Dict) -> List[Dict]:
    """
    Build the failure-episode dropdown list from the loaded full_output.

    Source of truth is `phase_a.all_rollouts` filtered by
    `phase_a.failure_episode_ids`.  Every field is read with `.get()` so that
    missing keys never crash the dashboard — the dropdown still populates
    even if some metadata fields weren't saved.
    """
    phase_a = full_output.get("phase_a", {}) or {}
    failure_ids = set(phase_a.get("failure_episode_ids", []) or [])
    out: List[Dict] = []
    for e in phase_a.get("all_rollouts", []) or []:
        eid = e.get("episode_id")
        if eid is None or eid not in failure_ids:
            continue
        dyn  = e.get("dynamic_config", {}) or {}
        seed = e.get("seed", "?")
        steps = e.get("total_steps", "?")
        try:
            reward = float(e.get("total_reward", 0.0) or 0.0)
        except Exception:
            reward = 0.0
        label = (
            f"Episode {eid} (seed={seed}, steps={steps}, reward={reward:.2f}) "
            f"start={dyn.get('start_pos','?')} goal={dyn.get('goal_pos','?')}"
        )
        out.append({
            "episode_id":     eid,
            "seed":           seed,
            "maze_name":      e.get("maze_name", ""),
            "total_steps":    steps,
            "total_reward":   reward,
            "label":          label,
            "frame_paths":    e.get("frame_paths", {}) or {},
            "key_frames":     e.get("key_frames", []) or [],
            "dynamic_config": dyn,
            "ascii_grid":     e.get("ascii_grid", "") or "",
        })
    return out


def dropdown_choices(failure_episodes: List[Dict]) -> List[Tuple[str, int]]:
    return [(ep["label"], ep["episode_id"]) for ep in failure_episodes]


# ---------------------------------------------------------------------------
# Frame resolution
# ---------------------------------------------------------------------------

# Frames are saved by `pipeline/rollout.py::_save_key_frames` as:
#   start.png       (role=start_frame)
#   high_loss.png   (role=highest_loss_frame)
#   end.png         (role=end_frame)
# We also accept a few historical / alternative names just in case.
_FILENAME_CANDIDATES: Dict[str, List[str]] = {
    "start_frame": [
        "start.png", "start_frame.png", "frame_start.png",
        "frame_0000.png", "frame_0.png", "step_0.png",
    ],
    "highest_loss_frame": [
        "high_loss.png", "highest_loss.png", "high_loss_frame.png",
        "highest_loss_frame.png", "loss_peak.png", "worst.png",
    ],
    "end_frame": [
        "end.png", "end_frame.png", "frame_end.png",
        "frame_last.png", "last.png",
    ],
}


def _candidate_frames_dirs(run_dir: Path, episode_id: int) -> List[Path]:
    """
    Return every plausible directory where frames for this episode might live,
    in priority order.  We walk up to 3 ancestors looking for a `rollout/`
    sibling — this handles the ablation-suite layout
    (`results/ablations/<run>/<profile>/`) and the single-run layout
    (`results/runs/<run>/`) without hard-coding either.
    """
    leaf = f"episodes/episode_{episode_id}/frames"
    cands: List[Path] = []

    # 1. Inside the profile/run dir itself.
    cands.append(run_dir / leaf)

    # 2. Sibling rollout/ (ablation suite — the common case).
    cands.append(run_dir.parent / "rollout" / leaf)

    # 3. Walk up further in case of deeper nesting.
    for ancestor in [run_dir.parent.parent, run_dir.parent.parent.parent]:
        if ancestor and ancestor != ancestor.parent:
            cands.append(ancestor / "rollout" / leaf)

    # 4. Inside the run dir itself, with a `rollout/` child (single-run layout
    #    where Phase A and Phase B are co-located).
    cands.append(run_dir / "rollout" / leaf)

    # Deduplicate while preserving order.
    seen = set()
    unique: List[Path] = []
    for c in cands:
        s = str(c)
        if s not in seen:
            seen.add(s)
            unique.append(c)
    return unique


def _resolve_frame_path_value(value: str, run_dir: Path) -> Optional[str]:
    """A path stored in `frame_paths` may be absolute or relative.  Try both."""
    if not value:
        return None
    p = Path(value)
    if p.is_absolute() and p.exists():
        return str(p)
    # Relative: try it as-is (cwd), then relative to run_dir, then relative to
    # the run's parent (suite root).
    for base in [Path.cwd(), run_dir, run_dir.parent]:
        cand = (base / p) if not p.is_absolute() else p
        if cand.exists():
            return str(cand)
    return None


def _resolve_frame(
    run_dir: Path,
    episode_id: int,
    frame_key: str,
    frame_paths: Dict,
    key_frames: Optional[List[Dict]] = None,
) -> Optional[str]:
    """
    Locate a single frame image by trying, in order:

      1. The path recorded in `frame_paths[frame_key]` (absolute or relative).
      2. Each candidate `episodes/episode_<id>/frames/` directory across the
         profile dir, the sibling `rollout/`, and ancestor rollout dirs —
         using the known filename candidates.
      3. The same candidate dirs but globbing for `frame_<idx>.png` /
         `step_<idx>.png` where `<idx>` comes from `key_frames`.
      4. As a final fallback, list any `*.png` in the first existing frames
         dir and pick by filename heuristic.
    """
    # 1) Recorded path.
    p = _resolve_frame_path_value(frame_paths.get(frame_key, ""), run_dir)
    if p:
        return p

    candidate_dirs = _candidate_frames_dirs(run_dir, episode_id)

    # 2) Known filenames in any candidate dir.
    for fdir in candidate_dirs:
        if not fdir.exists():
            continue
        for fname in _FILENAME_CANDIDATES.get(frame_key, []):
            p = fdir / fname
            if p.exists():
                return str(p)

    # 3) Use key_frames step_idx as a last-resort filename hint.
    role_to_idx: Dict[str, int] = {}
    for kf in (key_frames or []):
        role = kf.get("role")
        if role and "step_idx" in kf:
            role_to_idx[role] = int(kf["step_idx"])
    idx = role_to_idx.get(frame_key)
    if idx is not None:
        idx_names = [
            f"frame_{idx:04d}.png", f"frame_{idx}.png",
            f"step_{idx:04d}.png", f"step_{idx}.png",
            f"{idx:04d}.png", f"{idx}.png",
        ]
        for fdir in candidate_dirs:
            if not fdir.exists():
                continue
            for fname in idx_names:
                p = fdir / fname
                if p.exists():
                    return str(p)

    # 4) Glob fallback — any PNG, sorted, pick by heuristic.
    for fdir in candidate_dirs:
        if not fdir.exists():
            continue
        pngs = sorted(fdir.glob("*.png"))
        if not pngs:
            continue
        if frame_key == "start_frame":
            return str(pngs[0])
        if frame_key == "end_frame":
            return str(pngs[-1])
        # highest_loss_frame: use the middle file as a best-effort guess.
        return str(pngs[len(pngs) // 2])

    return None


def get_episode_frames(run_dir: str, episode_id: int) -> Dict[str, Optional[str]]:
    """
    Resolve filepaths for {start, highest_loss, end} frames of a failure
    episode.  Per spec, the resolution sources are tried in this order:

      A. `frame_paths` from `phase_b.per_episode[i]`  (rerun output)
      B. `frame_paths` from `phase_a.all_rollouts[i]` (rollout output)
      C. `key_frames`  from `phase_a.all_rollouts[i]` (used as filename hint)
      D. Direct read of `<rollout>/episodes/episode_<id>/episode_data.json`
         which holds the canonical key_frames + frame_paths.

    Each frame is then located on disk via `_resolve_frame`, which itself
    tries multiple directories and filename conventions.  Returns `None` for
    any frame that cannot be located — never crashes.
    """
    run = Path(run_dir)
    frame_paths: Dict = {}
    key_frames: List[Dict] = []

    # A + B + C — read from full_output.json.
    full_out_path = run / "full_output.json"
    if full_out_path.exists():
        try:
            with open(full_out_path, "r") as f:
                fo = json.load(f)
            # A) phase_b first
            for ep in (fo.get("phase_b", {}) or {}).get("per_episode", []) or []:
                if ep.get("episode_id") == episode_id:
                    fp = ep.get("frame_paths") or {}
                    if fp:
                        frame_paths = dict(fp)
                    break
            # B + C) phase_a
            for ep in (fo.get("phase_a", {}) or {}).get("all_rollouts", []) or []:
                if ep.get("episode_id") == episode_id:
                    if not frame_paths:
                        frame_paths = dict(ep.get("frame_paths") or {})
                    key_frames = list(ep.get("key_frames") or [])
                    break
        except Exception:
            pass

    # D) Direct read of episode_data.json from the rollout dir, walking up.
    if not frame_paths or not key_frames:
        for fdir in _candidate_frames_dirs(run, episode_id):
            ed_path = fdir.parent / "episode_data.json"   # episodes/episode_<id>/episode_data.json
            if ed_path.exists():
                try:
                    with open(ed_path, "r") as f:
                        ed = json.load(f)
                    if not frame_paths:
                        frame_paths = dict(ed.get("frame_paths") or {})
                    if not key_frames:
                        key_frames = list(ed.get("key_frames") or [])
                except Exception:
                    pass
                break

    return {
        key: _resolve_frame(run, episode_id, key, frame_paths, key_frames)
        for key in ("start_frame", "highest_loss_frame", "end_frame")
    }


# ---------------------------------------------------------------------------
# Per-component output (VLM / KAG / RAG / Reasoning / TKF / Prescription)
# ---------------------------------------------------------------------------

# UI key  ->  (txt-file basename in <run>/episodes/episode_<id>/,
#              field name in phase_b.per_episode[i])
_COMPONENT_FIELDS: Dict[str, Tuple[str, str]] = {
    "vlm_report":         ("vlm_report.txt",         "vlm_report"),
    "kag_context":        ("kag_context.txt",        "kag_context"),
    "rag_retrieved":      ("rag_retrieved.txt",      "rag_context"),
    "reasoning":          ("reasoning.txt",          "reasoning_combined"),
    "tkf_result":         ("tkf_result.json",        "tkf_result"),
    "final_prescription": ("final_prescription.txt", "prescription"),
}


def _read_text_file(p: Path) -> Optional[str]:
    """Return file contents, or None if missing/empty/unreadable."""
    if not p.exists():
        return None
    try:
        if p.suffix == ".json":
            with open(p, "r") as f:
                data = json.load(f)
            return json.dumps(data, indent=2, default=str)
        s = p.read_text(encoding="utf-8")
        return s if s.strip() else None
    except Exception:
        return None


def _read_from_phase_b(run: Path, episode_id: int, phase_b_field: str) -> Optional[str]:
    """Read a single field from `phase_b.per_episode[i]` in full_output.json."""
    full_out_path = run / "full_output.json"
    if not full_out_path.exists():
        return None
    try:
        with open(full_out_path, "r") as f:
            fo = json.load(f)
    except Exception:
        return None
    for ep in (fo.get("phase_b", {}) or {}).get("per_episode", []) or []:
        if ep.get("episode_id") != episode_id:
            continue
        val = ep.get(phase_b_field)
        if val is None:
            return None
        if isinstance(val, (dict, list)):
            return json.dumps(val, indent=2, default=str)
        s = str(val)
        return s if s.strip() else None
    return None


def get_episode_component_outputs(run_dir: str, episode_id: int) -> Dict[str, str]:
    """
    Read VLM/KAG/RAG/Reasoning/TKF/Prescription outputs for an episode.

    Resolution order per field:
      1. The pipeline's saved text file under
         `<run_dir>/episodes/episode_<id>/<field>.txt` (or `.json` for TKF).
      2. The corresponding field inside
         `<run_dir>/full_output.json -> phase_b.per_episode[i]`.

    Empty / whitespace-only file content is treated as missing and falls
    through to source 2.  Truly absent values render as "(not available)" so
    the user can tell the difference from a blank/disabled component.
    """
    run = Path(run_dir)
    edir = run / "episodes" / f"episode_{episode_id}"
    out: Dict[str, str] = {}
    for ui_key, (filename, phase_b_field) in _COMPONENT_FIELDS.items():
        # 1) on-disk text/json file
        val = _read_text_file(edir / filename)
        # 2) phase_b fallback
        if val is None:
            val = _read_from_phase_b(run, episode_id, phase_b_field)
        out[ui_key] = val if val is not None else "(not available)"
    return out


# ---------------------------------------------------------------------------
# Episode metadata
# ---------------------------------------------------------------------------

def get_episode_metadata(
    failure_episodes: List[Dict],
    episode_id: int,
    run_dir: Optional[str] = None,
) -> Dict:
    """
    Look up episode metadata (seed, dynamic_config, ascii_grid, maze_name).

    Resolution order:
      1. The pre-built `failure_episodes` list (built from `phase_a` at load
         time — this is the fast path).
      2. `phase_a.all_rollouts` directly inside `<run_dir>/full_output.json`.
      3. `phase_b.per_episode` in the same file (carries seed +
         dynamic_config but not ascii_grid / maze_name).
      4. `<rollout>/episodes/episode_<id>/episode_data.json` — the canonical
         per-episode artefact written by Phase A; reachable from any profile
         dir by walking up to the sibling `rollout/`.
    """
    # 1) failure_episodes
    for ep in failure_episodes or []:
        if ep.get("episode_id") == episode_id:
            return ep

    if not run_dir:
        return {}

    run = Path(run_dir)
    full_out_path = run / "full_output.json"

    # 2 + 3) full_output.json
    if full_out_path.exists():
        try:
            with open(full_out_path, "r") as f:
                fo = json.load(f)
            for ep in (fo.get("phase_a", {}) or {}).get("all_rollouts", []) or []:
                if ep.get("episode_id") == episode_id:
                    return ep
            for ep in (fo.get("phase_b", {}) or {}).get("per_episode", []) or []:
                if ep.get("episode_id") == episode_id:
                    return ep
        except Exception:
            pass

    # 4) episode_data.json from the rollout dir
    for fdir in _candidate_frames_dirs(run, episode_id):
        ed_path = fdir.parent / "episode_data.json"
        if ed_path.exists():
            try:
                with open(ed_path, "r") as f:
                    return json.load(f)
            except Exception:
                continue

    return {}