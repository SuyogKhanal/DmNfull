"""
dashboard/app.py
=================================================================================
MazeNav Profile Comparator — Gradio dashboard.

Layout
------
* Suite auto-loads the LATEST run under results/ablations/.
* Top bar: shared failure-episode selector + 3 frame thumbnails.
* Six profile tabs (P1 … P6), each with component sub-tabs:
    VLM Report | KAG Context | RAG Retrieved | Reasoning | TKF Verdict
    Per-Episode Prescription | Cross-Episode Reasoning | Final Prescription
  Tabs whose component was DISABLED for that profile show a gray notice.
* Bottom: Final Prescription row (all 6 profiles side-by-side in one tab each).
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import gradio as gr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ABLATIONS_DIR = REPO_ROOT / "results" / "ablations"

PROFILE_ORDER = [
    "p1_vlm_plain_llm",
    "p2_vlm_reasoning_plain_llm",
    "p3_vlm_reasoning_cross_plain_llm",
    "p4_vlm_reasoning_kag_cross_plain_llm",
    "p5_vlm_reasoning_kag_rag_cross_plain_llm",
    "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm",
]

PROFILE_LABELS = {
    "p1_vlm_plain_llm":                              "P1 · VLM + Plain LLM",
    "p2_vlm_reasoning_plain_llm":                    "P2 · VLM + Reasoning + Plain LLM",
    "p3_vlm_reasoning_cross_plain_llm":              "P3 · P2 + Cross-Ep Reasoning",
    "p4_vlm_reasoning_kag_cross_plain_llm":          "P4 · P3 + KAG",
    "p5_vlm_reasoning_kag_rag_cross_plain_llm":      "P5 · P4 + RAG",
    "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm": "P6 · P5 + TKF (Full)",
}

# Map pipeline flag → human label
FLAG_LABELS = {
    "use_vlm":                    "VLM",
    "use_reasoning":              "Reasoning",
    "use_cross_episode_reasoning":"Cross-Ep Reasoning",
    "use_kag":                    "KAG",
    "use_rag":                    "RAG",
    "use_tkf":                    "TKF",
    "use_plain_llm":              "Plain LLM",
    "use_aggregator":             "Aggregator",
}

# Which component tab maps to which pipeline flag (None = always shown)
COMPONENT_FLAG = {
    "VLM Report":               "use_vlm",
    "KAG Context":              "use_kag",
    "RAG Retrieved":            "use_rag",
    "Reasoning":                "use_reasoning",
    "TKF Verdict":              "use_tkf",
    "Per-Episode Prescription": None,
    "Cross-Ep Reasoning":       "use_cross_episode_reasoning",
    "Final Prescription":       None,
}

DISABLED_NOTICE = "— DISABLED for this profile —"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _latest_suite_dir() -> Optional[Path]:
    """Return the most-recently-created suite dir under results/ablations/."""
    if not ABLATIONS_DIR.exists():
        return None
    candidates = sorted(
        [d for d in ABLATIONS_DIR.iterdir() if d.is_dir() and (d / "suite_manifest.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_suite(suite_dir: Path) -> Dict:
    """
    Load all six profile full_output.json files from a suite directory.
    Returns a dict keyed by profile name.
    """
    data: Dict[str, Dict] = {}
    for profile in PROFILE_ORDER:
        fo_path = suite_dir / profile / "full_output.json"
        if fo_path.exists():
            try:
                with open(fo_path, "r") as f:
                    data[profile] = json.load(f)
            except Exception as e:
                data[profile] = {"_load_error": str(e)}
        else:
            data[profile] = {"_missing": True}
    return data


def _failure_episode_choices(suite_data: Dict) -> List[Tuple[str, int]]:
    """Extract failure episode IDs shared across profiles (from phase_a of any loaded profile)."""
    for profile_data in suite_data.values():
        phase_a = profile_data.get("phase_a", {})
        failure_ids = phase_a.get("failure_episode_ids", [])
        if failure_ids:
            return [(f"Episode {eid}", eid) for eid in sorted(failure_ids)]
    # Fallback: also check rollout/full_output.json
    return []


def _resolve_rollout_dir(suite_dir: Path) -> Optional[Path]:
    rd = suite_dir / "rollout"
    return rd if rd.exists() else None


def _get_frames(suite_dir: Path, episode_id: int) -> Dict:
    """Resolve start / highest-loss / end frame paths for an episode."""
    frames: Dict[str, Optional[str]] = {"start": None, "mid": None, "end": None}

    # Try rollout dir first
    rollout_dir = _resolve_rollout_dir(suite_dir)
    search_dirs = [rollout_dir] if rollout_dir else []
    # Also check each profile dir
    for profile in PROFILE_ORDER:
        pd = suite_dir / profile
        if pd.exists():
            search_dirs.append(pd)

    for base in search_dirs:
        if base is None:
            continue
        ep_dir = base / "episodes" / f"episode_{episode_id}"
        if not ep_dir.exists():
            # Try frames directly under base
            frames_dir = base / "frames" / f"episode_{episode_id}"
            if frames_dir.exists():
                ep_dir = frames_dir
            else:
                continue

        for fname, key in [("start_frame.png", "start"), ("frame_start.png", "start"),
                           ("highest_loss_frame.png", "mid"), ("frame_mid.png", "mid"),
                           ("end_frame.png", "end"), ("frame_end.png", "end")]:
            candidate = ep_dir / fname
            if candidate.exists() and frames[key] is None:
                frames[key] = str(candidate)

        if all(v is not None for v in frames.values()):
            break

    # Fallback: read frame_paths from full_output.json of any profile
    if any(v is None for v in frames.values()):
        for profile in PROFILE_ORDER:
            fo = suite_dir / profile / "full_output.json"
            if not fo.exists():
                continue
            try:
                with open(fo) as f:
                    fo_data = json.load(f)
                for ep in fo_data.get("phase_a", {}).get("all_rollouts", []):
                    if ep.get("episode_id") == episode_id:
                        fp = ep.get("frame_paths", {})
                        frames["start"] = frames["start"] or fp.get("start_frame")
                        frames["mid"]   = frames["mid"]   or fp.get("highest_loss_frame")
                        frames["end"]   = frames["end"]   or fp.get("end_frame")
                        break
            except Exception:
                pass
            if all(v is not None for v in frames.values()):
                break

    return frames


def _get_episode_outputs(profile_data: Dict, episode_id: int) -> Dict:
    """
    Extract all per-component text outputs for one episode from a profile's full_output.json.
    Returns a dict with keys matching COMPONENT_FLAG.
    """
    flags = profile_data.get("config", {}).get("pipeline", {})
    # Also check metadata.pipeline_flags as fallback
    if not flags:
        flags = profile_data.get("metadata", {}).get("pipeline_flags", {})

    per_ep = profile_data.get("phase_b", {}).get("per_episode", [])
    ep_data = next((e for e in per_ep if e.get("episode_id") == episode_id), {})

    phase_c = profile_data.get("phase_c", {})

    def _disabled(flag_key):
        return not bool(flags.get(flag_key, True))

    def _val(text, flag_key):
        if flag_key and _disabled(flag_key):
            return DISABLED_NOTICE
        return text or ""

    return {
        "VLM Report":               _val(ep_data.get("vlm_report", ""),        "use_vlm"),
        "KAG Context":              _val(ep_data.get("kag_context", ""),       "use_kag"),
        "RAG Retrieved":            _val(ep_data.get("rag_context", ""),       "use_rag"),
        "Reasoning":                _val(ep_data.get("reasoning_combined", ""),"use_reasoning"),
        "TKF Verdict":              _val(json.dumps(ep_data.get("tkf_result") or {}, indent=2), "use_tkf"),
        "Per-Episode Prescription": ep_data.get("prescription", ep_data.get("summary", "")),
        "Cross-Ep Reasoning":       _val(phase_c.get("cross_episode_reasoning", ""), "use_cross_episode_reasoning"),
        "Final Prescription":       json.dumps(phase_c.get("parsed_prescription", {}), indent=2),
        "_flags":                   flags,
    }


def _flags_badge_html(flags: Dict) -> str:
    """Render a small coloured badge row showing which flags are ON/OFF."""
    parts = []
    for flag, label in FLAG_LABELS.items():
        on = bool(flags.get(flag, False))
        colour = "#2d7a4f" if on else "#888"
        bg     = "#d4f0e0" if on else "#f0f0f0"
        parts.append(
            f'<span style="background:{bg};color:{colour};border:1px solid {colour};'
            f'border-radius:4px;padding:2px 7px;margin:2px;font-size:12px;'
            f'font-family:monospace;display:inline-block;">'
            f'{"✓" if on else "✗"} {label}</span>'
        )
    return "<div style='padding:6px 0'>" + "".join(parts) + "</div>"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _state_init() -> Dict:
    return {"suite_dir": None, "suite_data": {}, "episode_id": None}


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def on_load_suite(state: Dict):
    """Called on page load — find the latest suite and populate everything."""
    suite_dir = _latest_suite_dir()
    if suite_dir is None:
        status = "No completed suite found under results/ablations/. Run the ablation suite first."
        return _empty_outputs(state, status)

    suite_data = _load_suite(suite_dir)
    failures   = _failure_episode_choices(suite_data)
    first_eid  = failures[0][1] if failures else None

    state = {"suite_dir": str(suite_dir), "suite_data": suite_data, "episode_id": first_eid}
    status = (
        f"Suite loaded: {suite_dir.name}\n"
        f"Profiles found: {sum(1 for v in suite_data.values() if '_missing' not in v)} / {len(PROFILE_ORDER)}\n"
        f"Failure episodes: {len(failures)}"
    )

    ep_choices = gr.update(choices=failures, value=first_eid)
    ep_outputs = _build_all_episode_outputs(suite_data, first_eid, str(suite_dir))
    return (state, status, ep_choices) + ep_outputs


def on_select_episode(episode_id, state: Dict):
    if not state or not state.get("suite_data") or episode_id is None:
        return _empty_episode_outputs()
    state["episode_id"] = int(episode_id)
    suite_data = state["suite_data"]
    suite_dir  = state["suite_dir"]
    return _build_all_episode_outputs(suite_data, int(episode_id), suite_dir)


def _build_all_episode_outputs(suite_data: Dict, episode_id: Optional[int], suite_dir_str: str):
    """
    Returns a flat tuple of all Gradio output values:
      (start_img, mid_img, end_img,
       p1_badge, p1_vlm, p1_kag, p1_rag, p1_rsn, p1_tkf, p1_per, p1_cer, p1_final,
       p2_badge, ... x6 profiles)
    """
    suite_dir = Path(suite_dir_str) if suite_dir_str else None
    frames = _get_frames(suite_dir, episode_id) if suite_dir and episode_id is not None else {}

    result = [
        frames.get("start"),
        frames.get("mid"),
        frames.get("end"),
    ]

    for profile in PROFILE_ORDER:
        pdata   = suite_data.get(profile, {})
        outputs = _get_episode_outputs(pdata, episode_id) if episode_id is not None else {}
        flags   = outputs.get("_flags", {})
        result.append(_flags_badge_html(flags))                          # badge
        result.append(outputs.get("VLM Report",               ""))       # vlm
        result.append(outputs.get("KAG Context",              ""))       # kag
        result.append(outputs.get("RAG Retrieved",            ""))       # rag
        result.append(outputs.get("Reasoning",                ""))       # rsn
        result.append(outputs.get("TKF Verdict",              ""))       # tkf
        result.append(outputs.get("Per-Episode Prescription", ""))       # per_ep
        result.append(outputs.get("Cross-Ep Reasoning",       ""))       # cer
        result.append(outputs.get("Final Prescription",       ""))       # final

    return tuple(result)


def _empty_outputs(state, status):
    n_profile_outputs = len(PROFILE_ORDER) * 9  # badge + 8 text boxes
    return (
        state, status,
        gr.update(choices=[], value=None),
        None, None, None,
    ) + ("",) * n_profile_outputs


def _empty_episode_outputs():
    n_profile_outputs = len(PROFILE_ORDER) * 9
    return (None, None, None) + ("",) * n_profile_outputs


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="MazeNav Profile Comparator", theme=gr.themes.Soft()) as app:

        gr.Markdown("# MazeNav Profile Comparator")
        gr.Markdown(
            "Automatically loads the **latest suite run** from `results/ablations/`. "
            "Select a failure episode to see what each of the six profiles produced."
        )

        state = gr.State(_state_init())

        # ---- Top bar -------------------------------------------------------
        with gr.Row():
            load_status = gr.Textbox(
                label="Suite status", lines=3, interactive=False, scale=3
            )
            episode_dd = gr.Dropdown(
                label="Failure episode", choices=[], interactive=True, scale=1
            )

        with gr.Row():
            start_img = gr.Image(label="Start frame",        type="filepath", height=200)
            mid_img   = gr.Image(label="Highest-loss frame", type="filepath", height=200)
            end_img   = gr.Image(label="End frame",          type="filepath", height=200)

        # ---- Per-profile tabs ----------------------------------------------
        # We build one gr.Tab per profile.  Each tab contains:
        #   - a flag-badge HTML row
        #   - sub-tabs for each component
        # All textbox handles are stored in order so we can wire them up.

        all_badge_html:  List[gr.HTML]    = []
        all_vlm:         List[gr.Textbox] = []
        all_kag:         List[gr.Textbox] = []
        all_rag:         List[gr.Textbox] = []
        all_rsn:         List[gr.Textbox] = []
        all_tkf:         List[gr.Textbox] = []
        all_per_ep:      List[gr.Textbox] = []
        all_cer:         List[gr.Textbox] = []
        all_final:       List[gr.Textbox] = []

        with gr.Tabs():
            for profile in PROFILE_ORDER:
                label = PROFILE_LABELS.get(profile, profile)
                with gr.Tab(label):
                    badge_html = gr.HTML()
                    all_badge_html.append(badge_html)

                    with gr.Tabs():
                        with gr.Tab("VLM Report"):
                            tb = gr.Textbox(
                                lines=14, interactive=False,
                                placeholder="VLM frame-by-frame analysis of this failure."
                            )
                            all_vlm.append(tb)

                        with gr.Tab("KAG Context"):
                            tb = gr.Textbox(
                                lines=14, interactive=False,
                                placeholder="Domain knowledge injected via KAG."
                            )
                            all_kag.append(tb)

                        with gr.Tab("RAG Retrieved"):
                            tb = gr.Textbox(
                                lines=14, interactive=False,
                                placeholder="Past failure retrievals from the RAG bank."
                            )
                            all_rag.append(tb)

                        with gr.Tab("Reasoning"):
                            tb = gr.Textbox(
                                lines=20, interactive=False,
                                placeholder="Per-episode chain-of-thought reasoning."
                            )
                            all_rsn.append(tb)

                        with gr.Tab("TKF Verdict"):
                            tb = gr.Textbox(
                                lines=14, interactive=False,
                                placeholder="Training-Knowledge Fetcher result."
                            )
                            all_tkf.append(tb)

                        with gr.Tab("Per-Episode Prescription"):
                            tb = gr.Textbox(
                                lines=14, interactive=False,
                                placeholder="Plain LLM per-episode prescription."
                            )
                            all_per_ep.append(tb)

                        with gr.Tab("Cross-Ep Reasoning"):
                            tb = gr.Textbox(
                                lines=20, interactive=False,
                                placeholder="Cross-episode reasoning (Phase C)."
                            )
                            all_cer.append(tb)

                        with gr.Tab("Final Prescription"):
                            tb = gr.Textbox(
                                lines=20, interactive=False,
                                placeholder="Structured JSON prescription from the aggregator."
                            )
                            all_final.append(tb)

        # ---- Build the flat output list in the same order as _build_all_episode_outputs ----
        # (start_img, mid_img, end_img, then per profile: badge,vlm,kag,rag,rsn,tkf,per,cer,final)
        all_outputs = [start_img, mid_img, end_img]
        for i in range(len(PROFILE_ORDER)):
            all_outputs += [
                all_badge_html[i],
                all_vlm[i],
                all_kag[i],
                all_rag[i],
                all_rsn[i],
                all_tkf[i],
                all_per_ep[i],
                all_cer[i],
                all_final[i],
            ]

        # on_load_suite returns: state, status, ep_dd_update, + all_outputs
        load_outputs = [state, load_status, episode_dd] + all_outputs

        app.load(
            fn=on_load_suite,
            inputs=[state],
            outputs=load_outputs,
        )

        episode_dd.change(
            fn=on_select_episode,
            inputs=[episode_dd, state],
            outputs=all_outputs,
        )

    return app


def main():
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )


if __name__ == "__main__":
    main()
