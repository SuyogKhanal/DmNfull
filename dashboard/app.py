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

from pipeline.pipeline_runner import rerun_pipeline_only
from dashboard.components.episode_viewer import (
    load_full_output,
    list_failure_episodes,
    get_episode_frames,
    get_episode_component_outputs,
    get_episode_metadata,
)
from dashboard.components.prescription_diff import prescription_pair, html_side_by_side, unified_diff
from dashboard.components.performance_graph import build_performance_figure


RUNS_DIR_DEFAULT      = str(REPO_ROOT / "results" / "runs")
ABLATIONS_DIR_DEFAULT = str(REPO_ROOT / "results" / "ablations")
BASELINES_DIR_DEFAULT = str(REPO_ROOT / "results" / "baselines")


def _state_init() -> Dict:
    return {
        "run_dir":          None,
        "rerun_dir":        None,
        "failure_episodes": [],
    }


def _list_saved_runs() -> List[str]:
    """
    Scan results/runs, results/ablations, results/baselines for any directory
    (up to 2 levels deep) that contains a full_output.json file.
    Ablation suite layout: results/ablations/<run_TIMESTAMP>/<profile>/full_output.json
    Single run layout:     results/runs/<run_TIMESTAMP>/full_output.json
    """
    out = []
    for base in [RUNS_DIR_DEFAULT, ABLATIONS_DIR_DEFAULT, BASELINES_DIR_DEFAULT]:
        p = Path(base)
        if not p.exists():
            continue
        for d in sorted(p.iterdir()):
            if not d.is_dir():
                continue
            # Direct child has full_output.json (single run)
            if (d / "full_output.json").exists():
                out.append(str(d))
            else:
                # One level deeper: profile subdirs inside an ablation suite run
                for sub in sorted(d.iterdir()):
                    if sub.is_dir() and (sub / "full_output.json").exists():
                        out.append(str(sub))
    return out


def on_load_run(run_dir: str, state: Dict):
    state = dict(state or {})
    state["run_dir"] = run_dir
    state["rerun_dir"] = None
    try:
        full = load_full_output(run_dir)
    except Exception as e:
        state["failure_episodes"] = []
        return (
            state,
            gr.update(choices=[], value=None),
            f"Error loading run: {e}",
            None, None, None,
            "", "", "", "", "", "",
            "", "",
        )
    failures = list_failure_episodes(full)
    state["failure_episodes"] = failures
    choices = [(f["label"], f["episode_id"]) for f in failures]
    value = choices[0][1] if choices else None
    status = f"Loaded run: {run_dir}\n{len(failures)} failure episodes found."
    return (
        state,
        gr.update(choices=choices, value=value),
        status,
        None, None, None,
        "", "", "", "", "", "",
        "", "",
    )


def on_select_episode(episode_id: Optional[int], state: Dict):
    if not state or not state.get("run_dir") or episode_id is None:
        return (None, None, None, "", "", "", "", "", "", "", "")
    run_dir = state["run_dir"]
    frames = get_episode_frames(run_dir, int(episode_id))
    outputs = get_episode_component_outputs(run_dir, int(episode_id))
    meta = get_episode_metadata(state.get("failure_episodes", []), int(episode_id))
    meta_text = json.dumps({
        "episode_id":     meta.get("episode_id"),
        "seed":           meta.get("seed"),
        "dynamic_config": meta.get("dynamic_config", {}),
        "ascii_grid":     meta.get("ascii_grid", ""),
    }, indent=2, default=str)
    return (
        frames.get("start_frame"),
        frames.get("highest_loss_frame"),
        frames.get("end_frame"),
        outputs.get("vlm_report", ""),
        outputs.get("kag_context", ""),
        outputs.get("rag_retrieved", ""),
        outputs.get("reasoning", ""),
        outputs.get("tkf_result", ""),
        outputs.get("final_prescription", ""),
        meta_text,
        "",
    )


def on_rerun(
    use_vlm: bool,
    use_kag: bool,
    use_rag: bool,
    use_reasoning: bool,
    use_tkf: bool,
    use_aggregator: bool,
    state: Dict,
):
    state = dict(state or {})
    if not state.get("run_dir"):
        return state, "Load a saved run first.", "", "", ""

    # The rerun source must be the rollout dir (shared Phase A data).
    # For profile dirs inside a suite run, walk up to find the sibling rollout/.
    run_dir = Path(state["run_dir"])
    rollout_dir = run_dir / "rollout"
    if not rollout_dir.exists():
        parent_rollout = run_dir.parent / "rollout"
        if parent_rollout.exists() and (parent_rollout / "full_output.json").exists():
            rollout_dir = parent_rollout
        else:
            rollout_dir = run_dir  # fallback: use as-is

    overrides = {"pipeline": {
        "use_vlm":        bool(use_vlm),
        "use_kag":        bool(use_kag),
        "use_rag":        bool(use_rag),
        "use_reasoning":  bool(use_reasoning),
        "use_tkf":        bool(use_tkf),
        "use_aggregator": bool(use_aggregator),
    }}

    # Write rerun output inside the suite root (sibling of profile dirs)
    suite_root = run_dir.parent if (run_dir.parent / "suite_manifest.json").exists() else run_dir.parent
    from datetime import datetime
    rerun_name = f"rerun_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_run_dir = str(suite_root / rerun_name)

    master_cfg_path = str(REPO_ROOT / "configs" / "experiment_config.yaml")

    try:
        rerun_pipeline_only(
            saved_run_dir=str(rollout_dir),
            overrides=overrides,
            out_run_dir=out_run_dir,
            master_config_path=master_cfg_path,
        )
    except Exception as e:
        return state, f"Rerun failed: {e}", "", "", ""

    state["rerun_dir"] = out_run_dir
    status = f"Rerun complete \u2192 {out_run_dir}"
    return state, status, out_run_dir, "", ""


def on_diff(episode_id: Optional[int], state: Dict, mode: str):
    if not state or not state.get("run_dir"):
        return "", "", "(load a run first)"
    orig_dir = state["run_dir"]
    rerun_dir = state.get("rerun_dir")
    if mode == "Episode":
        eid = int(episode_id) if episode_id is not None else None
        orig, new = prescription_pair(orig_dir, rerun_dir, eid)
    else:
        orig, new = prescription_pair(orig_dir, rerun_dir, None)
    html = html_side_by_side(orig, new, "Original (loaded run)", "Re-run (current toggles)")
    diff = unified_diff(orig, new)
    return orig, new, html + "\n\n<pre>" + diff + "</pre>"


def on_refresh_graph():
    fig, _stats = build_performance_figure(ABLATIONS_DIR_DEFAULT, BASELINES_DIR_DEFAULT, RUNS_DIR_DEFAULT)
    return fig


def build_app():
    with gr.Blocks(title="MazeNav Research Dashboard") as app:
        gr.Markdown("# MazeNav Research Dashboard")
        gr.Markdown("Load a saved run \u2192 pick a failure episode \u2192 toggle components \u2192 re-run Phase B + C \u2192 compare prescriptions.")

        state = gr.State(_state_init())

        with gr.Tab("Explorer"):
            with gr.Row():
                with gr.Column(scale=2):
                    run_picker = gr.Dropdown(
                        choices=_list_saved_runs(),
                        label="Saved run directory",
                        allow_custom_value=True,
                    )
                    run_refresh = gr.Button("Refresh run list")
                    load_btn = gr.Button("Load Saved Run", variant="primary")
                    load_status = gr.Textbox(label="Status", lines=3, interactive=False)

                    gr.Markdown("### Component toggles (for re-run)")
                    use_vlm        = gr.Checkbox(label="VLM",        value=True)
                    use_kag        = gr.Checkbox(label="KAG",        value=True)
                    use_rag        = gr.Checkbox(label="RAG",        value=True)
                    use_reasoning  = gr.Checkbox(label="Reasoning",  value=True)
                    use_tkf        = gr.Checkbox(label="TKF",        value=True)
                    use_aggregator = gr.Checkbox(label="Aggregator", value=True)
                    rerun_btn      = gr.Button("Re-run Pipeline (Phases B + C)", variant="primary")
                    rerun_status   = gr.Textbox(label="Re-run status", lines=2, interactive=False)
                    rerun_dir_tb   = gr.Textbox(label="Re-run dir", interactive=False)

                with gr.Column(scale=3):
                    episode_dd = gr.Dropdown(label="Failure episode", choices=[], interactive=True)

                    with gr.Row():
                        start_img    = gr.Image(label="Start frame",        type="filepath", height=220)
                        highloss_img = gr.Image(label="Highest-loss frame", type="filepath", height=220)
                        end_img      = gr.Image(label="End frame",          type="filepath", height=220)

                    meta_box = gr.Textbox(label="Episode metadata", lines=8, interactive=False)

                    with gr.Tabs():
                        with gr.Tab("VLM Report"):
                            vlm_box = gr.Textbox(lines=14, interactive=False)
                        with gr.Tab("KAG Context"):
                            kag_box = gr.Textbox(lines=14, interactive=False)
                        with gr.Tab("RAG Retrieved"):
                            rag_box = gr.Textbox(lines=14, interactive=False)
                        with gr.Tab("Reasoning"):
                            reasoning_box = gr.Textbox(lines=20, interactive=False)
                        with gr.Tab("TKF Verdict"):
                            tkf_box = gr.Textbox(lines=14, interactive=False)
                        with gr.Tab("Final Prescription"):
                            final_box = gr.Textbox(lines=14, interactive=False)

        with gr.Tab("Prescription Diff"):
            with gr.Row():
                diff_mode = gr.Radio(choices=["Episode", "Cross-episode (Phase C)"], value="Episode", label="Diff target")
                diff_btn  = gr.Button("Compute Diff", variant="primary")
            with gr.Row():
                orig_box = gr.Textbox(label="Original prescription", lines=16)
                new_box  = gr.Textbox(label="Re-run prescription",   lines=16)
            diff_html = gr.HTML()

        with gr.Tab("Performance Graph"):
            perf_btn  = gr.Button("Refresh from results/ablations + results/baselines", variant="primary")
            perf_plot = gr.Plot()

        def _refresh_runs():
            return gr.update(choices=_list_saved_runs())

        run_refresh.click(_refresh_runs, inputs=None, outputs=run_picker)

        load_btn.click(
            fn=on_load_run,
            inputs=[run_picker, state],
            outputs=[state, episode_dd, load_status,
                     start_img, highloss_img, end_img,
                     vlm_box, kag_box, rag_box, reasoning_box, tkf_box, final_box,
                     meta_box, rerun_status],
        )

        episode_dd.change(
            fn=on_select_episode,
            inputs=[episode_dd, state],
            outputs=[start_img, highloss_img, end_img,
                     vlm_box, kag_box, rag_box, reasoning_box, tkf_box, final_box,
                     meta_box, rerun_status],
        )

        rerun_btn.click(
            fn=on_rerun,
            inputs=[use_vlm, use_kag, use_rag, use_reasoning, use_tkf, use_aggregator, state],
            outputs=[state, rerun_status, rerun_dir_tb, orig_box, new_box],
        )

        diff_btn.click(
            fn=on_diff,
            inputs=[episode_dd, state, diff_mode],
            outputs=[orig_box, new_box, diff_html],
        )

        perf_btn.click(fn=on_refresh_graph, inputs=None, outputs=perf_plot)

    return app


def main():
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False)


if __name__ == "__main__":
    main()
