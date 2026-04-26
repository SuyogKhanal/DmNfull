import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.rollout import run_rollouts
from pipeline.vlm_analyser import analyse_failure, save_vlm_report
from pipeline.kag_loader import load_and_format, save_kag_context
from pipeline.rag_bank import RAGBank, save_rag_retrieved
from pipeline.reasoning import run_reasoning, summarise_episode, save_reasoning
from pipeline.knowledge_fetcher import run_knowledge_check, format_tkf_block, save_tkf_result
from pipeline.aggregator import (
    run_aggregator,
    save_final_prescription,
    save_episode_final_prescription,
)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(data: Dict, path: Path):
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_config(master_path: str, ablation_path: Optional[str] = None) -> Dict:
    base = _load_yaml(master_path)
    if ablation_path:
        override = _load_yaml(ablation_path)
        return _deep_merge(base, override)
    return base


def _make_run_dir(config: Dict, tag: Optional[str] = None) -> Path:
    base_out = Path(config.get("tracking", {}).get("output_dir", "results/runs"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}" if not tag else f"run_{ts}_{tag}"
    p = base_out / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _snapshot_config(config: Dict, run_dir: Path):
    _dump_yaml(config, run_dir / "config_used.yaml")


def _build_phaseB_for_episode(
    episode: Dict,
    episode_dir: Path,
    config: Dict,
    kag_context: str,
    rag_bank: Optional[RAGBank],
    run_id: str,
    cache=None,
) -> Dict:
    pipeline_flags = config.get("pipeline", {})
    llm_cfg = config.get("llm", {})
    tkf_cfg = config.get("tkf", {})
    track_cfg = config.get("tracking", {})

    use_vlm        = bool(pipeline_flags.get("use_vlm", True))
    use_kag        = bool(pipeline_flags.get("use_kag", True))
    use_rag        = bool(pipeline_flags.get("use_rag", True))
    use_reasoning  = bool(pipeline_flags.get("use_reasoning", True))
    use_tkf        = bool(pipeline_flags.get("use_tkf", True))
    use_plain_llm   = bool(pipeline_flags.get("use_plain_llm", True))

    episode_id = episode["episode_id"]

    vlm_report = ""
    per_frame: List[Dict] = []
    if use_vlm:
        print(f"  [Phase B][ep {episode_id}] VLM...")
        vlm_report, per_frame = analyse_failure(episode, llm_cfg, cache=cache)
        if track_cfg.get("save_prescriptions", True):
            save_vlm_report(vlm_report, episode_dir)
    else:
        vlm_report = "[VLM DISABLED]"
        if track_cfg.get("save_prescriptions", True):
            save_vlm_report("DISABLED", episode_dir)

    if use_kag:
        kag_ctx_for_ep = kag_context
        if track_cfg.get("save_prescriptions", True):
            save_kag_context(kag_ctx_for_ep or "DISABLED", episode_dir)
    else:
        kag_ctx_for_ep = ""
        if track_cfg.get("save_prescriptions", True):
            save_kag_context("DISABLED", episode_dir)

    rag_ctx = ""
    if use_rag and rag_bank is not None:
        print(f"  [Phase B][ep {episode_id}] RAG retrieve...")
        end_frame = episode.get("frame_paths", {}).get("end_frame")
        rag_ctx = rag_bank.retrieve(vlm_report or "", end_frame)
        if track_cfg.get("save_prescriptions", True):
            save_rag_retrieved(rag_ctx or "(no matches above threshold)", episode_dir)
    else:
        if track_cfg.get("save_prescriptions", True):
            save_rag_retrieved("DISABLED", episode_dir)

    analysis_text = ""
    prescription_text = ""
    combined_text = ""
    tkf_result: Optional[Dict] = None
    tkf_block = ""

    if use_reasoning:
        print(f"  [Phase B][ep {episode_id}] Reasoning (analysis)...")
        analysis_text, _prelim_rx, _prelim_combined = run_reasoning(
            episode=episode,
            vision_report=vlm_report if use_vlm else "",
            kag_context=kag_ctx_for_ep,
            rag_context=rag_ctx,
            tkf_block="",
            llm_cfg=llm_cfg,
            cache=cache,
            use_vlm=use_vlm,
            use_kag=use_kag,
            use_rag=use_rag,
        )
    else:
        analysis_text = "[REASONING DISABLED]"

    if use_tkf:
        print(f"  [Phase B][ep {episode_id}] TKF...")
        try:
            tkf_result = run_knowledge_check(
                analysis_text,
                tkf_cfg,
                llm_cfg,
                cache=cache,
                episode_id=episode_id,
            )
            tkf_block = format_tkf_block(tkf_result)
            if track_cfg.get("save_prescriptions", True):
                save_tkf_result(tkf_result, episode_dir)
        except Exception as e:
            print(f"  [Phase B][ep {episode_id}] TKF error: {e}")
            tkf_block = ""
            if track_cfg.get("save_prescriptions", True):
                save_tkf_result({"verdict": "ERROR", "error": str(e)}, episode_dir)
    else:
        if track_cfg.get("save_prescriptions", True):
            save_tkf_result({"verdict": "DISABLED"}, episode_dir)

    if use_reasoning:
        print(f"  [Phase B][ep {episode_id}] Reasoning (prescription)...")
        analysis_text, prescription_text, combined_text = run_reasoning(
            episode=episode,
            vision_report=vlm_report if use_vlm else "",
            kag_context=kag_ctx_for_ep,
            rag_context=rag_ctx,
            tkf_block=tkf_block,
            llm_cfg=llm_cfg,
            cache=cache,
            use_vlm=use_vlm,
            use_kag=use_kag,
            use_rag=use_rag,
        )
        if track_cfg.get("save_prescriptions", True):
            save_reasoning(combined_text, episode_dir)
    else:
        combined_text = "[REASONING DISABLED]\n" + tkf_block
        if track_cfg.get("save_prescriptions", True):
            save_reasoning(combined_text, episode_dir)

    if use_reasoning and use_plain_llm:
        summary = summarise_episode(
            combined_text,
            episode_id,
            llm_cfg,
            cache=cache,
            use_vlm=use_vlm,
            use_kag=use_kag,
            use_rag=use_rag,
        )

    elif use_reasoning:
        summary = combined_text
    else:
        summary = f"Reasoning disabled. Failure at episode {episode_id} with config {episode.get('dynamic_config', {})}."

    if rag_bank is not None and use_rag:
        end_frame = episode.get("frame_paths", {}).get("end_frame")
        rag_bank.store(
            run_id=run_id,
            episode=episode,
            vision_report=vlm_report or "",
            prescription={"summary": summary, "root_cause": "pending"},
            end_frame_path=end_frame,
        )

    if track_cfg.get("save_prescriptions", True):
        save_episode_final_prescription(prescription_text or "[NO PRESCRIPTION]", episode_dir)

    return {
        "episode_id":       episode_id,
        "seed":             episode["seed"],
        "total_steps":      episode["total_steps"],
        "total_reward":     episode["total_reward"],
        "success":          episode["success"],
        "dynamic_config":   episode.get("dynamic_config", {}),
        "summary":          summary,
        "vlm_report":       vlm_report,
        "kag_context":      kag_ctx_for_ep,
        "rag_context":      rag_ctx,
        "analysis_text":    analysis_text,
        "prescription":     prescription_text,
        "reasoning_combined":combined_text,
        "tkf_result":       tkf_result,
        "frame_paths":      episode.get("frame_paths", {}),
    }


def run_pipeline(config: Dict, run_dir: Optional[Path] = None, tag: Optional[str] = None, cache=None) -> Dict:
    """Full Phase A + Phase B + Phase C run. Returns the full_output dict and writes all files to run_dir."""
    if run_dir is None:
        run_dir = _make_run_dir(config, tag=tag)
    run_id = run_dir.name
    if config.get("tracking", {}).get("save_config_snapshot", True):
        _snapshot_config(config, run_dir)

    print(f"\n{'='*70}\n[PipelineRunner] PHASE A — rollouts\n{'='*70}")
    rollout_result = run_rollouts(config, run_dir)

    kag_cfg = config.get("kag", {})
    kag_context = ""
    if config.get("pipeline", {}).get("use_kag", True):
        try:
            kag_context = load_and_format(kag_cfg.get("document_path", "knowledge/kag_maze_knowledge.json"))
        except Exception as e:
            print(f"[PipelineRunner] KAG load failed: {e}")
            kag_context = ""

    rag_bank: Optional[RAGBank] = None
    if config.get("pipeline", {}).get("use_rag", True):
        try:
            rag_cfg = config.get("rag", {})
            rag_bank = RAGBank(
                bank_path=rag_cfg.get("bank_path", "results/rag_bank"),
                top_k=int(rag_cfg.get("top_k", 3)),
                sim_threshold=float(rag_cfg.get("sim_threshold", 0.3)),
                clip_model=rag_cfg.get("clip_model", "openai/clip-vit-large-patch14"),
            )
        except Exception as e:
            print(f"[PipelineRunner] RAG init failed: {e}")
            rag_bank = None

    failures = [e for e in rollout_result["all_episodes"] if not e["success"]]
    failure_ids = [e["episode_id"] for e in failures]
    print(f"\n{'='*70}\n[PipelineRunner] PHASE B — {len(failures)} failures\n{'='*70}")

    per_episode: List[Dict] = []
    for ed in failures:
        episode_dir = run_dir / "episodes" / f"episode_{ed['episode_id']}"
        per_episode.append(_build_phaseB_for_episode(ed, episode_dir, config, kag_context, rag_bank, run_id, cache=cache))

    cross_text = ""
    structured: Dict = {}
    if config.get("pipeline", {}).get("use_aggregator", True) and failures:
        print(f"\n{'='*70}\n[PipelineRunner] PHASE C — aggregator\n{'='*70}")
        maze_ascii = failures[0].get("ascii_grid", "")
        cross_text, _raw, structured = run_aggregator(
            failure_summaries=per_episode,
            failure_ids=failure_ids,
            maze_ascii=maze_ascii,
            llm_cfg=config.get("llm", {}),
            cache=cache,
        )

    full_output = {
        "metadata": {
            "run_id":        run_id,
            "timestamp":     datetime.now().isoformat(),
            "n_episodes":    rollout_result["n_episodes"],
            "seed_base":     rollout_result["seed_base"],
            "n_successes":   len(rollout_result["success_episode_ids"]),
            "n_failures":    len(failure_ids),
            "pipeline_flags":config.get("pipeline", {}),
        },
        "config":        config,
        "phase_a": {
            "all_rollouts": [
                {
                    "episode_id":   e["episode_id"],
                    "maze_name":    e["maze_name"],
                    "seed":         e["seed"],
                    "total_steps":  e["total_steps"],
                    "total_reward": e["total_reward"],
                    "success":      e["success"],
                    "ascii_grid":   e["ascii_grid"],
                    "dynamic_config":e["dynamic_config"],
                    "key_frames":   e["key_frames"],
                    "frame_paths":  e.get("frame_paths", {}),
                }
                for e in rollout_result["all_episodes"]
            ],
            "success_episode_ids": rollout_result["success_episode_ids"],
            "failure_episode_ids": failure_ids,
        },
        "phase_b": {
            "per_episode": per_episode,
        },
        "phase_c": {
            "cross_episode_reasoning": cross_text,
            "parsed_prescription":     structured,
        },
    }

    with open(run_dir / "full_output.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)

    save_final_prescription(cross_text, structured, run_dir, per_episode)

    print(f"\n[PipelineRunner] Done. Results → {run_dir}")
    return full_output


def _load_episode_from_saved_run(run_dir: Path, episode_id: int) -> Optional[Dict]:
    ed_path = run_dir / "episodes" / f"episode_{episode_id}" / "episode_data.json"
    if not ed_path.exists():
        return None
    with open(ed_path, "r") as f:
        ed = json.load(f)
    for s in ed.get("steps", []):
        s.setdefault("rgb", None)
    return ed


def rerun_pipeline_only(saved_run_dir: str, overrides: Dict, out_run_dir: Optional[str] = None, master_config_path: Optional[str] = None, cache=None) -> Dict:
    """Re-run Phase B + Phase C over a saved run's stored episodes/frames, with new toggle overrides."""
    saved = Path(saved_run_dir)
    if not saved.exists():
        raise FileNotFoundError(saved)

    base_cfg_path = saved / "config_used.yaml"
    if not base_cfg_path.exists():
        if master_config_path is None:
            raise FileNotFoundError(f"config_used.yaml missing in {saved} and no master config provided")
        base_cfg = _load_yaml(master_config_path)
    else:
        base_cfg = _load_yaml(str(base_cfg_path))

    config = _deep_merge(base_cfg, overrides or {})

    if out_run_dir is None:
        out_run_dir = str(_make_run_dir(config, tag="rerun"))
    out_dir = Path(out_run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_config(config, out_dir)

    full_out_src = saved / "full_output.json"
    if not full_out_src.exists():
        raise FileNotFoundError(full_out_src)
    with open(full_out_src, "r") as f:
        saved_full = json.load(f)

    saved_failure_ids = saved_full.get("phase_a", {}).get("failure_episode_ids", [])

    failures = []
    for eid in saved_failure_ids:
        ed = _load_episode_from_saved_run(saved, eid)
        if ed is not None:
            failures.append(ed)

    kag_context = ""
    if config.get("pipeline", {}).get("use_kag", True):
        try:
            kag_context = load_and_format(config.get("kag", {}).get("document_path", "knowledge/kag_maze_knowledge.json"))
        except Exception as e:
            print(f"[Rerun] KAG load failed: {e}")

    rag_bank = None
    if config.get("pipeline", {}).get("use_rag", True):
        try:
            rag_cfg = config.get("rag", {})
            rag_bank = RAGBank(
                bank_path=rag_cfg.get("bank_path", "results/rag_bank"),
                top_k=int(rag_cfg.get("top_k", 3)),
                sim_threshold=float(rag_cfg.get("sim_threshold", 0.3)),
                clip_model=rag_cfg.get("clip_model", "openai/clip-vit-large-patch14"),
            )
        except Exception as e:
            print(f"[Rerun] RAG init failed: {e}")

    per_episode = []
    for ed in failures:
        episode_dir = out_dir / "episodes" / f"episode_{ed['episode_id']}"
        per_episode.append(_build_phaseB_for_episode(ed, episode_dir, config, kag_context, rag_bank, out_dir.name, cache=cache))

    cross_text = ""
    structured: Dict = {}
    if config.get("pipeline", {}).get("use_aggregator", True) and failures:
        maze_ascii = failures[0].get("ascii_grid", "")
        cross_text, _raw, structured = run_aggregator(
            failure_summaries=per_episode,
            failure_ids=[f["episode_id"] for f in failures],
            maze_ascii=maze_ascii,
            llm_cfg=config.get("llm", {}),
            cache=cache,
        )

    full_output = {
        "metadata": {
            "run_id":          out_dir.name,
            "timestamp":       datetime.now().isoformat(),
            "rerun_of":        str(saved),
            "pipeline_flags":  config.get("pipeline", {}),
            "n_episodes":      saved_full.get("metadata", {}).get("n_episodes", 0),
            "n_successes":     saved_full.get("metadata", {}).get("n_successes", 0),
            "n_failures":      saved_full.get("metadata", {}).get("n_failures", len(saved_failure_ids)),
            "seed_base":       saved_full.get("metadata", {}).get("seed_base"),
        },
        "config":    config,
        "phase_a":   saved_full.get("phase_a", {}),
        "phase_b":   {"per_episode": per_episode},
        "phase_c":   {"cross_episode_reasoning": cross_text, "parsed_prescription": structured},
    }

    with open(out_dir / "full_output.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)

    save_final_prescription(cross_text, structured, out_dir, per_episode)
    return full_output