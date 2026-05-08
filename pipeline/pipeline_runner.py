import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm.auto import tqdm


def _phase_b_max_workers(config: Dict, n_failures: int) -> int:
    """Decide how many Phase B episodes to run in parallel.

    `pipeline.phase_b_max_workers` in the config overrides; when unset we
    fan out to one worker per failure (so 5 failures => 5 concurrent
    chains). Capped only by n_failures to avoid empty workers; setting
    the config value to 1 restores the legacy serial behaviour.
    """
    pf = config.get("pipeline", {}) or {}
    requested = pf.get("phase_b_max_workers", None)
    if requested is None:
        return max(1, n_failures)
    try:
        n = int(requested)
    except (TypeError, ValueError):
        return max(1, n_failures)
    if n <= 0:
        return max(1, n_failures)
    return max(1, min(n, n_failures))


def _run_phase_b_parallel(
    failures: List[Dict],
    run_dir: Path,
    config: Dict,
    kag_context: str,
    rag_bank,
    run_id: str,
    cache,
    rollout_id: str,
    log_prefix: str,
) -> List[Dict]:
    """Run Phase B for every failure concurrently and return per-episode
    summaries in the SAME order as `failures`.

    Cross-episode reasoning / Phase C must NOT be invoked from inside
    this function — the caller runs it AFTER this returns, which gives
    the desired ordering: per-episode chains fan out in parallel, then
    cross-episode reasoning waits for all of them before consuming the
    summaries.
    """
    if not failures:
        return []

    max_workers = _phase_b_max_workers(config, len(failures))
    failure_ids = [e["episode_id"] for e in failures]
    print(
        f"[{log_prefix}] Phase B fan-out: {len(failures)} failures, "
        f"max_workers={max_workers}, ids={failure_ids}"
    )

    results: Dict[int, Dict] = {}
    phase_b_t0 = time.time()

    def _one(idx: int, ed: Dict) -> Dict:
        ep_t0 = time.time()
        episode_dir = run_dir / "episodes" / f"episode_{ed['episode_id']}"
        out = _build_phaseB_for_episode(
            ed, episode_dir, config, kag_context, rag_bank,
            run_id, cache=cache, rollout_id=rollout_id,
        )
        tqdm.write(
            f"[{log_prefix}] Phase B ep {ed['episode_id']} done in "
            f"{time.time()-ep_t0:.1f}s"
        )
        return out

    if max_workers == 1:
        for idx, ed in enumerate(failures):
            results[idx] = _one(idx, ed)
    else:
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="phaseB") as pool:
            future_to_idx = {
                pool.submit(_one, idx, ed): idx
                for idx, ed in enumerate(failures)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()

    print(
        f"[{log_prefix}] Phase B total: {time.time()-phase_b_t0:.1f}s for "
        f"{len(failures)} failures (max_workers={max_workers})"
    )
    return [results[i] for i in range(len(failures))]

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.rollout import run_rollouts
from pipeline.vlm_analyser import analyse_failure, save_vlm_report
from pipeline.kag_loader import load_and_format, save_kag_context
from pipeline.rag_bank import RAGBank, save_rag_retrieved
from pipeline.reasoning import (
    run_analysis,
    run_prescription,
    run_plain_prescription,
    summarise_episode,
    save_reasoning,
)
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
    rollout_id: str = "default",
) -> Dict:
    """
    Phase B per-episode flow — strict 3-pass order:

      1. VLM  (cached per (rollout_id, episode_id) — VLM ONLY)
      2. KAG  (no cache)
      3. RAG  (no cache)
      4. run_analysis(VLM + KAG + RAG)  -> analysis_text   [no cache]
      5. TKF(analysis_text)             -> tkf_block        [no cache]
      6. run_prescription(analysis_text, tkf_block) -> prescription_text [no cache]

    This guarantees reasoning flows correctly into the final prescription:
    the TKF block is built from the REAL analysis, and the prescription
    is built from BOTH the real analysis AND the real TKF block.

    rollout_id must match the identifier used by all profiles in an ablation
    suite so that VLM results are shared correctly across profiles without
    polluting results from other suites.
    """
    pipeline_flags = config.get("pipeline", {})
    llm_cfg = config.get("llm", {})
    tkf_cfg = config.get("tkf", {})
    track_cfg = config.get("tracking", {})

    use_vlm       = bool(pipeline_flags.get("use_vlm",       True))
    use_kag       = bool(pipeline_flags.get("use_kag",       True))
    use_rag       = bool(pipeline_flags.get("use_rag",       True))
    use_reasoning = bool(pipeline_flags.get("use_reasoning", True))
    use_tkf       = bool(pipeline_flags.get("use_tkf",       True))
    use_plain_llm = bool(pipeline_flags.get("use_plain_llm", True))

    episode_id = episode["episode_id"]
    episode_dyn_cfg = episode.get("dynamic_config", {}) or {}

    # ------------------------------------------------------------------ #
    # PASS 0-A: VLM  (cached per (rollout_id, episode_id) — VLM ONLY)   #
    # rollout_id is propagated so all profiles within the same ablation   #
    # suite share VLM results (same frames) without cross-suite pollution.#
    # ------------------------------------------------------------------ #
    vlm_report = ""
    per_frame: List[Dict] = []
    if use_vlm:
        print(f"  [Phase B][ep {episode_id}] VLM...")
        vlm_report, per_frame = analyse_failure(
            episode, llm_cfg, cache=cache, rollout_id=rollout_id
        )
        if track_cfg.get("save_prescriptions", True):
            save_vlm_report(vlm_report, episode_dir)
    else:
        vlm_report = "[VLM DISABLED]"
        if track_cfg.get("save_prescriptions", True):
            save_vlm_report("DISABLED", episode_dir)

    # ------------------------------------------------------------------ #
    # PASS 0-B: KAG (no cache)                                            #
    # ------------------------------------------------------------------ #
    if use_kag:
        kag_ctx_for_ep = kag_context
        if track_cfg.get("save_prescriptions", True):
            save_kag_context(kag_ctx_for_ep or "DISABLED", episode_dir)
    else:
        kag_ctx_for_ep = ""
        if track_cfg.get("save_prescriptions", True):
            save_kag_context("DISABLED", episode_dir)

    # ------------------------------------------------------------------ #
    # PASS 0-C: RAG (no cache)                                            #
    # ------------------------------------------------------------------ #
    # exclude_episode_id alone is insufficient across profiles in the     #
    # ablation suite: p5 stores ep4 with episode_id=4 BEFORE p6 runs ep4, #
    # and although both share id=4 the cross-profile carry-over still     #
    # leaks because the bank persists between profile invocations.        #
    # exclude_episode_config additionally skips any stored entry whose    #
    # (start, goal, fires) matches the current episode — catching the     #
    # cross-profile "same physical episode" case directly.                #
    # ------------------------------------------------------------------ #
    rag_ctx = ""
    if use_rag and rag_bank is not None:
        print(f"  [Phase B][ep {episode_id}] RAG retrieve...")
        end_frame = episode.get("frame_paths", {}).get("end_frame")
        rag_ctx = rag_bank.retrieve(
            vlm_report or "",
            end_frame,
            exclude_episode_id=episode_id,
            exclude_episode_config=episode_dyn_cfg,
        )
        if track_cfg.get("save_prescriptions", True):
            save_rag_retrieved(rag_ctx or "(no matches above threshold)", episode_dir)
    else:
        if track_cfg.get("save_prescriptions", True):
            save_rag_retrieved("DISABLED", episode_dir)

    # ------------------------------------------------------------------ #
    # PASS 1: Analysis (no cache) — MUST run before TKF                   #
    # ------------------------------------------------------------------ #
    analysis_text = ""
    if use_reasoning:
        print(f"  [Phase B][ep {episode_id}] Reasoning — analysis pass...")
        analysis_text = run_analysis(
            episode=episode,
            vision_report=vlm_report if use_vlm else "",
            kag_context=kag_ctx_for_ep,
            rag_context=rag_ctx,
            llm_cfg=llm_cfg,
        )
    else:
        analysis_text = "[REASONING DISABLED]"

    # ------------------------------------------------------------------ #
    # PASS 2: TKF — receives the REAL analysis_text (no cache)            #
    # When no demo meets the threshold, top-2 closest demos are passed    #
    # WITH sim scores to the LLM so it can reason about variety vs new.   #
    # ------------------------------------------------------------------ #
    tkf_result: Optional[Dict] = None
    tkf_block = ""
    if use_tkf:
        print(f"  [Phase B][ep {episode_id}] TKF (knowledge check)...")
        try:
            tkf_result = run_knowledge_check(
                reasoning_text=analysis_text,
                tkf_cfg=tkf_cfg,
                llm_cfg=llm_cfg,
                cache=None,          # TKF is never cached
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

    # ------------------------------------------------------------------ #
    # PASS 3: Prescription — DERIVES FINAL_REC from analysis + KAG + RAG  #
    # + TKF (no cache). Analysis no longer emits FINAL_REC, so toggling   #
    # use_kag / use_rag / use_tkf actually changes the structured output, #
    # not just the prose around it. KAG and RAG context are passed in     #
    # explicitly so the grounding rules in build_prescription_prompt can  #
    # cite them by name and rank.                                         #
    # ------------------------------------------------------------------ #
    prescription_text = ""
    combined_text = ""
    used_plain_only = False
    if use_reasoning:
        print(f"  [Phase B][ep {episode_id}] Reasoning — prescription pass...")
        prescription_text, combined_text = run_prescription(
            episode=episode,
            analysis_text=analysis_text,
            tkf_block=tkf_block,
            llm_cfg=llm_cfg,
            kag_context=kag_ctx_for_ep if use_kag else "",
            rag_context=rag_ctx if use_rag else "",
        )
        if track_cfg.get("save_prescriptions", True):
            save_reasoning(combined_text, episode_dir)
    elif use_vlm and use_plain_llm:
        # P1 baseline: no reasoning chain, but a plain LLM still reads the VLM
        # report and emits a FINAL_REC + plain-English explanation. Without
        # this, P1's per-episode prescription is empty and the aggregator has
        # nothing to cluster — so P1 produces no actionable demos.
        print(f"  [Phase B][ep {episode_id}] Plain LLM prescription (P1 baseline)...")
        prescription_text, combined_text = run_plain_prescription(
            episode=episode,
            vision_report=vlm_report if use_vlm else "",
            llm_cfg=llm_cfg,
        )
        used_plain_only = True
        if track_cfg.get("save_prescriptions", True):
            save_reasoning(combined_text, episode_dir)
    else:
        combined_text = "[REASONING DISABLED]\n" + tkf_block
        if track_cfg.get("save_prescriptions", True):
            save_reasoning(combined_text, episode_dir)

    # ------------------------------------------------------------------ #
    # PASS 4: Plain LLM summariser (per-episode, no cache)                #
    # The summariser takes the prescription text and condenses it into a  #
    # 3-5 sentence summary used by the aggregator. P1 (used_plain_only)   #
    # already produced a compact LLM output, so we pass that summary text #
    # through directly rather than burning another LLM call.              #
    # ------------------------------------------------------------------ #
    if use_reasoning and use_plain_llm:
        summary = summarise_episode(
            combined_text, episode_id, llm_cfg,
        )
    elif use_reasoning:
        summary = combined_text
    elif used_plain_only:
        summary = prescription_text or combined_text
    else:
        summary = (
            f"Reasoning disabled. Failure at episode {episode_id} "
            f"with config {episode.get('dynamic_config', {})}."
        )

    # ------------------------------------------------------------------ #
    # RAG store (for future episodes to retrieve from)                    #
    # ------------------------------------------------------------------ #
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
        "episode_id":        episode_id,
        "seed":              episode["seed"],
        "total_steps":       episode["total_steps"],
        "total_reward":      episode["total_reward"],
        "success":           episode["success"],
        "dynamic_config":    episode.get("dynamic_config", {}),
        "summary":           summary,
        "vlm_report":        vlm_report,
        "kag_context":       kag_ctx_for_ep,
        "rag_context":       rag_ctx,
        "analysis_text":     analysis_text,
        "prescription":      prescription_text,
        "reasoning_combined":combined_text,
        "tkf_result":        tkf_result,
        "adjusted_prescription": (tkf_result or {}).get("adjusted_prescription", ""),
        "frame_paths":       episode.get("frame_paths", {}),
    }


def run_pipeline(config: Dict, run_dir: Optional[Path] = None, tag: Optional[str] = None, cache=None) -> Dict:
    """Full Phase A + B + C run."""
    if run_dir is None:
        run_dir = _make_run_dir(config, tag=tag)
    run_id = run_dir.name
    if config.get("tracking", {}).get("save_config_snapshot", True):
        _snapshot_config(config, run_dir)

    print(f"\n{'='*70}\n[PipelineRunner] PHASE A — rollouts\n{'='*70}")
    rollout_result = run_rollouts(config, run_dir)

    kag_context = ""
    if config.get("pipeline", {}).get("use_kag", True):
        try:
            kag_context = load_and_format(
                config.get("kag", {}).get("document_path", "knowledge/kag_maze_knowledge.json")
            )
        except Exception as e:
            print(f"[PipelineRunner] KAG load failed: {e}")

    rag_bank: Optional[RAGBank] = None
    if config.get("pipeline", {}).get("use_rag", True):
        try:
            rag_cfg = config.get("rag", {})
            rag_bank = RAGBank(
                bank_path=rag_cfg.get("bank_path", "results/rag_bank"),
                top_k=int(rag_cfg.get("top_k", 3)),
                sim_threshold=float(rag_cfg.get("sim_threshold", 0.3)),
                clip_model=rag_cfg.get("clip_model", "openai/clip-vit-large-patch14"),
                owner_run_id=run_id,
            )
        except Exception as e:
            print(f"[PipelineRunner] RAG init failed: {e}")

    failures = [e for e in rollout_result["all_episodes"] if not e["success"]]
    failure_ids = [e["episode_id"] for e in failures]
    print(f"\n{'='*70}\n[PipelineRunner] PHASE B — {len(failures)} failures\n{'='*70}")
    if failures:
        print(f"[PipelineRunner] failure_ids={failure_ids}")

    per_episode: List[Dict] = _run_phase_b_parallel(
        failures=failures,
        run_dir=run_dir,
        config=config,
        kag_context=kag_context,
        rag_bank=rag_bank,
        run_id=run_id,
        cache=cache,
        rollout_id=run_id,
        log_prefix="PipelineRunner",
    )

    cross_text = ""
    structured: Dict = {}
    if config.get("pipeline", {}).get("use_aggregator", True) and failures:
        print(f"\n{'='*70}\n[PipelineRunner] PHASE C — aggregator\n{'='*70}")
        phase_c_t0 = time.time()
        maze_ascii = failures[0].get("ascii_grid", "")
        cross_text, _raw, structured = run_aggregator(
            failure_summaries=per_episode,
            failure_ids=failure_ids,
            maze_ascii=maze_ascii,
            llm_cfg=config.get("llm", {}),
            pipeline_flags=config.get("pipeline", {}),
            cache=cache,
            kag_context=kag_context,
        )
        n_clusters = len((structured or {}).get("failure_clusters", []) or [])
        n_pres     = len((structured or {}).get("demonstration_prescriptions", []) or [])
        n_layouts  = sum(len(p.get("recommended_layouts", []) or []) for p in (structured or {}).get("demonstration_prescriptions", []) or [])
        total_demos = (structured or {}).get("total_demonstrations_needed", "?")
        print(
            f"[PipelineRunner] Phase C done in {time.time()-phase_c_t0:.1f}s | "
            f"clusters={n_clusters} | prescriptions={n_pres} | layouts={n_layouts} | "
            f"total_demos_needed={total_demos}"
        )

    full_output = {
        "metadata": {
            "run_id":         run_id,
            "timestamp":      datetime.now().isoformat(),
            "n_episodes":     rollout_result["n_episodes"],
            "seed_base":      rollout_result["seed_base"],
            "n_successes":    len(rollout_result["success_episode_ids"]),
            "n_failures":     len(failure_ids),
            "pipeline_flags": config.get("pipeline", {}),
        },
        "config":  config,
        "phase_a": {
            "all_rollouts": [
                {
                    "episode_id":    e["episode_id"],
                    "maze_name":     e["maze_name"],
                    "seed":          e["seed"],
                    "total_steps":   e["total_steps"],
                    "total_reward":  e["total_reward"],
                    "success":       e["success"],
                    "ascii_grid":    e["ascii_grid"],
                    "dynamic_config":e["dynamic_config"],
                    "key_frames":    e["key_frames"],
                    "frame_paths":   e.get("frame_paths", {}),
                }
                for e in rollout_result["all_episodes"]
            ],
            "success_episode_ids": rollout_result["success_episode_ids"],
            "failure_episode_ids": failure_ids,
        },
        "phase_b": {"per_episode": per_episode},
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


def _load_episodes_from_saved_run(run_dir: Path, episode_ids: List[int]) -> List[Dict]:
    """
    Load episode dicts for the given IDs from a saved run directory.

    Resolution order (first hit wins):
      1. full_output.json  phase_a.all_rollouts   — always present in ablation
         suite rollout dirs (written by _save_rollout_only).
      2. episodes/episode_N/episode_data.json — present in full pipeline runs.

    This dual-source lookup means both run_ablation_suite (which stores
    everything in full_output.json) and standalone run_pipeline runs (which
    write individual episode_data.json files) are handled correctly.
    """
    # Source 1: full_output.json
    full_out_path = run_dir / "full_output.json"
    rollout_map: Dict[int, Dict] = {}
    if full_out_path.exists():
        with open(full_out_path, "r") as f:
            saved = json.load(f)
        for ep in saved.get("phase_a", {}).get("all_rollouts", []):
            rollout_map[int(ep["episode_id"])] = ep

    episodes: List[Dict] = []
    for eid in episode_ids:
        ep = rollout_map.get(int(eid))
        if ep is not None:
            episodes.append(ep)
            continue
        # Source 2: individual episode_data.json (standalone runs)
        ed_path = run_dir / "episodes" / f"episode_{eid}" / "episode_data.json"
        if ed_path.exists():
            with open(ed_path, "r") as f:
                ed = json.load(f)
            for s in ed.get("steps", []):
                s.setdefault("rgb", None)
            episodes.append(ed)
        else:
            print(f"[Rerun] WARNING: episode {eid} not found in {run_dir} — skipping.")
    return episodes


def rerun_pipeline_only(
    saved_run_dir: str,
    overrides: Dict,
    out_run_dir: Optional[str] = None,
    master_config_path: Optional[str] = None,
    cache=None,
) -> Dict:
    """Re-run Phase B + Phase C over a saved rollout, with new toggle overrides."""
    saved = Path(saved_run_dir)
    if not saved.exists():
        raise FileNotFoundError(saved)

    base_cfg_path = saved / "config_used.yaml"
    if not base_cfg_path.exists():
        if master_config_path is None:
            raise FileNotFoundError(
                f"config_used.yaml missing in {saved} and no master config provided"
            )
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

    # Use the rollout dir name as rollout_id so VLM cache is shared correctly
    # across all profiles in the same ablation suite.
    rollout_id = saved.name

    failures = _load_episodes_from_saved_run(saved, saved_failure_ids)
    if not failures:
        print(f"[Rerun] WARNING: no failure episodes could be loaded from {saved}.")

    kag_context = ""
    if config.get("pipeline", {}).get("use_kag", True):
        try:
            kag_context = load_and_format(
                config.get("kag", {}).get("document_path", "knowledge/kag_maze_knowledge.json")
            )
        except Exception as e:
            print(f"[Rerun] KAG load failed: {e}")

    rag_bank = None
    if config.get("pipeline", {}).get("use_rag", True):
        try:
            rag_cfg = config.get("rag", {})
            # owner_run_id MUST equal the run_id passed to rag_bank.store() below
            # (out_dir.name) so that retrieve() returns only entries from THIS
            # profile. In the ablation suite out_dir.name is the profile name
            # (e.g. "p5_vlm_reasoning_kag_rag_cross_plain_llm"), so each profile
            # is fully isolated from every other profile.
            rag_bank = RAGBank(
                bank_path=rag_cfg.get("bank_path", "results/rag_bank"),
                top_k=int(rag_cfg.get("top_k", 3)),
                sim_threshold=float(rag_cfg.get("sim_threshold", 0.3)),
                clip_model=rag_cfg.get("clip_model", "openai/clip-vit-large-patch14"),
                owner_run_id=out_dir.name,
            )
        except Exception as e:
            print(f"[Rerun] RAG init failed: {e}")

    print(f"\n{'='*70}\n[Rerun] PHASE B — {len(failures)} failures\n{'='*70}")
    if failures:
        print(f"[Rerun] failure_ids={[f['episode_id'] for f in failures]}")

    per_episode = _run_phase_b_parallel(
        failures=failures,
        run_dir=out_dir,
        config=config,
        kag_context=kag_context,
        rag_bank=rag_bank,
        run_id=out_dir.name,
        cache=cache,
        rollout_id=rollout_id,
        log_prefix="Rerun",
    )

    cross_text = ""
    structured: Dict = {}
    if config.get("pipeline", {}).get("use_aggregator", True) and failures:
        print(f"\n{'='*70}\n[Rerun] PHASE C — aggregator\n{'='*70}")
        phase_c_t0 = time.time()
        maze_ascii = failures[0].get("ascii_grid", "")
        cross_text, _raw, structured = run_aggregator(
            failure_summaries=per_episode,
            failure_ids=[f["episode_id"] for f in failures],
            maze_ascii=maze_ascii,
            llm_cfg=config.get("llm", {}),
            pipeline_flags=config.get("pipeline", {}),
            cache=cache,
            kag_context=kag_context,
        )
        n_clusters = len((structured or {}).get("failure_clusters", []) or [])
        n_pres     = len((structured or {}).get("demonstration_prescriptions", []) or [])
        n_layouts  = sum(len(p.get("recommended_layouts", []) or []) for p in (structured or {}).get("demonstration_prescriptions", []) or [])
        total_demos = (structured or {}).get("total_demonstrations_needed", "?")
        print(
            f"[Rerun] Phase C done in {time.time()-phase_c_t0:.1f}s | "
            f"clusters={n_clusters} | prescriptions={n_pres} | layouts={n_layouts} | "
            f"total_demos_needed={total_demos}"
        )

    full_output = {
        "metadata": {
            "run_id":         out_dir.name,
            "timestamp":      datetime.now().isoformat(),
            "rerun_of":       str(saved),
            "pipeline_flags": config.get("pipeline", {}),
            "n_episodes":     saved_full.get("metadata", {}).get("n_episodes", 0),
            "n_successes":    saved_full.get("metadata", {}).get("n_successes", 0),
            "n_failures":     saved_full.get("metadata", {}).get("n_failures", len(saved_failure_ids)),
            "seed_base":      saved_full.get("metadata", {}).get("seed_base"),
        },
        "config":  config,
        "phase_a": saved_full.get("phase_a", {}),
        "phase_b": {"per_episode": per_episode},
        "phase_c": {
            "cross_episode_reasoning": cross_text,
            "parsed_prescription":     structured,
        },
    }

    with open(out_dir / "full_output.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)

    save_final_prescription(cross_text, structured, out_dir, per_episode)
    return full_output