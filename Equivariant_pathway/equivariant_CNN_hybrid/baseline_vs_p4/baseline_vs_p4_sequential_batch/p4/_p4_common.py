"""Shared round-loop logic for P4 sequential and batch variants.

Both modes share 95% of their code: rollout → LLM analysis with mode-
specific addendum → cap → corridor-aware demo collection → replay fine-
tune → heldout eval → log → repeat. The only differences are the prompt
addendum text and the demos-per-round cap, which the caller passes in.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import (  # noqa: E402
    p4_budget as _upstream_p4,
)

from ..layouts.layout_setup import ensure_correction_layouts_for_round
from ..logging_ext.compression_log import CompressionLog
from ..logging_ext.prescription_overlap import PrescriptionOverlapLog
from ..logging_ext.training_log import TrainingLog
from . import demo_collector, prompts, runner


def _info(tag: str, msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def _finetune(
    cumulative_demo_dir: Path,
    new_demo_dir: Optional[Path],
    ckpt_dir: Path,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    replay_mix: float,
    replay_mix_floor: float,
) -> int:
    cmd = [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4."
        "baseline_vs_p4_sequential_batch.trainer.finetune_replay",
        "--demo_dir", str(cumulative_demo_dir),
        "--checkpoint_dir", str(ckpt_dir),
        "--epochs", str(int(epochs)),
        "--batch_size", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--weight_decay", str(float(weight_decay)),
        "--replay_mix", str(float(replay_mix)),
        "--replay_mix_floor", str(float(replay_mix_floor)),
        "--seed", str(int(seed)),
        "--resume",
    ]
    if new_demo_dir is not None and new_demo_dir.exists():
        cmd += ["--new_demos_dir", str(new_demo_dir)]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _persist_curve(
    method: str, results_dir: Path, history: List[Dict],
    budget: int, target_sr: float, demo_dir: Path, ckpt_dir: Path,
    run_shared_dir: Path, heldout_yaml: Path, run_id: int,
) -> None:
    """Persist the learning curve. Correction pool rotates per round, so
    we store the run's shared directory (which holds the per-round
    yamls) instead of a single yaml path; each history row already
    carries its own ``correction_yaml`` field.
    """
    out = {
        "method": method,
        "run_id": int(run_id),
        "budget": int(budget),
        "target_sr": float(target_sr),
        "demo_dir": str(demo_dir),
        "checkpoint_dir": str(ckpt_dir),
        "correction_yaml_dir": str(run_shared_dir),
        "heldout_yaml": str(heldout_yaml),
        "history": history,
    }
    with open(results_dir / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def run_loop(
    *,
    mode: str,                          # "sequential" or "batch"
    run_id: int,
    method_root: Path,
    shared_demo_dir: Path,
    shared_ckpt_dir: Path,
    train_yaml: Path,
    heldout_yaml: Path,
    run_shared_dir: Path,
    correction_n: int,
    config: Dict,
) -> Dict:
    """Execute the budget-constrained P4 loop for one method.

    The correction pool rotates per round: at the top of each round we
    call ``ensure_correction_layouts_for_round(run_id, rnd, ...)`` which
    samples 50 fresh layouts (blocked from training + heldout + prior
    rounds in this run) and writes them to
    ``run_shared_dir/round_NNN/correction_layouts.yaml``. baseline /
    p4_sequential / p4_batch all call the same function with the same
    (run_id, rnd); whichever runs first does the sampling, the others
    re-read the cached yaml.

    Args:
        mode: "sequential" (1 layout/round) or "batch" (LLM decides,
              capped to remaining budget).
        run_id: run identifier (used for seeds + log tags).
        method_root: e.g. results/run_5/p4_sequential/.
        shared_demo_dir / shared_ckpt_dir: per-run bootstrap source.
        train_yaml / heldout_yaml: global layouts used as blocked set.
        run_shared_dir: the per-run ``shared/`` dir that holds the
            ``round_NNN/`` subtrees.
        correction_n: number of layouts to sample for each round.
        config: parsed config.yaml dict.
    """
    assert mode in ("sequential", "batch"), f"unknown mode: {mode}"
    method_label = (
        runner.LABEL_SEQ if mode == "sequential" else runner.LABEL_BATCH
    )
    method_tag = f"p4-{mode[:3]}"

    demo_dir = method_root / "demos"
    ckpt_dir = method_root / "checkpoints"
    results_dir = method_root / "results"
    for d in (demo_dir, ckpt_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Seed cumulative demo pool + initial checkpoint from per-run shared.
    for src in sorted(shared_demo_dir.glob("*.json")):
        dst = demo_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    for name in ("best_hybrid_policy.pth", "last_hybrid_policy.pth"):
        s = shared_ckpt_dir / name
        if s.exists():
            shutil.copy2(s, ckpt_dir / name)

    # Config knobs.
    budget = int(config.get("budget", 15))
    target_sr = float(config.get("target_sr", 0.90))
    max_rounds = int(config.get("max_rounds", 50))
    max_steps = int(config.get("max_steps", 60))
    seed = int(config.get("seed", 0)) + int(run_id)
    corridor_blocking = bool(config.get("corridor_blocking", True))

    p4cfg = (config.get("p4") or {})
    mode_max: Optional[int]
    if mode == "sequential":
        mode_max = int((p4cfg.get("sequential") or {}).get("demos_per_round", 1) or 1)
    else:
        mode_max_raw = (p4cfg.get("batch") or {}).get("demos_per_round")
        mode_max = int(mode_max_raw) if mode_max_raw is not None else None

    tcfg = (config.get("trainer") or {})
    finetune_epochs = int(tcfg.get("finetune_epochs", 20))
    lr = float(tcfg.get("lr", 5e-5))
    batch_size = int(tcfg.get("batch_size", 64))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    replay_mix = float(tcfg.get("replay_mix", 0.5))
    replay_mix_floor = float(tcfg.get("replay_mix_floor", 0.2))

    overlap_runs = int((config.get("logging") or {}).get("prescription_overlap_runs", 3))
    overlap_log = PrescriptionOverlapLog(
        path=results_dir / "prescription_overlap.json",
        enabled=(int(run_id) <= overlap_runs),
    )
    comp_log = CompressionLog(results_dir / "compression_log.csv")
    tlog = TrainingLog(results_dir / "training_log.csv")

    # Round 0 — heldout eval on the initial policy.
    round0 = results_dir / "round_000_heldout_eval"
    rc = _upstream_p4._rollout(ckpt_dir, heldout_yaml, round0, seed=seed, max_steps=max_steps)
    if rc != 0:
        raise RuntimeError(f"initial heldout rollout failed (rc={rc})")
    initial = _upstream_p4._read_sr(round0)

    history: List[Dict] = [{
        "round": 0,
        "cum_demos": _upstream_p4._count_demos(demo_dir),
        "extra_demos": 0,
        "budget_remaining": budget,
        "heldout_sr": initial["success_rate"],
        "heldout_n_successes": initial["n_successes"],
        "heldout_n_episodes": initial["n_episodes"],
        "correction_sr": None,
        "n_prescribed_layouts": 0,
        "n_capped_by_budget": 0,
        "n_new_demos": 0,
    }]
    method_full_label = f"p4_{'sequential' if mode == 'sequential' else 'batch'}_seqbatch"
    history[0]["correction_yaml"] = None
    _persist_curve(
        method_full_label, results_dir, history, budget, target_sr,
        demo_dir, ckpt_dir, run_shared_dir, heldout_yaml, run_id,
    )
    tlog.write_row(
        round_idx=0, demos_added_total=0, training_rounds_total=0,
        heldout_sr=initial["success_rate"], finetune_epochs=0,
        lr=lr, replay_mix=replay_mix,
    )

    if initial["success_rate"] >= target_sr:
        return {"history": history, "stopped_reason": "initial>=target"}

    extras_saved = 0
    training_rounds_total = 0

    for rnd in range(1, max_rounds + 1):
        remaining = budget - extras_saved
        if remaining <= 0:
            _info(method_tag, f"round {rnd}: budget exhausted ({extras_saved}/{budget}); stop")
            break

        _info(method_tag, f"ROUND {rnd}  (remaining budget = {remaining})")
        round_dir = results_dir / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # ---- Pre-Phase A: sample THIS round's fresh correction pool ------
        # Idempotent — within the same (run_id, rnd), baseline and the
        # two P4 methods all retrieve the same yaml so the comparison
        # stays apples-to-apples for the round.
        correction_yaml_rnd = ensure_correction_layouts_for_round(
            run_id=run_id, round_idx=rnd, correction_n=correction_n,
            train_yaml=train_yaml, heldout_yaml=heldout_yaml,
            run_shared_dir=run_shared_dir,
        )

        # ---- Phase A: rollout the correction pool -------------------------
        corr_dir = round_dir / "correction_rollout"
        rc = _upstream_p4._rollout(
            ckpt_dir, correction_yaml_rnd, corr_dir,
            seed=seed + rnd * 1000, max_steps=max_steps,
        )
        if rc != 0:
            raise RuntimeError(f"correction rollout failed (rc={rc})")
        corr_metrics = _upstream_p4._read_sr(corr_dir)
        # The rollout output has per-episode results we can summarize.
        n_failures = int(corr_metrics["n_episodes"]) - int(corr_metrics["n_successes"])
        failure_mode_counts: Dict[str, int] = _summarize_failures_from_rollout(corr_dir)

        # ---- Phase B+C: LLM analysis with our addendums -------------------
        analysis_dir = round_dir / "p4_analysis"
        reasoning_add = prompts.reasoning_addendum(
            mode=mode, round_idx=rnd,
            demos_added_this_round=0,
            demos_remaining_in_budget=remaining,
            n_failures_observed=n_failures,
            failure_mode_counts=failure_mode_counts,
            budget_total=budget, already_used=extras_saved,
        )
        aggregator_add = prompts.aggregator_addendum(
            mode=mode, round_idx=rnd,
            demos_added_this_round=0,
            demos_remaining_in_budget=remaining,
            n_failures_observed=n_failures,
            failure_mode_counts=failure_mode_counts,
            budget_total=budget, already_used=extras_saved,
        )
        try:
            runner.run_analysis(
                rollout_dir=corr_dir, out_dir=analysis_dir,
                demo_dir=demo_dir, label=method_label,
                reasoning_addendum=reasoning_add,
                aggregator_addendum=aggregator_add,
            )
        except Exception as exc:  # noqa: BLE001
            _info(method_tag, f"round {rnd}: LLM analysis FAILED ({exc!r}); skipping round")
            comp_log.write_row(
                round_idx=rnd,
                n_failures_observed=n_failures,
                n_layouts_prescribed=0,
                n_demos_collected=0,
                n_corridor_infeasible=0,
            )
            history.append(_round_history_entry(
                rnd, demo_dir, extras_saved, budget, 0, 0,
                0, 0, corr_metrics, _llm_failed_post(initial),
                correction_yaml_rnd,
            ))
            _persist_curve(method_full_label, results_dir, history, budget, target_sr,
                           demo_dir, ckpt_dir, run_shared_dir, heldout_yaml, run_id)
            return {"history": history, "stopped_reason": "llm_error"}

        rec_path = runner.flatten_recommendations(analysis_dir, label=method_label)
        cap_audit = {"proposed": 0, "kept": 0, "capped": 0, "cap": 0}
        coll_result: Optional[demo_collector.CollectResult] = None
        if rec_path is not None:
            cap_audit = runner.cap_layouts(
                rec_path, remaining=remaining, mode_max=mode_max,
            )
            if cap_audit["kept"] > 0:
                round_demo_dir = demo_dir / f"round_{rnd:03d}"
                round_demo_dir.mkdir(parents=True, exist_ok=True)
                coll_result = demo_collector.collect(
                    rec_path=rec_path, demo_dir=round_demo_dir,
                    seed=seed + 1000 + rnd,
                    corridor_blocking=corridor_blocking,
                )
                demo_collector.write_summary(
                    coll_result, round_dir / "collect_summary.json",
                )

        n_collected = coll_result.n_saved if coll_result else 0
        n_infeasible = coll_result.n_infeasible if coll_result else 0
        extras_saved += n_collected
        _info(method_tag, (
            f"round {rnd}: prescribed={cap_audit['proposed']} "
            f"kept={cap_audit['kept']} "
            f"collected={n_collected} infeasible={n_infeasible} "
            f"cum_extras={extras_saved}/{budget}"
        ))

        # ---- compression + overlap logs -----------------------------------
        comp_log.write_row(
            round_idx=rnd,
            n_failures_observed=n_failures,
            n_layouts_prescribed=cap_audit["proposed"],
            n_demos_collected=n_collected,
            n_corridor_infeasible=n_infeasible,
        )
        if overlap_log.enabled and coll_result is not None:
            overlap_log.write_round(
                round_idx=rnd,
                prescribed_layouts=[
                    {
                        "start_pos": d.start_pos, "goal_pos": d.goal_pos,
                        "fire_positions": d.fire_positions,
                        "steps": d.steps_str, "rationale": d.rationale,
                    }
                    for d in (coll_result.saved + coll_result.infeasible)
                ],
                llm_reasoning_text=runner.aggregator_reasoning_text(
                    analysis_dir, label=method_label,
                ),
                compression_ratio=(
                    float(n_failures) / max(1, cap_audit["proposed"])
                    if cap_audit["proposed"] > 0 else 0.0
                ),
                extras={
                    "cap_audit": cap_audit,
                    "n_corridor_infeasible": n_infeasible,
                },
            )

        # ---- Fine-tune with replay buffer ---------------------------------
        if n_collected > 0:
            new_demos_dir = demo_dir / f"round_{rnd:03d}"
            rc = _finetune(
                cumulative_demo_dir=demo_dir, new_demo_dir=new_demos_dir,
                ckpt_dir=ckpt_dir, seed=seed,
                epochs=finetune_epochs, lr=lr, batch_size=batch_size,
                weight_decay=weight_decay, replay_mix=replay_mix,
                replay_mix_floor=replay_mix_floor,
            )
            if rc != 0:
                _info(method_tag, f"round {rnd}: finetune rc={rc} (continuing)")
            training_rounds_total += 1

        # ---- Heldout eval -------------------------------------------------
        post_dir = round_dir / "heldout_eval"
        rc = _upstream_p4._rollout(
            ckpt_dir, heldout_yaml, post_dir,
            seed=seed + rnd, max_steps=max_steps,
        )
        if rc != 0:
            raise RuntimeError(f"post heldout rollout failed (rc={rc})")
        post = _upstream_p4._read_sr(post_dir)

        history.append(_round_history_entry(
            rnd, demo_dir, extras_saved, budget,
            cap_audit["proposed"], cap_audit["capped"],
            n_collected, n_infeasible,
            corr_metrics, post,
            correction_yaml_rnd,
        ))
        _persist_curve(
            method_full_label, results_dir, history, budget, target_sr,
            demo_dir, ckpt_dir, run_shared_dir, heldout_yaml, run_id,
        )
        tlog.write_row(
            round_idx=rnd, demos_added_total=extras_saved,
            training_rounds_total=training_rounds_total,
            heldout_sr=post["success_rate"],
            finetune_epochs=(finetune_epochs if n_collected > 0 else 0),
            lr=lr, replay_mix=replay_mix,
        )
        _info(method_tag, f"round {rnd}: heldout_sr={post['success_rate']:.3f}")

        if post["success_rate"] >= target_sr:
            return {"history": history, "stopped_reason": "target_hit"}
        if n_collected == 0:
            return {"history": history, "stopped_reason": "no_new_demos"}
        if extras_saved >= budget:
            return {"history": history, "stopped_reason": "budget_exhausted"}

    return {"history": history, "stopped_reason": "max_rounds"}


def _summarize_failures_from_rollout(rollout_dir: Path) -> Dict[str, int]:
    """Coarse quadrant-bucket failure summary from a rollout's full_output.json.

    Returns an empty dict if the file is malformed — the LLM addendum
    handles the empty case with "n/a" text.
    """
    p = rollout_dir / "full_output.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    records = []
    for ep in (obj.get("episodes") or []):
        if ep.get("success"):
            continue
        records.append({"layout": {
            "start_pos": ep.get("start_pos", [0, 0]),
            "goal_pos": ep.get("goal_pos", [0, 0]),
        }})
    return prompts.summarize_failure_modes(records)


def _round_history_entry(
    rnd: int, demo_dir: Path, extras_saved: int, budget: int,
    n_prescribed: int, n_capped: int,
    n_collected: int, n_infeasible: int,
    corr_metrics: Dict, post: Dict,
    correction_yaml_rnd: Path,
) -> Dict:
    return {
        "round": int(rnd),
        "cum_demos": _upstream_p4._count_demos(demo_dir),
        "extra_demos": int(extras_saved),
        "budget_remaining": int(budget) - int(extras_saved),
        "heldout_sr": float(post["success_rate"]),
        "heldout_n_successes": int(post["n_successes"]),
        "heldout_n_episodes": int(post["n_episodes"]),
        "correction_sr": float(corr_metrics["success_rate"]),
        "n_prescribed_layouts": int(n_prescribed),
        "n_capped_by_budget": int(n_capped),
        "n_corridor_infeasible": int(n_infeasible),
        "n_new_demos": int(n_collected),
        "correction_yaml": str(correction_yaml_rnd),
    }


def _llm_failed_post(initial: Dict) -> Dict:
    """Synthesize a 'post' dict mirroring initial when the LLM call
    failed — keeps history rows uniform across the early-return path.
    """
    return {
        "success_rate": float(initial["success_rate"]),
        "n_successes": int(initial["n_successes"]),
        "n_episodes": int(initial["n_episodes"]),
    }
