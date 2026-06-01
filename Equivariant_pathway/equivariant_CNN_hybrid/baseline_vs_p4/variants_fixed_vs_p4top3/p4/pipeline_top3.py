"""p4_top3 — P4 with a top-3 loss-informed prescription on a FIXED pool.

Each round, against the run's single fixed correction pool:
  1. roll out the pool WITH per-step loss (same primitive the baseline uses),
  2. rank the failures by loss and take the top-3,
  3. feed ONLY those 3 failures to the P4 LLM pipeline, which prescribes
     EXACTLY ONE corrective layout (hard-capped to 1),
  4. corridor-collect that demo, replay fine-tune, eval held-out.

Differences from p4_sequential: the pool is fixed (not rotated per round),
and the LLM sees only the 3 highest-loss failures rather than the whole
pool. Stopping: target_hit / budget_exhausted / pool_solved (no failures
left in the fixed pool) / no_progress (LLM gave only infeasible/empty
prescriptions for several rounds) / max_rounds. Infeasible corridors are
fed back into later rounds' prompts (same mechanism as the other P4 loop).
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import (  # noqa: E402
    baseline_budget as _bb,   # _load_model, _rollout_with_loss, _load_correction_layouts
    p4_budget as _pp,         # _rollout, _read_sr, _count_demos
)
from Equivariant_pathway.layout_sampler import write_yaml  # noqa: E402

from ..logging_ext.compression_log import CompressionLog
from ..logging_ext.training_log import TrainingLog
from ..selection.rank import rank_failures
from . import demo_collector, prompts, runner
from ._p4_common import _finetune, _persist_curve

TAG = "p4-top3"
LABEL = runner.LABEL_SEQ          # reuse the sequential profile/label
METHOD_FULL = "p4_top3"
TOP_K = 3


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{TAG}] {msg}", flush=True)


def run(
    *,
    run_id: int,
    method_root: Path,
    shared_demo_dir: Path,
    shared_ckpt_dir: Path,
    train_yaml: Path,
    heldout_yaml: Path,
    run_shared_dir: Path,
    correction_n: int,
    config: Dict,
    fixed_pool_yaml: Path,
) -> Dict:
    demo_dir = method_root / "demos"
    ckpt_dir = method_root / "checkpoints"
    results_dir = method_root / "results"
    for d in (demo_dir, ckpt_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Seed cumulative demo pool + initial checkpoint from per-run bootstrap.
    for src in sorted(shared_demo_dir.glob("*.json")):
        dst = demo_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    for name in ("best_hybrid_policy.pth", "last_hybrid_policy.pth"):
        s = shared_ckpt_dir / name
        if s.exists():
            shutil.copy2(s, ckpt_dir / name)

    budget = int(config.get("budget", 15))
    target_sr = float(config.get("target_sr", 0.90))
    max_rounds = int(config.get("max_rounds", 50))
    max_steps = int(config.get("max_steps", 60))
    seed = int(config.get("seed", 0)) + int(run_id)
    corridor_blocking = bool(config.get("corridor_blocking", True))
    tcfg = config.get("trainer", {}) or {}
    finetune_epochs = int(tcfg.get("finetune_epochs", 20))
    lr = float(tcfg.get("lr", 5e-5))
    batch_size = int(tcfg.get("batch_size", 64))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    replay_mix = float(tcfg.get("replay_mix", 0.5))
    replay_mix_floor = float(tcfg.get("replay_mix_floor", 0.2))
    max_consecutive_empty = int(config.get("max_consecutive_empty", 8))

    comp_log = CompressionLog(results_dir / "compression_log.csv")
    tlog = TrainingLog(results_dir / "training_log.csv")

    fixed_layouts = _bb._load_correction_layouts(fixed_pool_yaml)
    if not fixed_layouts:
        raise SystemExit(f"no layouts in fixed pool {fixed_pool_yaml}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Round 0 — held-out eval on the initial policy.
    round0 = results_dir / "round_000_heldout_eval"
    rc = _pp._rollout(ckpt_dir, heldout_yaml, round0, seed=seed, max_steps=max_steps)
    if rc != 0:
        raise RuntimeError(f"initial heldout rollout failed (rc={rc})")
    initial = _pp._read_sr(round0)

    history: List[Dict] = [{
        "round": 0,
        "cum_demos": _pp._count_demos(demo_dir),
        "extra_demos": 0,
        "budget_remaining": budget,
        "heldout_sr": initial["success_rate"],
        "heldout_n_successes": initial["n_successes"],
        "heldout_n_episodes": initial["n_episodes"],
        "correction_sr": None,
        "n_prescribed_layouts": 0,
        "n_new_demos": 0,
        "correction_yaml": str(fixed_pool_yaml),
    }]
    _persist_curve(METHOD_FULL, results_dir, history, budget, target_sr,
                   demo_dir, ckpt_dir, run_shared_dir, heldout_yaml, run_id)
    tlog.write_row(round_idx=0, demos_added_total=0, training_rounds_total=0,
                   heldout_sr=initial["success_rate"], finetune_epochs=0,
                   lr=lr, replay_mix=replay_mix)
    if initial["success_rate"] >= target_sr:
        return {"history": history, "stopped_reason": "initial>=target"}

    extras_saved = 0
    training_rounds_total = 0
    infeasible_memo: List[Dict] = []
    consecutive_empty = 0

    for rnd in range(1, max_rounds + 1):
        remaining = budget - extras_saved
        if remaining <= 0:
            break
        _info(f"ROUND {rnd}  (remaining budget = {remaining})")
        round_dir = results_dir / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 1. loss rollout over the FIXED pool
        model = _bb._load_model(ckpt_dir / "best_hybrid_policy.pth", device)
        records: List[Dict] = []
        n_success = 0
        for ep_idx, layout in enumerate(fixed_layouts):
            ep_seed = seed + rnd * 1000 + ep_idx
            rec = _bb._rollout_with_loss(model, layout, ep_seed, max_steps, device)
            rec.update({"ep_idx": ep_idx, "layout": layout, "ep_seed": ep_seed})
            records.append(rec)
            if rec["success"]:
                n_success += 1
        failures = [r for r in records if not r["success"]]
        correction_sr = n_success / len(records) if records else 0.0
        _info(f"round {rnd}: pool={len(records)} successes={n_success} "
              f"failures={len(failures)} correction_sr={correction_sr:.3f}")

        # 2. top-3 highest-loss failures
        ranked = rank_failures(failures, rng_seed=7_000_000 + int(run_id) * 1000 + rnd)
        top = [d[3] for d in ranked[:TOP_K]]

        proposed = kept = n_collected = n_infeasible = 0
        if top:
            # 3. write the top-3 layouts + P4-style rollout of just those
            top_layouts = [r["layout"] for r in top]
            top_yaml = round_dir / "top3_layouts.yaml"
            write_yaml(top_layouts, top_yaml, grid_size=5)
            top_roll = round_dir / "top3_rollout"
            rc = _pp._rollout(ckpt_dir, top_yaml, top_roll, seed=seed + rnd, max_steps=max_steps)
            if rc != 0:
                raise RuntimeError(f"top3 rollout failed (rc={rc})")

            # 4. P4 analysis — sequential directive caps the prescription to 1
            analysis_dir = round_dir / "p4_analysis"
            infeasible_fb = prompts.infeasible_feedback_block(infeasible_memo)
            common = dict(
                mode="sequential", round_idx=rnd, demos_added_this_round=0,
                demos_remaining_in_budget=remaining,
                n_failures_observed=len(top),
                budget_total=budget, already_used=extras_saved,
                infeasible_feedback=infeasible_fb,
            )
            try:
                runner.run_analysis(
                    rollout_dir=top_roll, out_dir=analysis_dir,
                    demo_dir=demo_dir, label=LABEL,
                    reasoning_addendum=prompts.reasoning_addendum(**common),
                    aggregator_addendum=prompts.aggregator_addendum(**common),
                )
            except Exception as exc:  # noqa: BLE001
                _info(f"round {rnd}: LLM analysis FAILED ({exc!r}); skipping round")
                comp_log.write_row(round_idx=rnd, n_failures_observed=len(failures),
                                   n_layouts_prescribed=0, n_demos_collected=0,
                                   n_corridor_infeasible=0)
                history.append(_row(rnd, demo_dir, extras_saved, budget, 0, 0,
                                    correction_sr, initial, fixed_pool_yaml))
                _persist_curve(METHOD_FULL, results_dir, history, budget, target_sr,
                               demo_dir, ckpt_dir, run_shared_dir, heldout_yaml, run_id)
                return {"history": history, "stopped_reason": "llm_error"}

            rec_path = runner.flatten_recommendations(analysis_dir, label=LABEL)
            if rec_path is not None:
                cap = runner.cap_layouts(rec_path, remaining=remaining, mode_max=1)
                proposed, kept = cap["proposed"], cap["kept"]
                if kept > 0:
                    rdd = demo_dir / f"round_{rnd:03d}"
                    rdd.mkdir(parents=True, exist_ok=True)
                    cr = demo_collector.collect(
                        rec_path=rec_path, demo_dir=rdd, seed=seed + 1000 + rnd,
                        corridor_blocking=corridor_blocking,
                    )
                    demo_collector.write_summary(cr, round_dir / "collect_summary.json")
                    n_collected, n_infeasible = cr.n_saved, cr.n_infeasible
                    for dd in cr.infeasible:
                        infeasible_memo.append({
                            "round": rnd, "start_pos": dd.start_pos,
                            "goal_pos": dd.goal_pos, "fire_positions": dd.fire_positions,
                            "steps": dd.steps_str, "reason": dd.reason,
                        })

        extras_saved += n_collected
        _info(f"round {rnd}: top3={len(top)} prescribed={proposed} kept={kept} "
              f"collected={n_collected} infeasible={n_infeasible} "
              f"cum_extras={extras_saved}/{budget}")
        comp_log.write_row(round_idx=rnd, n_failures_observed=len(failures),
                           n_layouts_prescribed=proposed, n_demos_collected=n_collected,
                           n_corridor_infeasible=n_infeasible)

        if n_collected > 0:
            rc = _finetune(
                cumulative_demo_dir=demo_dir, new_demo_dir=demo_dir / f"round_{rnd:03d}",
                ckpt_dir=ckpt_dir, seed=seed, epochs=finetune_epochs, lr=lr,
                batch_size=batch_size, weight_decay=weight_decay,
                replay_mix=replay_mix, replay_mix_floor=replay_mix_floor,
            )
            if rc != 0:
                _info(f"round {rnd}: finetune rc={rc} (continuing)")
            training_rounds_total += 1

        post_dir = round_dir / "heldout_eval"
        rc = _pp._rollout(ckpt_dir, heldout_yaml, post_dir, seed=seed + rnd, max_steps=max_steps)
        if rc != 0:
            raise RuntimeError(f"post heldout rollout failed (rc={rc})")
        post = _pp._read_sr(post_dir)

        history.append(_row(rnd, demo_dir, extras_saved, budget, proposed,
                            n_collected, correction_sr, post, fixed_pool_yaml,
                            n_infeasible=n_infeasible))
        _persist_curve(METHOD_FULL, results_dir, history, budget, target_sr,
                       demo_dir, ckpt_dir, run_shared_dir, heldout_yaml, run_id)
        tlog.write_row(round_idx=rnd, demos_added_total=extras_saved,
                       training_rounds_total=training_rounds_total,
                       heldout_sr=post["success_rate"],
                       finetune_epochs=(finetune_epochs if n_collected > 0 else 0),
                       lr=lr, replay_mix=replay_mix)
        _info(f"round {rnd}: heldout_sr={post['success_rate']:.3f}")

        if post["success_rate"] >= target_sr:
            return {"history": history, "stopped_reason": "target_hit"}
        if extras_saved >= budget:
            return {"history": history, "stopped_reason": "budget_exhausted"}
        if not failures:
            # Fixed pool fully solved — it can never produce new failures
            # (the policy only improves), so this is genuine saturation.
            return {"history": history, "stopped_reason": "pool_solved"}
        if n_collected == 0:
            # Failures remain but the prescription was infeasible/empty.
            # Keep going (the infeasible feedback should help) up to a cap.
            consecutive_empty += 1
            if consecutive_empty >= max_consecutive_empty:
                return {"history": history, "stopped_reason": "no_progress"}
        else:
            consecutive_empty = 0

    return {"history": history, "stopped_reason": "max_rounds"}


def _row(rnd, demo_dir, extras_saved, budget, proposed, n_collected,
         correction_sr, sr, fixed_pool_yaml, n_infeasible=0):
    return {
        "round": rnd,
        "cum_demos": _pp._count_demos(demo_dir),
        "extra_demos": extras_saved,
        "budget_remaining": budget - extras_saved,
        "heldout_sr": sr["success_rate"],
        "heldout_n_successes": sr["n_successes"],
        "heldout_n_episodes": sr["n_episodes"],
        "correction_sr": correction_sr,
        "n_prescribed_layouts": proposed,
        "n_corridor_infeasible": n_infeasible,
        "n_new_demos": n_collected,
        "correction_yaml": str(fixed_pool_yaml),
    }
