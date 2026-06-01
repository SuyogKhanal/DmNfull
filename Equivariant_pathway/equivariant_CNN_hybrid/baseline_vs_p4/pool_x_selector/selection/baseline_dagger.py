"""Baseline DAgger driver for the sequential-batch suite.

Mirrors upstream ``baseline_budget.run_baseline_budget`` but with three
behavioural changes:

  1. Uses our replay-buffer fine-tuner instead of the cold-start trainer.
     Each round's update is a warm-start fine-tune over ``D_old`` ∪
     ``D_new`` with a configurable replay mix.
  2. Always picks exactly ONE failure per round — the highest-loss
     episode — via :func:`selection.rank.top_one`. This matches the user
     spec; upstream supports ``demos_per_round=1`` but we hard-code it
     here so callers can't accidentally change it.
  3. Writes a ``training_log.csv`` alongside ``learning_curve.json`` so
     plots can use ``training_rounds`` as an alternate x-axis.

Upstream helpers (``_rollout_with_loss``, ``_save_corrective_demo``,
``_load_model``, ``_eval_heldout``, ``_find_deviation_step``, etc.) are
imported as read-only utilities. We don't modify them.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import (  # noqa: E402
    baseline_budget as _upstream,
)

from ..layouts.layout_setup import ensure_correction_layouts_for_round
from ..logging_ext.training_log import TrainingLog
from .rank import rank_failures


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [baseline-dagger] {msg}", flush=True)


def _section(title: str, char: str = "-") -> None:
    print(f"\n{char * 80}\n{title}\n{char * 80}", flush=True)


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
    """Invoke the replay-buffer fine-tuner as a subprocess.

    Returns the subprocess return code so the caller can decide whether
    to abort the round.
    """
    cmd = [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4."
        "pool_x_selector.trainer.finetune_replay",
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
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def run(
    run_id: int,
    bo_root: Path,
    shared_demo_dir: Path,
    shared_ckpt_dir: Path,
    train_yaml: Path,
    heldout_yaml: Path,
    run_shared_dir: Path,
    correction_n: int,
    budget: int,
    target_sr: float,
    max_rounds: int,
    max_steps: int,
    seed: int,
    # fine-tune knobs (passed through to finetune_replay):
    finetune_epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    replay_mix: float,
    replay_mix_floor: float,
    fixed_pool_yaml: Optional[Path] = None,
    selection: str = "highest_loss",
    method_label: str = "baseline_dagger",
) -> Dict:
    """Execute the budget-constrained baseline DAgger loop for one method.

    Two pool modes:
      * ``fixed_pool_yaml`` given  -> use that single fixed pool EVERY round
        (sampled once per run; shared across methods for apples-to-apples).
        With a fixed pool, ``no_new_demos`` (all failures solved) is a
        legitimate terminal state.
      * ``fixed_pool_yaml`` None   -> rotate a fresh pool per round via
        ``ensure_correction_layouts_for_round`` (the original behaviour).

    Two selection modes:
      * ``selection="highest_loss"`` -> pick the top failure by per-step
        BCE loss (``selection.rank.rank_failures``).
      * ``selection="random"``       -> pick a uniformly random failure
        (seeded for reproducibility).
    Either way, exactly ONE corrective demo is saved per round.

    Writes:
      - bo_root/demos/round_NNN/...
      - bo_root/checkpoints/{best,last}_hybrid_policy.pth, training_log.json
      - bo_root/results/learning_curve.json (legacy schema, plus a
        ``correction_yaml`` field per history row)
      - bo_root/results/round_NNN/ranking.json
      - bo_root/results/training_log.csv
    """
    demo_dir = bo_root / "demos"
    ckpt_dir = bo_root / "checkpoints"
    results_dir = bo_root / "results"
    for d in (demo_dir, ckpt_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Seed initial demos and checkpoint from the shared bootstrap.
    for src in sorted(shared_demo_dir.glob("*.json")):
        dst = demo_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    for name in ("best_hybrid_policy.pth", "last_hybrid_policy.pth"):
        s = shared_ckpt_dir / name
        if s.exists():
            shutil.copy2(s, ckpt_dir / name)

    # Round 0 heldout eval (no correction pool yet — sampled per round below).
    initial = _upstream._eval_heldout(
        ckpt_dir, heldout_yaml, results_dir,
        tag="round_000", seed=seed, max_steps=max_steps,
    )

    history: List[Dict] = [{
        "round": 0,
        "cum_demos": _upstream._count_demos(demo_dir),
        "extra_demos": 0,
        "budget_remaining": budget,
        "heldout_sr": initial["success_rate"],
        "heldout_n_successes": initial["n_successes"],
        "heldout_n_episodes": initial["n_episodes"],
        "correction_sr": None,
        "n_failures_in_pool": None,
        "n_picked": 0,
        "n_new_demos": 0,
        "correction_yaml": None,
    }]
    _persist_curve(
        results_dir, history, budget, target_sr, demo_dir, ckpt_dir,
        run_shared_dir, heldout_yaml, run_id, method_label,
    )
    tlog = TrainingLog(results_dir / "training_log.csv")
    tlog.write_row(
        round_idx=0, demos_added_total=0, training_rounds_total=0,
        heldout_sr=initial["success_rate"], finetune_epochs=0,
        lr=lr, replay_mix=replay_mix,
    )

    if initial["success_rate"] >= target_sr:
        return {"history": history, "stopped_reason": "initial>=target"}

    extras_saved = 0
    training_rounds_total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for rnd in range(1, max_rounds + 1):
        remaining = budget - extras_saved
        if remaining <= 0:
            _info(f"round {rnd}: budget exhausted ({extras_saved}/{budget}); stopping.")
            break

        _section(f"BASELINE-DAGGER ROUND {rnd}  (remaining budget = {remaining})")

        # Pass 0: get this round's correction pool. Fixed mode reuses one
        # pool every round; rotation mode samples a fresh disjoint pool.
        if fixed_pool_yaml is not None:
            correction_yaml_rnd = fixed_pool_yaml
        else:
            correction_yaml_rnd = ensure_correction_layouts_for_round(
                run_id=run_id, round_idx=rnd, correction_n=correction_n,
                train_yaml=train_yaml, heldout_yaml=heldout_yaml,
                run_shared_dir=run_shared_dir,
            )
        layouts = _upstream._load_correction_layouts(correction_yaml_rnd)
        if not layouts:
            raise SystemExit(f"no correction layouts in {correction_yaml_rnd}")

        # Pass 1: rollout correction pool, score each episode.
        model = _upstream._load_model(ckpt_dir / "best_hybrid_policy.pth", device)
        round_records: List[Dict] = []
        n_success = 0
        for ep_idx, layout in enumerate(layouts):
            ep_seed = seed + rnd * 1000 + ep_idx
            rec = _upstream._rollout_with_loss(model, layout, ep_seed, max_steps, device)
            rec.update({"ep_idx": ep_idx, "layout": layout, "ep_seed": ep_seed})
            round_records.append(rec)
            if rec["success"]:
                n_success += 1

        failures = [r for r in round_records if not r["success"]]
        correction_sr = n_success / len(round_records) if round_records else 0.0
        _info(
            f"round {rnd}: pool={len(round_records)} successes={n_success} "
            f"failures={len(failures)} correction_sr={correction_sr:.3f}"
        )

        # Pass 2: order the failures by the selection rule.
        rng_seed = 0xBADF00D + 1009 * (int(run_id) + 1) + rnd
        if selection == "random":
            # Uniformly random order; same 4-tuple shape as rank_failures
            # so the downstream pick loop is identical. Loss fields are kept
            # in the record for the audit log.
            order = list(failures)
            random.Random(rng_seed).shuffle(order)
            ranked = [(0.0, 0.0, 0.0, rec) for rec in order]
        else:
            ranked = rank_failures(failures, rng_seed=rng_seed)

        round_dir = results_dir / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_demo_dir = demo_dir / f"round_{rnd:03d}"
        round_demo_dir.mkdir(parents=True, exist_ok=True)

        # Pass 3: emit exactly one corrective demo for the highest-priority
        # failure that we can produce a BFS path for.
        saved = 0
        picked_ranks: List[int] = []
        for i, (_loss, _steps, _jitter, rec) in enumerate(ranked):
            if saved >= 1:
                break
            dev = _upstream._find_deviation_step(
                rec["layout"], rec["actions"], rec["trajectory"],
            )
            if dev < 0:
                dev = 0
            out = _upstream._save_corrective_demo(
                rec["layout"], dev, rec["ep_seed"], rec["actions"],
                demo_dir=round_demo_dir,
                layout_name=rec["layout"].get("name", f"ep{rec['ep_idx']}"),
                suffix=f"correct_r{rnd}_ep{rec['ep_idx']}_dev{dev}",
            )
            if out is not None:
                picked_ranks.append(i)
                saved += 1
            else:
                _info(f"round {rnd}: skip ep{rec['ep_idx']} (no BFS path)")

        # Persist the full ranking for audit.
        with open(round_dir / "ranking.json", "w") as f:
            json.dump([
                {
                    "rank": i + 1,
                    "ep_idx": d[3]["ep_idx"],
                    "policy_loss": d[3]["policy_loss"],
                    "policy_loss_sum": d[3]["policy_loss_sum"],
                    "n_steps": d[3]["n_steps"],
                    "jitter": d[2],
                    "layout_name": d[3]["layout"].get("name"),
                    "picked": (i in picked_ranks),
                } for i, d in enumerate(ranked)
            ], f, indent=2)

        # Uniform info-gain proxy: the picked failure's pre-finetune loss.
        # Same schema as p4/pipeline_p4.py so the notebook reads ONE filename
        # across methods. (Picks list has 0 or 1 entries — baseline picks 1/rnd.)
        picks_payload: List[Dict] = []
        for i in picked_ranks:
            r = ranked[i][3]
            picks_payload.append({
                "layout": r["layout"],
                "policy_loss": r["policy_loss"],
                "policy_loss_sum": r["policy_loss_sum"],
                "n_steps": r["n_steps"],
                "ep_seed": r["ep_seed"],
                "success": r["success"],
            })
        if picks_payload:
            with open(round_dir / "prescribed_loss.json", "w") as f:
                json.dump({"round": rnd, "method": method_label, "picks": picks_payload},
                          f, indent=2, default=str)

        extras_saved += saved
        _info(f"round {rnd}: picked={len(picked_ranks)} saved={saved} "
              f"cum_extras={extras_saved}/{budget}")

        # Pass 4: replay-buffer fine-tune on cumulative pool, with the
        # round's new demos given extra weight via the sampler.
        if saved > 0:
            rc = _finetune(
                cumulative_demo_dir=demo_dir,
                new_demo_dir=round_demo_dir,
                ckpt_dir=ckpt_dir,
                seed=seed,
                epochs=finetune_epochs,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                replay_mix=replay_mix,
                replay_mix_floor=replay_mix_floor,
            )
            if rc != 0:
                _info(f"round {rnd}: finetune rc={rc} (continuing)")
            training_rounds_total += 1

        post = _upstream._eval_heldout(
            ckpt_dir, heldout_yaml, results_dir,
            tag=f"round_{rnd:03d}", seed=seed + rnd, max_steps=max_steps,
        )
        history.append({
            "round": rnd,
            "cum_demos": _upstream._count_demos(demo_dir),
            "extra_demos": extras_saved,
            "budget_remaining": budget - extras_saved,
            "heldout_sr": post["success_rate"],
            "heldout_n_successes": post["n_successes"],
            "heldout_n_episodes": post["n_episodes"],
            "correction_sr": correction_sr,
            "n_failures_in_pool": len(failures),
            "n_picked": len(picked_ranks),
            "n_new_demos": saved,
            "correction_yaml": str(correction_yaml_rnd),
        })
        _persist_curve(
            results_dir, history, budget, target_sr, demo_dir, ckpt_dir,
            run_shared_dir, heldout_yaml, run_id, method_label,
        )
        tlog.write_row(
            round_idx=rnd, demos_added_total=extras_saved,
            training_rounds_total=training_rounds_total,
            heldout_sr=post["success_rate"],
            finetune_epochs=finetune_epochs if saved > 0 else 0,
            lr=lr, replay_mix=replay_mix,
        )
        _info(f"round {rnd}: heldout_sr={post['success_rate']:.3f}")

        if post["success_rate"] >= target_sr:
            return {"history": history, "stopped_reason": "target_hit"}
        if saved == 0:
            return {"history": history, "stopped_reason": "no_new_demos"}
        if extras_saved >= budget:
            return {"history": history, "stopped_reason": "budget_exhausted"}

    return {"history": history, "stopped_reason": "max_rounds"}


def _persist_curve(
    results_dir: Path, history: List[Dict], budget: int, target_sr: float,
    demo_dir: Path, ckpt_dir: Path, run_shared_dir: Path, heldout_yaml: Path,
    run_id: int, method_label: str = "baseline_dagger",
) -> None:
    """Match upstream's learning_curve.json schema so existing
    aggregation tools keep working. The correction pool now rotates per
    round, so we record the shared *directory* (which contains
    ``round_NNN/correction_layouts.yaml`` subtrees) instead of a single
    yaml path. Each history row also carries its round's yaml path.
    """
    out = {
        "method": method_label,
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
