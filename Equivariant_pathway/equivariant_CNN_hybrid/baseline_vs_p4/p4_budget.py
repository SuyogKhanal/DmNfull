"""Budget-constrained P4 loop for the hybrid pathway.

The standard hybrid p4_only pipeline lets the LLM decide how many
demonstrations to prescribe each round, with a soft "scale with pool
diversity" guideline. Here we bound the global cost so a comparison
against the budget-baseline is fair: across ALL rounds the LLM may
prescribe at most ``budget`` additional layouts. We tell it that
constraint up-front and refresh it every round with how much budget
is still available, then hard-cap the post-aggregator output to the
remaining budget. The LLM concentrating its budget across rounds is
exactly the behaviour we want to compare against the baseline's
"pick top-K by loss" rule.

Other than the budget addendum + post-hoc cap, the analyzer and
collect_demos calls are byte-identical to the existing
``equivariant_CNN_hybrid.p4_only`` pipeline.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway._analysis_common import run_profile_analysis

PROFILE_YAML = "p4_vlm_reasoning_kag_cross_plain_llm.yaml"
LABEL = "p4_only_budget_hybrid"
OUT_SUBDIR = "p4_analysis"

REASONING_ADDENDUM_BASE = (
    "SAMPLE-EFFICIENCY DIRECTIVE — pick the smallest n_demos that closes the failure mode,\n"
    "but never zero when a failure is present.\n"
    "HARD FLOOR: this episode IS a failure. n_demos for this episode must be >= 1."
)

AGGREGATOR_ADDENDUM_BASE = (
    "HOLISTIC SAMPLE-EFFICIENCY DIRECTIVE — minimise total layouts but NEVER zero, and\n"
    "scale recommendations with the diversity of the failure pool.\n"
    "1. Cluster the failures into the smallest set of distinct modes.\n"
    "2. Recommend the SMALLEST set of layouts per cluster that fixes the missing behaviour.\n"
    "3. HARD FLOOR: if n_failure_episodes >= 1, the response MUST contain at least one\n"
    "   cluster, one demonstration_prescription, one recommended_layout, and\n"
    "   total_demonstrations_needed >= 1.\n"
)


def _info(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [p4-budget] {msg}", flush=True)


def _section(title: str, char: str = "="):
    print(f"\n{char*80}\n{title}\n{char*80}", flush=True)


def _budget_addendum(budget_total: int, already_used: int) -> str:
    remaining = max(0, budget_total - already_used)
    return (
        "BUDGET CONSTRAINT (hard) — this comparison runs on a finite demo\n"
        f"budget of {budget_total} extra demonstrations on top of the initial 20.\n"
        f"  * demonstrations already prescribed in earlier rounds: {already_used}\n"
        f"  * demonstrations REMAINING for this round AND every future round: {remaining}\n"
        "Recommend AT MOST that many layouts here. The orchestrator will hard-\n"
        "cap your output to that count, so any layouts beyond it are wasted.\n"
        "If you can defer some learnings to a later round (after the model is\n"
        "retrained on what you prescribe now), prescribe FEWER now and revisit.\n"
        "If you choose to spend your entire remaining budget here, do so on the\n"
        "layouts that will close the most failure modes per demo."
    )


def _run_analysis(rollout_dir: Path, out_dir: Path, demo_dir: Path,
                  budget_total: int, already_used: int) -> Dict:
    extra: Dict = {
        "llm": {
            "prompt_addendum_reasoning":
                REASONING_ADDENDUM_BASE + "\n\n" + _budget_addendum(budget_total, already_used),
            "prompt_addendum_aggregator":
                AGGREGATOR_ADDENDUM_BASE + "\n\n" + _budget_addendum(budget_total, already_used),
        },
        "tkf": {"demo_dir": str(demo_dir)},
    }
    return run_profile_analysis(
        profile_yaml_name=PROFILE_YAML,
        rollout_dir=str(rollout_dir),
        out_subdir_name=OUT_SUBDIR,
        out_dir_override=str(out_dir),
        master_config_path=None,
        label=LABEL,
        extra_overrides=extra,
    )


def _flatten_recommended_layouts(analysis_dir: Path) -> Optional[Path]:
    rec_path = analysis_dir / "recommended_layouts.json"
    if rec_path.exists():
        return rec_path
    report_path = analysis_dir / f"{LABEL}_prescription_report.json"
    if not report_path.exists():
        return None
    report = json.load(open(report_path))
    flat: List[Dict] = []
    for pres in report.get("prescription", {}).get("demonstration_prescriptions", []) or []:
        cluster = pres.get("cluster", "?")
        for li, lay in enumerate(pres.get("recommended_layouts", []) or []):
            flat.append({
                "parent_demo_id": cluster, "layout_index": li,
                "repetition": 1,
                "n_repetitions": int(lay.get("n_repetitions", 1) or 1),
                "start_pos": list(lay.get("start_pos", [])),
                "goal_pos":  list(lay.get("goal_pos", [])),
                "fire_positions": [list(p) for p in lay.get("fire_positions", []) or []],
                "rationale": lay.get("rationale", ""),
            })
    with open(rec_path, "w") as f:
        json.dump({"layouts": flat, "n_layouts": len(flat)}, f, indent=2, default=str)
    return rec_path


def _cap_recommendations(rec_path: Path, remaining: int) -> Dict:
    """Hard-cap the flattened layouts list to ``remaining`` items.

    Returns a small audit dict so the curve can record how many the
    LLM proposed vs how many we ended up collecting.
    """
    payload = json.load(open(rec_path))
    layouts = payload.get("layouts", []) or []
    proposed = len(layouts)
    if remaining <= 0:
        kept: List[Dict] = []
    else:
        kept = layouts[:remaining]
    payload["layouts"] = kept
    payload["n_layouts"] = len(kept)
    payload["n_layouts_proposed"] = proposed
    payload["n_layouts_capped_by_budget"] = max(0, proposed - len(kept))
    with open(rec_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return {
        "proposed": proposed,
        "kept": len(kept),
        "capped": max(0, proposed - len(kept)),
    }


def _rollout(ckpt_dir: Path, layouts_yaml: Path, out_dir: Path,
             seed: int, max_steps: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.rollout_test",
        "--checkpoint", str(ckpt_dir / "best_hybrid_policy.pth"),
        "--layouts", str(layouts_yaml),
        "--out_dir", str(out_dir),
        "--seed", str(seed),
        "--max_steps", str(max_steps),
    ]
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _read_sr(rollout_dir: Path) -> Dict:
    full = json.load(open(rollout_dir / "full_output.json"))
    md = full.get("metadata", {}) or {}
    n = int(md.get("n_episodes", 0) or 0)
    ns = int(md.get("n_successes", 0) or 0)
    return {"n_episodes": n, "n_successes": ns,
            "success_rate": ns / n if n else 0.0}


def _collect_prescribed(rec_path: Path, round_demo_dir: Path, seed: int) -> int:
    pre = sum(1 for _ in round_demo_dir.rglob("*.json")) if round_demo_dir.exists() else 0
    round_demo_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "Equivariant_pathway.collect_demos",
        "--layouts_from", str(rec_path),
        "--demo_dir", str(round_demo_dir),
        "--seed", str(seed),
    ]
    _info("$ " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        _info(f"collect_demos rc={rc}")
    post = sum(1 for _ in round_demo_dir.rglob("*.json"))
    return post - pre


def _count_demos(demo_dir: Path) -> int:
    return sum(1 for _ in demo_dir.rglob("*.json")) if demo_dir.exists() else 0


def _retrain(demo_dir: Path, ckpt_dir: Path, seed: int, epochs: int,
             train_from_scratch: bool) -> int:
    cmd = [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.train",
        "--demo_dir", str(demo_dir),
        "--checkpoint_dir", str(ckpt_dir),
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    if not train_from_scratch:
        cmd.append("--resume")
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def run_p4_budget(
    run_index: int,
    p4_root: Path,
    shared_demo_dir: Path,
    shared_ckpt_dir: Path,
    correction_yaml: Path,
    heldout_yaml: Path,
    budget: int,
    target_sr: float,
    round_epochs: int,
    max_rounds: int,
    max_steps: int,
    seed: int,
    train_from_scratch: bool,
) -> Dict:
    demo_dir    = p4_root / "demos"
    ckpt_dir    = p4_root / "checkpoints"
    results_dir = p4_root / "results"
    for d in (demo_dir, ckpt_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Seed initial demos and checkpoint from the shared bootstrap.
    for src in sorted(shared_demo_dir.glob("*.json")):
        dst = demo_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    init_best = shared_ckpt_dir / "best_hybrid_policy.pth"
    init_last = shared_ckpt_dir / "last_hybrid_policy.pth"
    if init_best.exists():
        shutil.copy2(init_best, ckpt_dir / "best_hybrid_policy.pth")
    if init_last.exists():
        shutil.copy2(init_last, ckpt_dir / "last_hybrid_policy.pth")

    round0 = results_dir / "round_000_heldout_eval"
    rc = _rollout(ckpt_dir, heldout_yaml, round0, seed=seed, max_steps=max_steps)
    if rc != 0:
        raise RuntimeError(f"initial heldout rollout failed (rc={rc})")
    initial = _read_sr(round0)

    history: List[Dict] = [{
        "round": 0,
        "cum_demos": _count_demos(demo_dir),
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
    _persist_curve(results_dir, history, budget, target_sr, demo_dir, ckpt_dir,
                   correction_yaml, heldout_yaml, run_index)

    if initial["success_rate"] >= target_sr:
        return {"history": history, "stopped_reason": "initial>=target"}

    extras_saved = 0
    for rnd in range(1, max_rounds + 1):
        remaining = budget - extras_saved
        if remaining <= 0:
            _info(f"round {rnd}: budget exhausted ({extras_saved}/{budget}); stopping.")
            break

        _section(f"P4-BUDGET ROUND {rnd}  (remaining budget = {remaining})", char="-")

        round_dir = results_dir / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        corr_dir = round_dir / "correction_rollout"
        rc = _rollout(ckpt_dir, correction_yaml, corr_dir,
                      seed=seed + rnd * 1000, max_steps=max_steps)
        if rc != 0:
            raise RuntimeError(f"correction rollout failed (rc={rc})")
        corr_metrics = _read_sr(corr_dir)

        analysis_dir = round_dir / "p4_analysis"
        _run_analysis(corr_dir, analysis_dir, demo_dir,
                      budget_total=budget, already_used=extras_saved)

        rec_path = _flatten_recommended_layouts(analysis_dir)
        cap_audit = {"proposed": 0, "kept": 0, "capped": 0}
        n_new_demos = 0
        if rec_path is not None:
            cap_audit = _cap_recommendations(rec_path, remaining=remaining)
            if cap_audit["kept"] > 0:
                round_demo_dir = demo_dir / f"round_{rnd:03d}"
                n_new_demos = _collect_prescribed(
                    rec_path, round_demo_dir, seed=seed + 1000 + rnd,
                )
                # Saved on disk may exceed cap if the recommended_layouts
                # carry n_repetitions > 1; clip extras to budget here.
                if n_new_demos > (budget - extras_saved):
                    paths = sorted(round_demo_dir.rglob("*.json"))
                    over = n_new_demos - (budget - extras_saved)
                    for p in paths[-over:]:
                        try:
                            p.unlink()
                        except OSError:
                            pass
                    n_new_demos = sum(1 for _ in round_demo_dir.rglob("*.json"))
        extras_saved += n_new_demos
        _info(
            f"round {rnd}: proposed={cap_audit['proposed']} "
            f"kept_after_cap={cap_audit['kept']} "
            f"capped={cap_audit['capped']} "
            f"new_demos={n_new_demos} cum_extras={extras_saved}/{budget}"
        )

        if n_new_demos > 0:
            rc = _retrain(demo_dir, ckpt_dir, seed=seed, epochs=round_epochs,
                          train_from_scratch=train_from_scratch)
            if rc != 0:
                _info(f"round {rnd}: retrain rc={rc} (continuing)")

        post_dir = round_dir / "heldout_eval"
        rc = _rollout(ckpt_dir, heldout_yaml, post_dir,
                      seed=seed + rnd, max_steps=max_steps)
        if rc != 0:
            raise RuntimeError(f"post heldout rollout failed (rc={rc})")
        post = _read_sr(post_dir)
        history.append({
            "round": rnd,
            "cum_demos": _count_demos(demo_dir),
            "extra_demos": extras_saved,
            "budget_remaining": budget - extras_saved,
            "heldout_sr": post["success_rate"],
            "heldout_n_successes": post["n_successes"],
            "heldout_n_episodes": post["n_episodes"],
            "correction_sr": corr_metrics["success_rate"],
            "n_prescribed_layouts": cap_audit["proposed"],
            "n_capped_by_budget": cap_audit["capped"],
            "n_new_demos": n_new_demos,
        })
        _persist_curve(results_dir, history, budget, target_sr, demo_dir, ckpt_dir,
                       correction_yaml, heldout_yaml, run_index)
        _info(f"round {rnd}: heldout_sr={post['success_rate']:.3f}")

        if post["success_rate"] >= target_sr:
            return {"history": history, "stopped_reason": "target_hit"}
        if n_new_demos == 0:
            return {"history": history, "stopped_reason": "no_new_demos"}
        if extras_saved >= budget:
            return {"history": history, "stopped_reason": "budget_exhausted"}

    return {"history": history, "stopped_reason": "max_rounds"}


def _persist_curve(results_dir: Path, history: List[Dict],
                   budget: int, target_sr: float,
                   demo_dir: Path, ckpt_dir: Path,
                   correction_yaml: Path, heldout_yaml: Path,
                   run_index: int):
    out = {
        "method": "p4_only_budget_hybrid",
        "run_index": int(run_index),
        "budget": int(budget),
        "target_sr": float(target_sr),
        "demo_dir": str(demo_dir),
        "checkpoint_dir": str(ckpt_dir),
        "correction_yaml": str(correction_yaml),
        "heldout_yaml": str(heldout_yaml),
        "history": history,
    }
    with open(results_dir / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_index", type=int, required=True)
    p.add_argument("--p4_root", type=str, required=True)
    p.add_argument("--shared_demo_dir", type=str, required=True)
    p.add_argument("--shared_ckpt_dir", type=str, required=True)
    p.add_argument("--correction_yaml", type=str, required=True)
    p.add_argument("--heldout_yaml", type=str, required=True)
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--target_sr", type=float, default=0.90)
    p.add_argument("--round_epochs", type=int, default=500)
    p.add_argument("--max_rounds", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_from_scratch", type=lambda s: str(s).lower() in ("1","true","yes","on"),
                   default=True)
    args = p.parse_args()
    result = run_p4_budget(
        run_index=args.run_index,
        p4_root=Path(args.p4_root).resolve(),
        shared_demo_dir=Path(args.shared_demo_dir).resolve(),
        shared_ckpt_dir=Path(args.shared_ckpt_dir).resolve(),
        correction_yaml=Path(args.correction_yaml).resolve(),
        heldout_yaml=Path(args.heldout_yaml).resolve(),
        budget=args.budget,
        target_sr=args.target_sr,
        round_epochs=args.round_epochs,
        max_rounds=args.max_rounds,
        max_steps=args.max_steps,
        seed=args.seed,
        train_from_scratch=args.train_from_scratch,
    )
    print(f"[p4-budget] DONE stopped_reason={result.get('stopped_reason')}")
